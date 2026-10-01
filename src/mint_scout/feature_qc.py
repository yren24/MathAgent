from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.data.manifest_io import load_configured_protein_ligand_records
from mint_scout.invariants.manifest import stable_hash
from mint_scout.invariants.plbind_tools import expected_feature_shape
from mint_scout.representation import RepresentationSpec


DEFAULT_FAIL_ON = (
    "duplicate_sample",
    "failed_generation",
    "invariant_mismatch",
    "manifest_sample_mismatch",
    "missing_file",
    "wrong_representation_hash",
    "shape_mismatch",
    "empty",
    "nonfinite",
)
DEFAULT_WARN_ON = (
    "all_zero",
    "constant",
    "high_sparsity",
    "large_magnitude",
    "robust_outlier",
    "many_all_zero_coordinates",
    "many_constant_coordinates",
)


@dataclass(frozen=True)
class FeatureQCConfig:
    fail_on: tuple[str, ...] = DEFAULT_FAIL_ON
    warn_on: tuple[str, ...] = DEFAULT_WARN_ON
    absolute_max: float = 1.0e12
    sparsity_warning_fraction: float = 0.999
    robust_z_warning: float = 12.0
    robust_min_samples: int = 20
    coordinate_zero_epsilon: float = 1.0e-12
    population_all_zero_warning_fraction: float = 0.95
    population_constant_warning_fraction: float = 0.99

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "FeatureQCConfig":
        raw = dict(data or {})
        if "enabled" in raw:
            raw.pop("enabled")
        return cls(
            fail_on=tuple(str(value) for value in raw.pop("fail_on", DEFAULT_FAIL_ON)),
            warn_on=tuple(str(value) for value in raw.pop("warn_on", DEFAULT_WARN_ON)),
            absolute_max=float(raw.pop("absolute_max", 1.0e12)),
            sparsity_warning_fraction=float(raw.pop("sparsity_warning_fraction", 0.999)),
            robust_z_warning=float(raw.pop("robust_z_warning", 12.0)),
            robust_min_samples=int(raw.pop("robust_min_samples", 20)),
            coordinate_zero_epsilon=float(raw.pop("coordinate_zero_epsilon", 1.0e-12)),
            population_all_zero_warning_fraction=float(
                raw.pop("population_all_zero_warning_fraction", 0.95)
            ),
            population_constant_warning_fraction=float(
                raw.pop("population_constant_warning_fraction", 0.99)
            ),
        )

    def __post_init__(self) -> None:
        if self.absolute_max <= 0:
            raise ValueError("feature_qc.absolute_max must be positive")
        if not 0 <= self.sparsity_warning_fraction <= 1:
            raise ValueError("feature_qc.sparsity_warning_fraction must be in [0, 1]")
        if self.robust_z_warning <= 0:
            raise ValueError("feature_qc.robust_z_warning must be positive")
        if self.robust_min_samples < 1:
            raise ValueError("feature_qc.robust_min_samples must be positive")
        if self.coordinate_zero_epsilon < 0:
            raise ValueError("feature_qc.coordinate_zero_epsilon must be non-negative")
        for name, value in (
            ("population_all_zero_warning_fraction", self.population_all_zero_warning_fraction),
            ("population_constant_warning_fraction", self.population_constant_warning_fraction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"feature_qc.{name} must be in [0, 1]")
        overlap = set(self.fail_on) & set(self.warn_on)
        if overlap:
            raise ValueError(f"feature_qc issues cannot be both errors and warnings: {sorted(overlap)}")


@dataclass(frozen=True)
class FeatureIssue:
    invariant: str
    sample_id: str | None
    issue: str
    severity: str
    message: str


@dataclass(frozen=True)
class FeatureStats:
    sample_id: str
    shape: tuple[int, ...]
    dtype: str
    size: int
    finite: bool
    nonzero_count: int
    zero_fraction: float
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    l2_norm: float | None
    max_abs: float | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QC frozen feature manifests before Scout modeling.")
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--probe-selection", type=Path, default=None)
    parser.add_argument("--sample-id-file", type=Path, default=None)
    parser.add_argument("--task-config", type=Path, default=None)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default=None
    )
    parser.add_argument(
        "--feature-manifest",
        action="append",
        required=True,
        metavar="INVARIANT=PATH",
    )
    parser.add_argument("--qc-config", type=Path, default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--evidence-scope", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    representation = RepresentationSpec.read(args.representation_spec)
    representation.assert_frozen()
    if args.task_config is not None:
        if args.probe_selection is not None or args.sample_id_file is not None:
            raise ValueError(
                "--task-config cannot be combined with --probe-selection or --sample-id-file"
            )
        if args.split is None:
            raise ValueError("--task-config requires --split")
        task = load_yaml(args.task_config)
        records = load_configured_protein_ligand_records(
            task, config_path=args.task_config, split=args.split
        )
        sample_ids = tuple(record.pdb_id for record in records)
        selection_hash = None
    else:
        if args.split is not None:
            raise ValueError("--split requires --task-config")
        sample_ids, selection_hash = _resolve_sample_ids(
            args.probe_selection, args.sample_id_file
        )
    config = _load_qc_config(args.qc_config)
    manifest_paths = _parse_manifest_args(args.feature_manifest)
    report = run_feature_qc(
        representation=representation,
        sample_ids=sample_ids,
        manifest_paths=manifest_paths,
        config=config,
        selection_hash=selection_hash,
        dataset_id=args.dataset_id,
        evidence_scope=args.evidence_scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"status={report['status']} invariants={','.join(sorted(manifest_paths))} "
        f"blocking={report['blocking_issue_count']} warnings={report['warning_issue_count']} output={args.output}"
    )
    return 0 if report["status"] != "FAIL" else 1


def run_feature_qc(
    *,
    representation: RepresentationSpec,
    sample_ids: tuple[str, ...],
    manifest_paths: Mapping[str, Path],
    config: FeatureQCConfig,
    selection_hash: str | None = None,
    dataset_id: str | None = None,
    evidence_scope: str | None = None,
) -> dict[str, Any]:
    if not sample_ids:
        raise ValueError("sample_ids cannot be empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    invariant_reports: dict[str, Any] = {}
    all_issues: list[FeatureIssue] = []
    for invariant, path in manifest_paths.items():
        invariant_name = invariant.upper()
        expected = expected_feature_shape(invariant_name, representation)
        stats, issues, population = _qc_manifest(
            path=path,
            invariant=invariant_name,
            sample_ids=sample_ids,
            representation=representation,
            expected_shape=expected,
            config=config,
        )
        issues = issues + _population_issues(
            invariant_name, stats, population, config
        )
        all_issues.extend(issues)
        invariant_reports[invariant_name] = _invariant_report(
            path=path,
            expected_shape=expected,
            stats=stats,
            issues=issues,
            population=population,
        )

    blocking = [issue for issue in all_issues if issue.severity == "error"]
    warnings = [issue for issue in all_issues if issue.severity == "warning"]
    information = [issue for issue in all_issues if issue.severity == "info"]
    status = "FAIL" if blocking else ("WARN" if warnings else "PASS")
    payload = {
        "report_schema": "mint-agent.feature-qc.v1",
        "status": status,
        "dataset_id": dataset_id,
        "evidence_scope": evidence_scope,
        "representation_hash": representation.spec_hash,
        "selection_hash": selection_hash,
        "sample_count": len(sample_ids),
        "sample_ids": list(sample_ids),
        "feature_manifests": {
            str(name).upper(): str(path) for name, path in manifest_paths.items()
        },
        "policy": asdict(config),
        "invariants": invariant_reports,
        "blocking_issue_count": len(blocking),
        "warning_issue_count": len(warnings),
        "information_issue_count": len(information),
        "issues": [asdict(issue) for issue in all_issues],
    }
    payload["qc_hash"] = stable_hash(
        {
            "representation_hash": representation.spec_hash,
            "selection_hash": selection_hash,
            "dataset_id": dataset_id,
            "evidence_scope": evidence_scope,
            "sample_ids": sample_ids,
            "feature_manifests": payload["feature_manifests"],
            "policy": payload["policy"],
            "invariants": {
                name: {
                    "file_hashes": report["file_hashes"],
                    "status": report["status"],
                    "aggregate": report["aggregate"],
                }
                for name, report in invariant_reports.items()
            },
        }
    )
    return payload


def assert_qc_report_compatible(
    *,
    path: Path,
    representation: RepresentationSpec,
    sample_ids: tuple[str, ...],
    manifest_paths: Mapping[str, Path],
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("report_schema") != "mint-agent.feature-qc.v1":
        raise ValueError(f"Unsupported feature QC report schema in {path}")
    if report.get("status") == "FAIL":
        raise ValueError(f"Feature QC failed; refusing to run Scout: {path}")
    if report.get("representation_hash") != representation.spec_hash:
        raise ValueError("Feature QC report representation hash does not match")
    if tuple(str(value) for value in report.get("sample_ids", ())) != sample_ids:
        raise ValueError("Feature QC report sample IDs do not match")
    recorded = {str(key).upper(): str(value) for key, value in report.get("feature_manifests", {}).items()}
    expected = {str(key).upper(): str(value) for key, value in manifest_paths.items()}
    if any(recorded.get(name) != manifest_path for name, manifest_path in expected.items()):
        raise ValueError("Feature QC report feature manifests do not match")
    return report


def _qc_manifest(
    *,
    path: Path,
    invariant: str,
    sample_ids: tuple[str, ...],
    representation: RepresentationSpec,
    expected_shape: tuple[int, ...],
    config: FeatureQCConfig,
) -> tuple[list[FeatureStats], list[FeatureIssue], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id: dict[str, dict[str, Any]] = {}
    issues: list[FeatureIssue] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id in by_id:
            issues.append(_issue(invariant, sample_id, "duplicate_sample", config, f"Duplicate sample in {path}"))
            continue
        by_id[sample_id] = row
    if set(by_id) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(by_id))
        extra = sorted(set(by_id) - set(sample_ids))
        issues.append(
            _issue(
                invariant,
                None,
                "manifest_sample_mismatch",
                config,
                f"Manifest sample IDs differ from expected set; missing={missing[:5]} extra={extra[:5]}",
            )
        )

    stats: list[FeatureStats] = []
    coordinate_sum = np.zeros(expected_shape, dtype=np.float64)
    coordinate_sum_square = np.zeros(expected_shape, dtype=np.float64)
    coordinate_nonzero_count = np.zeros(expected_shape, dtype=np.int64)
    population_sample_count = 0
    for sample_id in sample_ids:
        row = by_id.get(sample_id)
        if row is None:
            continue
        row_stats, row_issues, array = _qc_row(
            row=row,
            invariant=invariant,
            sample_id=sample_id,
            representation_hash=representation.spec_hash,
            expected_shape=expected_shape,
            config=config,
        )
        stats.extend(row_stats)
        issues.extend(row_issues)
        if array is not None and tuple(array.shape) == expected_shape and np.isfinite(array).all():
            values = np.asarray(array, dtype=np.float64)
            coordinate_sum += values
            coordinate_sum_square += values * values
            coordinate_nonzero_count += np.abs(values) > config.coordinate_zero_epsilon
            population_sample_count += 1
    population = _coordinate_population_summary(
        sample_count=population_sample_count,
        coordinate_sum=coordinate_sum,
        coordinate_sum_square=coordinate_sum_square,
        coordinate_nonzero_count=coordinate_nonzero_count,
        config=config,
    )
    return stats, issues, population


def _qc_row(
    *,
    row: Mapping[str, Any],
    invariant: str,
    sample_id: str,
    representation_hash: str,
    expected_shape: tuple[int, ...],
    config: FeatureQCConfig,
) -> tuple[list[FeatureStats], list[FeatureIssue], np.ndarray | None]:
    issues: list[FeatureIssue] = []
    if str(row.get("invariant", "")).upper() != invariant:
        issues.append(_issue(invariant, sample_id, "invariant_mismatch", config, "Manifest invariant mismatch"))
    if row.get("status") not in {"computed", "cached"}:
        issues.append(_issue(invariant, sample_id, "failed_generation", config, "Feature generation did not succeed"))
        return [], issues, None
    feature_path = Path(str(row.get("output_path", "")))
    if not feature_path.exists():
        issues.append(_issue(invariant, sample_id, "missing_file", config, f"Missing feature file: {feature_path}"))
        return [], issues, None
    if f"/repr-{representation_hash}/" not in feature_path.as_posix():
        issues.append(
            _issue(
                invariant,
                sample_id,
                "wrong_representation_hash",
                config,
                f"Feature path does not contain repr-{representation_hash}",
            )
        )
    try:
        array = np.load(feature_path, allow_pickle=False)
    except Exception as exc:
        issues.append(_issue(invariant, sample_id, "missing_file", config, f"Could not load feature file: {exc}"))
        return [], issues, None
    shape = tuple(int(dim) for dim in array.shape)
    if shape != expected_shape:
        issues.append(
            _issue(
                invariant,
                sample_id,
                "shape_mismatch",
                config,
                f"Expected shape {expected_shape}, got {shape}",
            )
        )
    if array.size == 0:
        issues.append(_issue(invariant, sample_id, "empty", config, "Feature array is empty"))
        return [_empty_stats(sample_id, shape, str(array.dtype))], issues, None
    finite = bool(np.isfinite(array).all())
    if not finite:
        issues.append(_issue(invariant, sample_id, "nonfinite", config, "Feature contains NaN or Inf"))
        return [_nonfinite_stats(sample_id, shape, str(array.dtype), array)], issues, None
    nonzero_count = int(np.count_nonzero(array))
    zero_fraction = 1.0 - (nonzero_count / float(array.size))
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    std = float(np.std(array))
    max_abs = float(np.max(np.abs(array)))
    stats = FeatureStats(
        sample_id=sample_id,
        shape=shape,
        dtype=str(array.dtype),
        size=int(array.size),
        finite=True,
        nonzero_count=nonzero_count,
        zero_fraction=zero_fraction,
        minimum=minimum,
        maximum=maximum,
        mean=float(np.mean(array)),
        std=std,
        l2_norm=float(np.linalg.norm(array.ravel())),
        max_abs=max_abs,
    )
    if nonzero_count == 0:
        issues.append(_issue(invariant, sample_id, "all_zero", config, "Feature is entirely zero"))
    if std == 0.0:
        issues.append(_issue(invariant, sample_id, "constant", config, "Feature has zero within-sample variance"))
    if zero_fraction >= config.sparsity_warning_fraction:
        issues.append(
            _issue(
                invariant,
                sample_id,
                "high_sparsity",
                config,
                f"Zero fraction {zero_fraction:.6g} exceeds {config.sparsity_warning_fraction:.6g}",
            )
        )
    if max_abs > config.absolute_max:
        issues.append(
            _issue(
                invariant,
                sample_id,
                "large_magnitude",
                config,
                f"max(abs(feature)) {max_abs:.6g} exceeds {config.absolute_max:.6g}",
            )
        )
    return [stats], issues, array


def _population_issues(
    invariant: str,
    stats: list[FeatureStats],
    population: Mapping[str, Any],
    config: FeatureQCConfig,
) -> list[FeatureIssue]:
    issues: list[FeatureIssue] = []
    if int(population.get("valid_sample_count") or 0) >= config.robust_min_samples:
        all_zero_fraction = float(population["all_zero_coordinate_fraction"])
        constant_fraction = float(population["constant_coordinate_fraction"])
        if all_zero_fraction >= config.population_all_zero_warning_fraction:
            issues.append(
                _issue(
                    invariant,
                    None,
                    "many_all_zero_coordinates",
                    config,
                    f"All-zero coordinate fraction {all_zero_fraction:.6g} exceeds "
                    f"{config.population_all_zero_warning_fraction:.6g}",
                )
            )
        if constant_fraction >= config.population_constant_warning_fraction:
            issues.append(
                _issue(
                    invariant,
                    None,
                    "many_constant_coordinates",
                    config,
                    f"Constant coordinate fraction {constant_fraction:.6g} exceeds "
                    f"{config.population_constant_warning_fraction:.6g}",
                )
            )
    if len(stats) < config.robust_min_samples:
        return issues
    norms = np.asarray([item.l2_norm for item in stats if item.l2_norm is not None], dtype=float)
    if norms.size < config.robust_min_samples:
        return issues
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    if mad == 0.0:
        return issues
    robust_z = 0.6745 * (norms - median) / mad
    for item, value in zip(stats, robust_z):
        if abs(float(value)) > config.robust_z_warning:
            issues.append(
                _issue(
                    invariant,
                    item.sample_id,
                    "robust_outlier",
                    config,
                    f"L2 norm robust z-score {float(value):.6g} exceeds {config.robust_z_warning:.6g}",
                )
            )
    return issues


def _invariant_report(
    *,
    path: Path,
    expected_shape: tuple[int, ...],
    stats: list[FeatureStats],
    issues: list[FeatureIssue],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    blocking = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    max_abs_values = [item.max_abs for item in stats if item.max_abs is not None]
    zero_fractions = [item.zero_fraction for item in stats]
    l2_norms = [item.l2_norm for item in stats if item.l2_norm is not None]
    return {
        "status": "FAIL" if blocking else ("WARN" if warnings else "PASS"),
        "manifest": str(path),
        "expected_shape": list(expected_shape),
        "loaded_sample_count": len(stats),
        "blocking_issue_count": len(blocking),
        "warning_issue_count": len(warnings),
        "aggregate": {
            "max_abs_max": _safe_max(max_abs_values),
            "zero_fraction_mean": _safe_mean(zero_fractions),
            "l2_norm_median": _safe_median(l2_norms),
            "l2_norm_mad": _safe_mad(l2_norms),
            "coordinate_population": dict(population),
        },
        "file_hashes": {
            item.sample_id: stable_hash(
                {
                    "sample_id": item.sample_id,
                    "shape": item.shape,
                    "dtype": item.dtype,
                    "nonzero_count": item.nonzero_count,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "mean": item.mean,
                    "std": item.std,
                    "l2_norm": item.l2_norm,
                    "max_abs": item.max_abs,
                }
            )
            for item in stats
        },
    }


def _issue(
    invariant: str,
    sample_id: str | None,
    issue: str,
    config: FeatureQCConfig,
    message: str,
) -> FeatureIssue:
    if issue in set(config.fail_on):
        severity = "error"
    elif issue in set(config.warn_on):
        severity = "warning"
    else:
        severity = "info"
    return FeatureIssue(invariant=invariant, sample_id=sample_id, issue=issue, severity=severity, message=message)


def _coordinate_population_summary(
    *,
    sample_count: int,
    coordinate_sum: np.ndarray,
    coordinate_sum_square: np.ndarray,
    coordinate_nonzero_count: np.ndarray,
    config: FeatureQCConfig,
) -> dict[str, Any]:
    coordinate_count = int(coordinate_sum.size)
    if sample_count == 0:
        return {
            "valid_sample_count": 0,
            "coordinate_count": coordinate_count,
            "all_zero_coordinate_count": None,
            "all_zero_coordinate_fraction": None,
            "constant_coordinate_count": None,
            "constant_coordinate_fraction": None,
            "coordinate_zero_epsilon": config.coordinate_zero_epsilon,
        }
    means = coordinate_sum / float(sample_count)
    variances = np.maximum(
        0.0, coordinate_sum_square / float(sample_count) - means * means
    )
    all_zero_count = int(np.count_nonzero(coordinate_nonzero_count == 0))
    constant_count = int(
        np.count_nonzero(np.sqrt(variances) <= config.coordinate_zero_epsilon)
    )
    return {
        "valid_sample_count": sample_count,
        "coordinate_count": coordinate_count,
        "all_zero_coordinate_count": all_zero_count,
        "all_zero_coordinate_fraction": all_zero_count / float(coordinate_count),
        "constant_coordinate_count": constant_count,
        "constant_coordinate_fraction": constant_count / float(coordinate_count),
        "coordinate_zero_epsilon": config.coordinate_zero_epsilon,
    }


def _empty_stats(sample_id: str, shape: tuple[int, ...], dtype: str) -> FeatureStats:
    return FeatureStats(sample_id, shape, dtype, 0, True, 0, 1.0, None, None, None, None, None, None)


def _nonfinite_stats(sample_id: str, shape: tuple[int, ...], dtype: str, array: np.ndarray) -> FeatureStats:
    finite_values = array[np.isfinite(array)]
    return FeatureStats(
        sample_id=sample_id,
        shape=shape,
        dtype=dtype,
        size=int(array.size),
        finite=False,
        nonzero_count=int(np.count_nonzero(array)),
        zero_fraction=1.0 - (int(np.count_nonzero(array)) / float(array.size)) if array.size else 1.0,
        minimum=float(np.min(finite_values)) if finite_values.size else None,
        maximum=float(np.max(finite_values)) if finite_values.size else None,
        mean=float(np.mean(finite_values)) if finite_values.size else None,
        std=float(np.std(finite_values)) if finite_values.size else None,
        l2_norm=None,
        max_abs=float(np.max(np.abs(finite_values))) if finite_values.size else None,
    )


def _resolve_sample_ids(probe_selection: Path | None, sample_id_file: Path | None) -> tuple[tuple[str, ...], str | None]:
    if probe_selection and sample_id_file:
        raise ValueError("Use either --probe-selection or --sample-id-file, not both")
    if probe_selection is None and sample_id_file is None:
        raise ValueError("One of --probe-selection or --sample-id-file is required")
    path = probe_selection or sample_id_file
    assert path is not None
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        rows: list[Any] = []
        try:
            rows = [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            rows = []
        if rows and all(isinstance(row, Mapping) and row.get("sample_id") for row in rows):
            sample_ids = tuple(str(row["sample_id"]) for row in rows)
        else:
            sample_ids = tuple(lines)
        return sample_ids, None
    if isinstance(payload, dict):
        if probe_selection is not None:
            values = payload.get("probe_sample_ids")
            selection_hash = payload.get("selection_hash")
        else:
            values = payload.get("sample_ids", payload.get("modeling_sample_ids"))
            selection_hash = None
    else:
        values = payload
        selection_hash = None
    if not isinstance(values, list):
        expected = "probe_sample_ids" if probe_selection is not None else "sample_ids or modeling_sample_ids"
        raise ValueError(f"sample-id JSON must contain a {expected} list")
    return tuple(str(value) for value in values), str(selection_hash) if selection_hash else None


def _parse_manifest_args(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Feature manifest must use INVARIANT=PATH: {value!r}")
        invariant, raw_path = value.split("=", 1)
        name = invariant.upper()
        if name in result:
            raise ValueError(f"Duplicate feature manifest for {name}")
        result[name] = Path(raw_path)
    return result


def _load_qc_config(path: Path | None) -> FeatureQCConfig:
    if path is None:
        return FeatureQCConfig()
    raw = load_yaml(path)
    data = raw.get("feature_qc", raw)
    if not isinstance(data, Mapping):
        raise ValueError("feature_qc config must be a mapping")
    return FeatureQCConfig.from_mapping(data)


def _safe_max(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def _safe_median(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.median(clean)) if clean else None


def _safe_mad(values: list[float | None]) -> float | None:
    clean = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if clean.size == 0:
        return None
    median = float(np.median(clean))
    return float(np.median(np.abs(clean - median)))


if __name__ == "__main__":
    raise SystemExit(main())
