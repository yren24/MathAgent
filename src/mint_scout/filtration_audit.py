from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mint_scout.config import load_yaml
from mint_scout.feature_qc import _parse_manifest_args, _resolve_sample_ids
from mint_scout.invariants.manifest import stable_hash
from mint_scout.invariants.plbind_tools import expected_feature_shape
from mint_scout.representation import RepresentationSpec


@dataclass(frozen=True)
class FiltrationAuditConfig:
    zero_epsilon: float = 1.0e-12
    tail_fraction: float = 0.20
    min_tail_points: int = 5
    sparse_zero_fraction: float = 0.995
    saturation_relative_l2: float = 0.005
    boundary_relative_l2: float = 0.05
    boundary_window: int = 3

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "FiltrationAuditConfig":
        raw = dict(data or {})
        raw.pop("enabled", None)
        return cls(**raw)

    def __post_init__(self) -> None:
        if self.zero_epsilon < 0.0:
            raise ValueError("filtration_audit.zero_epsilon must be non-negative")
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("filtration_audit.tail_fraction must be in (0, 1]")
        if self.min_tail_points < 2:
            raise ValueError("filtration_audit.min_tail_points must be at least 2")
        if not 0.0 <= self.sparse_zero_fraction <= 1.0:
            raise ValueError("filtration_audit.sparse_zero_fraction must be in [0, 1]")
        if self.saturation_relative_l2 < 0.0 or self.boundary_relative_l2 < 0.0:
            raise ValueError("filtration_audit change thresholds must be non-negative")
        if self.boundary_window < 1:
            raise ValueError("filtration_audit.boundary_window must be positive")


@dataclass
class _AxisAccumulator:
    value_count: np.ndarray
    nonzero_count: np.ndarray
    active_channel_count: np.ndarray
    channel_count: np.ndarray
    sum_abs: np.ndarray
    sum_value: np.ndarray
    sum_square: np.ndarray
    transition_diff_square: np.ndarray
    transition_base_square: np.ndarray
    sample_count: int = 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit feature behavior along a frozen filtration axis.")
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--probe-selection", type=Path, default=None)
    parser.add_argument("--sample-id-file", type=Path, default=None)
    parser.add_argument("--feature-manifest", action="append", required=True, metavar="INVARIANT=PATH")
    parser.add_argument("--audit-config", type=Path, default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--evidence-scope", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    representation = RepresentationSpec.read(args.representation_spec)
    representation.assert_frozen()
    sample_ids, selection_hash = _resolve_sample_ids(args.probe_selection, args.sample_id_file)
    config = _load_config(args.audit_config)
    manifests = _parse_manifest_args(args.feature_manifest)
    report = run_filtration_audit(
        representation=representation,
        sample_ids=sample_ids,
        manifest_paths=manifests,
        config=config,
        selection_hash=selection_hash,
        dataset_id=args.dataset_id,
        evidence_scope=args.evidence_scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    recommendations = ",".join(
        f"{name}:{item['recommendation']}" for name, item in sorted(report["invariants"].items())
    )
    print(f"status={report['status']} recommendations={recommendations} output={args.output}")
    return 1 if report["status"] == "FAIL" else 0


def run_filtration_audit(
    *,
    representation: RepresentationSpec,
    sample_ids: tuple[str, ...],
    manifest_paths: Mapping[str, Path],
    config: FiltrationAuditConfig = FiltrationAuditConfig(),
    selection_hash: str | None = None,
    dataset_id: str | None = None,
    evidence_scope: str | None = None,
) -> dict[str, Any]:
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be nonempty and unique")
    invariant_reports: dict[str, dict[str, Any]] = {}
    for raw_name, path in sorted(manifest_paths.items()):
        name = raw_name.upper()
        if name not in representation.filtration_profiles:
            raise ValueError(f"Representation has no filtration profile for {name}")
        invariant_reports[name] = _audit_manifest(
            invariant=name,
            path=path,
            sample_ids=sample_ids,
            representation=representation,
            config=config,
        )
    statuses = {item["status"] for item in invariant_reports.values()}
    status = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    payload: dict[str, Any] = {
        "report_schema": "mint-agent.filtration-axis-audit.v1",
        "status": status,
        "dataset_id": dataset_id,
        "evidence_scope": evidence_scope,
        "representation_hash": representation.spec_hash,
        "selection_hash": selection_hash,
        "sample_count": len(sample_ids),
        "sample_ids": list(sample_ids),
        "policy": asdict(config),
        "feature_manifests": {name.upper(): str(path) for name, path in sorted(manifest_paths.items())},
        "invariants": invariant_reports,
    }
    payload["audit_hash"] = stable_hash(
        {
            "representation_hash": representation.spec_hash,
            "selection_hash": selection_hash,
            "dataset_id": dataset_id,
            "evidence_scope": evidence_scope,
            "sample_ids": sample_ids,
            "policy": payload["policy"],
            "invariants": invariant_reports,
        }
    )
    return payload


def assert_filtration_audit_compatible(
    *,
    path: Path,
    representation: RepresentationSpec,
    sample_ids: tuple[str, ...],
    manifest_paths: Mapping[str, Path],
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("report_schema") != "mint-agent.filtration-axis-audit.v1":
        raise ValueError(f"Unsupported filtration audit report schema in {path}")
    if report.get("status") == "FAIL":
        raise ValueError(f"Filtration audit failed; refusing to run Scout: {path}")
    if report.get("representation_hash") != representation.spec_hash:
        raise ValueError("Filtration audit representation hash does not match")
    if tuple(str(value) for value in report.get("sample_ids", ())) != sample_ids:
        raise ValueError("Filtration audit sample IDs do not match")
    recorded = {str(key).upper(): str(value) for key, value in report.get("feature_manifests", {}).items()}
    expected = {str(key).upper(): str(value) for key, value in manifest_paths.items()}
    if any(recorded.get(name) != manifest_path for name, manifest_path in expected.items()):
        raise ValueError("Filtration audit feature manifests do not match")
    return report


def _audit_manifest(
    *,
    invariant: str,
    path: Path,
    sample_ids: tuple[str, ...],
    representation: RepresentationSpec,
    config: FiltrationAuditConfig,
) -> dict[str, Any]:
    profile = representation.filtration_profiles[invariant]
    expected_shape = expected_feature_shape(invariant, representation)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row.get("sample_id", "")): row for row in rows}
    errors: list[str] = []
    if len(by_id) != len(rows):
        errors.append("duplicate sample IDs in manifest")
    if set(by_id) != set(sample_ids):
        errors.append("manifest sample IDs do not match the requested sample set")
    accumulator = _make_accumulator(profile.num_points)
    for sample_id in sample_ids:
        row = by_id.get(sample_id)
        if row is None:
            continue
        if row.get("status") not in {"computed", "cached"}:
            errors.append(f"{sample_id}: generation status is {row.get('status')!r}")
            continue
        feature_path = Path(str(row.get("output_path", "")))
        if f"/repr-{representation.spec_hash}/" not in feature_path.as_posix():
            errors.append(f"{sample_id}: feature path has the wrong representation hash")
            continue
        try:
            array = np.load(feature_path, allow_pickle=False)
        except Exception as exc:
            errors.append(f"{sample_id}: cannot load feature: {exc}")
            continue
        if tuple(array.shape) != expected_shape:
            errors.append(f"{sample_id}: expected shape {expected_shape}, got {tuple(array.shape)}")
            continue
        if not np.isfinite(array).all():
            errors.append(f"{sample_id}: feature contains NaN or Inf")
            continue
        _update_accumulator(accumulator, np.asarray(array, dtype=np.float64), config.zero_epsilon)

    if errors or accumulator.sample_count != len(sample_ids):
        return {
            "status": "FAIL",
            "recommendation": "REVIEW",
            "reason_codes": ["INVALID_FEATURE_INPUT"],
            "errors": errors[:50],
            "loaded_sample_count": accumulator.sample_count,
            "expected_shape": list(expected_shape),
            "scale_kind": profile.scale_kind,
        }
    return _summarize_axis(invariant, accumulator, profile, expected_shape, config)


def _make_accumulator(num_points: int) -> _AxisAccumulator:
    zeros = lambda size: np.zeros(size, dtype=np.float64)
    return _AxisAccumulator(
        value_count=zeros(num_points),
        nonzero_count=zeros(num_points),
        active_channel_count=zeros(num_points),
        channel_count=zeros(num_points),
        sum_abs=zeros(num_points),
        sum_value=zeros(num_points),
        sum_square=zeros(num_points),
        transition_diff_square=zeros(max(0, num_points - 1)),
        transition_base_square=zeros(max(0, num_points - 1)),
    )


def _update_accumulator(acc: _AxisAccumulator, array: np.ndarray, epsilon: float) -> None:
    flattened = array.reshape(array.shape[0], -1)
    active_channels = np.any(np.abs(array) > epsilon, axis=tuple(range(2, array.ndim)))
    acc.value_count += flattened.shape[1]
    acc.nonzero_count += np.count_nonzero(np.abs(flattened) > epsilon, axis=1)
    acc.active_channel_count += np.count_nonzero(active_channels, axis=1)
    acc.channel_count += active_channels.shape[1]
    acc.sum_abs += np.sum(np.abs(flattened), axis=1)
    acc.sum_value += np.sum(flattened, axis=1)
    acc.sum_square += np.sum(flattened * flattened, axis=1)
    if flattened.shape[0] > 1:
        differences = np.diff(flattened, axis=0)
        acc.transition_diff_square += np.sum(differences * differences, axis=1)
        acc.transition_base_square += np.sum(flattened[:-1] * flattened[:-1], axis=1)
    acc.sample_count += 1


def _summarize_axis(invariant, acc, profile, expected_shape, config) -> dict[str, Any]:
    zero_fraction = 1.0 - np.divide(acc.nonzero_count, acc.value_count)
    active_fraction = np.divide(acc.active_channel_count, acc.channel_count)
    mean = np.divide(acc.sum_value, acc.value_count)
    variance = np.maximum(0.0, np.divide(acc.sum_square, acc.value_count) - mean * mean)
    relative_change = np.sqrt(acc.transition_diff_square) / np.maximum(
        np.sqrt(acc.transition_base_square), config.zero_epsilon or np.finfo(float).tiny
    )
    point_count = len(zero_fraction)
    tail_points = min(point_count, max(config.min_tail_points, int(math.ceil(point_count * config.tail_fraction))))
    tail_start = point_count - tail_points
    tail_changes = relative_change[max(0, tail_start - 1) :]
    boundary_changes = relative_change[-min(config.boundary_window, len(relative_change)) :]
    tail_sparse = bool(float(np.mean(zero_fraction[tail_start:])) >= config.sparse_zero_fraction)
    tail_saturated = bool(
        len(tail_changes) >= config.min_tail_points - 1
        and float(np.max(tail_changes)) <= config.saturation_relative_l2
    )
    boundary_active = bool(
        len(boundary_changes) > 0 and float(np.mean(boundary_changes)) >= config.boundary_relative_l2
    )
    reason_codes: list[str] = []
    if tail_sparse:
        reason_codes.append("SPARSE_TAIL")
    if tail_saturated:
        reason_codes.append("SATURATED_TAIL")
    if boundary_active:
        reason_codes.append("ACTIVE_AT_BOUNDARY")
    if tail_sparse or tail_saturated:
        recommendation = "SHORTEN"
    elif boundary_active:
        recommendation = "EXTEND"
    else:
        recommendation = "KEEP"
    axis_values = [profile.start + index * profile.step for index in range(point_count)]
    effective_transitions = relative_change > config.saturation_relative_l2
    effective_indices = np.flatnonzero(effective_transitions)
    last_effective_point_index = (
        int(effective_indices[-1] + 1) if effective_indices.size else None
    )
    trailing_stable_transition_count = 0
    for changed in effective_transitions[::-1]:
        if changed:
            break
        trailing_stable_transition_count += 1
    points = []
    for index, value in enumerate(axis_values):
        points.append(
            {
                "index": index,
                "axis_value": float(value),
                "zero_fraction": float(zero_fraction[index]),
                "active_channel_fraction": float(active_fraction[index]),
                "mean_abs": float(acc.sum_abs[index] / acc.value_count[index]),
                "std": float(math.sqrt(variance[index])),
                "relative_l2_change_from_previous": (
                    None if index == 0 else float(relative_change[index - 1])
                ),
            }
        )
    return {
        "status": "PASS" if recommendation == "KEEP" else "WARN",
        "recommendation": recommendation,
        "reason_codes": reason_codes,
        "loaded_sample_count": acc.sample_count,
        "expected_shape": list(expected_shape),
        "scale_kind": profile.scale_kind,
        "axis_start": float(profile.start),
        "axis_stop": float(profile.stop),
        "axis_step": float(profile.step),
        "tail_start_index": tail_start,
        "tail_start_value": float(axis_values[tail_start]),
        "tail_point_count": tail_points,
        "tail_zero_fraction_mean": float(np.mean(zero_fraction[tail_start:])),
        "tail_relative_l2_change_max": float(np.max(tail_changes)) if len(tail_changes) else None,
        "effective_transition_count": int(np.count_nonzero(effective_transitions)),
        "effective_transition_fraction": (
            float(np.mean(effective_transitions)) if len(effective_transitions) else None
        ),
        "last_effective_point_index": last_effective_point_index,
        "last_effective_axis_value": (
            float(axis_values[last_effective_point_index])
            if last_effective_point_index is not None
            else None
        ),
        "trailing_stable_transition_count": trailing_stable_transition_count,
        "boundary_relative_l2_change_mean": (
            float(np.mean(boundary_changes)) if len(boundary_changes) else None
        ),
        "points": points,
    }


def _load_config(path: Path | None) -> FiltrationAuditConfig:
    if path is None:
        return FiltrationAuditConfig()
    raw = load_yaml(path)
    data = raw.get("filtration_audit", raw)
    if not isinstance(data, Mapping):
        raise ValueError("filtration_audit config must be a mapping")
    return FiltrationAuditConfig.from_mapping(data)


if __name__ == "__main__":
    raise SystemExit(main())
