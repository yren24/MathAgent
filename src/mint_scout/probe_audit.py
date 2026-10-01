from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mint_scout.compute_feature_batch import _selected_records
from mint_scout.config import load_yaml
from mint_scout.data.casf_index import CasfRecord
from mint_scout.data.element_inventory import iter_elements_from_file
from mint_scout.invariants.manifest import stable_hash
from mint_scout.scout.sampling import quantile_bin_assignments


@dataclass(frozen=True)
class ProbeAuditConfig:
    max_ks_distance: float = 0.10
    max_abs_standardized_mean_difference: float = 0.20
    max_joint_cell_share_error: float = 0.03
    require_exact_extremes: bool = False

    def __post_init__(self) -> None:
        for name in ("max_ks_distance", "max_joint_cell_share_error"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"probe_audit.{name} must be in [0, 1]")
        if self.max_abs_standardized_mean_difference < 0.0:
            raise ValueError(
                "probe_audit.max_abs_standardized_mean_difference must be non-negative"
            )
        if not isinstance(self.require_exact_extremes, bool):
            raise ValueError("probe_audit.require_exact_extremes must be boolean")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ProbeAuditConfig":
        raw = config.get("probe_audit", {})
        if not isinstance(raw, Mapping):
            raise ValueError("probe_audit config must be a mapping")
        return cls(**dict(raw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit how well a frozen CASF probe represents its training pool."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--scout-config", type=Path, required=True)
    parser.add_argument("--probe-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    task_config = load_yaml(args.task_config)
    scout_mapping = load_yaml(args.scout_config)
    audit_config = ProbeAuditConfig.from_mapping(scout_mapping)
    probe_config = scout_mapping.get("probe", {})
    if not isinstance(probe_config, Mapping):
        raise ValueError("probe config must be a mapping")

    train_records = _selected_records(
        task_config,
        sample_ids=None,
        split="train",
        offset=0,
        limit=None,
        config_path=args.task_config,
    )
    test_records = _selected_records(
        task_config,
        sample_ids=None,
        split="test",
        offset=0,
        limit=None,
        config_path=args.task_config,
    )
    selection = _load_json_object(args.probe_selection)
    report = build_probe_audit(
        train_records=train_records,
        test_records=test_records,
        selection=selection,
        target_quantile_bins=int(probe_config.get("target_quantile_bins", 10)),
        size_quantile_bins=int(probe_config.get("size_quantile_bins", 5)),
        config=audit_config,
        task_id=str(task_config.get("task_id") or "") or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"probe_audit status={report['status']} modeling={report['modeling_sample_count']} "
        f"probe={report['probe_sample_count']} warnings={len(report['warnings'])} "
        f"output={args.output}"
    )
    return 0


def build_probe_audit(
    *,
    train_records: Sequence[CasfRecord],
    test_records: Sequence[CasfRecord],
    selection: Mapping[str, Any],
    target_quantile_bins: int,
    size_quantile_bins: int,
    config: ProbeAuditConfig,
    task_id: str | None = None,
) -> dict[str, Any]:
    modeling_ids = tuple(record.pdb_id for record in train_records)
    test_ids = {record.pdb_id for record in test_records}
    probe_ids = tuple(str(value) for value in selection.get("probe_sample_ids", ()))
    _validate_selection(selection, modeling_ids, probe_ids, test_ids)

    labels: dict[str, float] = {}
    protein_sizes: dict[str, float] = {}
    ligand_sizes: dict[str, float] = {}
    protein_elements: dict[str, set[str]] = {}
    ligand_elements: dict[str, set[str]] = {}
    for record in train_records:
        if record.protein_path is None or record.ligand_path is None:
            raise ValueError(f"Sample {record.pdb_id!r} is missing protein/ligand paths")
        protein = tuple(iter_elements_from_file(record.protein_path))
        ligand = tuple(iter_elements_from_file(record.ligand_path))
        labels[record.pdb_id] = float(record.label)
        protein_sizes[record.pdb_id] = float(len(protein))
        ligand_sizes[record.pdb_id] = float(len(ligand))
        protein_elements[record.pdb_id] = set(protein)
        ligand_elements[record.pdb_id] = set(ligand)

    total_sizes = {
        sample_id: protein_sizes[sample_id] + ligand_sizes[sample_id]
        for sample_id in modeling_ids
    }
    vectors = {
        "target": labels,
        "protein_atom_count": protein_sizes,
        "ligand_atom_count": ligand_sizes,
        "total_atom_count": total_sizes,
    }
    distributions = {
        name: compare_distribution(
            [values[sample_id] for sample_id in modeling_ids],
            [values[sample_id] for sample_id in probe_ids],
        )
        for name, values in vectors.items()
    }

    target_bins = quantile_bin_assignments(
        [labels[sample_id] for sample_id in modeling_ids],
        modeling_ids,
        max_bins=target_quantile_bins,
    )
    size_bins = quantile_bin_assignments(
        [total_sizes[sample_id] for sample_id in modeling_ids],
        modeling_ids,
        max_bins=size_quantile_bins,
    )
    strata = compare_joint_strata(
        modeling_ids=modeling_ids,
        probe_ids=probe_ids,
        target_bins=target_bins,
        size_bins=size_bins,
    )
    pair_coverage = _pair_coverage(
        selection.get("pair_support", {}),
        probe_ids=probe_ids,
        protein_elements=protein_elements,
        ligand_elements=ligand_elements,
        minimum=int(selection.get("probe_config", {}).get("min_pair_support", 0)),
    )
    warnings = _audit_warnings(
        distributions=distributions,
        strata=strata,
        pair_coverage=pair_coverage,
        config=config,
    )
    report: dict[str, Any] = {
        "report_schema": "mint-agent.probe-representativeness-audit.v1",
        "task_id": task_id,
        "selection_hash": selection.get("selection_hash"),
        "representation_hash": selection.get("representation_hash"),
        "modeling_scope": "train",
        "modeling_sample_count": len(modeling_ids),
        "probe_sample_count": len(probe_ids),
        "modeling_sample_ids": list(modeling_ids),
        "probe_sample_ids": list(probe_ids),
        "test_sample_count_excluded": len(test_ids),
        "test_leakage_detected": False,
        "atom_count_definition": {
            "protein_atom_count": "parsed atom records in the protein pocket file",
            "ligand_atom_count": "parsed atom records in the ligand structure file",
            "total_atom_count": "protein_atom_count + ligand_atom_count",
        },
        "thresholds": asdict(config),
        "threshold_interpretation": "Configurable diagnostic thresholds, not scientific constants.",
        "distribution_comparisons": distributions,
        "joint_target_total_size_strata": strata,
        "element_pair_coverage": pair_coverage,
        "warnings": warnings,
        "status": "PASS" if not warnings else "WARN",
    }
    report["audit_hash"] = stable_hash(report)
    return report


def compare_distribution(
    modeling_values: Sequence[float],
    probe_values: Sequence[float],
) -> dict[str, Any]:
    modeling = np.asarray(modeling_values, dtype=float)
    probe = np.asarray(probe_values, dtype=float)
    if modeling.ndim != 1 or probe.ndim != 1 or modeling.size == 0 or probe.size == 0:
        raise ValueError("distribution inputs must be nonempty one-dimensional vectors")
    if not np.all(np.isfinite(modeling)) or not np.all(np.isfinite(probe)):
        raise ValueError("distribution inputs must be finite")
    modeling_std = float(np.std(modeling))
    mean_difference = float(np.mean(probe) - np.mean(modeling))
    standardized = mean_difference / modeling_std if modeling_std > 0.0 else 0.0
    return {
        "modeling": _distribution_summary(modeling),
        "probe": _distribution_summary(probe),
        "mean_difference": mean_difference,
        "standardized_mean_difference": float(standardized),
        "ks_distance": _ks_distance(modeling, probe),
        "extreme_coverage": {
            "minimum_retained": bool(np.min(probe) == np.min(modeling)),
            "maximum_retained": bool(np.max(probe) == np.max(modeling)),
        },
    }


def compare_joint_strata(
    *,
    modeling_ids: Sequence[str],
    probe_ids: Sequence[str],
    target_bins: Mapping[str, int],
    size_bins: Mapping[str, int],
) -> dict[str, Any]:
    modeling_counts = Counter((target_bins[value], size_bins[value]) for value in modeling_ids)
    probe_counts = Counter((target_bins[value], size_bins[value]) for value in probe_ids)
    keys = tuple(sorted(set(modeling_counts) | set(probe_counts)))
    rows = []
    absolute_errors = []
    for target_bin, size_bin in keys:
        modeling_share = modeling_counts[(target_bin, size_bin)] / len(modeling_ids)
        probe_share = probe_counts[(target_bin, size_bin)] / len(probe_ids)
        error = probe_share - modeling_share
        absolute_errors.append(abs(error))
        rows.append(
            {
                "target_bin": target_bin,
                "size_bin": size_bin,
                "modeling_count": modeling_counts[(target_bin, size_bin)],
                "probe_count": probe_counts[(target_bin, size_bin)],
                "modeling_share": modeling_share,
                "probe_share": probe_share,
                "share_error": error,
            }
        )
    target_values = tuple(sorted(set(target_bins.values())))
    size_values = tuple(sorted(set(size_bins.values())))
    return {
        "target_bin_count": len(target_values),
        "size_bin_count": len(size_values),
        "all_target_bins_covered": all(
            sum(probe_counts[(target_bin, size_bin)] for size_bin in size_values) > 0
            for target_bin in target_values
        ),
        "all_size_bins_covered": all(
            sum(probe_counts[(target_bin, size_bin)] for target_bin in target_values) > 0
            for size_bin in size_values
        ),
        "max_abs_cell_share_error": max(absolute_errors, default=0.0),
        "total_variation_distance": 0.5 * sum(absolute_errors),
        "cells": rows,
    }


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    quantiles = np.quantile(values, (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0))
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "quantiles": {
            name: float(value)
            for name, value in zip(("q00", "q05", "q25", "q50", "q75", "q95", "q100"), quantiles)
        },
    }


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    values = np.unique(np.concatenate((left, right)))
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    left_cdf = np.searchsorted(left_sorted, values, side="right") / left.size
    right_cdf = np.searchsorted(right_sorted, values, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _pair_coverage(
    declared_support: Any,
    *,
    probe_ids: Sequence[str],
    protein_elements: Mapping[str, set[str]],
    ligand_elements: Mapping[str, set[str]],
    minimum: int,
) -> dict[str, Any]:
    if not isinstance(declared_support, Mapping):
        raise ValueError("probe selection pair_support must be a mapping")
    rows = []
    for name, declared in sorted(declared_support.items()):
        protein_element, ligand_element = _parse_pair_name(str(name))
        recomputed = sum(
            protein_element in protein_elements[sample_id]
            and ligand_element in ligand_elements[sample_id]
            for sample_id in probe_ids
        )
        rows.append(
            {
                "pair": str(name),
                "declared_support": int(declared),
                "recomputed_support": recomputed,
                "support_consistent": int(declared) == recomputed,
                "minimum_support": minimum,
                "minimum_met": recomputed >= minimum,
            }
        )
    return {
        "pair_count": len(rows),
        "all_support_consistent": all(row["support_consistent"] for row in rows),
        "all_minimum_support_met": all(row["minimum_met"] for row in rows),
        "pairs": rows,
    }


def _audit_warnings(
    *,
    distributions: Mapping[str, Mapping[str, Any]],
    strata: Mapping[str, Any],
    pair_coverage: Mapping[str, Any],
    config: ProbeAuditConfig,
) -> list[str]:
    warnings = []
    for name, comparison in distributions.items():
        if comparison["ks_distance"] > config.max_ks_distance:
            warnings.append(f"PROBE_DISTRIBUTION_KS: {name}")
        if (
            abs(comparison["standardized_mean_difference"])
            > config.max_abs_standardized_mean_difference
        ):
            warnings.append(f"PROBE_DISTRIBUTION_MEAN_SHIFT: {name}")
        if config.require_exact_extremes and not all(comparison["extreme_coverage"].values()):
            warnings.append(f"PROBE_EXTREME_NOT_EXACTLY_RETAINED: {name}")
    if strata["max_abs_cell_share_error"] > config.max_joint_cell_share_error:
        warnings.append("PROBE_JOINT_STRATUM_SHARE_SHIFT")
    if not strata["all_target_bins_covered"]:
        warnings.append("PROBE_TARGET_BIN_MISSING")
    if not strata["all_size_bins_covered"]:
        warnings.append("PROBE_SIZE_BIN_MISSING")
    if not pair_coverage["all_support_consistent"]:
        warnings.append("PROBE_PAIR_SUPPORT_MISMATCH")
    if not pair_coverage["all_minimum_support_met"]:
        warnings.append("PROBE_PAIR_UNDERCOVERED")
    return warnings


def _validate_selection(
    selection: Mapping[str, Any],
    modeling_ids: tuple[str, ...],
    probe_ids: tuple[str, ...],
    test_ids: set[str],
) -> None:
    if selection.get("modeling_scope") != "train":
        raise ValueError("Probe selection must have modeling_scope=train")
    declared_modeling = tuple(str(value) for value in selection.get("modeling_sample_ids", ()))
    if declared_modeling != modeling_ids:
        raise ValueError("Probe selection modeling sample IDs do not match the training pool")
    if not probe_ids or len(set(probe_ids)) != len(probe_ids):
        raise ValueError("Probe sample IDs must be nonempty and unique")
    if not set(probe_ids).issubset(modeling_ids):
        raise ValueError("Probe contains samples outside the training pool")
    leakage = sorted(set(probe_ids) & test_ids)
    if leakage:
        raise ValueError(f"Probe contains held-out test samples: {leakage[:10]}")


def _parse_pair_name(name: str) -> tuple[str, str]:
    try:
        protein, ligand = name.split("|", maxsplit=1)
        protein_role, protein_element = protein.split(":", maxsplit=1)
        ligand_role, ligand_element = ligand.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"Invalid role-aware element-pair name: {name!r}") from exc
    if protein_role != "protein" or ligand_role != "ligand":
        raise ValueError(f"Invalid role-aware element-pair name: {name!r}")
    return protein_element, ligand_element


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
