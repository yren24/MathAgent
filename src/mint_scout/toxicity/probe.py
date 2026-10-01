from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mint_scout.data.geometry import AtomCloud, load_atom_cloud
from mint_scout.data.manifest_io import load_dataset_manifest
from mint_scout.data.splits import make_kfold_assignments
from mint_scout.invariants.manifest import stable_hash
from mint_scout.probe_audit import compare_distribution, compare_joint_strata
from mint_scout.representation import RepresentationSpec
from mint_scout.schemas import SampleSplit
from mint_scout.scout.sampling import (
    ProbeSamplingConfig,
    quantile_bin_assignments,
    select_probe_samples,
)


PROBE_SCHEMA = "mint-agent.toxicity-probe-selection.v1"


@dataclass(frozen=True)
class ToxicityProbePolicy:
    candidate_sizes: tuple[int, ...] = (300, 500, 750, 1000)
    target_quantile_bins: int = 10
    size_quantile_bins: int = 5
    min_pair_support: int = 3
    augmentation_fraction_limit: float = 0.10
    random_seed: int = 2026
    cv_folds: int = 5
    max_ks_distance: float = 0.10
    max_abs_standardized_mean_difference: float = 0.20
    max_joint_cell_share_error: float = 0.03

    def __post_init__(self) -> None:
        if not self.candidate_sizes or any(size < 2 for size in self.candidate_sizes):
            raise ValueError("toxicity Probe candidate sizes must be at least two")
        if len(set(self.candidate_sizes)) != len(self.candidate_sizes):
            raise ValueError("toxicity Probe candidate sizes must be unique")
        if tuple(sorted(self.candidate_sizes)) != self.candidate_sizes:
            raise ValueError("toxicity Probe candidate sizes must be increasing")
        if self.cv_folds < 2:
            raise ValueError("toxicity Probe cv_folds must be at least two")


def select_toxicity_probe(
    *,
    training_targets: Mapping[str, float],
    training_clouds: Mapping[str, AtomCloud],
    representation_specs: Sequence[RepresentationSpec],
    policy: ToxicityProbePolicy = ToxicityProbePolicy(),
    policy_source: str = "deterministic_default",
    policy_rationale: str | None = None,
) -> dict[str, Any]:
    sample_ids = tuple(sorted(training_targets))
    if not sample_ids or set(sample_ids) != set(training_clouds):
        raise ValueError("toxicity Probe targets and structures must be aligned")
    if any(not np.isfinite(training_targets[sample_id]) for sample_id in sample_ids):
        raise ValueError("toxicity Probe targets must be finite")
    if not representation_specs:
        raise ValueError("toxicity Probe requires at least one representation")
    for spec in representation_specs:
        spec.assert_frozen()
        if spec.system_type != "small_molecule":
            raise ValueError("toxicity Probe representations must be small_molecule")

    retained_pairs = tuple(
        dict.fromkeys(
            pair for spec in representation_specs for pair in spec.pair_order
        )
    )
    structure_sizes = {
        sample_id: float(len(training_clouds[sample_id].elements))
        for sample_id in sample_ids
    }
    pair_presence = {
        sample_id: tuple(
            pair
            for pair in retained_pairs
            if pair[0] in set(training_clouds[sample_id].elements)
            and pair[1] in set(training_clouds[sample_id].elements)
        )
        for sample_id in sample_ids
    }
    targets = [float(training_targets[sample_id]) for sample_id in sample_ids]
    sizes = [structure_sizes[sample_id] for sample_id in sample_ids]
    configured_sizes = tuple(
        size for size in policy.candidate_sizes if size <= len(sample_ids)
    )
    candidate_sizes = configured_sizes or (len(sample_ids),)
    size_source = (
        "configured_candidate_size"
        if configured_sizes
        else "small_data_full_train_fallback"
    )
    candidates = []
    for requested_size in candidate_sizes:
        config = ProbeSamplingConfig(
            fraction=requested_size / len(sample_ids),
            min_samples=requested_size,
            max_samples=requested_size,
            target_quantile_bins=policy.target_quantile_bins,
            size_quantile_bins=policy.size_quantile_bins,
            min_pair_support=policy.min_pair_support,
            augmentation_fraction_limit=policy.augmentation_fraction_limit,
            random_seed=policy.random_seed,
        )
        selection = select_probe_samples(
            sample_ids=sample_ids,
            targets=targets,
            structure_sizes=sizes,
            retained_pairs=retained_pairs,
            pair_presence_by_sample=pair_presence,
            config=config,
        )
        audit = _audit_probe(
            sample_ids=sample_ids,
            targets=training_targets,
            structure_sizes=structure_sizes,
            selection=selection,
            policy=policy,
        )
        candidates.append(
            {
                "candidate_id": f"hierarchical-probe-{requested_size}",
                "size_source": size_source,
                "requested_base_size": requested_size,
                "final_size": selection.final_size,
                "sample_ids": list(selection.sample_ids),
                "pair_support": {
                    _pair_name(pair): count
                    for pair, count in selection.pair_support.items()
                },
                "undercovered_pairs": [
                    list(pair) for pair in selection.undercovered_pairs
                ],
                "warnings": list(selection.warnings),
                "audit": audit,
                "eligible": audit["all_gates_passed"],
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected_under_pair_support_fallback = False
    if eligible:
        selected = min(
            eligible,
            key=lambda candidate: (candidate["final_size"], candidate["candidate_id"]),
        )
        selection_basis = "representativeness_hard_gates_then_minimum_sample_cost"
    else:
        pair_support_relaxed = [
            candidate
            for candidate in candidates
            if set(candidate["audit"]["gate_failures"]) <= {"minimum_pair_support"}
        ]
        if pair_support_relaxed:
            selected = min(
                pair_support_relaxed,
                key=lambda candidate: (
                    _pair_support_deficit(candidate, policy.min_pair_support),
                    candidate["final_size"],
                    candidate["candidate_id"],
                ),
            )
            selection_basis = (
                "representativeness_hard_gates_with_pair_support_fallback"
            )
            selected_under_pair_support_fallback = True
        else:
            selected = None
    if selected is None:
        failures = {
            candidate["candidate_id"]: candidate["audit"]["gate_failures"]
            for candidate in candidates
        }
        raise ValueError(
            "no toxicity Probe candidate passed every representativeness gate: "
            f"{failures}"
        )
    probe_ids = tuple(selected["sample_ids"])
    cv_folds = min(policy.cv_folds, len(probe_ids))
    fold_assignment = make_kfold_assignments(
        probe_ids,
        n_splits=cv_folds,
        seed=policy.random_seed,
    )
    representation_hashes = [spec.spec_hash for spec in representation_specs]
    payload: dict[str, Any] = {
        "report_schema": PROBE_SCHEMA,
        "status": "COMPLETE",
        "evidence_scope": "train_only",
        "test_structures_used": False,
        "test_labels_used": False,
        "selection_basis": selection_basis,
        "policy_source": policy_source,
        "policy_rationale": policy_rationale,
        "policy": asdict(policy),
        "training_sample_count": len(sample_ids),
        "representation_count": len(representation_specs),
        "representation_hashes": representation_hashes,
        "retained_pair_union": [list(pair) for pair in retained_pairs],
        "retained_pair_union_count": len(retained_pairs),
        "candidates": candidates,
        "decision_trace": [
            {
                "stage": "representativeness_hard_gates",
                "survivors": [
                    candidate["candidate_id"] for candidate in eligible
                ],
            },
            *(
                [
                    {
                        "stage": "pair_support_fallback",
                        "reason": (
                            "no candidate passed every gate, but at least one "
                            "candidate failed only the minimum_pair_support gate"
                        ),
                        "survivors": [
                            candidate["candidate_id"]
                            for candidate in pair_support_relaxed
                        ],
                        "selected": selected["candidate_id"],
                        "selected_undercoverage": _pair_support_deficit(
                            selected, policy.min_pair_support
                        ),
                    }
                ]
                if selected_under_pair_support_fallback
                else []
            ),
            {
                "stage": "minimum_sample_cost",
                "selected": selected["candidate_id"],
                "selected_size": selected["final_size"],
            },
        ],
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_all_gates_passed": bool(selected["eligible"]),
        "selected_under_pair_support_fallback": selected_under_pair_support_fallback,
        "probe_sample_ids": list(probe_ids),
        "probe_sample_count": len(probe_ids),
        "probe_fold_assignment": fold_assignment,
        "cv_folds": cv_folds,
        "fold_seed": policy.random_seed,
    }
    payload["selection_hash"] = stable_hash(payload)
    return payload


def _audit_probe(
    *,
    sample_ids: Sequence[str],
    targets: Mapping[str, float],
    structure_sizes: Mapping[str, float],
    selection: Any,
    policy: ToxicityProbePolicy,
) -> dict[str, Any]:
    probe_ids = selection.sample_ids
    target_distribution = compare_distribution(
        [targets[sample_id] for sample_id in sample_ids],
        [targets[sample_id] for sample_id in probe_ids],
    )
    size_distribution = compare_distribution(
        [structure_sizes[sample_id] for sample_id in sample_ids],
        [structure_sizes[sample_id] for sample_id in probe_ids],
    )
    joint = compare_joint_strata(
        modeling_ids=sample_ids,
        probe_ids=probe_ids,
        target_bins=_all_bins(
            sample_ids, targets, policy.target_quantile_bins
        ),
        size_bins=_all_bins(
            sample_ids, structure_sizes, policy.size_quantile_bins
        ),
    )
    gate_results = {
        "target_ks": target_distribution["ks_distance"]
        <= policy.max_ks_distance,
        "size_ks": size_distribution["ks_distance"] <= policy.max_ks_distance,
        "target_mean_alignment": abs(
            target_distribution["standardized_mean_difference"]
        )
        <= policy.max_abs_standardized_mean_difference,
        "size_mean_alignment": abs(
            size_distribution["standardized_mean_difference"]
        )
        <= policy.max_abs_standardized_mean_difference,
        "joint_strata_share": joint["max_abs_cell_share_error"]
        <= policy.max_joint_cell_share_error,
        "target_bins_covered": joint["all_target_bins_covered"],
        "size_bins_covered": joint["all_size_bins_covered"],
        "minimum_pair_support": not selection.undercovered_pairs,
    }
    failures = [name for name, passed in gate_results.items() if not passed]
    return {
        "target_distribution": target_distribution,
        "structure_size_distribution": size_distribution,
        "joint_target_size_strata": joint,
        "gate_results": gate_results,
        "gate_failures": failures,
        "all_gates_passed": not failures,
    }


def _pair_support_deficit(
    candidate: Mapping[str, Any],
    min_pair_support: int,
) -> tuple[int, int]:
    pair_support = candidate.get("pair_support")
    undercovered_pairs = candidate.get("undercovered_pairs")
    if not isinstance(pair_support, Mapping) or not isinstance(
        undercovered_pairs, list
    ):
        raise ValueError("toxicity Probe candidate is missing pair support metadata")
    deficits = []
    for raw_pair in undercovered_pairs:
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise ValueError("toxicity undercovered pair must be a two-item list")
        count = int(pair_support.get(_pair_name((str(raw_pair[0]), str(raw_pair[1]))), 0))
        deficits.append(max(0, min_pair_support - count))
    return (len(deficits), int(sum(deficits)))


def _all_bins(
    sample_ids: Sequence[str],
    values: Mapping[str, float],
    max_bins: int,
) -> dict[str, int]:
    return quantile_bin_assignments(
        [values[sample_id] for sample_id in sample_ids],
        sample_ids,
        max_bins=max_bins,
    )


def _pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


def _load_representation_specs(
    design_report: Path, representation_dir: Path
) -> tuple[RepresentationSpec, ...]:
    payload = json.loads(design_report.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("toxicity design report must contain a JSON object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("toxicity design report has no candidates")
    specs = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("toxicity design candidate must be an object")
        candidate_id = str(candidate.get("candidate_id") or "")
        spec = RepresentationSpec.read(representation_dir / f"{candidate_id}.json")
        expected_hash = str(candidate.get("representation_hash") or "")
        if spec.spec_hash != expected_hash:
            raise ValueError(
                f"toxicity representation hash mismatch for {candidate_id}"
            )
        specs.append(spec)
    return tuple(specs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a shared hierarchical train-only Probe for toxicity."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-report", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--advisory-artifact", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", default="LD50")
    args = parser.parse_args(argv)
    manifest = load_dataset_manifest(
        args.manifest,
        dataset_id=args.dataset_id,
        system_type="small_molecule",
        columns={
            "identifiers": {
                "smiles": "smiles",
                "source_filename": "source_filename",
            }
        },
        label_name=args.dataset_id,
    )
    training = manifest.samples_in_split(SampleSplit.TRAIN)
    targets = {
        sample.sample_id: float(sample.target)
        for sample in training
        if sample.target is not None
    }
    if len(targets) != len(training):
        raise ValueError("toxicity Probe requires labels for every training sample")
    clouds = {
        sample.sample_id: load_atom_cloud(sample.role_paths["molecule"])
        for sample in training
    }
    policy = ToxicityProbePolicy()
    policy_source = "deterministic_default"
    policy_rationale = None
    if args.advisory_artifact is not None:
        policy, policy_rationale = _policy_from_advisory_artifact(
            args.advisory_artifact
        )
        policy_source = "llm_advisory"
    report = select_toxicity_probe(
        training_targets=targets,
        training_clouds=clouds,
        representation_specs=_load_representation_specs(
            args.design_report, args.representation_dir
        ),
        policy=policy,
        policy_source=policy_source,
        policy_rationale=policy_rationale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"status={report['status']} selected={report['selected_candidate_id']} "
        f"probe={report['probe_sample_count']} output={args.output}"
    )
    return 0


def _policy_from_advisory_artifact(path: Path) -> tuple[ToxicityProbePolicy, str | None]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("report_schema")
        != "mint-agent.toxicity-representation-advisory.v1"
        or artifact.get("status") != "VALIDATED"
    ):
        raise ValueError("toxicity advisory artifact is not validated")
    advice = artifact.get("advice")
    if not isinstance(advice, Mapping):
        raise ValueError("toxicity advisory artifact has no advice")
    strategy = advice.get("probe_strategy")
    if not isinstance(strategy, Mapping):
        raise ValueError("toxicity advisory artifact has no probe_strategy")
    candidate_sizes = strategy.get("candidate_sizes")
    if not isinstance(candidate_sizes, list) or not candidate_sizes:
        raise ValueError("toxicity probe_strategy has no candidate_sizes")
    rationale = strategy.get("rationale")
    return (
        ToxicityProbePolicy(
            candidate_sizes=tuple(int(value) for value in candidate_sizes),
            target_quantile_bins=int(strategy["target_quantile_bins"]),
            size_quantile_bins=int(strategy["size_quantile_bins"]),
            min_pair_support=int(strategy["min_pair_support"]),
        ),
        str(rationale) if rationale is not None else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
