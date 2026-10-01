from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_scout.config import load_yaml
from mint_scout.invariants.manifest import stable_hash

MATRIX_SCHEMA = "mint-agent.controlled-experiment-matrix.v2"
PROBE_SELECTION_SCHEMA = "mint-agent.controlled-probe-selection.v1"
REPRESENTATION_SELECTION_SCHEMA = "mint-agent.controlled-representation-selection.v2"
REPRESENTATION_REPORT_SCHEMAS = {
    "mint-agent.representation-design.v1",
    "mint-agent.filtration-repair.v1",
}


@dataclass(frozen=True)
class ProbeComparisonPolicy:
    max_ks_distance: float = 0.10
    max_abs_standardized_mean_difference: float = 0.20
    max_joint_cell_share_error: float = 0.03
    require_all_target_bins_covered: bool = True
    require_all_size_bins_covered: bool = True
    require_pair_support_consistent: bool = True
    require_minimum_pair_support: bool = True

    def __post_init__(self) -> None:
        for name in ("max_ks_distance", "max_joint_cell_share_error"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            not math.isfinite(float(self.max_abs_standardized_mean_difference))
            or self.max_abs_standardized_mean_difference < 0.0
        ):
            raise ValueError(
                "max_abs_standardized_mean_difference must be finite and non-negative"
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "ProbeComparisonPolicy":
        if value is None:
            return cls()
        return cls(
            max_ks_distance=float(
                value.get("max_ks_distance", cls.max_ks_distance)
            ),
            max_abs_standardized_mean_difference=float(
                value.get(
                    "max_abs_standardized_mean_difference",
                    cls.max_abs_standardized_mean_difference,
                )
            ),
            max_joint_cell_share_error=float(
                value.get(
                    "max_joint_cell_share_error", cls.max_joint_cell_share_error
                )
            ),
            require_all_target_bins_covered=bool(
                value.get(
                    "require_all_target_bins_covered",
                    cls.require_all_target_bins_covered,
                )
            ),
            require_all_size_bins_covered=bool(
                value.get(
                    "require_all_size_bins_covered",
                    cls.require_all_size_bins_covered,
                )
            ),
            require_pair_support_consistent=bool(
                value.get(
                    "require_pair_support_consistent",
                    cls.require_pair_support_consistent,
                )
            ),
            require_minimum_pair_support=bool(
                value.get(
                    "require_minimum_pair_support",
                    cls.require_minimum_pair_support,
                )
            ),
        )


@dataclass(frozen=True)
class RepresentationComparisonPolicy:
    require_train_only: bool = True
    minimum_pair_coverage: float = 0.0
    pair_coverage_tolerance: float = 0.0
    filtration_issue_tolerance: int = 0
    feature_warning_tolerance: int = 0
    max_total_feature_dimensions: int | None = None
    dimension_tolerance: int = 0

    def __post_init__(self) -> None:
        for name in ("minimum_pair_coverage", "pair_coverage_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in (
            "filtration_issue_tolerance",
            "feature_warning_tolerance",
            "dimension_tolerance",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.max_total_feature_dimensions is not None
            and int(self.max_total_feature_dimensions) < 1
        ):
            raise ValueError("max_total_feature_dimensions must be positive or null")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "RepresentationComparisonPolicy":
        if value is None:
            return cls()
        return cls(
            require_train_only=bool(
                value.get("require_train_only", cls.require_train_only)
            ),
            minimum_pair_coverage=float(
                value.get("minimum_pair_coverage", cls.minimum_pair_coverage)
            ),
            pair_coverage_tolerance=float(
                value.get("pair_coverage_tolerance", cls.pair_coverage_tolerance)
            ),
            filtration_issue_tolerance=int(
                value.get("filtration_issue_tolerance", cls.filtration_issue_tolerance)
            ),
            feature_warning_tolerance=int(
                value.get("feature_warning_tolerance", cls.feature_warning_tolerance)
            ),
            max_total_feature_dimensions=(
                int(value["max_total_feature_dimensions"])
                if value.get("max_total_feature_dimensions") is not None
                else None
            ),
            dimension_tolerance=int(
                value.get("dimension_tolerance", cls.dimension_tolerance)
            ),
        )


def build_controlled_experiment_matrix(
    *,
    task_config: Mapping[str, Any],
    scout_config: Mapping[str, Any],
    experiment_plan: Mapping[str, Any],
) -> dict[str, Any]:
    advice = _validated_advice(experiment_plan)
    controlled = scout_config.get("controlled_experiment", {})
    if not isinstance(controlled, Mapping):
        raise ValueError("controlled_experiment configuration must be an object")
    max_probe_alternatives = int(controlled.get("max_probe_alternatives", 2))
    max_representation_alternatives = int(
        controlled.get("max_representation_alternatives", 2)
    )
    if not 0 <= max_probe_alternatives <= 2:
        raise ValueError("max_probe_alternatives must be in [0, 2]")
    if not 0 <= max_representation_alternatives <= 2:
        raise ValueError("max_representation_alternatives must be in [0, 2]")
    if len(advice.get("probe_variants", ())) > max_probe_alternatives:
        raise ValueError("LLM probe variants exceed the configured experiment budget")
    if len(advice.get("representation_variants", ())) > max_representation_alternatives:
        raise ValueError(
            "LLM representation variants exceed the configured experiment budget"
        )
    audit_thresholds = scout_config.get("probe_audit", {})
    if not isinstance(audit_thresholds, Mapping):
        raise ValueError("probe_audit configuration must be an object")
    probe_policy_values = {
        key: audit_thresholds[key]
        for key in (
            "max_ks_distance",
            "max_abs_standardized_mean_difference",
            "max_joint_cell_share_error",
        )
        if key in audit_thresholds
    }
    configured_probe_policy = controlled.get("probe_comparison")
    if configured_probe_policy is not None:
        if not isinstance(configured_probe_policy, Mapping):
            raise ValueError("controlled_experiment.probe_comparison must be an object")
        probe_policy_values.update(configured_probe_policy)
    probe_policy = ProbeComparisonPolicy.from_mapping(probe_policy_values)
    representation_policy = RepresentationComparisonPolicy.from_mapping(
        controlled.get("representation_comparison")
        if isinstance(controlled.get("representation_comparison"), Mapping)
        else None
    )
    baseline_design = _required_mapping(task_config, "representation_design")
    _required_mapping(scout_config, "probe")
    _required_mapping(baseline_design, "support")
    filtration = _required_mapping(baseline_design, "filtration")
    if int(filtration.get("fixed_point_count", 0)) != 50:
        raise ValueError(
            "controlled representation experiments require fixed_point_count=50"
        )

    probe_candidates = [
        _candidate(
            "baseline-probe",
            "deterministic_baseline",
            dict(scout_config),
            llm_prior_rank=None,
        )
    ]
    for index, variant in enumerate(advice.get("probe_variants", ()), start=1):
        candidate_config = copy.deepcopy(dict(scout_config))
        candidate_probe = dict(_required_mapping(candidate_config, "probe"))
        for key in (
            "fraction",
            "max_samples",
            "target_quantile_bins",
            "size_quantile_bins",
            "min_pair_support",
            "augmentation_fraction_limit",
        ):
            candidate_probe[key] = variant[key]
        candidate_config["probe"] = candidate_probe
        probe_candidates.append(
            _candidate(
                str(variant["id"]),
                str(variant.get("rationale") or "LLM-proposed bounded probe variant."),
                candidate_config,
                llm_prior_rank=index,
            )
        )

    representation_candidates = [
        _candidate(
            "baseline-representation",
            "deterministic_baseline",
            dict(task_config),
            llm_prior_rank=None,
        )
    ]
    for index, variant in enumerate(advice.get("representation_variants", ()), start=1):
        candidate_config = copy.deepcopy(dict(task_config))
        candidate_design = dict(
            _required_mapping(candidate_config, "representation_design")
        )
        candidate_support = dict(_required_mapping(candidate_design, "support"))
        candidate_filtration = dict(_required_mapping(candidate_design, "filtration"))
        for key in (
            "min_pair_support_fraction",
            "min_pair_support_samples",
            "max_element_pair_channels",
        ):
            candidate_support[key] = variant[key]
        for key in (
            "local_distance_quantile",
            "dataset_distance_quantile",
            "margin_factor",
            "fixed_point_count",
        ):
            candidate_filtration[key] = variant[key]
        candidate_design["support"] = candidate_support
        candidate_design["filtration"] = candidate_filtration
        candidate_config["representation_design"] = candidate_design
        representation_candidates.append(
            _candidate(
                str(variant["id"]),
                str(
                    variant.get("rationale")
                    or "LLM-proposed bounded representation variant."
                ),
                candidate_config,
                llm_prior_rank=index,
            )
        )

    _reject_duplicate_candidate_configs(probe_candidates, "probe")
    _reject_duplicate_candidate_configs(representation_candidates, "representation")
    payload: dict[str, Any] = {
        "report_schema": MATRIX_SCHEMA,
        "status": "PLANNED",
        "mode": experiment_plan.get("mode"),
        "execution_allowed": bool(
            experiment_plan.get("mode") == "advisory"
            and experiment_plan.get("applied_to_execution", False)
        ),
        "llm_input_hash": experiment_plan.get("input_hash"),
        "scientific_controls": {
            "baseline_included": True,
            "train_only_design": True,
            "test_evidence_allowed": False,
            "probe_selected_before_representation_comparison": True,
            "shared_probe_required_for_representation_feature_audits": True,
            "fixed_filtration_point_count": 50,
            "max_probe_alternatives": max_probe_alternatives,
            "max_representation_alternatives": max_representation_alternatives,
        },
        "probe_comparison_policy": {
            "uses_model_performance": False,
            "uses_weighted_composite": False,
            "uses_test_evidence": False,
            "decision_rule": "representativeness_gates_then_minimum_sample_cost",
            "cost_proxy": "probe_sample_count",
            "thresholds": asdict(probe_policy),
            "baseline_tie_break": True,
        },
        "representation_comparison_policy": {
            "uses_train_only_structure_and_feature_evidence": True,
            "uses_model_performance": False,
            "uses_ranking_stability": False,
            "uses_runtime_cost": False,
            "uses_test_evidence": False,
            "decision_rule": "hierarchical_representation_suitability",
            "llm_prior_used": False,
            **asdict(representation_policy),
            "baseline_tie_break": True,
        },
        "probe_candidates": probe_candidates,
        "representation_candidates": representation_candidates,
        "source_summary": {
            "probe_strategy": advice.get("probe_strategy"),
            "representation_hypotheses": advice.get("representation_hypotheses"),
        },
    }
    payload["experiment_hash"] = stable_hash(payload)
    return payload


def materialize_controlled_experiment(
    matrix: Mapping[str, Any], *, output_dir: Path
) -> dict[str, Any]:
    _require_schema(matrix, MATRIX_SCHEMA)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    materialized = copy.deepcopy(dict(matrix))
    for group, suffix in (
        ("probe_candidates", "scout.json"),
        ("representation_candidates", "task.json"),
    ):
        for candidate in materialized[group]:
            path = destination / f"{candidate['id']}.{suffix}"
            _write_json(candidate["config"], path)
            candidate["config_path"] = str(path)
    manifest_path = destination / "controlled_experiment.v2.json"
    materialized["materialized_root"] = str(destination)
    materialized["manifest_path"] = str(manifest_path)
    _write_json(materialized, manifest_path)
    return materialized


def choose_probe_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_id: str = "baseline-probe",
    policy: ProbeComparisonPolicy = ProbeComparisonPolicy(),
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one probe candidate is required")
    rows: list[dict[str, Any]] = []
    modeling_ids: tuple[str, ...] | None = None
    full_folds: dict[str, int] | None = None
    for descriptor in candidates:
        candidate_id = str(descriptor.get("candidate_id") or "")
        selection = _load_payload(descriptor.get("selection"))
        audit = _load_payload(descriptor.get("audit"))
        _require_schema(selection, "mint-agent.casf-probe-selection.v1")
        _require_schema(audit, "mint-agent.probe-representativeness-audit.v1")
        if audit.get("selection_hash") != selection.get("selection_hash"):
            raise ValueError(f"probe audit does not match selection for {candidate_id}")
        current_ids = tuple(str(value) for value in selection["modeling_sample_ids"])
        current_folds = {
            str(key): int(value)
            for key, value in selection["modeling_fold_assignment"].items()
        }
        if modeling_ids is None:
            modeling_ids = current_ids
            full_folds = current_folds
        elif current_ids != modeling_ids or current_folds != full_folds:
            raise ValueError(
                "probe candidates must share one modeling pool and fold assignment"
            )
        distributions = _required_mapping(audit, "distribution_comparisons")
        ks_values = tuple(
            float(value["ks_distance"])
            for value in distributions.values()
            if isinstance(value, Mapping)
        )
        abs_smd_values = tuple(
            abs(float(value["standardized_mean_difference"]))
            for value in distributions.values()
            if isinstance(value, Mapping)
        )
        max_ks = max(ks_values, default=0.0)
        max_abs_smd = max(abs_smd_values, default=0.0)
        strata = _required_mapping(audit, "joint_target_total_size_strata")
        pair_coverage = _required_mapping(audit, "element_pair_coverage")
        gate_results = {
            "ks_distance": max_ks <= policy.max_ks_distance,
            "mean_alignment": (
                max_abs_smd <= policy.max_abs_standardized_mean_difference
            ),
            "joint_strata_share": (
                float(strata["max_abs_cell_share_error"])
                <= policy.max_joint_cell_share_error
            ),
            "target_bins_covered": (
                not policy.require_all_target_bins_covered
                or bool(strata.get("all_target_bins_covered"))
            ),
            "size_bins_covered": (
                not policy.require_all_size_bins_covered
                or bool(strata.get("all_size_bins_covered"))
            ),
            "pair_support_consistent": (
                not policy.require_pair_support_consistent
                or bool(pair_coverage.get("all_support_consistent"))
            ),
            "minimum_pair_support": (
                not policy.require_minimum_pair_support
                or bool(pair_coverage.get("all_minimum_support_met"))
            ),
        }
        failures = [name for name, passed in gate_results.items() if not passed]
        count = int(audit["probe_sample_count"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "selection_path": _path_value(descriptor.get("selection")),
                "audit_path": _path_value(descriptor.get("audit")),
                "sample_count": count,
                "cost_proxy": count,
                "max_ks_distance": max_ks,
                "max_abs_standardized_mean_difference": max_abs_smd,
                "max_joint_cell_share_error": float(
                    strata["max_abs_cell_share_error"]
                ),
                "pair_support_consistent": bool(
                    pair_coverage.get("all_support_consistent")
                ),
                "minimum_pair_support_met": bool(
                    pair_coverage.get("all_minimum_support_met")
                ),
                "gate_results": gate_results,
                "gate_failures": failures,
                "eligible": not failures,
                "warnings": list(audit.get("warnings", ())),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            not row["eligible"],
            row["sample_count"],
            row["candidate_id"],
        ),
    )
    eligible = [row for row in ranked if row["eligible"]]
    if not eligible:
        failures = {
            row["candidate_id"]: row["gate_failures"] for row in ranked
        }
        raise ValueError(
            "no Probe candidate passed every representativeness gate; "
            f"increase or redesign the Probe candidates: {failures}"
        )
    minimum_count = min(row["sample_count"] for row in eligible)
    cheapest = [row for row in eligible if row["sample_count"] == minimum_count]
    selected = next(
        (row for row in cheapest if row["candidate_id"] == baseline_id),
        min(cheapest, key=lambda row: row["candidate_id"]),
    )
    payload = {
        "report_schema": PROBE_SELECTION_SCHEMA,
        "status": "COMPLETE",
        "selection_basis": "representativeness_gates_then_minimum_sample_cost",
        "policy": asdict(policy),
        "decision_trace": [
            {
                "stage": "representativeness_hard_gates",
                "survivors": [row["candidate_id"] for row in eligible],
            },
            {
                "stage": "minimum_sample_cost",
                "minimum_probe_sample_count": minimum_count,
                "survivors": [row["candidate_id"] for row in cheapest],
            },
            {
                "stage": "deterministic_tie_break",
                "rule": "baseline_then_candidate_id",
                "selected": selected["candidate_id"],
            },
        ],
        "shared_modeling_fold_hash": stable_hash(full_folds),
        "candidates": ranked,
        "selected_candidate_id": selected["candidate_id"],
        "selected_probe_selection": selected["selection_path"],
        "selected_probe_audit": selected["audit_path"],
        "test_evidence_used": False,
    }
    payload["comparison_hash"] = stable_hash(payload)
    return payload


def choose_representation_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str | None = None,
    baseline_id: str = "baseline-representation",
    policy: RepresentationComparisonPolicy = RepresentationComparisonPolicy(),
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one representation candidate is required")
    rows: list[dict[str, Any]] = []
    probe_ids: tuple[str, ...] | None = None
    excluded: list[dict[str, Any]] = []
    for descriptor in candidates:
        candidate_id = str(descriptor.get("candidate_id") or "")
        representation = _load_payload(descriptor.get("representation"))
        probe = _load_payload(descriptor.get("probe"))
        qc = _load_payload(descriptor.get("qc"))
        filtration = _load_payload(descriptor.get("filtration"))
        _require_schema_in(representation, REPRESENTATION_REPORT_SCHEMAS)
        _require_schema(probe, "mint-agent.casf-probe-selection.v1")
        current_ids = tuple(str(value) for value in probe["probe_sample_ids"])
        if probe_ids is None:
            probe_ids = current_ids
        elif current_ids != probe_ids:
            raise ValueError(
                "representation candidates must use the exact same Probe samples"
            )
        suitability = _representation_suitability(
            representation=representation,
            qc=qc,
            filtration=filtration,
            policy=policy,
        )
        gate_failures = list(suitability.pop("gate_failures"))
        if gate_failures:
            excluded.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "representation_suitability_gate_failed",
                    "failed_gates": gate_failures,
                }
            )
            continue
        llm_prior_rank = descriptor.get("llm_prior_rank")
        if llm_prior_rank is not None:
            llm_prior_rank = int(llm_prior_rank)
            if llm_prior_rank < 1:
                raise ValueError(f"invalid LLM prior rank for {candidate_id}")
        rows.append(
            {
                "candidate_id": candidate_id,
                "representation_path": _path_value(descriptor.get("representation")),
                "probe_path": _path_value(descriptor.get("probe")),
                "scout_path": _path_value(descriptor.get("scout")),
                "scout_execution_path": _path_value(descriptor.get("scout_execution")),
                "qc_path": _path_value(descriptor.get("qc")),
                "filtration_path": _path_value(descriptor.get("filtration")),
                **suitability,
                "proposal_source": (
                    "baseline"
                    if candidate_id == baseline_id
                    else "validated_llm_proposal"
                ),
                "llm_proposal_rank": llm_prior_rank,
            }
        )
    if not rows:
        raise ValueError("all representation candidates failed suitability gates")
    selected, decision_trace = _hierarchical_representation_selection(
        rows, baseline_id=baseline_id, policy=policy,
    )
    selected_id = selected["candidate_id"]
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row["candidate_id"] == selected_id else 1,
            -row["pair_coverage"],
            row["filtration_adjustment_count"],
            row["feature_degeneracy_warning_count"],
            row["feature_warning_count"],
            (
                row["total_feature_dimensions"]
                if row["total_feature_dimensions"] is not None
                else math.inf
            ),
            row["candidate_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["decision_rank"] = rank
        row["selected"] = row["candidate_id"] == selected_id
    payload = {
        "report_schema": REPRESENTATION_SELECTION_SCHEMA,
        "status": "COMPLETE",
        "selection_basis": "hierarchical_train_only_representation_suitability",
        "primary_metric_received_but_not_used": (
            str(primary_metric).upper() if primary_metric is not None else None
        ),
        "decision_policy": asdict(policy),
        "decision_trace": decision_trace,
        "excluded_candidates": excluded,
        "llm_prior_used": False,
        "model_performance_used": False,
        "ranking_stability_used": False,
        "runtime_cost_used": False,
        "shared_probe_hash": stable_hash({"sample_ids": probe_ids}),
        "candidates": ranked,
        "selected_candidate_id": selected["candidate_id"],
        "selected_representation": selected["representation_path"],
        "selected_probe": selected["probe_path"],
        "selected_scout": selected["scout_path"],
        "selected_scout_execution": selected["scout_execution_path"],
        "selected_qc": selected["qc_path"],
        "selected_filtration": selected["filtration_path"],
        "test_evidence_used": False,
    }
    payload["comparison_hash"] = stable_hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize and compare bounded LLM-proposed scientific settings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--task-config", type=Path, required=True)
    materialize.add_argument("--scout-config", type=Path, required=True)
    materialize.add_argument("--experiment-plan", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    choose_probe = subparsers.add_parser("choose-probe")
    choose_probe.add_argument("--candidate", type=Path, action="append", required=True)
    choose_probe.add_argument("--matrix", type=Path)
    choose_probe.add_argument("--output", type=Path, required=True)
    choose_representation = subparsers.add_parser("choose-representation")
    choose_representation.add_argument(
        "--candidate", type=Path, action="append", required=True
    )
    choose_representation.add_argument("--matrix", type=Path)
    choose_representation.add_argument("--primary-metric", required=True)
    choose_representation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "materialize":
        plan_payload = _load_payload(args.experiment_plan)
        experiment_plan = plan_payload.get("experiment_plan", plan_payload)
        matrix = build_controlled_experiment_matrix(
            task_config=load_yaml(args.task_config),
            scout_config=load_yaml(args.scout_config),
            experiment_plan=experiment_plan,
        )
        result = materialize_controlled_experiment(matrix, output_dir=args.output_dir)
    else:
        descriptors = [_load_payload(path) for path in args.candidate]
        matrix = _load_payload(args.matrix) if args.matrix is not None else None
        if args.command == "choose-probe":
            probe_policy = ProbeComparisonPolicy()
            if matrix is not None:
                _require_schema(matrix, MATRIX_SCHEMA)
                policy = _required_mapping(matrix, "probe_comparison_policy")
                probe_policy = ProbeComparisonPolicy.from_mapping(
                    _required_mapping(policy, "thresholds")
                )
            result = choose_probe_candidate(
                descriptors, policy=probe_policy,
            )
        else:
            representation_policy = RepresentationComparisonPolicy()
            if matrix is not None:
                _require_schema(matrix, MATRIX_SCHEMA)
                representation_policy = RepresentationComparisonPolicy.from_mapping(
                    _required_mapping(matrix, "representation_comparison_policy")
                )
            result = choose_representation_candidate(
                descriptors,
                primary_metric=args.primary_metric,
                policy=representation_policy,
            )
    _write_json(result, args.output)
    print(f"status={result['status']} output={args.output}")
    return 0


def _validated_advice(experiment_plan: Mapping[str, Any]) -> Mapping[str, Any]:
    if experiment_plan.get("status") not in {"VALIDATED", "DISABLED"}:
        raise ValueError("controlled experiment requires validated scientific advice")
    advice = experiment_plan.get("advice")
    if not isinstance(advice, Mapping):
        raise ValueError("experiment plan has no structured advice")
    for key in ("probe_variants", "representation_variants"):
        if not isinstance(advice.get(key), list):
            raise ValueError(f"experiment advice is missing {key}")
    return advice


def _candidate(
    candidate_id: str,
    rationale: str,
    config: Mapping[str, Any],
    *,
    llm_prior_rank: int | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    return {
        "id": candidate_id,
        "source": "baseline" if llm_prior_rank is None else "validated_llm_proposal",
        "llm_prior_rank": llm_prior_rank,
        "rationale": rationale,
        "config_hash": stable_hash(payload),
        "config": payload,
    }


def _reject_duplicate_candidate_configs(
    candidates: Sequence[Mapping[str, Any]], label: str
) -> None:
    seen: dict[str, str] = {}
    for candidate in candidates:
        config_hash = str(candidate["config_hash"])
        prior = seen.get(config_hash)
        if prior is not None:
            raise ValueError(
                f"{label} candidate {candidate['id']!r} duplicates candidate {prior!r}"
            )
        seen[config_hash] = str(candidate["id"])


def _load_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, (str, Path)):
        raise ValueError("artifact reference must be a path or object")
    payload = json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {value}")
    return payload


def _path_value(value: Any) -> str | None:
    return str(value) if isinstance(value, (str, Path)) else None


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"{key} must be an object")
    return nested


def _require_schema(payload: Mapping[str, Any], schema: str) -> None:
    if payload.get("report_schema") != schema:
        raise ValueError(f"expected {schema}, found {payload.get('report_schema')!r}")


def _require_schema_in(payload: Mapping[str, Any], schemas: set[str]) -> None:
    schema = payload.get("report_schema")
    if schema not in schemas:
        expected = ", ".join(sorted(schemas))
        raise ValueError(f"expected one of {expected}, found {schema!r}")


def _representation_suitability(
    *,
    representation: Mapping[str, Any],
    qc: Mapping[str, Any],
    filtration: Mapping[str, Any],
    policy: RepresentationComparisonPolicy,
) -> dict[str, Any]:
    spec = _required_mapping(representation, "representation_spec")
    parameters = spec.get("parameters", {})
    parameters = parameters if isinstance(parameters, Mapping) else {}
    gate_failures: list[str] = []
    if not bool(spec.get("frozen")):
        gate_failures.append("representation_not_frozen")
    schema = str(representation.get("report_schema") or "")
    train_only_verified = True
    if policy.require_train_only:
        if schema == "mint-agent.filtration-repair.v1":
            evidence_scope = str(representation.get("evidence_scope") or "").lower()
            audit_scope = str(representation.get("audit_evidence_scope") or "").lower()
            train_only_verified = (
                evidence_scope == "design"
                and audit_scope in {"probe", "train", "design"}
            )
            if not train_only_verified:
                gate_failures.append("representation_not_verified_train_only")
        else:
            modeling_scope = str(representation.get("modeling_scope") or "").lower()
            train_only_verified = modeling_scope == "train"
            if not train_only_verified:
                gate_failures.append("representation_not_verified_train_only")
    for label, report in (("feature_qc", qc), ("filtration", filtration)):
        if str(report.get("status") or "").upper() == "FAIL":
            gate_failures.append(f"{label}_failed")
        evidence_scope = str(report.get("evidence_scope") or "").lower()
        if evidence_scope == "test":
            gate_failures.append(f"{label}_uses_test_evidence")
        recorded_hash = report.get("representation_hash")
        expected_hash = representation.get("representation_hash")
        if recorded_hash is not None and expected_hash is not None and recorded_hash != expected_hash:
            gate_failures.append(f"{label}_representation_hash_mismatch")

    pair_coverage, pair_coverage_details = _pair_coverage(parameters, representation)
    if pair_coverage < policy.minimum_pair_coverage:
        gate_failures.append("pair_coverage_below_minimum")

    filtration_reports = filtration.get("invariants", {})
    filtration_reports = (
        filtration_reports if isinstance(filtration_reports, Mapping) else {}
    )
    filtration_adjustments = sum(
        1
        for value in filtration_reports.values()
        if isinstance(value, Mapping)
        and str(value.get("recommendation") or "REVIEW").upper() != "KEEP"
    )
    filtration_keep_fraction = (
        1.0 - filtration_adjustments / len(filtration_reports)
        if filtration_reports
        else 0.0
    )

    issues = [
        issue for issue in qc.get("issues", ()) if isinstance(issue, Mapping)
    ]
    warning_count = int(
        qc.get("warning_issue_count")
        if qc.get("warning_issue_count") is not None
        else sum(str(issue.get("severity") or "").lower() == "warning" for issue in issues)
    )
    degeneracy_codes = {
        "all_zero",
        "constant",
        "high_sparsity",
        "many_all_zero_coordinates",
        "many_constant_coordinates",
    }
    degeneracy_count = sum(
        str(issue.get("issue") or issue.get("reason_code") or "") in degeneracy_codes
        for issue in issues
    )
    total_dimensions, dimension_evidence = _total_feature_dimensions(qc)
    if (
        policy.max_total_feature_dimensions is not None
        and total_dimensions is not None
        and total_dimensions > policy.max_total_feature_dimensions
    ):
        gate_failures.append("feature_dimension_budget_exceeded")

    return {
        "gate_failures": gate_failures,
        "train_only_verified": train_only_verified,
        "pair_coverage": pair_coverage,
        "pair_coverage_details": pair_coverage_details,
        "filtration_adjustment_count": filtration_adjustments,
        "filtration_keep_fraction": filtration_keep_fraction,
        "feature_degeneracy_warning_count": degeneracy_count,
        "feature_warning_count": warning_count,
        "total_feature_dimensions": total_dimensions,
        "dimension_evidence_available": dimension_evidence,
    }


def _pair_coverage(
    parameters: Mapping[str, Any], representation: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    records = [
        row
        for row in parameters.get("pair_support", ())
        if isinstance(row, Mapping) and bool(row.get("retained"))
    ]
    if records:
        selected = [row for row in records if bool(row.get("selected", True))]
        channel_fraction = len(selected) / len(records)
        total_mass = sum(max(float(row.get("sample_presence") or 0.0), 0.0) for row in records)
        selected_mass = sum(
            max(float(row.get("sample_presence") or 0.0), 0.0) for row in selected
        )
        support_mass_fraction = selected_mass / total_mass if total_mass else channel_fraction
    else:
        pair_order = _required_mapping(representation, "representation_spec").get(
            "pair_order", ()
        )
        channel_fraction = 1.0 if pair_order else 0.0
        support_mass_fraction = channel_fraction

    sample_count = max(int(representation.get("sample_count") or 0), 0)
    compatibility = parameters.get("adapter_compatibility", {})
    compatibility = compatibility if isinstance(compatibility, Mapping) else {}
    unsupported = compatibility.get("unsupported_observed", {})
    unsupported = unsupported if isinstance(unsupported, Mapping) else {}
    max_unsupported_presence = max(
        (
            int(row.get("sample_presence") or 0)
            for rows in unsupported.values()
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes))
            for row in rows
            if isinstance(row, Mapping) and not bool(row.get("tolerated", False))
        ),
        default=0,
    )
    adapter_coverage = (
        max(0.0, 1.0 - max_unsupported_presence / sample_count)
        if sample_count
        else 1.0
    )
    # Channel count is diagnostic only: pruning rare pairs should be judged by
    # retained support mass, not by whether a candidate kept 30 or 40 channels.
    coverage = float(min(support_mass_fraction, adapter_coverage))
    return coverage, {
        "selected_channel_fraction": float(channel_fraction),
        "selected_support_mass_fraction": float(support_mass_fraction),
        "adapter_supported_sample_fraction": float(adapter_coverage),
        "coverage_basis": "support_mass_and_non_tolerated_adapter_coverage",
        "evidence_available": bool(records),
    }


def _total_feature_dimensions(qc: Mapping[str, Any]) -> tuple[int | None, bool]:
    invariant_reports = qc.get("invariants", {})
    if not isinstance(invariant_reports, Mapping) or not invariant_reports:
        return None, False
    total = 0
    for report in invariant_reports.values():
        if not isinstance(report, Mapping):
            return None, False
        shape = report.get("expected_shape")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or not shape:
            return None, False
        dimensions = [int(value) for value in shape]
        if any(value < 1 for value in dimensions):
            return None, False
        total += math.prod(dimensions)
    return total, True


def _hierarchical_representation_selection(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_id: str,
    policy: RepresentationComparisonPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    survivors = list(rows)
    trace: list[dict[str, Any]] = [
        {
            "stage": "train_only_and_quality_gates",
            "survivors": [row["candidate_id"] for row in survivors],
        }
    ]
    survivors = _keep_near_best(
        survivors,
        field="pair_coverage",
        higher_is_better=True,
        tolerance=policy.pair_coverage_tolerance,
    )
    trace.append(
        {
            "stage": "element_pair_coverage",
            "tolerance": policy.pair_coverage_tolerance,
            "survivors": [row["candidate_id"] for row in survivors],
        }
    )
    survivors = _keep_near_best(
        survivors,
        field="filtration_adjustment_count",
        higher_is_better=False,
        tolerance=float(policy.filtration_issue_tolerance),
    )
    trace.append(
        {
            "stage": "filtration_adequacy",
            "tolerance": policy.filtration_issue_tolerance,
            "survivors": [row["candidate_id"] for row in survivors],
        }
    )
    survivors = _keep_near_best(
        survivors,
        field="feature_degeneracy_warning_count",
        higher_is_better=False,
        tolerance=float(policy.feature_warning_tolerance),
    )
    trace.append(
        {
            "stage": "feature_degeneracy",
            "tolerance": policy.feature_warning_tolerance,
            "survivors": [row["candidate_id"] for row in survivors],
        }
    )
    survivors = _keep_near_best(
        survivors,
        field="feature_warning_count",
        higher_is_better=False,
        tolerance=float(policy.feature_warning_tolerance),
    )
    trace.append(
        {
            "stage": "feature_health",
            "tolerance": policy.feature_warning_tolerance,
            "survivors": [row["candidate_id"] for row in survivors],
        }
    )
    if survivors and all(row["total_feature_dimensions"] is not None for row in survivors):
        survivors = _keep_near_best(
            survivors,
            field="total_feature_dimensions",
            higher_is_better=False,
            tolerance=float(policy.dimension_tolerance),
        )
        dimension_stage = "representation_dimension_tie_break"
    else:
        dimension_stage = "representation_dimension_tie_break_skipped"
    trace.append(
        {
            "stage": dimension_stage,
            "tolerance": policy.dimension_tolerance,
            "survivors": [row["candidate_id"] for row in survivors],
        }
    )
    baseline = next(
        (row for row in survivors if row["candidate_id"] == baseline_id), None
    )
    selected = baseline or min(survivors, key=lambda row: row["candidate_id"])
    trace.append(
        {
            "stage": "deterministic_tie_break",
            "rule": "baseline_then_candidate_id",
            "survivors": [row["candidate_id"] for row in survivors],
            "selected": selected["candidate_id"],
        }
    )
    return selected, trace


def _keep_near_best(
    rows: Sequence[dict[str, Any]],
    *,
    field: str,
    higher_is_better: bool,
    tolerance: float,
) -> list[dict[str, Any]]:
    values = [float(row[field]) for row in rows]
    best = max(values) if higher_is_better else min(values)
    return [
        row
        for row in rows
        if (
            best - float(row[field]) <= tolerance
            if higher_is_better
            else float(row[field]) - best <= tolerance
        )
    ]
def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.is_file():
        existing = destination.read_text(encoding="utf-8")
        if existing == content:
            return
        raise ValueError(
            f"refusing to overwrite existing controlled artifact: {destination}"
        )
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    raise SystemExit(main())
