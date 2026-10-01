from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationSpec


LEGACY_PROTEIN_ELEMENTS = ("C", "N", "O", "S")
LEGACY_LIGAND_ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H")
LEGACY_PAIR_SET = frozenset((p, l) for p in LEGACY_PROTEIN_ELEMENTS for l in LEGACY_LIGAND_ELEMENTS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare legacy and adaptive PL evidence without test tuning.")
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--adaptive-cv", type=Path, required=True)
    parser.add_argument("--scout-execution", type=Path, required=True)
    parser.add_argument("--legacy-cv", type=Path, required=True)
    parser.add_argument("--legacy-fixed-test", type=Path, required=True)
    parser.add_argument("--filtration-audit", type=Path, required=True)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_representation_comparison(
        representation=RepresentationSpec.read(args.representation_spec),
        adaptive_cv=_read_object(args.adaptive_cv),
        scout_execution=_read_object(args.scout_execution),
        legacy_cv=_read_object(args.legacy_cv),
        legacy_fixed_test=_read_object(args.legacy_fixed_test),
        filtration_audit=_read_object(args.filtration_audit),
        min_delta=args.min_delta,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"status={report['status']} delta={report['same_protocol_comparison']['pcc_delta_legacy_minus_adaptive']:.6g} "
        f"material={report['same_protocol_comparison']['material_difference']} output={args.output_json}"
    )
    return 0


def build_representation_comparison(
    *,
    representation: RepresentationSpec,
    adaptive_cv: Mapping[str, Any],
    scout_execution: Mapping[str, Any],
    legacy_cv: Mapping[str, Any],
    legacy_fixed_test: Mapping[str, Any],
    filtration_audit: Mapping[str, Any],
    min_delta: float,
) -> dict[str, Any]:
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")
    representation.assert_frozen()
    adaptive_pcc = float(adaptive_cv["result"]["selected_score"])
    legacy_cv_pcc = float(legacy_cv["metrics"]["PCC"])
    legacy_test_pcc = float(legacy_fixed_test["metrics"]["PCC"])
    adaptive_ids = tuple(str(value) for value in adaptive_cv["sample_ids"])
    legacy_ids = tuple(str(value) for value in legacy_cv["evaluation_sample_ids"])
    if adaptive_ids != legacy_ids:
        raise ValueError("Legacy and adaptive CV sample order differs")
    scout_ids = tuple(str(value) for value in scout_execution["modeling_sample_ids"])
    if adaptive_ids != scout_ids:
        raise ValueError("Adaptive evaluation and Scout modeling sample order differs")
    adaptive_folds = {
        str(key): int(value) for key, value in scout_execution["full_fold_assignment"].items()
    }
    legacy_folds = {str(key): int(value) for key, value in legacy_cv["fold_assignment"].items()}
    if adaptive_folds != legacy_folds:
        raise ValueError("Legacy and adaptive CV fold assignments differ")
    adaptive_pair_set = frozenset(tuple(pair) for pair in representation.pair_order)
    delta = legacy_cv_pcc - adaptive_pcc
    material = abs(delta) >= min_delta
    audit_recommendations = {
        str(name): str(item["recommendation"])
        for name, item in filtration_audit["invariants"].items()
    }
    payload: dict[str, Any] = {
        "report_schema": "mint-agent.representation-comparison.v1",
        "dataset_id": adaptive_cv.get("task_id", "casf2016"),
        "split": "train",
        "evidence_scope": "full_train",
        "status": "COMPLETE",
        "representation_hash": representation.spec_hash,
        "sample_count": len(adaptive_ids),
        "invariants": ["PL"],
        "same_protocol_comparison": {
            "protocol": "shared_frozen_5fold_full_train_oof",
            "adaptive_pl_pcc": adaptive_pcc,
            "legacy_pl_pcc": legacy_cv_pcc,
            "pcc_delta_legacy_minus_adaptive": delta,
            "min_material_delta": min_delta,
            "material_difference": material,
        },
        "historical_parity": {
            "protocol": "casf2016_fixed_test",
            "legacy_pl_pcc": legacy_test_pcc,
            "selection_use_permitted": False,
        },
        "representation_difference": {
            "element_pair_set_changed": adaptive_pair_set != LEGACY_PAIR_SET,
            "adaptive_pair_count": len(adaptive_pair_set),
            "legacy_pair_count": len(LEGACY_PAIR_SET),
            "filtration_changed": True,
            "legacy_filtration": legacy_cv["representation"],
            "adaptive_filtration": representation.filtration_profiles["PL"].to_dict(),
        },
        "filtration_audit": {
            "audit_hash": filtration_audit.get("audit_hash"),
            "recommendations": audit_recommendations,
        },
        "conclusion": (
            "The shared-fold PL difference is below the prespecified material-difference threshold; "
            "the historical 0.83 fixed-test result must not be attributed to element-pair selection or "
            "adaptive filtration performance."
            if not material and adaptive_pair_set == LEGACY_PAIR_SET
            else "The shared-fold comparison requires scientific review before attribution."
        ),
    }
    payload["comparison_hash"] = stable_hash(payload)
    return payload


def render_markdown(report: Mapping[str, Any]) -> str:
    comparison = report["same_protocol_comparison"]
    difference = report["representation_difference"]
    historical = report["historical_parity"]
    recommendations = report["filtration_audit"]["recommendations"]
    recommendation_text = ", ".join(f"{key}={value}" for key, value in sorted(recommendations.items()))
    return (
        "# PL Representation Comparison\n\n"
        f"- Dataset: `{report['dataset_id']}`\n"
        f"- Modeling samples: `{report['sample_count']}`\n"
        f"- Representation hash: `{report['representation_hash']}`\n"
        f"- Adaptive PL full-train OOF PCC: `{comparison['adaptive_pl_pcc']:.6f}`\n"
        f"- Legacy PL full-train OOF PCC: `{comparison['legacy_pl_pcc']:.6f}`\n"
        f"- Legacy minus adaptive PCC: `{comparison['pcc_delta_legacy_minus_adaptive']:.6f}`\n"
        f"- Material difference at `{comparison['min_material_delta']:.6f}`: "
        f"`{comparison['material_difference']}`\n"
        f"- Element-pair set changed: `{difference['element_pair_set_changed']}`\n"
        f"- Historical fixed-test legacy PCC: `{historical['legacy_pl_pcc']:.6f}`\n"
        f"- Filtration audit: `{recommendation_text}`\n\n"
        "## Interpretation\n\n"
        f"{report['conclusion']}\n"
    )


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
