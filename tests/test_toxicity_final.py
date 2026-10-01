from __future__ import annotations

import json

import numpy as np

from mint_scout.toxicity.final_gbt import (
    combine_final_reports,
    run_final_method_task,
)
from mint_scout.toxicity.final_plan import build_final_plan


def test_final_plan_freezes_probe_priority_before_test(tmp_path):
    feature_plan = {
        "plan_hash": "feature-plan-1",
        "candidate_feature_signatures": {
            "candidate": {"EIC": "eic-signature", "FPRC": "fprc-signature"},
        },
        "tasks": [
            {
                "invariant": "EIC",
                "feature_signature": "eic-signature",
                "representation_hash": "eic-representation",
                "representation_spec": "/tmp/eic.json",
            },
            {
                "invariant": "FPRC",
                "feature_signature": "fprc-signature",
                "representation_hash": "fprc-representation",
                "representation_spec": "/tmp/fprc.json",
            },
        ],
    }
    scout = {
        "status": "COMPLETE",
        "test_labels_used": False,
        "feature_plan_hash": "feature-plan-1",
        "scout_hash": "scout-1",
        "nominal_best": ["EIC", "FPRC"],
        "priority_order": [["EIC"], ["EIC", "FPRC"], ["FPRC"]],
        "subset_summaries": [
            {"invariants": ["EIC"], "nominal_primary_score": 0.6},
            {"invariants": ["EIC", "FPRC"], "nominal_primary_score": 0.61},
            {"invariants": ["FPRC"], "nominal_primary_score": 0.5},
        ],
    }
    feature_plan_path = tmp_path / "feature_plan.json"
    scout_path = tmp_path / "scout.json"
    feature_plan_path.write_text(json.dumps(feature_plan), encoding="utf-8")
    scout_path.write_text(json.dumps(scout), encoding="utf-8")

    report = build_final_plan(
        scout_report=scout_path,
        feature_plan=feature_plan_path,
        output_root=tmp_path / "final",
    )

    assert report["selected_subset"] == ["EIC"]
    assert report["selection_policy"] == "probe_priority_rank1"
    assert report["test_labels_used_for_selection"] is False
    assert report["test_query_budget"] == 1
    assert [(row["invariant"], row["split"]) for row in report["tasks"]] == [
        ("EIC", "train"),
        ("EIC", "test"),
    ]


def test_final_plan_target_uses_probe_scores_without_test_labels(tmp_path):
    feature_plan = {
        "plan_hash": "feature-plan-1",
        "selection_contract": {
            "requested_invariants": ["EIC", "FPRC"],
            "primary_metric": "R2",
            "selection_objective": "satisfy_target",
            "user_target": 0.7,
        },
        "tasks": [
            {"invariant": method, "feature_signature": method.lower(),
             "representation_hash": method, "representation_spec": f"/{method}.json"}
            for method in ("EIC", "FPRC")
        ],
    }
    scout = {
        "status": "COMPLETE", "test_labels_used": False,
        "feature_plan_hash": "feature-plan-1", "scout_hash": "scout-1",
        "primary_metric": "R2", "requested_invariants": ["EIC", "FPRC"],
        "priority_order": [["EIC"], ["FPRC"], ["EIC", "FPRC"]],
        "nominal_best": ["EIC", "FPRC"],
        "subset_summaries": [
            {"invariants": ["EIC"], "nominal_primary_score": 0.65},
            {"invariants": ["FPRC"], "nominal_primary_score": 0.72},
            {"invariants": ["EIC", "FPRC"], "nominal_primary_score": 0.75},
        ],
    }
    feature_path = tmp_path / "features.json"
    scout_path = tmp_path / "scout.json"
    feature_path.write_text(json.dumps(feature_plan), encoding="utf-8")
    scout_path.write_text(json.dumps(scout), encoding="utf-8")

    report = build_final_plan(
        scout_report=scout_path, feature_plan=feature_path,
        output_root=tmp_path / "final",
    )
    assert report["selected_subset"] == ["FPRC"]
    assert report["priority_index"] == 1
    assert report["probe_target_met"] is True
    assert report["primary_metric"] == "R2"
    assert report["test_labels_used_for_selection"] is False

    feature_plan["selection_contract"]["user_target"] = 0.9
    feature_path.write_text(json.dumps(feature_plan), encoding="utf-8")
    unmet = build_final_plan(
        scout_report=scout_path, feature_plan=feature_path,
        output_root=tmp_path / "unmet",
    )
    assert unmet["selected_subset"] == ["EIC"]
    assert unmet["probe_target_met"] is False


def test_final_method_and_combine_report_fixed_test_metrics(tmp_path):
    train_eic = _write_manifest(
        tmp_path,
        name="train-eic",
        invariant="EIC",
        split="train",
        labels=np.arange(12, dtype=float),
    )
    test_eic = _write_manifest(
        tmp_path,
        name="test-eic",
        invariant="EIC",
        split="test",
        labels=np.arange(4, dtype=float) + 0.5,
    )
    plan = {
        "plan_hash": "final-plan-1",
        "probe_scout_hash": "scout-1",
        "selected_subset": ["EIC"],
        "tasks": [
            {
                "invariant": "EIC",
                "split": "train",
                "manifest_path": str(train_eic),
                "feature_signature": "eic-signature",
                "representation_hash": "eic-representation",
            },
            {
                "invariant": "EIC",
                "split": "test",
                "manifest_path": str(test_eic),
                "feature_signature": "eic-signature",
                "representation_hash": "eic-representation",
            },
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    gbt_path = tmp_path / "gbt.yaml"
    gbt_path.write_text(
        "\n".join(
            (
                "config_id: test",
                "n_estimators: 5",
                "max_depth: 2",
                "min_samples_split: 2",
                "learning_rate: 0.1",
                "subsample: 1.0",
                "max_features: sqrt",
                "random_state: 7",
                "n_runs: 1",
                "normalize: StandardScaler",
            )
        ),
        encoding="utf-8",
    )
    method_dir = tmp_path / "methods"

    method = run_final_method_task(
        plan_path=plan_path,
        gbt_config_path=gbt_path,
        method_index=0,
        output_dir=method_dir,
    )
    report = combine_final_reports(
        plan_path=plan_path,
        method_report_dir=method_dir,
        n_bootstrap=20,
        bootstrap_seed=9,
    )

    assert method["status"] == "COMPLETE"
    assert report["status"] == "COMPLETE"
    assert report["protocol"]["test_used_for_selection"] is False
    assert report["protocol"]["full_train_cross_validation_used"] is False
    assert set(report["final_test"]["metrics"]) == {
        "PCC2",
        "PCC",
        "R2",
        "RMSE",
        "MAE",
    }

    plan["selection_contract"] = {
        "requested_invariants": ["EIC"],
        "primary_metric": "R2",
        "selection_objective": "satisfy_target",
        "user_target": 1.0,
    }
    plan["primary_metric"] = "R2"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    target_report = combine_final_reports(
        plan_path=plan_path,
        method_report_dir=method_dir,
        n_bootstrap=0,
    )
    assert target_report["primary_metric"] == "R2"
    assert target_report["status"] == "TARGET_NOT_REACHED"
    assert target_report["final_target_met"] is False
    assert target_report["protocol"]["test_query_count"] == 1
    assert target_report["final_test"]["bootstrap_primary"]["metric"] == "R2"


def _write_manifest(tmp_path, *, name, invariant, split, labels):
    rows = []
    for index, label in enumerate(labels):
        sample_id = f"{split}-{index}"
        feature = tmp_path / f"{name}-{index}.npy"
        np.save(feature, np.asarray([[label, label**2 + 1.0]], dtype=np.float32))
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "label": float(label),
                "invariant": invariant,
                "status": "computed",
                "output_path": str(feature),
            }
        )
    manifest = tmp_path / f"{name}.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return manifest
