from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mint_scout.scout.stability import BootstrapConfig
from mint_scout.toxicity.probe_gbt import (
    _load_feature_matrix,
    _load_gbt_config,
    combine_toxicity_probe_gbt,
    run_toxicity_oof_task,
)


def test_load_feature_matrix_uses_frozen_probe_order(tmp_path):
    rows = []
    for sample_id, label, value in (("b", 2.0, 20.0), ("a", 1.0, 10.0)):
        feature = tmp_path / f"{sample_id}.npy"
        np.save(feature, np.full((2, 1, 1), value, dtype=np.float32))
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "invariant": "PH",
                "status": "computed",
                "output_path": str(feature),
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    features, targets = _load_feature_matrix(
        manifest, sample_ids=("a", "b"), invariant="PH"
    )

    assert targets.tolist() == [1.0, 2.0]
    assert features[:, 0, 0, 0].tolist() == [10.0, 20.0]


def test_probe_scout_uses_requested_methods_and_r2(tmp_path):
    gbt_path = Path("configs/gbt/toxicity_probe_gbt.yaml").resolve()
    gbt = _load_gbt_config(gbt_path)
    ids = [f"s{index}" for index in range(30)]
    y = np.arange(30, dtype=float) / 10
    plan = {
        "plan_hash": "plan-1",
        "selection_contract": {
            "requested_invariants": ["PL", "EIC"],
            "primary_metric": "R2",
            "selection_objective": "maximize_rank1",
            "user_target": None,
        },
        "candidate_feature_signatures": {
            "candidate": {"PL": "pl-signature", "EIC": "eic-signature"}
        },
        "tasks": [
            {"invariant": "PL", "feature_signature": "pl-signature"},
            {"invariant": "EIC", "feature_signature": "eic-signature"},
        ],
    }
    qc = {
        "plan_hash": "plan-1", "qc_hash": "qc-1", "test_labels_used": False,
        "candidates": [{"hard_gate_passed": True}],
    }
    probe = {
        "selection_hash": "probe-1", "test_labels_used": False,
        "probe_sample_ids": ids,
        "probe_fold_assignment": {sample_id: index % 5 for index, sample_id in enumerate(ids)},
    }
    paths = {}
    for name, payload in (("plan", plan), ("qc", qc), ("probe", probe)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    for method, signature, prediction in (
        ("PL", "pl-signature", y * 0.2),
        ("EIC", "eic-signature", y + np.sin(y) * 0.02),
    ):
        oof = {
            "report_schema": "mint-agent.toxicity-probe-oof.v1",
            "status": "COMPLETE", "invariant": method,
            "feature_signature": signature, "sample_ids": ids,
            "feature_plan_hash": "plan-1", "feature_qc_hash": "qc-1",
            "probe_selection_hash": "probe-1",
            "gbt_parameter_hash": gbt.parameter_hash,
            "test_labels_used": False,
            "targets": y.tolist(), "oof_predictions": prediction.tolist(),
            "fold_ids": probe["probe_fold_assignment"],
            "cost": {"wall_seconds": 1.0},
        }
        (oof_dir / f"{signature}.json").write_text(json.dumps(oof), encoding="utf-8")

    scout = combine_toxicity_probe_gbt(
        feature_plan_path=paths["plan"], feature_qc_path=paths["qc"],
        probe_selection_path=paths["probe"], gbt_config_path=gbt_path,
        oof_dir=oof_dir, bootstrap=BootstrapConfig(replicates=30),
    )
    assert scout["primary_metric"] == "R2"
    assert scout["requested_invariants"] == ["PL", "EIC"]
    assert len(scout["subset_summaries"]) == 3
    assert scout["nominal_best"] == ["EIC"]
    assert all(set(row["invariants"]) <= {"PL", "EIC"} for row in scout["subset_summaries"])
    assert scout["test_labels_used"] is False

    skipped = run_toxicity_oof_task(
        feature_plan_path=paths["plan"], feature_qc_path=paths["qc"],
        probe_selection_path=paths["probe"], gbt_config_path=gbt_path,
        task_index=4, output_dir=oof_dir,
    )
    assert skipped["status"] == "SKIPPED"
