from __future__ import annotations

import json

import numpy as np

from mint_scout.representation import make_legacy_toxicity_representation_spec
from mint_scout.toxicity.feature_qc import (
    ToxicityFeatureQCPolicy,
    audit_toxicity_feature_plan,
)
from mint_scout.toxicity.feature_plan import toxicity_feature_signature


def test_toxicity_feature_qc_hard_gate_and_scientific_warning(tmp_path):
    spec = make_legacy_toxicity_representation_spec()
    spec_path = tmp_path / "spec.json"
    spec.write(spec_path)
    signature = toxicity_feature_signature(spec, "PH")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    rows = []
    for index in range(3):
        path = feature_dir / f"sample-{index}.npy"
        values = np.zeros((50, 30, 1), dtype=np.float32)
        values[:, 0, 0] = np.arange(50, dtype=np.float32) + index
        np.save(path, values)
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "status": "computed",
                "output_path": str(path),
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    plan = {
        "plan_hash": "test-plan",
        "tasks": [
            {
                "task_index": 0,
                "feature_signature": signature,
                "invariant": "PH",
                "candidate_ids": ["legacy"],
                "representation_spec": str(spec_path),
                "manifest_path": str(manifest),
            }
        ],
        "candidate_feature_signatures": {"legacy": {"PH": signature}},
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = audit_toxicity_feature_plan(
        plan_path=plan_path,
        policy=ToxicityFeatureQCPolicy(expected_sample_count=3),
    )

    task = report["tasks"][0]
    assert task["hard_gate_passed"] is True
    assert task["status"] == "PASS"
    assert task["filtration"]["recommendation"] == "KEEP"
    assert report["candidates"][0]["hard_gate_passed"] is True


def test_toxicity_feature_qc_rejects_wrong_shape(tmp_path):
    spec = make_legacy_toxicity_representation_spec()
    spec_path = tmp_path / "spec.json"
    spec.write(spec_path)
    signature = toxicity_feature_signature(spec, "PL")
    feature = tmp_path / "wrong.npy"
    np.save(feature, np.ones((2, 2), dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample-0",
                "status": "computed",
                "output_path": str(feature),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan = {
        "tasks": [
            {
                "task_index": 0,
                "feature_signature": signature,
                "invariant": "PL",
                "candidate_ids": ["legacy"],
                "representation_spec": str(spec_path),
                "manifest_path": str(manifest),
            }
        ],
        "candidate_feature_signatures": {"legacy": {"PL": signature}},
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = audit_toxicity_feature_plan(plan_path=plan_path)

    assert report["tasks"][0]["status"] == "FAIL"
    assert "SHAPE_MISMATCH" in report["tasks"][0]["hard_failures"]
