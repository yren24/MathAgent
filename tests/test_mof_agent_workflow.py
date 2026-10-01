import csv
import json
from pathlib import Path

import numpy as np

from mint_scout.mof.agent_workflow import (
    WORKFLOW_SCHEMA,
    prepare_workflow,
    select_stratified_probe,
    write_workflow,
)
from mint_scout.mof.manifest import MofManifestRecord, write_manifest
from mint_scout.mof.probe_gbt import run_probe_gbt
from mint_scout.mof.selection import select_probe_method


def _records(tmp_path, count=20):
    records = [
        MofManifestRecord(f"MOF{index:03d}", float(index), tmp_path / f"MOF{index:03d}.cif")
        for index in range(count)
    ]
    for record in records:
        record.cif_path.write_text("data_mof\n")
    manifest = write_manifest(records, tmp_path / "manifest.csv")
    return records, manifest


def test_probe_is_train_only_stratified_and_deterministic(tmp_path):
    records, _ = _records(tmp_path)

    first = select_stratified_probe(records, probe_size=8, strata=4, seed=7)
    second = select_stratified_probe(records, probe_size=8, strata=4, seed=7)

    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8
    assert any(int(sample[-3:]) < 5 for sample in first)
    assert any(int(sample[-3:]) >= 15 for sample in first)


def test_workflow_reports_partial_tool_availability_without_selecting_winner(tmp_path):
    records, manifest = _records(tmp_path)
    feature_root = tmp_path / "features"
    results_root = tmp_path / "results"
    for record in records:
        path = feature_root / "homology" / "O2" / f"{record.sample_id}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"feature")
    metric_path = results_root / "O2_homology_repeat0_gbt_standard_metrics.json"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps({"mean_rmse": 0.1, "mean_r2_standard": 0.8}))

    report = prepare_workflow(
        manifest_path=manifest,
        feature_dir=feature_root,
        result_dir=results_root,
        property_name="O2",
        probe_size=8,
        probe_strata=4,
        seed=7,
    )
    report_path, probe_path = write_workflow(report, tmp_path / "workflow.json")

    assert report["report_schema"] == WORKFLOW_SCHEMA
    assert report["workflow_status"] == "PROBE_SCOUT_READY_PARTIAL_TOOLSET"
    assert report["tools"][0]["full_feature_ready"] is True
    assert report["tools"][0]["standard_metrics"]["mean_rmse"] == 0.1
    assert report["tools"][1]["full_feature_ready"] is False
    assert json.loads(report_path.read_text())["probe_selection"]["evidence_scope"] == "train_only"
    with probe_path.open() as handle:
        assert len([line for line in handle if line.strip()]) == 8


def test_probe_gbt_uses_only_frozen_train_probe_and_writes_standard_metrics(tmp_path):
    records, manifest = _records(tmp_path, count=12)
    feature_dir = tmp_path / "features"
    for index, record in enumerate(records):
        path = feature_dir / "homology" / "O2" / f"{record.sample_id}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray([index, index % 3], dtype=np.float32))
    probe_ids = tmp_path / "probe.txt"
    probe_ids.write_text("\n".join(record.sample_id for record in records[:10]) + "\n")

    report = run_probe_gbt(
        manifest_path=manifest,
        probe_id_path=probe_ids,
        feature_dir=feature_dir,
        topology="homology",
        property_name="O2",
        output_dir=tmp_path / "output",
        folds=2,
        repeats=2,
        n_estimators=2,
        max_depth=2,
    )

    assert report["evidence_scope"] == "train_only_probe"
    assert len(report["fold_metrics"]) == 4
    assert len(report["repeat_summaries"]) == 2
    assert report["summary"]["mean_rmse"] is not None
    assert Path(report["predictions_path"]).exists()


def test_selection_uses_standard_r2_then_repeat_stability_then_feature_cost():
    def status(name, r2, repeat_r2, dimension):
        return {
            "tool_name": name,
            "probe_feature_ready": True,
            "feature_dimension": dimension,
            "probe_metrics": {
                "summary": {"mean_r2_standard": r2},
                "repeat_summaries": [
                    {"mean_r2_standard": value} for value in repeat_r2
                ],
            },
        }

    report = select_probe_method(
        [
            status("A", 0.80, [0.80, 0.80, 0.80], 100),
            status("B", 0.797, [0.76, 0.83, 0.80], 10),
        ]
    )

    assert report["selected_tool"] == "A"
    assert report["is_final_selection"] is True
    assert report["performance_tier"] == ["A", "B"]
    assert report["stability_tier"] == ["A"]
