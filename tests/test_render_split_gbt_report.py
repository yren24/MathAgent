from __future__ import annotations

import json

import pytest

from mint_scout.render_split_gbt_report import main, render_split_gbt_markdown


def _report() -> dict[str, object]:
    metrics = {"PCC": 0.7, "RMSE": 1.2, "MAE": 0.9, "R2": 0.45}
    return {
        "report_schema": "mint-agent.split-gbt-evaluation.v1",
        "status": "COMPLETE",
        "dataset_id": "official-data",
        "primary_metric": "PCC",
        "protocol": {
            "selection_split": "validation",
            "final_evaluation_split": "test",
            "representation_design_split": "train",
            "selection_frozen_before_test_scoring": True,
            "test_used_for_selection": False,
            "test_query_count": 1,
        },
        "representation_hash": "repr-1",
        "gbt_parameter_hash": "gbt-1",
        "selection_hash": "selection-1",
        "evaluation_hash": "evaluation-1",
        "sample_counts": {"train": 10, "validation": 4, "test": 5},
        "invariants": ["PH", "PL"],
        "selection": {
            "selected_subset": ["PH", "PL"],
            "selected_score": 0.8,
            "by_invariant_metrics": {
                "PH": metrics,
                "PL": {**metrics, "PCC": 0.75},
            },
            "subset_scores": [
                {"subset": ["PH"], "score": 0.7},
                {"subset": ["PL"], "score": 0.75},
                {"subset": ["PH", "PL"], "score": 0.8},
            ],
        },
        "final_test": {
            "metrics": {**metrics, "PCC": 0.78},
            "bootstrap_pcc_95": {
                "n_bootstrap": 1000,
                "seed": 2026,
                "lower": 0.66,
                "upper": 0.86,
            },
        },
        "fit_predict_elapsed_seconds": 12.5,
    }


def test_render_split_gbt_markdown_explains_strict_protocol():
    markdown = render_split_gbt_markdown(_report())

    assert "Selected subset: `PH+PL`" in markdown
    assert "| 1 | PH+PL | 0.8 |" in markdown
    assert "PCC: `0.78`" in markdown
    assert "one final test scoring query" in markdown


def test_render_split_gbt_markdown_rejects_test_selection():
    report = _report()
    protocol = report["protocol"]
    assert isinstance(protocol, dict)
    protocol["test_used_for_selection"] = True

    with pytest.raises(ValueError, match="protocol guard failed"):
        render_split_gbt_markdown(report)


def test_render_split_gbt_report_cli_writes_markdown(tmp_path):
    source = tmp_path / "evaluation.json"
    output = tmp_path / "evaluation.md"
    source.write_text(json.dumps(_report()), encoding="utf-8")

    assert main(["--input", str(source), "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8").startswith(
        "# MathAgent Official Split GBT Report"
    )
