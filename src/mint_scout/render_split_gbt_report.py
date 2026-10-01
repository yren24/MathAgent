from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mint_scout.evaluation.metrics import higher_is_better
from mint_scout.evaluate_split_gbt import REPORT_SCHEMA


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a completed strict split GBT evaluation as Markdown."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Split GBT report must be a JSON object")
    markdown = render_split_gbt_markdown(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"input={args.input} output={args.output}")
    return 0


def render_split_gbt_markdown(report: Mapping[str, Any]) -> str:
    _assert_report_is_final(report)
    dataset_id = str(report.get("dataset_id") or "unknown")
    metric = str(report.get("primary_metric") or "PCC").upper()
    counts = _mapping(report, "sample_counts")
    selection = _mapping(report, "selection")
    final_test = _mapping(report, "final_test")
    test_metrics = _mapping(final_test, "metrics")
    bootstrap = _mapping(final_test, "bootstrap_pcc_95")
    selected = tuple(str(value) for value in selection.get("selected_subset", ()))
    if not selected:
        raise ValueError("Completed split GBT report has no selected subset")

    by_invariant = _mapping(selection, "by_invariant_metrics")
    subset_rows = selection.get("subset_scores")
    if not isinstance(subset_rows, list) or not subset_rows:
        raise ValueError("Completed split GBT report has no validation subset scores")
    ranked = sorted(
        (_mapping_value(row, "validation subset score") for row in subset_rows),
        key=lambda row: (
            -_finite_float(row.get("score"), "validation subset score")
            if higher_is_better(metric)
            else _finite_float(row.get("score"), "validation subset score"),
            len(row.get("subset", ())),
            tuple(str(value) for value in row.get("subset", ())),
        ),
    )

    lines = [
        "# MathAgent Official Split GBT Report",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{dataset_id}`",
        f"- Train samples: `{counts.get('train')}`",
        f"- Validation samples: `{counts.get('validation')}`",
        f"- Test samples: `{counts.get('test')}`",
        "- Representation design: `train only`",
        "- Invariant-subset selection: `validation only`",
        "- Final evaluation: `test once after selection freeze`",
        f"- Representation hash: `{report.get('representation_hash')}`",
        f"- GBT parameter hash: `{report.get('gbt_parameter_hash')}`",
        f"- Selection hash: `{report.get('selection_hash')}`",
        f"- Evaluation hash: `{report.get('evaluation_hash')}`",
        "",
        "## Validation Selection",
        "",
        f"- Primary metric: `{metric}`",
        f"- Selected subset: `{'+'.join(selected)}`",
        f"- Selected validation score: `{_number(selection.get('selected_score'))}`",
        "",
        "### Single Invariants",
        "",
        "| Invariant | PCC | RMSE | MAE | R2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for invariant in report.get("invariants", ()):
        values = by_invariant.get(str(invariant), {})
        if not isinstance(values, Mapping):
            values = {}
        lines.append(
            f"| {invariant} | {_number(values.get('PCC'))} | "
            f"{_number(values.get('RMSE'))} | {_number(values.get('MAE'))} | "
            f"{_number(values.get('R2'))} |"
        )

    lines.extend(
        [
            "",
            "### Top Validation Consensus Subsets",
            "",
            f"| Rank | Subset | {metric} |",
            "| ---: | --- | ---: |",
        ]
    )
    for rank, row in enumerate(ranked[:10], start=1):
        subset = "+".join(str(value) for value in row.get("subset", ()))
        lines.append(f"| {rank} | {subset} | {_number(row.get('score'))} |")

    lines.extend(
        [
            "",
            "## Final Test",
            "",
            f"- Selected subset: `{'+'.join(selected)}`",
            f"- PCC: `{_number(test_metrics.get('PCC'))}`",
            f"- RMSE: `{_number(test_metrics.get('RMSE'))}`",
            f"- MAE: `{_number(test_metrics.get('MAE'))}`",
            f"- R2: `{_number(test_metrics.get('R2'))}`",
            f"- Bootstrap PCC 95% interval: "
            f"`[{_number(bootstrap.get('lower'))}, {_number(bootstrap.get('upper'))}]`",
            f"- Bootstrap replicates: `{bootstrap.get('n_bootstrap')}`",
            f"- Fit/predict wall time: `{_number(report.get('fit_predict_elapsed_seconds'))} seconds`",
            "",
            "The test labels were not used for representation design or subset selection. "
            "This report records one final test scoring query after the selection hash was frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def _assert_report_is_final(report: Mapping[str, Any]) -> None:
    if report.get("report_schema") != REPORT_SCHEMA:
        raise ValueError(f"Expected report_schema={REPORT_SCHEMA!r}")
    if report.get("status") != "COMPLETE":
        raise ValueError("Only a COMPLETE split GBT report can be rendered")
    protocol = _mapping(report, "protocol")
    required = {
        "selection_split": "validation",
        "final_evaluation_split": "test",
        "representation_design_split": "train",
        "selection_frozen_before_test_scoring": True,
        "test_used_for_selection": False,
        "test_query_count": 1,
    }
    mismatches = {
        key: {"expected": expected, "actual": protocol.get(key)}
        for key, expected in required.items()
        if protocol.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Split GBT final-test protocol guard failed: {mismatches}")


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Split GBT report field {key!r} must be a mapping")
    return value


def _mapping_value(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6g}" if math.isfinite(number) else str(number)


if __name__ == "__main__":
    raise SystemExit(main())
