from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


FEATURE_HEALTH_SCHEMA = "mint-agent.feature-health.v1"


def summarize_feature_health(
    qc_report: Mapping[str, Any],
    filtration_report: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_schema(
        qc_report,
        expected="mint-agent.feature-qc.v1",
        label="feature QC",
    )
    _validate_schema(
        filtration_report,
        expected="mint-agent.filtration-axis-audit.v1",
        label="filtration audit",
    )
    identity = _validated_identity(qc_report, filtration_report)
    actions = _qc_actions(qc_report) + _filtration_actions(filtration_report)
    blocking = (
        str(qc_report.get("status")) == "FAIL"
        or str(filtration_report.get("status")) == "FAIL"
        or any(action["severity"] == "error" for action in actions)
    )
    warned = (
        str(qc_report.get("status")) == "WARN"
        or str(filtration_report.get("status")) == "WARN"
        or bool(actions)
    )
    if blocking:
        status = "BLOCKED"
    elif warned:
        status = "PROCEED_WITH_WARNINGS"
    else:
        status = "PASS"

    return {
        "report_schema": FEATURE_HEALTH_SCHEMA,
        "status": status,
        "proceed_allowed": not blocking,
        "automatic_sample_removal": False,
        "automatic_representation_mutation": False,
        "representation_change_requires_new_spec": any(
            action["action"]
            in {"COMPARE_EXTENDED_FILTRATION", "COMPARE_SHORTER_FILTRATION"}
            for action in actions
        ),
        "identity": identity,
        "source_status": {
            "feature_qc": qc_report.get("status"),
            "filtration_audit": filtration_report.get("status"),
        },
        "source_hashes": {
            "feature_qc": qc_report.get("qc_hash"),
            "filtration_audit": filtration_report.get("audit_hash"),
        },
        "action_count": len(actions),
        "actions": actions,
    }


def summarize_feature_health_files(
    qc_path: str | Path,
    filtration_path: str | Path,
    *,
    identity_hint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = {
        "feature_qc": Path(qc_path).expanduser(),
        "filtration_audit": Path(filtration_path).expanduser(),
    }
    try:
        reports = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items()
        }
        if not all(isinstance(report, dict) for report in reports.values()):
            raise ValueError("feature-health inputs must contain JSON objects")
        summary = summarize_feature_health(
            reports["feature_qc"], reports["filtration_audit"]
        )
        if identity_hint is not None:
            identity = summary["identity"]
            for field in (
                "dataset_id",
                "evidence_scope",
                "representation_hash",
                "selection_hash",
                "sample_count",
            ):
                if identity.get(field) is None and identity_hint.get(field) is not None:
                    identity[field] = identity_hint[field]
        return summary
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "report_schema": FEATURE_HEALTH_SCHEMA,
            "status": "UNAVAILABLE",
            "proceed_allowed": None,
            "automatic_sample_removal": False,
            "automatic_representation_mutation": False,
            "input_paths": {name: str(path) for name, path in paths.items()},
            "reason": str(exc),
            "action_count": 0,
            "actions": [],
        }


def _qc_actions(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = []
    for issue in report.get("issues", ()):
        if not isinstance(issue, Mapping):
            continue
        code = str(issue.get("issue") or "unknown_feature_issue")
        severity = str(issue.get("severity") or "warning")
        actions.append(
            {
                "source": "feature_qc",
                "severity": severity,
                "invariant": issue.get("invariant"),
                "sample_id": issue.get("sample_id"),
                "reason_code": code,
                "action": _qc_action(code, severity),
                "message": issue.get("message"),
                "automatic_mutation": False,
            }
        )
    return actions


def _filtration_actions(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = []
    invariants = report.get("invariants", {})
    if not isinstance(invariants, Mapping):
        return actions
    for invariant in sorted(invariants):
        item = invariants[invariant]
        if not isinstance(item, Mapping):
            continue
        recommendation = str(item.get("recommendation") or "REVIEW")
        if recommendation == "KEEP" and item.get("status") != "FAIL":
            continue
        reason_codes = [str(code) for code in item.get("reason_codes", ())]
        actions.append(
            {
                "source": "filtration_audit",
                "severity": "error" if item.get("status") == "FAIL" else "warning",
                "invariant": str(invariant),
                "sample_id": None,
                "reason_codes": reason_codes,
                "recommendation": recommendation,
                "action": _filtration_action(recommendation),
                "automatic_mutation": False,
            }
        )
    return actions


def _qc_action(code: str, severity: str) -> str:
    if severity == "error":
        return "BLOCK_AND_REGENERATE_FEATURE_INPUT"
    if code == "robust_outlier":
        return "REVIEW_SAMPLE_FEATURE_NORM"
    if code == "large_magnitude":
        return "REVIEW_NUMERICAL_SCALE"
    if code in {
        "all_zero",
        "constant",
        "high_sparsity",
        "many_all_zero_coordinates",
        "many_constant_coordinates",
    }:
        return "REVIEW_FEATURE_DEGENERACY"
    return "REVIEW_FEATURE_WARNING"


def _filtration_action(recommendation: str) -> str:
    if recommendation == "EXTEND":
        return "COMPARE_EXTENDED_FILTRATION"
    if recommendation == "SHORTEN":
        return "COMPARE_SHORTER_FILTRATION"
    return "REVIEW_FILTRATION_INPUT"


def _validated_identity(
    qc_report: Mapping[str, Any],
    filtration_report: Mapping[str, Any],
) -> dict[str, Any]:
    fields = (
        "dataset_id",
        "evidence_scope",
        "representation_hash",
        "selection_hash",
        "sample_count",
        "sample_ids",
    )
    for field in fields:
        if qc_report.get(field) != filtration_report.get(field):
            raise ValueError(f"feature-health source mismatch: {field}")
    qc_invariants = _invariant_names(qc_report)
    filtration_invariants = _invariant_names(filtration_report)
    if qc_invariants != filtration_invariants:
        raise ValueError("feature-health source mismatch: invariants")
    return {
        "dataset_id": qc_report.get("dataset_id"),
        "evidence_scope": qc_report.get("evidence_scope"),
        "representation_hash": qc_report.get("representation_hash"),
        "selection_hash": qc_report.get("selection_hash"),
        "sample_count": qc_report.get("sample_count"),
        "invariants": list(qc_invariants),
    }


def _invariant_names(report: Mapping[str, Any]) -> tuple[str, ...]:
    raw = report.get("invariants", {})
    if not isinstance(raw, Mapping):
        raise ValueError("feature-health source invariants must be a mapping")
    return tuple(sorted(str(name) for name in raw))


def _validate_schema(
    report: Mapping[str, Any], *, expected: str, label: str
) -> None:
    if report.get("report_schema") != expected:
        raise ValueError(f"unsupported {label} schema: {report.get('report_schema')!r}")
