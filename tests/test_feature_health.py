from __future__ import annotations

import json
from copy import deepcopy

import pytest

from mint_scout.feature_health import (
    summarize_feature_health,
    summarize_feature_health_files,
)


def _reports():
    identity = {
        "dataset_id": "toy",
        "evidence_scope": "probe",
        "representation_hash": "repr",
        "selection_hash": "selection",
        "sample_count": 2,
        "sample_ids": ["a", "b"],
    }
    qc = {
        **identity,
        "report_schema": "mint-agent.feature-qc.v1",
        "status": "WARN",
        "qc_hash": "qc",
        "invariants": {"FPRC": {"status": "WARN"}, "PH": {"status": "PASS"}},
        "issues": [
            {
                "invariant": "FPRC",
                "sample_id": "b",
                "issue": "robust_outlier",
                "severity": "warning",
                "message": "large robust z score",
            }
        ],
    }
    filtration = {
        **identity,
        "report_schema": "mint-agent.filtration-axis-audit.v1",
        "status": "WARN",
        "audit_hash": "filtration",
        "invariants": {
            "FPRC": {
                "status": "WARN",
                "recommendation": "EXTEND",
                "reason_codes": ["ACTIVE_AT_BOUNDARY"],
            },
            "PH": {
                "status": "PASS",
                "recommendation": "KEEP",
                "reason_codes": [],
            },
        },
    }
    return qc, filtration


def test_feature_health_preserves_warnings_without_automatic_mutation():
    qc, filtration = _reports()

    result = summarize_feature_health(qc, filtration)

    assert result["status"] == "PROCEED_WITH_WARNINGS"
    assert result["proceed_allowed"] is True
    assert result["automatic_sample_removal"] is False
    assert result["automatic_representation_mutation"] is False
    assert result["representation_change_requires_new_spec"] is True
    assert [action["action"] for action in result["actions"]] == [
        "REVIEW_SAMPLE_FEATURE_NORM",
        "COMPARE_EXTENDED_FILTRATION",
    ]


def test_feature_health_blocks_error_but_does_not_mutate_input():
    qc, filtration = _reports()
    qc["status"] = "FAIL"
    qc["issues"][0]["severity"] = "error"
    qc_before = deepcopy(qc)

    result = summarize_feature_health(qc, filtration)

    assert result["status"] == "BLOCKED"
    assert result["proceed_allowed"] is False
    assert result["actions"][0]["action"] == "BLOCK_AND_REGENERATE_FEATURE_INPUT"
    assert qc == qc_before


def test_feature_health_rejects_mismatched_scientific_identity():
    qc, filtration = _reports()
    filtration["selection_hash"] = "different-selection"

    with pytest.raises(ValueError, match="selection_hash"):
        summarize_feature_health(qc, filtration)


def test_feature_health_files_can_fill_legacy_optional_identity(tmp_path):
    qc, filtration = _reports()
    qc["dataset_id"] = None
    qc["evidence_scope"] = None
    filtration["dataset_id"] = None
    filtration["evidence_scope"] = None
    qc_path = tmp_path / "qc.json"
    filtration_path = tmp_path / "filtration.json"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    filtration_path.write_text(json.dumps(filtration), encoding="utf-8")

    result = summarize_feature_health_files(
        qc_path,
        filtration_path,
        identity_hint={"dataset_id": "toy", "evidence_scope": "probe"},
    )

    assert result["identity"]["dataset_id"] == "toy"
    assert result["identity"]["evidence_scope"] == "probe"
