from __future__ import annotations

import json

import pytest

from mint_scout.agent.graph import (
    _materialize_filtration_repair,
    _resume_representation_from_downstream,
)
from mint_scout.artifact_memory import ArtifactRegistry
from mint_scout.filtration_repair import (
    FiltrationRepairConfig,
    REPAIR_SCHEMA,
    repair_filtration_profiles,
)
from mint_scout.representation import make_legacy_casf_representation_spec


def _audit(spec):
    return {
        "report_schema": "mint-agent.filtration-axis-audit.v1",
        "status": "WARN",
        "representation_hash": spec.spec_hash,
        "evidence_scope": "probe",
        "invariants": {
            "PH": {
                "recommendation": "EXTEND",
                "reason_codes": ["ACTIVE_AT_BOUNDARY"],
            },
            "PL": {"recommendation": "KEEP", "reason_codes": []},
            "CA": {
                "recommendation": "SHORTEN",
                "reason_codes": ["SATURATED_TAIL"],
                "last_effective_axis_value": 9.0,
            },
            "FPRC": {"recommendation": "KEEP", "reason_codes": []},
            "EIC": {
                "recommendation": "EXTEND",
                "reason_codes": ["ACTIVE_AT_BOUNDARY"],
            },
        },
    }


def test_method_specific_repair_changes_only_flagged_distance_profiles():
    baseline = make_legacy_casf_representation_spec()
    result = repair_filtration_profiles(
        baseline,
        _audit(baseline),
        requested_invariants=("PH", "PL", "CA", "FPRC", "EIC"),
        config=FiltrationRepairConfig(
            enabled=True,
            extend_factor=1.20,
            shorten_factor=0.85,
            trailing_buffer_points=4,
        ),
    )

    repaired = result.representation_spec
    assert result.changed is True
    assert result.repair_round == 1
    assert repaired.spec_hash != baseline.spec_hash
    assert repaired.frozen is True
    assert repaired.filtration_profiles["PH"].stop == pytest.approx(17.88)
    assert repaired.filtration_profiles["CA"].stop == pytest.approx(12.75)
    assert repaired.filtration_profiles["PL"] == baseline.filtration_profiles["PL"]
    assert repaired.filtration_profiles["FPRC"] == baseline.filtration_profiles["FPRC"]
    assert repaired.filtration_profiles["EIC"] == baseline.filtration_profiles["EIC"]
    for invariant in ("PH", "PL", "CA", "FPRC"):
        assert repaired.filtration_profiles[invariant].num_points == baseline.filtration_profiles[
            invariant
        ].num_points

    report = result.report(
        task_id="toy",
        audit=_audit(baseline),
        requested_invariants=("PH", "PL", "CA", "FPRC", "EIC"),
    )
    assert report["report_schema"] == REPAIR_SCHEMA
    assert report["parent_representation_hash"] == baseline.spec_hash
    assert report["representation_hash"] == repaired.spec_hash


def test_repair_refuses_test_or_other_representation_evidence():
    baseline = make_legacy_casf_representation_spec()
    audit = _audit(baseline)
    audit["representation_hash"] = "other-representation"

    with pytest.raises(ValueError, match="does not match"):
        repair_filtration_profiles(
            baseline,
            audit,
            requested_invariants=("PH",),
            config=FiltrationRepairConfig(enabled=True),
        )


def test_repair_stops_after_configured_round_limit():
    baseline = make_legacy_casf_representation_spec()
    first = repair_filtration_profiles(
        baseline,
        _audit(baseline),
        requested_invariants=("PH",),
        config=FiltrationRepairConfig(enabled=True, max_rounds=1),
    )
    second_audit = _audit(first.representation_spec)
    second = repair_filtration_profiles(
        first.representation_spec,
        second_audit,
        requested_invariants=("PH",),
        config=FiltrationRepairConfig(enabled=True, max_rounds=1),
    )

    assert first.changed is True
    assert second.changed is False
    assert second.representation_spec is first.representation_spec


def test_graph_materializes_and_registers_a_repair_near_its_audit(tmp_path):
    baseline = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "base-representation.json"
    baseline.write(representation_path)
    audit = _audit(baseline)
    audit["path"] = str(tmp_path / "probe-filtration-audit.json")
    registry = ArtifactRegistry(tmp_path / "artifacts.sqlite")

    repair = _materialize_filtration_repair(
        representation_spec_path=representation_path,
        representation_hash=baseline.spec_hash,
        audit=audit,
        task={
            "representation_design": {
                "filtration_repair": {"enabled": True, "max_rounds": 1}
            }
        },
        task_id="toy",
        requested_invariants=("PH", "PL", "CA", "FPRC", "EIC"),
        registry=registry,
    )

    assert repair is not None
    assert repair["kind"] == "filtration_repair"
    assert repair["path"].startswith(str(tmp_path / "filtration_repairs"))
    assert repair["metadata"]["parent_representation_hash"] == baseline.spec_hash
    assert repair["representation_hash"] != baseline.spec_hash


def test_graph_reuses_existing_equivalent_repair_for_new_run(tmp_path):
    baseline = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "base-representation.json"
    baseline.write(representation_path)
    registry = ArtifactRegistry(tmp_path / "artifacts.sqlite")

    first_audit = _audit(baseline)
    first_audit["path"] = str(tmp_path / "first-filtration-audit.json")
    first = _materialize_filtration_repair(
        representation_spec_path=representation_path,
        representation_hash=baseline.spec_hash,
        audit=first_audit,
        task={
            "representation_design": {
                "filtration_repair": {"enabled": True, "max_rounds": 1}
            }
        },
        task_id="first-run",
        requested_invariants=("PH", "PL", "CA", "FPRC", "EIC"),
        registry=registry,
    )
    assert first is not None

    second_audit = _audit(baseline)
    second_audit["path"] = str(tmp_path / "second-filtration-audit.json")
    second_audit["content_sha256"] = "different-audit-content"
    second = _materialize_filtration_repair(
        representation_spec_path=representation_path,
        representation_hash=baseline.spec_hash,
        audit=second_audit,
        task={
            "representation_design": {
                "filtration_repair": {"enabled": True, "max_rounds": 1}
            }
        },
        task_id="second-run",
        requested_invariants=("PH", "PL", "CA", "FPRC", "EIC"),
        registry=registry,
    )

    assert second is not None
    assert second["path"] == first["path"]
    assert second["dataset_id"] == "second-run"
    records = registry.query(
        artifact_kind="filtration_repair",
        dataset_id="second-run",
        status="COMPLETE",
    )
    assert len(records) == 1
    assert records[0].path == first["path"]


def test_graph_materializes_repair_from_artifact_summary(tmp_path):
    baseline = make_legacy_casf_representation_spec()
    representation_path = tmp_path / "base-representation.json"
    baseline.write(representation_path)
    audit_path = tmp_path / "probe-filtration-audit.json"
    full_audit = _audit(baseline)
    full_audit["path"] = str(audit_path)
    audit_path.write_text(json.dumps(full_audit), encoding="utf-8")
    summary = {
        "kind": "filtration_audit",
        "status": "WARN",
        "path": str(audit_path),
        "content_sha256": "fake",
        "representation_hash": baseline.spec_hash,
        "invariants": ["PH", "PL", "CA", "FPRC", "EIC"],
    }
    registry = ArtifactRegistry(tmp_path / "artifacts.sqlite")

    repair = _materialize_filtration_repair(
        representation_spec_path=representation_path,
        representation_hash=baseline.spec_hash,
        audit=summary,
        task={
            "representation_design": {
                "filtration_repair": {"enabled": True, "max_rounds": 1}
            }
        },
        task_id="toy",
        requested_invariants=("PH", "PL", "CA", "FPRC", "EIC"),
        registry=registry,
    )

    assert repair is not None
    assert repair["kind"] == "filtration_repair"
    assert repair["metadata"]["parent_representation_hash"] == baseline.spec_hash


def test_resume_representation_uses_downstream_evidence():
    artifacts = [
        {
            "kind": "representation_design",
            "status": "COMPLETE",
            "path": "representation.json",
            "representation_hash": "rep-a",
        },
        {
            "kind": "filtration_audit",
            "status": "WARN",
            "path": "filtration.json",
            "representation_hash": "rep-a",
        },
    ]

    representation = _resume_representation_from_downstream(artifacts)

    assert representation is not None
    assert representation["representation_hash"] == "rep-a"


def test_resume_representation_prefers_repair_downstream_evidence():
    artifacts = [
        {
            "kind": "representation_design",
            "status": "COMPLETE",
            "path": "representation.json",
            "representation_hash": "rep-a",
        },
        {
            "kind": "filtration_repair",
            "status": "COMPLETE",
            "path": "repair.json",
            "representation_hash": "rep-b",
        },
        {
            "kind": "probe_selection",
            "status": "COMPLETE",
            "path": "probe.json",
            "representation_hash": "rep-b",
        },
    ]

    representation = _resume_representation_from_downstream(artifacts)

    assert representation is not None
    assert representation["kind"] == "filtration_repair"
    assert representation["representation_hash"] == "rep-b"
