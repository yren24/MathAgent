from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mint_scout.data.element_pairs import ElementPairSchema
from mint_scout.filtration import FiltrationProfile
from mint_scout.filtration_audit import (
    FiltrationAuditConfig,
    assert_filtration_audit_compatible,
    run_filtration_audit,
)
from mint_scout.representation import RepresentationMode, RepresentationSpec


def _representation(points: int = 10) -> RepresentationSpec:
    schema = ElementPairSchema(
        schema_id="toy",
        system_type="protein_ligand",
        left_role="protein",
        right_role="ligand",
        left_elements=("C",),
        right_elements=("N",),
        global_element_order=None,
        pair_generation_rule="toy",
        role_aware=True,
        exclude_self_pairs=False,
        deduplicate_unordered_pairs=False,
        pair_order=(("C", "N"),),
        expected_pair_count=1,
    )
    profile = FiltrationProfile("distance", 0.0, float(points - 1), 1.0, points, "toy")
    return RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=schema,
        filtration_profiles={"PL": profile},
    ).freeze()


def _manifest(tmp_path: Path, representation: RepresentationSpec, arrays: dict[str, np.ndarray]) -> Path:
    root = tmp_path / f"repr-{representation.spec_hash}" / "PL"
    root.mkdir(parents=True)
    rows = []
    for sample_id, array in arrays.items():
        path = root / f"{sample_id}.npy"
        np.save(path, array)
        rows.append(json.dumps({"sample_id": sample_id, "invariant": "PL", "status": "cached", "output_path": str(path)}))
    manifest = tmp_path / "PL.jsonl"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def test_filtration_audit_detects_saturated_tail(tmp_path: Path):
    representation = _representation()
    values = np.repeat(
        np.asarray([0, 1, 2, 3, 4, 4, 4, 4, 4, 4], dtype=float).reshape(10, 1, 1),
        8,
        axis=2,
    )
    manifest = _manifest(tmp_path, representation, {"a": values, "b": values * 2})
    report = run_filtration_audit(
        representation=representation,
        sample_ids=("a", "b"),
        manifest_paths={"PL": manifest},
        config=FiltrationAuditConfig(tail_fraction=0.5, min_tail_points=5),
        dataset_id="toy",
        evidence_scope="full_train",
    )
    result = report["invariants"]["PL"]
    assert result["recommendation"] == "SHORTEN"
    assert "SATURATED_TAIL" in result["reason_codes"]
    assert result["effective_transition_count"] == 4
    assert result["last_effective_point_index"] == 4
    assert result["trailing_stable_transition_count"] == 5
    assert report["dataset_id"] == "toy"
    assert report["evidence_scope"] == "full_train"


def test_filtration_audit_detects_active_boundary(tmp_path: Path):
    representation = _representation()
    values = np.repeat(np.arange(1, 11, dtype=float).reshape(10, 1, 1), 8, axis=2)
    manifest = _manifest(tmp_path, representation, {"a": values, "b": values * 2})
    report = run_filtration_audit(
        representation=representation,
        sample_ids=("a", "b"),
        manifest_paths={"PL": manifest},
        config=FiltrationAuditConfig(saturation_relative_l2=0.001, boundary_relative_l2=0.05),
    )
    result = report["invariants"]["PL"]
    assert result["recommendation"] == "EXTEND"
    assert result["points"][-1]["axis_value"] == 9.0


def test_filtration_audit_detects_sparse_tail(tmp_path: Path):
    representation = _representation()
    values = np.ones((10, 1, 8), dtype=float)
    values[-5:] = 0.0
    manifest = _manifest(tmp_path, representation, {"a": values, "b": values})
    report = run_filtration_audit(
        representation=representation,
        sample_ids=("a", "b"),
        manifest_paths={"PL": manifest},
        config=FiltrationAuditConfig(tail_fraction=0.5, min_tail_points=5, sparse_zero_fraction=0.99),
    )
    result = report["invariants"]["PL"]
    assert result["recommendation"] == "SHORTEN"
    assert "SPARSE_TAIL" in result["reason_codes"]


def test_filtration_audit_compatibility_rejects_failed_report(tmp_path: Path):
    representation = _representation()
    report_path = tmp_path / "audit.json"
    report_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.filtration-axis-audit.v1",
                "status": "FAIL",
                "representation_hash": representation.spec_hash,
                "sample_ids": ["a"],
                "feature_manifests": {"PL": "PL.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="Filtration audit failed"):
        assert_filtration_audit_compatible(
            path=report_path,
            representation=representation,
            sample_ids=("a",),
            manifest_paths={"PL": Path("PL.jsonl")},
        )


def test_filtration_audit_compatibility_accepts_requested_subset(tmp_path: Path):
    representation = _representation()
    report_path = tmp_path / "audit.json"
    report_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.filtration-axis-audit.v1",
                "status": "WARN",
                "representation_hash": representation.spec_hash,
                "sample_ids": ["a"],
                "feature_manifests": {
                    "PL": "PL.jsonl",
                    "PH": "PH.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    report = assert_filtration_audit_compatible(
        path=report_path,
        representation=representation,
        sample_ids=("a",),
        manifest_paths={"PL": Path("PL.jsonl")},
    )

    assert report["status"] == "WARN"


def test_filtration_audit_compatibility_rejects_requested_manifest_mismatch(tmp_path: Path):
    representation = _representation()
    report_path = tmp_path / "audit.json"
    report_path.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.filtration-axis-audit.v1",
                "status": "PASS",
                "representation_hash": representation.spec_hash,
                "sample_ids": ["a"],
                "feature_manifests": {
                    "PL": "different.jsonl",
                    "PH": "PH.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="feature manifests do not match"):
        assert_filtration_audit_compatible(
            path=report_path,
            representation=representation,
            sample_ids=("a",),
            manifest_paths={"PL": Path("PL.jsonl")},
        )
