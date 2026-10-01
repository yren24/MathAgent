from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mint_scout.data.element_pairs import ElementPairSchema
from mint_scout.feature_qc import _resolve_sample_ids, assert_qc_report_compatible, main
from mint_scout.filtration import FiltrationProfile
from mint_scout.representation import RepresentationMode, RepresentationSpec


def _representation(tmp_path: Path) -> RepresentationSpec:
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
    profile = FiltrationProfile("distance", 0.0, 1.0, 1.0, 2, "toy")
    return RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=schema,
        filtration_profiles={"PH": profile, "PL": profile},
    ).freeze()


def _write_probe(path: Path, representation: RepresentationSpec, sample_ids: tuple[str, ...]) -> None:
    path.write_text(
        json.dumps(
            {
                "representation_hash": representation.spec_hash,
                "selection_hash": "selection-test",
                "probe_sample_ids": list(sample_ids),
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    *,
    representation: RepresentationSpec,
    sample_ids: tuple[str, ...],
    invariant: str = "PH",
    shape: tuple[int, int, int] = (2, 1, 2),
) -> None:
    rows = []
    feature_root = path.parent / f"repr-{representation.spec_hash}" / invariant
    feature_root.mkdir(parents=True)
    for index, sample_id in enumerate(sample_ids):
        feature_path = feature_root / f"{sample_id}.npy"
        values = np.arange(np.prod(shape), dtype=float).reshape(shape) + float(index + 1)
        np.save(feature_path, values)
        rows.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "invariant": invariant,
                    "status": "computed",
                    "output_path": str(feature_path),
                }
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_feature_qc_passes_valid_manifest(tmp_path: Path):
    representation = _representation(tmp_path)
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    sample_ids = ("a001", "a002", "a003")
    probe = tmp_path / "probe.json"
    _write_probe(probe, representation, sample_ids)
    manifest = tmp_path / "PH.jsonl"
    _write_manifest(manifest, representation=representation, sample_ids=sample_ids)
    output = tmp_path / "qc.json"

    code = main(
        [
            "--representation-spec",
            str(representation_path),
            "--probe-selection",
            str(probe),
            "--feature-manifest",
            f"PH={manifest}",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["status"] == "PASS"
    assert payload["invariants"]["PH"]["loaded_sample_count"] == 3
    assert payload["representation_hash"] == representation.spec_hash
    population = payload["invariants"]["PH"]["aggregate"]["coordinate_population"]
    assert population["valid_sample_count"] == 3
    assert population["coordinate_count"] == 4


def test_feature_qc_fails_shape_mismatch(tmp_path: Path):
    representation = _representation(tmp_path)
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    sample_ids = ("a001",)
    probe = tmp_path / "probe.json"
    _write_probe(probe, representation, sample_ids)
    manifest = tmp_path / "PH.jsonl"
    _write_manifest(manifest, representation=representation, sample_ids=sample_ids, shape=(3, 1, 2))
    output = tmp_path / "qc.json"

    code = main(
        [
            "--representation-spec",
            str(representation_path),
            "--probe-selection",
            str(probe),
            "--feature-manifest",
            f"PH={manifest}",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["status"] == "FAIL"
    assert payload["blocking_issue_count"] == 1
    assert payload["issues"][0]["issue"] == "shape_mismatch"


def test_qc_report_compatibility_blocks_failed_qc(tmp_path: Path):
    representation = _representation(tmp_path)
    report = tmp_path / "qc.json"
    report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.feature-qc.v1",
                "status": "FAIL",
                "representation_hash": representation.spec_hash,
                "sample_ids": ["a001"],
                "feature_manifests": {"PH": str(tmp_path / "PH.jsonl")},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Feature QC failed"):
        assert_qc_report_compatible(
            path=report,
            representation=representation,
            sample_ids=("a001",),
            manifest_paths={"PH": tmp_path / "PH.jsonl"},
        )


def test_qc_report_covering_superset_certifies_requested_subset(tmp_path: Path):
    representation = _representation(tmp_path)
    ph_manifest = tmp_path / "PH.jsonl"
    pl_manifest = tmp_path / "PL.jsonl"
    report = tmp_path / "qc.json"
    report.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.feature-qc.v1",
                "status": "PASS",
                "representation_hash": representation.spec_hash,
                "sample_ids": ["a001"],
                "feature_manifests": {
                    "PH": str(ph_manifest),
                    "PL": str(pl_manifest),
                },
            }
        ),
        encoding="utf-8",
    )

    compatible = assert_qc_report_compatible(
        path=report,
        representation=representation,
        sample_ids=("a001",),
        manifest_paths={"PL": pl_manifest},
    )

    assert compatible["status"] == "PASS"


def test_sample_id_file_can_use_frozen_modeling_pool(tmp_path: Path):
    path = tmp_path / "probe-selection.json"
    path.write_text(
        json.dumps(
            {
                "probe_sample_ids": ["probe-only"],
                "modeling_sample_ids": ["train-a", "train-b"],
                "selection_hash": "probe-hash",
            }
        ),
        encoding="utf-8",
    )

    sample_ids, selection_hash = _resolve_sample_ids(None, path)

    assert sample_ids == ("train-a", "train-b")
    assert selection_hash is None


def test_sample_id_file_can_be_feature_manifest(tmp_path: Path):
    path = tmp_path / "PH.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "a", "status": "computed"}),
                json.dumps({"sample_id": "b", "status": "cached"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sample_ids, selection_hash = _resolve_sample_ids(None, path)

    assert sample_ids == ("a", "b")
    assert selection_hash is None


def test_feature_qc_reports_population_dead_coordinates(tmp_path: Path):
    representation = _representation(tmp_path)
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    sample_ids = ("a001", "a002", "a003")
    manifest = tmp_path / "PH.jsonl"
    _write_manifest(manifest, representation=representation, sample_ids=sample_ids)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for index, row in enumerate(rows):
        values = np.zeros((2, 1, 2), dtype=float)
        values[0, 0, 0] = index + 1.0
        np.save(row["output_path"], values)
    qc_config = tmp_path / "qc.yaml"
    qc_config.write_text(
        "\n".join(
            [
                "feature_qc:",
                "  robust_min_samples: 2",
                "  population_all_zero_warning_fraction: 0.5",
                "  population_constant_warning_fraction: 0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "qc.json"

    code = main(
        [
            "--representation-spec",
            str(representation_path),
            "--sample-id-file",
            str(manifest),
            "--feature-manifest",
            f"PH={manifest}",
            "--qc-config",
            str(qc_config),
            "--dataset-id",
            "toy",
            "--evidence-scope",
            "full_train",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    population = payload["invariants"]["PH"]["aggregate"]["coordinate_population"]
    assert code == 0
    assert payload["status"] == "WARN"
    assert payload["dataset_id"] == "toy"
    assert population["all_zero_coordinate_count"] == 3
    assert {issue["issue"] for issue in payload["issues"]} >= {
        "many_all_zero_coordinates",
        "many_constant_coordinates",
    }
