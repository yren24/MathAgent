from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mint_scout.execute_casf import _load_gbt_config, _parse_priority_order, main
from mint_scout.execution import ScoutExecutionArtifact
from mint_scout.representation import RepresentationMode, make_legacy_casf_representation_spec


def _write_casf_fixture(tmp_path: Path, *, sample_count: int = 6) -> tuple[Path, Path]:
    index_root = tmp_path / "index"
    structures_root = tmp_path / "structures"
    index_root.mkdir()
    rows = []
    sample_ids = []
    for index in range(sample_count):
        sample_id = f"x{index:03d}"
        sample_ids.append(sample_id)
        rows.append(f"{sample_id}  2.0  2000  {5.0 + index / 10.0:.2f}  Kd=1nM\n")
        sample_root = structures_root / sample_id
        sample_root.mkdir(parents=True)
        (sample_root / f"{sample_id}_pocket.pdb").write_text("", encoding="utf-8")
        (sample_root / f"{sample_id}_ligand.mol2").write_text("", encoding="utf-8")
    (index_root / "2016_INDEX_refined.data").write_text("".join(rows), encoding="utf-8")
    (index_root / "train_data_2016.txt").write_text(repr(sample_ids), encoding="utf-8")
    (index_root / "test_data_2016.txt").write_text("[]", encoding="utf-8")

    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: toy\n"
        "primary_metric: pcc\n"
        "pair_schema:\n"
        "  schema_id: toy_40\n"
        "cv:\n"
        "  n_splits: 3\n"
        "stopping:\n"
        "  target_pcc: 0.75\n"
        "legacy:\n"
        f"  plbind_root: {tmp_path / 'legacy'}\n"
        "data_audit:\n"
        "  casf:\n"
        "    year: 2016\n"
        f"    index_root: {index_root}\n"
        f"    structures_root: {structures_root}\n"
        "feature_generation:\n"
        f"  output_root: {tmp_path / 'features'}\n",
        encoding="utf-8",
    )
    gbt = tmp_path / "gbt.yaml"
    gbt.write_text(
        "config_id: test\n"
        "normalize: StandardScaler\n"
        "n_estimators: 2\n"
        "max_depth: 2\n"
        "min_samples_split: 2\n"
        "learning_rate: 0.1\n"
        "subsample: 1.0\n"
        "max_features: sqrt\n"
        "random_state: 7\n"
        "n_runs: 1\n"
        "source: legacy.py\n",
        encoding="utf-8",
    )
    return config, gbt


def test_parse_priority_order_preserves_frozen_subset_order():
    assert _parse_priority_order("PL;PH;PL,PH") == (("PL",), ("PH",), ("PH", "PL"))


def test_parse_priority_order_rejects_unknown_invariant():
    with pytest.raises(KeyError):
        _parse_priority_order("PL;UNKNOWN")


def test_load_gbt_config_accepts_documentary_source_field(tmp_path: Path):
    _, path = _write_casf_fixture(tmp_path)
    config = _load_gbt_config(path)
    assert config.n_estimators == 2
    assert config.normalize == "StandardScaler"


def test_execute_casf_dry_run_writes_auditable_plan(tmp_path: Path):
    config, gbt = _write_casf_fixture(tmp_path)
    output = tmp_path / "phase4.json"

    code = main(
        [
            "--config",
            str(config),
            "--gbt-config",
            str(gbt),
            "--limit",
            "6",
            "--priority-order",
            "PL;PH;PL,PH",
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert report["status"] == "DRY_RUN"
    assert report["sample_count"] == 6
    assert report["cv_folds"] == 3
    assert report["target_value"] == 0.75
    assert report["gbt_config"]["normalize"] == "StandardScaler"
    assert report["priority_order"] == [["PL"], ["PH"], ["PH", "PL"]]
    assert report["run_kind"] == "engineering_smoke"


def test_execute_casf_dry_run_consumes_frozen_scout_artifact(tmp_path: Path):
    config, gbt = _write_casf_fixture(tmp_path)
    sample_ids = tuple(f"x{index:03d}" for index in range(6))
    gbt_config = _load_gbt_config(gbt)
    representation_hash = make_legacy_casf_representation_spec().spec_hash
    artifact = ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=sample_ids,
        representation_hash=representation_hash,
        frozen_priority_order=(("PH",), ("PL", "PH")),
        full_fold_assignment={sample_id: index % 3 for index, sample_id in enumerate(sample_ids)},
        target_metric="PCC",
        target_direction="higher",
        target_value=0.72,
        target_source="probe_derived",
        gbt_parameter_hash=gbt_config.parameter_hash,
        probe_hash="probe-hash",
    )
    artifact_path = tmp_path / "scout.json"
    artifact.write(artifact_path)
    output = tmp_path / "phase4-from-scout.json"

    code = main(
        [
            "--config",
            str(config),
            "--gbt-config",
            str(gbt),
            "--limit",
            "6",
            "--scout-artifact",
            str(artifact_path),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert report["priority_source"] == "scout_artifact"
    assert report["fold_source"] == "scout_artifact"
    assert report["target_value"] == 0.72
    assert report["representation_hash"] == representation_hash


def test_execute_casf_scout_artifact_uses_all_samples_when_limit_is_omitted(tmp_path: Path):
    sample_count = 20
    config, gbt = _write_casf_fixture(tmp_path, sample_count=sample_count)
    sample_ids = tuple(f"x{index:03d}" for index in range(sample_count))
    gbt_config = _load_gbt_config(gbt)
    representation_hash = make_legacy_casf_representation_spec().spec_hash
    artifact = ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=sample_ids,
        representation_hash=representation_hash,
        frozen_priority_order=(("PL",),),
        full_fold_assignment={sample_id: index % 5 for index, sample_id in enumerate(sample_ids)},
        target_metric="PCC",
        target_direction="higher",
        target_value=0.72,
        target_source="probe_derived",
        gbt_parameter_hash=gbt_config.parameter_hash,
        probe_hash="probe-hash",
    )
    artifact_path = tmp_path / "scout-all-samples.json"
    artifact.write(artifact_path)
    output = tmp_path / "phase4-all-samples.json"

    code = main(
        [
            "--config",
            str(config),
            "--gbt-config",
            str(gbt),
            "--scout-artifact",
            str(artifact_path),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert report["sample_count"] == sample_count


def test_execute_casf_accepts_explicit_frozen_representation_for_scout(tmp_path: Path):
    config, gbt = _write_casf_fixture(tmp_path)
    sample_ids = tuple(f"x{index:03d}" for index in range(6))
    representation = replace(
        make_legacy_casf_representation_spec(),
        representation_id="adaptive-test",
        mode=RepresentationMode.DATASET_ADAPTIVE,
    )
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    artifact = ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=sample_ids,
        representation_hash=representation.spec_hash,
        frozen_priority_order=(("PL",),),
        full_fold_assignment={sample_id: index % 3 for index, sample_id in enumerate(sample_ids)},
        target_metric="PCC",
        target_direction="higher",
        target_value=0.72,
        target_source="probe_derived",
        gbt_parameter_hash=_load_gbt_config(gbt).parameter_hash,
        probe_hash="probe",
    )
    artifact_path = tmp_path / "scout.json"
    artifact.write(artifact_path)
    output = tmp_path / "phase4.json"

    code = main(
        [
            "--config",
            str(config),
            "--gbt-config",
            str(gbt),
            "--limit",
            "6",
            "--scout-artifact",
            str(artifact_path),
            "--representation-spec",
            str(representation_path),
            "--max-acquisitions",
            "1",
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert report["representation_hash"] == representation.spec_hash
    assert report["representation_spec"]["mode"] == "dataset_adaptive"
    assert report["max_acquisitions"] == 1


def test_execute_casf_rejects_manual_target_with_scout_artifact(tmp_path: Path):
    config, gbt = _write_casf_fixture(tmp_path)
    sample_ids = tuple(f"x{index:03d}" for index in range(6))
    artifact = ScoutExecutionArtifact(
        artifact_version="mint-agent.scout-execution.v1",
        modeling_sample_ids=sample_ids,
        representation_hash=make_legacy_casf_representation_spec().spec_hash,
        frozen_priority_order=(("PL",),),
        full_fold_assignment={sample_id: index % 3 for index, sample_id in enumerate(sample_ids)},
        target_metric="PCC",
        target_direction="higher",
        target_value=0.72,
        target_source="user",
        gbt_parameter_hash=_load_gbt_config(gbt).parameter_hash,
        probe_hash="probe",
    )
    artifact_path = tmp_path / "scout.json"
    artifact.write(artifact_path)

    with pytest.raises(ValueError, match="remove overrides"):
        main(
            [
                "--config",
                str(config),
                "--gbt-config",
                str(gbt),
                "--limit",
                "6",
                "--scout-artifact",
                str(artifact_path),
                "--target-value",
                "0.9",
                "--output",
                str(tmp_path / "out.json"),
                "--dry-run",
            ]
        )
