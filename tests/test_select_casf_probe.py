from __future__ import annotations

import json

import pytest

import mint_scout.select_casf_probe as probe_module
from mint_scout.data.casf_index import CasfRecord
from mint_scout.data.element_pairs import ElementPairSchema
from mint_scout.filtration import make_distance_filtration_profile
from mint_scout.representation import RepresentationMode, RepresentationSpec


def _write_structure(root, sample_id: str, extra_atoms: int) -> None:
    sample_root = root / sample_id
    sample_root.mkdir(parents=True)
    protein_lines = [
        f"ATOM  {index:5d}  CA  ALA A{index:4d}    {index:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
        for index in range(1, extra_atoms + 2)
    ]
    (sample_root / f"{sample_id}_pocket.pdb").write_text("".join(protein_lines), encoding="utf-8")
    (sample_root / f"{sample_id}_ligand.mol2").write_text(
        "@<TRIPOS>MOLECULE\nligand\n@<TRIPOS>ATOM\n"
        "1 N1 3.000 0.000 0.000 N.3 1 LIG 0.0\n"
        "@<TRIPOS>BOND\n",
        encoding="utf-8",
    )


def _fixture(tmp_path):
    structures = tmp_path / "structures"
    index = tmp_path / "index"
    index.mkdir()
    train_ids = tuple(f"a{number:03d}" for number in range(1, 7))
    test_id = "b001"
    for number, sample_id in enumerate(train_ids + (test_id,), start=1):
        _write_structure(structures, sample_id, number)
    lines = [
        f"{sample_id} 2.0 2000 {4.0 + number:.2f} Kd=1nM"
        for number, sample_id in enumerate(train_ids + (test_id,))
    ]
    (index / "2016_INDEX_refined.data").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (index / "train_data_2016.txt").write_text(repr(list(train_ids)), encoding="utf-8")
    (index / "test_data_2016.txt").write_text(repr([test_id]), encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "data_audit:\n"
        "  casf:\n"
        "    year: 2016\n"
        f"    index_root: {index}\n"
        f"    structures_root: {structures}\n",
        encoding="utf-8",
    )
    scout = tmp_path / "scout.yaml"
    scout.write_text(
        "probe:\n"
        "  fraction: 0.5\n"
        "  min_samples: 3\n"
        "  max_samples: 3\n"
        "  target_quantile_bins: 3\n"
        "  size_quantile_bins: 2\n"
        "  min_pair_support: 1\n"
        "  augmentation_fraction_limit: 0.0\n"
        "  cv_folds: 3\n"
        "  random_seed: 17\n",
        encoding="utf-8",
    )
    schema = ElementPairSchema(
        schema_id="toy_adaptive",
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
    spec = RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=schema,
        filtration_profiles={"PL": make_distance_filtration_profile(rmax=5.0)},
    ).freeze()
    representation = tmp_path / "representation.json"
    representation.write_text(
        json.dumps(
            {
                "modeling_scope": "train",
                "sample_ids": list(train_ids),
                "representation_spec": spec.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return task, scout, representation, train_ids, test_id


def test_select_casf_probe_is_deterministic_and_excludes_test(tmp_path):
    task, scout, representation, train_ids, test_id = _fixture(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    common = [
        "--task-config",
        str(task),
        "--scout-config",
        str(scout),
        "--representation-spec",
        str(representation),
    ]

    assert probe_module.main(common + ["--output", str(first)]) == 0
    assert probe_module.main(common + ["--output", str(second)]) == 0
    payload = json.loads(first.read_text(encoding="utf-8"))
    repeated = json.loads(second.read_text(encoding="utf-8"))

    assert payload["modeling_sample_count"] == 6
    assert payload["test_sample_count_excluded"] == 1
    assert payload["probe_final_size"] == 3
    assert set(payload["modeling_fold_assignment"]) == set(train_ids)
    assert test_id not in payload["probe_sample_ids"]
    assert test_id not in payload["modeling_fold_assignment"]
    assert payload["pair_support"] == {"protein:C|ligand:N": 3}
    assert payload["selection_hash"] == repeated["selection_hash"]
    assert payload["probe_sample_ids"] == repeated["probe_sample_ids"]


def test_select_casf_probe_accepts_train_only_filtration_repair(tmp_path):
    task, scout, representation, train_ids, _test_id = _fixture(tmp_path)
    raw = json.loads(representation.read_text(encoding="utf-8"))
    repair = tmp_path / "repair.json"
    repair.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.filtration-repair.v1",
                "evidence_scope": "design",
                "audit_evidence_scope": "probe",
                "representation_spec": raw["representation_spec"],
                "representation_hash": "repair-hash",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "probe.json"

    assert (
        probe_module.main(
            [
                "--task-config",
                str(task),
                "--scout-config",
                str(scout),
                "--representation-spec",
                str(repair),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["modeling_sample_count"] == len(train_ids)


def test_select_casf_probe_rejects_test_evidence_repair(tmp_path):
    _task, _scout, representation, _train_ids, _test_id = _fixture(tmp_path)
    raw = json.loads(representation.read_text(encoding="utf-8"))
    repair = tmp_path / "repair-test.json"
    repair.write_text(
        json.dumps(
            {
                "report_schema": "mint-agent.filtration-repair.v1",
                "evidence_scope": "design",
                "audit_evidence_scope": "test",
                "representation_spec": raw["representation_spec"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not use test evidence"):
        probe_module._validate_design_scope(repair, tuple())


def test_select_casf_probe_supports_controlled_uniform_random_baseline(tmp_path):
    task, scout, representation, train_ids, test_id = _fixture(tmp_path)
    first = tmp_path / "random-first.json"
    second = tmp_path / "random-second.json"
    common = [
        "--task-config", str(task),
        "--scout-config", str(scout),
        "--representation-spec", str(representation),
        "--sampling-strategy", "uniform_random",
        "--random-seed", "91",
        "--probe-max-samples", "4",
    ]

    assert probe_module.main(common + ["--output", str(first)]) == 0
    assert probe_module.main(common + ["--output", str(second)]) == 0
    payload = json.loads(first.read_text(encoding="utf-8"))
    repeated = json.loads(second.read_text(encoding="utf-8"))

    assert payload["sampling_strategy"] == "uniform_random"
    assert payload["sampling_seed"] == 91
    assert payload["probe_config"]["max_samples"] == 4
    assert payload["probe_final_size"] == 3
    assert test_id not in payload["probe_sample_ids"]
    assert set(payload["probe_sample_ids"]).issubset(train_ids)
    assert payload["selection_hash"] == repeated["selection_hash"]


def test_select_casf_probe_rejects_representation_from_different_modeling_pool(tmp_path):
    task, scout, representation, _, _ = _fixture(tmp_path)
    payload = json.loads(representation.read_text(encoding="utf-8"))
    payload["sample_ids"] = payload["sample_ids"][:-1]
    representation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        probe_module.main(
            [
                "--task-config",
                str(task),
                "--scout-config",
                str(scout),
                "--representation-spec",
                str(representation),
                "--output",
                str(tmp_path / "probe.json"),
            ]
        )


def test_select_probe_can_reuse_frozen_samples_and_folds(tmp_path):
    task, scout, representation, _, _ = _fixture(tmp_path)
    source = tmp_path / "source.json"
    reused = tmp_path / "reused.json"
    common = [
        "--task-config",
        str(task),
        "--scout-config",
        str(scout),
        "--representation-spec",
        str(representation),
    ]

    assert probe_module.main(common + ["--output", str(source)]) == 0
    assert (
        probe_module.main(
            common
            + [
                "--frozen-sample-source",
                str(source),
                "--output",
                str(reused),
            ]
        )
        == 0
    )
    original = json.loads(source.read_text(encoding="utf-8"))
    copied = json.loads(reused.read_text(encoding="utf-8"))

    assert copied["probe_sample_ids"] == original["probe_sample_ids"]
    assert copied["probe_fold_assignment"] == original["probe_fold_assignment"]
    assert copied["shared_probe_hash"] == original["shared_probe_hash"]
    assert copied["frozen_sample_source"] == str(source.resolve())


def test_select_probe_uses_configured_grouped_folds(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    task, scout, representation, train_ids, test_id = _fixture(tmp_path)
    task.write_text(
        task.read_text(encoding="utf-8")
        + "cv:\n"
        + "  group_identifier: pdb_id\n",
        encoding="utf-8",
    )
    groups = {
        "a001": "1abc",
        "a002": "1abc",
        "a003": "2def",
        "a004": "3ghi",
        "a005": "4jkl",
        "a006": "5mno",
    }
    structures = tmp_path / "structures"

    def selected_records(_config, *, split, **_kwargs):
        if split == "test":
            sample_ids = (test_id,)
        else:
            sample_ids = train_ids
        return tuple(
            CasfRecord(
                pdb_id=sample_id,
                label=float(index + 1),
                split=split,
                protein_path=structures / sample_id / f"{sample_id}_pocket.pdb",
                ligand_path=structures / sample_id / f"{sample_id}_ligand.mol2",
                group_id=groups.get(sample_id, sample_id),
            )
            for index, sample_id in enumerate(sample_ids)
        )

    monkeypatch.setattr(probe_module, "_selected_records", selected_records)
    output = tmp_path / "grouped.json"

    assert (
        probe_module.main(
            [
                "--task-config",
                str(task),
                "--scout-config",
                str(scout),
                "--representation-spec",
                str(representation),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["fold_strategy"] == "balanced_group_kfold"
    assert payload["group_identifier"] == "pdb_id"
    assert payload["modeling_group_count"] == 5
    assert payload["fold_sample_counts"] == {"0": 2, "1": 2, "2": 2}
    assert sum(payload["fold_group_counts"].values()) == 5
    assert payload["group_leakage_check_passed"] is True
    folds = payload["modeling_fold_assignment"]
    assert folds["a001"] == folds["a002"]
