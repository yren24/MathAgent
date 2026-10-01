from __future__ import annotations

import json

import numpy as np
import pytest

from mint_scout.probe_audit import (
    ProbeAuditConfig,
    compare_distribution,
    compare_joint_strata,
    main,
)


def _write_structure(root, sample_id: str, protein_count: int, ligand_count: int) -> None:
    sample_root = root / sample_id
    sample_root.mkdir(parents=True)
    protein_lines = [
        f"ATOM  {index:5d}  CA  ALA A{index:4d}    {index:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
        for index in range(1, protein_count + 1)
    ]
    ligand_lines = [
        f"{index} N{index} {index:.3f} 0.000 0.000 N.3 1 LIG 0.0\n"
        for index in range(1, ligand_count + 1)
    ]
    (sample_root / f"{sample_id}_pocket.pdb").write_text(
        "".join(protein_lines), encoding="utf-8"
    )
    (sample_root / f"{sample_id}_ligand.mol2").write_text(
        "@<TRIPOS>MOLECULE\nligand\n@<TRIPOS>ATOM\n"
        + "".join(ligand_lines)
        + "@<TRIPOS>BOND\n",
        encoding="utf-8",
    )


def _fixture(tmp_path):
    structures = tmp_path / "structures"
    index = tmp_path / "index"
    index.mkdir()
    train_ids = tuple(f"a{number:03d}" for number in range(1, 9))
    test_id = "b001"
    for number, sample_id in enumerate(train_ids + (test_id,), start=1):
        _write_structure(structures, sample_id, protein_count=number + 2, ligand_count=number)
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
        "  target_quantile_bins: 2\n"
        "  size_quantile_bins: 2\n"
        "probe_audit:\n"
        "  max_ks_distance: 1.0\n"
        "  max_abs_standardized_mean_difference: 10.0\n"
        "  max_joint_cell_share_error: 1.0\n"
        "  require_exact_extremes: false\n",
        encoding="utf-8",
    )
    probe_ids = train_ids[::2]
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "modeling_scope": "train",
                "modeling_sample_ids": list(train_ids),
                "probe_sample_ids": list(probe_ids),
                "probe_config": {"min_pair_support": 1},
                "pair_support": {"protein:C|ligand:N": len(probe_ids)},
                "selection_hash": "selection-test",
                "representation_hash": "representation-test",
            }
        ),
        encoding="utf-8",
    )
    return task, scout, selection, train_ids, test_id


def test_compare_distribution_identical_vectors_have_zero_shift():
    comparison = compare_distribution([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert comparison["ks_distance"] == 0.0
    assert comparison["standardized_mean_difference"] == 0.0
    assert comparison["extreme_coverage"] == {
        "minimum_retained": True,
        "maximum_retained": True,
    }


def test_compare_joint_strata_reports_cell_share_error():
    modeling_ids = ("a", "b", "c", "d")
    target_bins = {"a": 0, "b": 0, "c": 1, "d": 1}
    size_bins = {"a": 0, "b": 1, "c": 0, "d": 1}

    report = compare_joint_strata(
        modeling_ids=modeling_ids,
        probe_ids=("a", "d"),
        target_bins=target_bins,
        size_bins=size_bins,
    )

    assert report["all_target_bins_covered"] is True
    assert report["all_size_bins_covered"] is True
    assert report["max_abs_cell_share_error"] == 0.25
    assert report["total_variation_distance"] == 0.5


def test_probe_audit_cli_compares_target_and_separate_structure_sizes(tmp_path):
    task, scout, selection, train_ids, test_id = _fixture(tmp_path)
    output = tmp_path / "audit.json"

    assert main(
        [
            "--task-config",
            str(task),
            "--scout-config",
            str(scout),
            "--probe-selection",
            str(selection),
            "--output",
            str(output),
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["modeling_sample_count"] == len(train_ids)
    assert report["probe_sample_count"] == len(train_ids[::2])
    assert report["test_sample_count_excluded"] == 1
    assert report["test_leakage_detected"] is False
    assert set(report["distribution_comparisons"]) == {
        "target",
        "protein_atom_count",
        "ligand_atom_count",
        "total_atom_count",
    }
    assert report["element_pair_coverage"]["all_support_consistent"] is True
    assert test_id not in json.dumps(report)


def test_probe_audit_rejects_modeling_pool_mismatch(tmp_path):
    task, scout, selection, _, _ = _fixture(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["modeling_sample_ids"] = payload["modeling_sample_ids"][:-1]
    selection.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        main(
            [
                "--task-config",
                str(task),
                "--scout-config",
                str(scout),
                "--probe-selection",
                str(selection),
                "--output",
                str(tmp_path / "audit.json"),
            ]
        )


def test_probe_audit_thresholds_are_validated():
    with pytest.raises(ValueError, match="max_ks_distance"):
        ProbeAuditConfig(max_ks_distance=1.1)
