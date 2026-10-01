from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mint_scout.data.atom3d_lba import prepare_atom3d_lba
from mint_scout.data.geometry import load_atom_cloud


lmdb = pytest.importorskip("lmdb")


def _frame(rows: list[list[object]]) -> dict[str, object]:
    return {
        "columns": [
            "x",
            "y",
            "z",
            "element",
            "name",
            "resname",
            "chain",
            "residue",
        ],
        "index": list(range(len(rows))),
        "data": rows,
    }


def _write_lmdb(path: Path, items: list[dict[str, object]]) -> None:
    path.mkdir(parents=True)
    environment = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        transaction.put(b"num_examples", str(len(items)).encode("ascii"))
        transaction.put(b"serialization_format", b"json")
        transaction.put(
            b"id_to_idx",
            json.dumps({item["id"]: index for index, item in enumerate(items)}).encode(
                "utf-8"
            ),
        )
        for index, item in enumerate(items):
            transaction.put(
                str(index).encode("ascii"),
                gzip.compress(json.dumps(item).encode("utf-8")),
            )
    environment.close()


def test_atom3d_lba_provider_preserves_coordinates_and_target(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_lmdb(
        source_root / "test",
        [
            {
                "id": "1abc",
                "scores": {"neglog_aff": 7.25},
                "atoms_pocket": _frame(
                    [
                        [0.0, 0.0, 0.0, "C", "CA", "ALA", "A", 1],
                        [1.0, 0.0, 0.0, "N", "N", "ALA", "A", 1],
                    ]
                ),
                "atoms_ligand": _frame(
                    [
                        [2.0, 0.0, 0.0, "C", "C1", "LIG", "L", 1],
                        [2.0, 1.0, 0.0, "Cl", "CL1", "LIG", "L", 1],
                    ]
                ),
            }
        ],
    )
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"already extracted fixture")
    archive_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    output_root = tmp_path / "prepared"
    manifest = tmp_path / "manifest.csv"
    audit = tmp_path / "audit.json"

    report = prepare_atom3d_lba(
        source_root=source_root,
        source_archive=archive,
        extract_root=tmp_path / "extract",
        output_root=output_root,
        manifest_path=manifest,
        audit_report_path=audit,
        source_splits=("test",),
        output_split=None,
        source_url="https://example.test/source.tar.gz",
        source_archive_md5=archive_md5,
        expected_sample_count=1,
    )

    assert report["status"] == "PASS"
    assert report["sample_count"] == 1
    assert report["expected_sample_count"] == 1
    assert report["samples"][0]["target_neglog_aff"] == 7.25
    assert report["preprocessing"]["coordinate_transform"] == "none"
    protein = load_atom_cloud(output_root / "1abc" / "1abc_pocket.pdb")
    ligand = load_atom_cloud(output_root / "1abc" / "1abc_ligand.mol2")
    assert protein.elements == ("C", "N")
    assert ligand.elements == ("C", "Cl")
    assert protein.coordinates.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert ligand.coordinates.tolist() == [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]
    assert "1abc,7.25,," in manifest.read_text(encoding="utf-8")


def test_atom3d_lba_provider_maps_each_official_source_split(tmp_path: Path):
    source_root = tmp_path / "source"
    item = {
        "scores": {"neglog_aff": 7.25},
        "atoms_pocket": _frame([[0.0, 0.0, 0.0, "C", "CA", "ALA", "A", 1]]),
        "atoms_ligand": _frame([[2.0, 0.0, 0.0, "N", "N1", "LIG", "L", 1]]),
    }
    for source_split, sample_id in (
        ("train", "train-a"),
        ("val", "val-a"),
        ("test", "test-a"),
    ):
        _write_lmdb(source_root / source_split, [{**item, "id": sample_id}])
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"already extracted fixture")
    report = prepare_atom3d_lba(
        source_root=source_root,
        source_archive=archive,
        extract_root=tmp_path / "extract",
        output_root=tmp_path / "prepared",
        manifest_path=tmp_path / "manifest.csv",
        audit_report_path=tmp_path / "audit.json",
        source_splits=("train", "val", "test"),
        output_split_by_source={
            "train": "train",
            "val": "validation",
            "test": "test",
        },
        source_url="https://example.test/source.tar.gz",
        source_archive_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
        expected_sample_count=3,
    )

    lines = (tmp_path / "manifest.csv").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("train-a,7.25,train,") for line in lines)
    assert any(line.startswith("val-a,7.25,validation,") for line in lines)
    assert any(line.startswith("test-a,7.25,test,") for line in lines)
    assert report["output_split_by_source"] == {
        "train": "train",
        "val": "validation",
        "test": "test",
    }


def test_atom3d_lba_provider_rejects_incomplete_split_mapping(tmp_path: Path):
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="mapping is incomplete"):
        prepare_atom3d_lba(
            source_root=tmp_path / "source",
            source_archive=archive,
            extract_root=tmp_path / "extract",
            output_root=tmp_path / "prepared",
            manifest_path=tmp_path / "manifest.csv",
            audit_report_path=tmp_path / "audit.json",
            source_splits=("train", "val"),
            output_split_by_source={"train": "train"},
            source_url="https://example.test/source.tar.gz",
            source_archive_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
        )
