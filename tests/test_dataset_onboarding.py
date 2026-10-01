from __future__ import annotations

import json
from pathlib import Path

from mint_scout.dataset_onboarding import (
    apply_column_mapping,
    profile_manifest,
    request_manifest_column_mapping,
)


class _FakeResponse:
    def __init__(self, payload: object):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _manifest(path: Path) -> Path:
    path.write_text(
        "complex,target_value,data_split,receptor_file,compound_file,pdb_code\n"
        "private-complex-001,7.25,train,/private/a/protein.pdb,/private/a/ligand.mol2,1abc\n"
        "private-complex-002,8.50,test,/private/b/protein.pdb,/private/b/ligand.sdf,2def\n",
        encoding="utf-8",
    )
    return path


def _request(manifest: Path) -> dict:
    return {
        "system_type": "protein_ligand",
        "task_type": "regression",
        "dataset": {
            "manifest_path": str(manifest),
            "label_name": None,
            "columns": {
                "sample_id": None,
                "target": None,
                "split": None,
                "roles": {"protein": None, "ligand": None},
                "identifiers": {"pdb_id": None},
            },
        },
    }


def _mapping(*, target_confidence: str = "high") -> dict:
    return {
        "mapping": {
            "sample_id": "complex",
            "target": "target_value",
            "split": "data_split",
            "protein": "receptor_file",
            "ligand": "compound_file",
            "pdb_id": "pdb_code",
        },
        "confidence": {
            "sample_id": "high",
            "target": target_confidence,
            "split": "high",
            "protein": "high",
            "ligand": "high",
            "pdb_id": "high",
        },
        "requires_confirmation": [] if target_confidence == "high" else ["target"],
        "evidence": {field: "Profile evidence." for field in (
            "sample_id", "target", "split", "protein", "ligand", "pdb_id"
        )},
        "summary": "The manifest roles are unambiguous.",
    }


def test_manifest_profile_is_bounded_and_value_redacted(tmp_path: Path):
    profile = profile_manifest(_manifest(tmp_path / "manifest.csv"), max_rows=1)

    serialized = json.dumps(profile)
    assert profile["row_count"] == 2
    assert profile["rows_profiled"] == 1
    assert profile["privacy"]["raw_values_included"] is False
    assert "private-complex-001" not in serialized
    assert "/private/a" not in serialized
    assert "7.25" not in serialized
    by_name = {item["name"]: item for item in profile["columns"]}
    assert by_name["target_value"]["numeric_fraction"] == 1.0
    assert by_name["receptor_file"]["file_suffixes"] == [".pdb"]
    assert by_name["data_split"]["recognized_split_values"] == ["test", "train"]
    assert by_name["data_split"]["recognized_split_values_scope"] == "full_manifest"


def test_column_mapper_uses_strict_actual_column_enum(tmp_path: Path):
    manifest = _manifest(tmp_path / "manifest.csv")
    profile = profile_manifest(manifest)
    captured: dict = {}

    def opener(request, *, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        assert timeout == 60.0
        return _FakeResponse(
            {
                "id": "resp_map",
                "model": "test-model",
                "status": "completed",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 30,
                },
                "output_text": json.dumps(_mapping()),
            }
        )

    mapping, provenance = request_manifest_column_mapping(
        profile,
        request=_request(manifest),
        model="test-model",
        api_key="secret-key",
        opener=opener,
    )

    allowed = captured["body"]["text"]["format"]["schema"]["properties"][
        "mapping"
    ]["properties"]["target"]["enum"]
    assert allowed == [
        None,
        "complex",
        "target_value",
        "data_split",
        "receptor_file",
        "compound_file",
        "pdb_code",
    ]
    assert mapping["mapping"]["protein"] == "receptor_file"
    assert provenance["usage"]["total_tokens"] == 30
    assert "secret-key" not in json.dumps(captured["body"])
    assert "private-complex-001" not in captured["body"]["input"]


def test_mapping_applies_only_high_confidence_required_fields(tmp_path: Path):
    manifest = _manifest(tmp_path / "manifest.csv")
    request = _request(manifest)
    available = tuple(
        item["name"] for item in profile_manifest(manifest)["columns"]
    )

    updated, errors = apply_column_mapping(
        request,
        _mapping(),
        available_columns=available,
    )
    assert errors == []
    assert updated["dataset"]["columns"]["target"] == "target_value"
    assert updated["dataset"]["columns"]["roles"] == {
        "protein": "receptor_file",
        "ligand": "compound_file",
    }

    unchanged, errors = apply_column_mapping(
        request,
        _mapping(target_confidence="medium"),
        available_columns=available,
    )
    assert unchanged["dataset"]["columns"]["target"] is None
    assert errors == [
        "Manifest column mapping for target requires user confirmation."
    ]
