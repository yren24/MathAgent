from pathlib import Path

import pytest

from mint_scout.compute_feature import main as compute_feature_main
from mint_scout.data.element_pairs import make_casf_protein_ligand_schema
from mint_scout.filtration import make_distance_filtration_profile
from mint_scout.invariants.plbind_tools import PLBindFeatureTool, PLBindToolConfig
from mint_scout.representation import RepresentationMode, RepresentationSpec


def test_compute_feature_cli_dry_run_uses_config_paths(tmp_path: Path):
    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "legacy:\n"
        "  plbind_root: " + str(tmp_path / "legacy") + "\n"
        "data_audit:\n"
        "  casf:\n"
        "    structures_root: " + str(tmp_path / "all-pdbs") + "\n"
        "feature_generation:\n"
        "  output_root: " + str(tmp_path / "features") + "\n",
        encoding="utf-8",
    )

    exit_code = compute_feature_main(
        [
            "--config",
            str(config),
            "--invariant",
            "PL",
            "--sample-id",
            "10GS",
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_compute_feature_rejects_missing_legacy_root_instead_of_using_cwd(tmp_path: Path):
    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: toy\n"
        "system_type: protein_ligand\n"
        "data_audit:\n"
        "  casf:\n"
        f"    structures_root: {tmp_path / 'all-pdbs'}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy.plbind_root"):
        compute_feature_main(
            [
                "--config",
                str(config),
                "--invariant",
                "PL",
                "--sample-id",
                "10GS",
                "--dry-run",
            ]
        )


def test_plbind_feature_tool_calls_legacy_feature_module(tmp_path: Path):
    legacy_root = tmp_path / "legacy"
    pdb_folder = tmp_path / "all-pdbs"
    output_root = tmp_path / "features"
    legacy_root.mkdir()
    pdb_folder.mkdir()
    (legacy_root / "feature.py").write_text(
        "from pathlib import Path\n"
        "class ProteinLigand:\n"
        "    def __init__(self, pdb, typ, pdb_folder=None, pdb_feature_folder=None):\n"
        "        Path(pdb_feature_folder).mkdir(parents=True, exist_ok=True)\n"
        "        Path(pdb_feature_folder, f'{pdb}.npy').write_bytes(typ.encode())\n",
        encoding="utf-8",
    )

    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=legacy_root,
            pdb_folder=pdb_folder,
            output_root=output_root,
            validate_output_shape=False,
        )
    )
    result = tool.compute_one("10gs", "PL")

    assert result.status == "computed"
    assert (output_root / "PL" / "10gs.npy").read_bytes() == b"lap"


def test_compute_feature_cli_accepts_frozen_adaptive_spec_for_dry_run(tmp_path: Path):
    config = tmp_path / "task.yaml"
    config.write_text(
        "task_id: toy\n"
        "representation_mode: dataset_adaptive\n"
        "legacy:\n"
        "  plbind_root: " + str(tmp_path / "legacy") + "\n"
        "data_audit:\n"
        "  casf:\n"
        "    structures_root: " + str(tmp_path / "all-pdbs") + "\n"
        "feature_generation:\n"
        "  output_root: " + str(tmp_path / "features") + "\n",
        encoding="utf-8",
    )
    spec = RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=make_casf_protein_ligand_schema(),
        filtration_profiles={"PL": make_distance_filtration_profile(rmax=12.0)},
    ).freeze()
    spec_path = tmp_path / "representation.json"
    spec.write(spec_path)

    exit_code = compute_feature_main(
        [
            "--config",
            str(config),
            "--representation-spec",
            str(spec_path),
            "--invariant",
            "PL",
            "--sample-id",
            "10GS",
            "--dry-run",
        ]
    )

    assert exit_code == 0
