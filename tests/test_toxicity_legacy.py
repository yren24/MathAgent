from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mint_scout.toxicity.legacy import (
    LEGACY_NAMES,
    LEGACY_TOXICITY_SHAPES,
    TOXICITY_INVARIANTS,
    ToxicityFeatureTool,
    ToxicityToolConfig,
)
from mint_scout.toxicity.manifest import build_ld50_manifest, write_ld50_manifest
from mint_scout.data.element_pairs import ElementPairSchema
from mint_scout.filtration import FiltrationProfile
from mint_scout.representation import (
    RepresentationMode,
    RepresentationSpec,
    make_legacy_toxicity_representation_spec,
)


def _write_fake_legacy(root: Path) -> None:
    root.mkdir()
    (root / "toxicity_topology_features.py").write_text(
        "from pathlib import Path\n"
        "import numpy as np\n"
        "PAIR_ELEMENT_NAMES = [('H', 'C')] * 30\n"
        "TRAILING_DIMS = {\n"
        "    'homology': 1, 'lap': 8, 'facet': 2, 'forman': 20, 'curvature': 10,\n"
        "}\n"
        "VALUES = {'homology': 1, 'lap': 2, 'facet': 3, 'forman': 4, 'curvature': 5}\n"
        "def generate_one(method, name, mol2_path, out_path, args):\n"
        "    assert Path(mol2_path).is_file()\n"
        "    if method == 'curvature':\n"
        "        span = args.curvature_tau_max - args.curvature_tau_min\n"
        "        points = int(round(span / args.curvature_tau_step)) + 1\n"
        "    elif method == 'homology':\n"
        "        points = int(round(args.max_filtration / args.homology_step))\n"
        "    elif method == 'facet':\n"
        "        span = args.max_filtration - args.facet_min_edge\n"
        "        points = int(round(span / args.facet_step))\n"
        "    else:\n"
        "        points = int(round(args.max_filtration / args.graph_step))\n"
        "    shape = (points, len(PAIR_ELEMENT_NAMES), TRAILING_DIMS[method])\n"
        "    Path(out_path).parent.mkdir(parents=True, exist_ok=True)\n"
        "    np.save(out_path, np.full(shape, VALUES[method], dtype=np.float32))\n"
        "    return True\n",
        encoding="utf-8",
    )


def _adaptive_spec() -> RepresentationSpec:
    pairs = (("C", "N"), ("C", "O"), ("N", "O"))
    schema = ElementPairSchema(
        schema_id="toxicity_test_adaptive_3",
        system_type="small_molecule",
        left_role=None,
        right_role=None,
        left_elements=("C", "N", "O"),
        right_elements=("C", "N", "O"),
        global_element_order=("C", "N", "O"),
        pair_generation_rule="test_unordered_pairs",
        role_aware=False,
        exclude_self_pairs=True,
        deduplicate_unordered_pairs=True,
        pair_order=pairs,
        expected_pair_count=len(pairs),
    )
    distance = FiltrationProfile("distance", 0.0, 4.9, 0.1, 50, "test")
    return RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=schema,
        filtration_profiles={
            "PH": distance,
            "PL": distance,
            "CA": distance,
            "FPRC": distance,
            "EIC": FiltrationProfile("tau", 0.2, 5.0, 0.2, 25, "test"),
        },
        parameters={"system_type": "small_molecule"},
    ).freeze()


def _tool(tmp_path: Path) -> ToxicityFeatureTool:
    legacy = tmp_path / "legacy"
    _write_fake_legacy(legacy)
    molecules = tmp_path / "molecules"
    molecules.mkdir()
    (molecules / "100-00-5.mol2").write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    return ToxicityFeatureTool(
        ToxicityToolConfig(
            legacy_root=legacy,
            molecule_dirs=(molecules,),
            output_root=tmp_path / "features",
            dataset_id="LD50",
        )
    )


def test_toxicity_tool_exposes_five_separate_legacy_adapters(tmp_path: Path):
    tool = _tool(tmp_path)

    adapters = tool.adapters()

    assert tuple(adapter.capability.name for adapter in adapters) == TOXICITY_INVARIANTS
    assert tuple(adapter.capability.legacy_name for adapter in adapters) == tuple(
        LEGACY_NAMES[name] for name in TOXICITY_INVARIANTS
    )
    assert all(adapter.capability.supported_systems == ("small_molecule",) for adapter in adapters)


@pytest.mark.parametrize("invariant", TOXICITY_INVARIANTS)
def test_each_toxicity_adapter_calls_only_its_legacy_method(tmp_path: Path, invariant: str):
    tool = _tool(tmp_path)

    result = tool.adapter_for(invariant).compute_one("100-00-5")
    feature = np.load(result.output_path)

    assert result.status == "computed"
    assert result.legacy_name == LEGACY_NAMES[invariant]
    assert feature.shape == LEGACY_TOXICITY_SHAPES[invariant]
    assert np.all(feature == TOXICITY_INVARIANTS.index(invariant) + 1)


def test_toxicity_feature_cache_is_content_and_parameter_addressed(tmp_path: Path):
    tool = _tool(tmp_path)

    first = tool.compute_one("100-00-5", "PL")
    second = tool.compute_one("100-00-5", "PL")

    assert first.status == "computed"
    assert second.status == "cached"
    assert "params-" in first.output_path.as_posix()
    assert "input-" in first.output_path.as_posix()


@pytest.mark.parametrize("invariant", TOXICITY_INVARIANTS)
def test_each_adapter_executes_a_frozen_adaptive_pair_and_grid_spec(
    tmp_path: Path, invariant: str
):
    tool = _tool(tmp_path)
    spec = _adaptive_spec()

    result = tool.adapter_for(invariant).compute_with_spec("100-00-5", spec)
    feature = np.load(result.output_path)

    expected_points = 25 if invariant == "EIC" else 50
    assert feature.shape == (expected_points, 3, LEGACY_TOXICITY_SHAPES[invariant][2])
    assert f"repr-{spec.spec_hash}" in result.output_path.as_posix()


def test_legacy_toxicity_representation_matches_audited_feature_shapes():
    spec = make_legacy_toxicity_representation_spec()

    assert spec.mode == RepresentationMode.LEGACY_TOXICITY
    assert spec.system_type == "small_molecule"
    assert len(spec.pair_order) == 30
    assert spec.filtration_profiles["PL"].num_points == 50
    assert spec.filtration_profiles["EIC"].num_points == 25
    assert spec.parameters["bond_delta"] == 0.45


def test_adaptive_ca_grid_keeps_fixed_size_with_nonterminating_float_step(
    tmp_path: Path,
):
    tool = _tool(tmp_path)
    spec = _adaptive_spec()
    step = 8.83466842622665 / 49
    profiles = dict(spec.filtration_profiles)
    profiles["CA"] = FiltrationProfile(
        "distance", 0.0, 8.83466842622665, step, 50, "repair_test"
    )
    repaired = replace(spec, filtration_profiles=profiles)

    result = tool.compute_with_spec("100-00-5", "CA", repaired)

    assert np.load(result.output_path).shape == (50, 3, 2)


def test_ld50_manifest_preserves_fixed_train_test_and_resolves_excel_date_names(
    tmp_path: Path,
):
    for split, rows in {
        "train": [("100-01-6", "CC", "2.265")],
        "test": [("8/7/1918", "CO", "2.574")],
    }.items():
        molecule_dir = tmp_path / f"LD50_{split}_x"
        molecule_dir.mkdir()
        molecule_name = "1918-8-7" if split == "test" else rows[0][0]
        (molecule_dir / f"{molecule_name}.mol2").write_text(
            "@<TRIPOS>MOLECULE\n", encoding="utf-8"
        )
        with (tmp_path / f"LD50_{split}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "smiles", "label"])
            writer.writerows(rows)

    records = build_ld50_manifest(tmp_path)
    manifest_path = write_ld50_manifest(records, tmp_path / "manifest.csv")

    assert [(record.sample_id, record.split) for record in records] == [
        ("100-01-6", "train"),
        ("1918-8-7", "test"),
    ]
    assert records[1].source_filename == "8/7/1918"
    assert records[1].molecule_path.name == "1918-8-7.mol2"
    assert manifest_path.read_text(encoding="utf-8").splitlines()[0] == (
        "sample_id,target,split,molecule_path,smiles,source_filename"
    )
