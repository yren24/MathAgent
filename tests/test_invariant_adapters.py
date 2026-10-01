from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mint_scout.invariants.contracts import assert_feature_shape, validate_feature_file
from mint_scout.data.element_pairs import ElementPairSchema
from mint_scout.filtration import FiltrationProfile
from mint_scout.invariants.plbind_tools import (
    PLBindFeatureTool,
    PLBindToolConfig,
    _load_adaptive_atoms,
)
from mint_scout.invariants.registry import INVARIANTS, get_capability
from mint_scout.representation import RepresentationMode, RepresentationSpec
from mint_scout.validate_plbind_parity import validate_parity


EXPECTED_LEGACY_TYPES = {
    "PH": "homology",
    "PL": "lap",
    "CA": "facet",
    "FPRC": "forman",
    "EIC": "curvature",
}


EXPECTED_LEGACY_CASF_SHAPES = {
    "PH": (150, 40, 2),
    "PL": (30, 40, 8),
    "CA": (141, 40, 2),
    "FPRC": (30, 40, 20),
    "EIC": (49, 40, 10),
}


def _write_fake_legacy_feature(root: Path) -> None:
    root.mkdir()
    (root / "feature.py").write_text(
        "from pathlib import Path\n"
        "class ProteinLigand:\n"
        "    def __init__(self, pdb, typ, pdb_folder=None, pdb_feature_folder=None):\n"
        "        Path(pdb_feature_folder).mkdir(parents=True, exist_ok=True)\n"
        "        Path(pdb_feature_folder, f'{pdb}.npy').write_bytes(typ.encode())\n",
        encoding="utf-8",
    )


def _write_fake_parameterized_legacy_feature(root: Path) -> None:
    root.mkdir()
    (root / "feature.py").write_text(
        "from pathlib import Path\n"
        "import numpy as np\n"
        "class ProteinLigand:\n"
        "    def __init__(self, pdb, typ, pdb_folder=None, pdb_feature_folder=None):\n"
        "        self.pdb = pdb\n"
        "        self.output = Path(pdb_feature_folder, f'{pdb}.npy')\n"
        "        self.output.parent.mkdir(parents=True, exist_ok=True)\n"
        "        dispatch = {'homology': self.get_persistent_homology, 'lap': self.get_euclidean_L0, 'facet': self.get_euclidean_facet, 'forman': self.get_forman_graph_feature, 'curvature': self.get_euclidean_curvature}\n"
        "        if typ in dispatch: dispatch[typ]()\n"
        "    def read_atom_from_pdb(self): pass\n"
        "    def set_euclidean_index_list(self): pass\n"
        "    def _save(self, points, width):\n"
        "        value = np.broadcast_to(np.arange(40)[None, :, None], (points, 40, width)).copy()\n"
        "        np.save(self.output, value)\n"
        "    def get_persistent_homology(self, grid_max=15.0, grid_step=0.1):\n"
        "        self._save(len(np.arange(0, grid_max, grid_step)), 2)\n"
        "    def get_euclidean_L0(self, grid_max=15.0, grid_step=0.5):\n"
        "        self._save(len(np.arange(0, grid_max, grid_step)), 8)\n"
        "    def get_euclidean_facet(self, min_edge=1.0, max_edge=15.0, grid_step=0.1):\n"
        "        self._save(len(np.arange(min_edge, max_edge + 1e-8, grid_step)), 2)\n"
        "    def get_forman_graph_feature(self, grid_max=15.0, grid_step=0.5):\n"
        "        self._save(len(np.arange(0, grid_max, grid_step)), 20)\n"
        "    def get_euclidean_curvature(self): self._save(49, 10)\n",
        encoding="utf-8",
    )


def _write_axis_parameterized_pl_legacy_feature(root: Path) -> None:
    root.mkdir()
    (root / "feature.py").write_text(
        "from pathlib import Path\n"
        "import numpy as np\n"
        "class ProteinLigand:\n"
        "    def __init__(self, pdb, typ, pdb_folder=None, pdb_feature_folder=None):\n"
        "        self.pdb = pdb\n"
        "        self.output = Path(pdb_feature_folder, f'{pdb}.npy')\n"
        "        self.output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    def read_atom_from_pdb(self): pass\n"
        "    def set_euclidean_index_list(self): pass\n"
        "    def get_euclidean_L0(self, grid_max=15.0, grid_step=0.5):\n"
        "        points = len(np.arange(0, grid_max, grid_step))\n"
        "        values = np.broadcast_to(np.arange(points)[:, None, None], (points, 40, 8)).copy()\n"
        "        np.save(self.output, values)\n",
        encoding="utf-8",
    )


def _write_endpoint_overflow_legacy_feature(root: Path) -> None:
    root.mkdir()
    (root / "feature.py").write_text(
        "from pathlib import Path\n"
        "import numpy as np\n"
        "class ProteinLigand:\n"
        "    def __init__(self, pdb, typ, pdb_folder=None, pdb_feature_folder=None):\n"
        "        self.pdb = pdb\n"
        "        self.output = Path(pdb_feature_folder, f'{pdb}.npy')\n"
        "        self.output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    def read_atom_from_pdb(self): pass\n"
        "    def set_euclidean_index_list(self): pass\n"
        "    def _save(self, points, width):\n"
        "        values = np.broadcast_to(np.arange(points)[:, None, None], (points, 40, width)).copy()\n"
        "        np.save(self.output, values)\n"
        "    def get_persistent_homology(self, grid_max=15.0, grid_step=0.1):\n"
        "        self._save(51, 2)\n"
        "    def get_euclidean_L0(self, grid_max=15.0, grid_step=0.5):\n"
        "        self._save(50, 8)\n"
        "    def get_euclidean_facet(self, min_edge=1.0, max_edge=15.0, grid_step=0.1):\n"
        "        self._save(50, 2)\n"
        "    def get_forman_graph_feature(self, grid_max=15.0, grid_step=0.5):\n"
        "        self._save(51, 20)\n"
        "    def get_euclidean_curvature(self): self._save(49, 10)\n",
        encoding="utf-8",
    )


def _adaptive_spec(
    pairs: tuple[tuple[str, str], ...],
    *,
    distance_points: int = 50,
    eic_profile: FiltrationProfile | None = None,
) -> RepresentationSpec:
    schema = ElementPairSchema(
        schema_id="test_adaptive_pairs",
        system_type="protein_ligand",
        left_role="protein",
        right_role="ligand",
        left_elements=tuple(dict.fromkeys(pair[0] for pair in pairs)),
        right_elements=tuple(dict.fromkeys(pair[1] for pair in pairs)),
        global_element_order=None,
        pair_generation_rule="test_frozen_order",
        role_aware=True,
        exclude_self_pairs=False,
        deduplicate_unordered_pairs=False,
        pair_order=pairs,
        expected_pair_count=len(pairs),
    )
    distance = FiltrationProfile(
        "distance",
        0.0,
        12.25,
        12.25 / (distance_points - 1),
        distance_points,
        "test_fixed_points",
    )
    return RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=schema,
        filtration_profiles={
            "PH": distance,
            "PL": distance,
            "CA": distance,
            "FPRC": distance,
            "EIC": eic_profile or FiltrationProfile("tau", 0.2, 5.0, 0.1, 49, "legacy"),
        },
    ).freeze()


def test_all_registered_invariants_have_legacy_casf_capabilities():
    for name in INVARIANTS:
        capability = get_capability(name)
        assert capability.legacy_name == EXPECTED_LEGACY_TYPES[name]
        assert capability.supported_systems == ("protein_ligand",)
        assert capability.supported_representation_modes == ("legacy_casf", "dataset_adaptive")
        assert capability.expected_shape("legacy_casf") == EXPECTED_LEGACY_CASF_SHAPES[name]
        assert capability.domain


def test_plbind_tool_exposes_one_adapter_per_invariant(tmp_path: Path):
    _write_fake_legacy_feature(tmp_path / "legacy")
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            validate_output_shape=False,
        )
    )

    adapters = tool.adapters()

    assert tuple(adapter.capability.name for adapter in adapters) == INVARIANTS


def test_each_plbind_adapter_calls_the_matching_legacy_type(tmp_path: Path):
    _write_fake_legacy_feature(tmp_path / "legacy")
    (tmp_path / "all-pdbs").mkdir()
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            validate_output_shape=False,
        )
    )

    for invariant, legacy_type in EXPECTED_LEGACY_TYPES.items():
        result = tool.adapter_for(invariant).compute_one("10gs")
        assert result.status == "computed"
        assert result.invariant_name == invariant
        assert result.legacy_name == legacy_type
        assert result.output_path.read_bytes() == legacy_type.encode()


def test_feature_shape_validation_accepts_expected_shape(tmp_path: Path):
    feature_path = tmp_path / "pl.npy"
    np.save(feature_path, np.zeros((30, 40, 8)))

    assert assert_feature_shape(feature_path, (30, 40, 8)) == (30, 40, 8)


def test_feature_shape_validation_rejects_mismatch(tmp_path: Path):
    feature_path = tmp_path / "pl.npy"
    np.save(feature_path, np.zeros((50, 40, 8)))

    with pytest.raises(AssertionError, match="Feature shape mismatch"):
        assert_feature_shape(feature_path, (30, 40, 8))


def test_feature_validation_reports_numeric_summary(tmp_path: Path):
    feature_path = tmp_path / "feature.npy"
    np.save(feature_path, np.asarray([[[0.0, -2.0], [3.0, 0.0]]], dtype=np.float32))

    report = validate_feature_file(feature_path, (1, 2, 2))

    assert report.shape == (1, 2, 2)
    assert report.dtype == "float32"
    assert report.all_finite is True
    assert report.nonzero_count == 2
    assert report.minimum == -2.0
    assert report.maximum == 3.0


def test_feature_validation_rejects_nan_and_inf(tmp_path: Path):
    feature_path = tmp_path / "invalid.npy"
    np.save(feature_path, np.asarray([[[np.nan, np.inf]]]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_feature_file(feature_path, (1, 1, 2))


def test_plbind_tool_validates_output_shape_by_invariant(tmp_path: Path):
    _write_fake_legacy_feature(tmp_path / "legacy")
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
        )
    )
    feature_path = tmp_path / "ph.npy"
    np.save(feature_path, np.zeros((150, 40, 2), dtype=np.float32))

    assert tool.assert_output_shape(feature_path, "PH") == (150, 40, 2)


def test_plbind_tool_rejects_wrong_shape_in_cached_output(tmp_path: Path):
    _write_fake_legacy_feature(tmp_path / "legacy")
    output_path = tmp_path / "features" / "PL" / "10gs.npy"
    output_path.parent.mkdir(parents=True)
    np.save(output_path, np.zeros((50, 40, 8)))
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
        )
    )

    with pytest.raises(AssertionError, match="Feature shape mismatch"):
        tool.compute_one("10gs", "PL")


def test_representation_aware_cache_paths_are_isolated(tmp_path: Path):
    _write_fake_legacy_feature(tmp_path / "legacy")
    (tmp_path / "all-pdbs").mkdir()
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            dataset_id="casf2016",
            validate_output_shape=False,
        )
    )

    first = tool.compute_one("10gs", "PL", representation_hash="hash-a")
    second = tool.compute_one("10gs", "PL", representation_hash="hash-b")

    assert first.output_path != second.output_path
    assert "casf2016/repr-hash-a/PL/params-" in first.output_path.as_posix()
    assert first.status == second.status == "computed"


def test_legacy_adapter_rejects_dataset_adaptive_before_computation(tmp_path: Path):
    _write_fake_legacy_feature(tmp_path / "legacy")
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            representation_mode="dataset_adaptive",
            validate_output_shape=False,
        )
    )

    with pytest.raises(ValueError, match="requires a frozen RepresentationSpec"):
        tool.compute_one("10gs", "PL", representation_hash="adaptive-hash")


@pytest.mark.parametrize(
    ("invariant", "width"),
    (("PH", 2), ("PL", 8), ("CA", 2), ("FPRC", 20)),
)
def test_adaptive_distance_adapter_uses_50_points_and_frozen_pair_order(
    tmp_path: Path,
    invariant: str,
    width: int,
):
    _write_fake_parameterized_legacy_feature(tmp_path / "legacy")
    spec = _adaptive_spec((("N", "O"), ("C", "C")))
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            representation_mode="dataset_adaptive",
        )
    )

    result = tool.compute_with_spec("10gs", invariant, spec)
    feature = np.load(result.output_path)

    assert feature.shape == (50, 2, width)
    assert np.all(feature[:, 0, :] == 12)
    assert np.all(feature[:, 1, :] == 0)


def test_adaptive_pl_supports_a_positive_grid_start_by_slicing_legacy_output(tmp_path: Path):
    _write_axis_parameterized_pl_legacy_feature(tmp_path / "legacy")
    base = _adaptive_spec((("C", "C"),))
    profiles = dict(base.filtration_profiles)
    profiles["PL"] = FiltrationProfile(
        "distance", 0.1, 10.0, 0.1, 100, "fixed_0_1_to_10_inclusive"
    )
    spec = RepresentationSpec(
        representation_id="positive-pl-start",
        mode=base.mode,
        system_type=base.system_type,
        schema_id=base.schema_id,
        pair_order=base.pair_order,
        pair_names=base.pair_names,
        filtration_profiles=profiles,
        parameters=dict(base.parameters),
        frozen=True,
    )
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            representation_mode="dataset_adaptive",
        )
    )

    result = tool.compute_with_spec("10gs", "PL", spec)
    feature = np.load(result.output_path)

    assert feature.shape == (100, 1, 8)
    assert np.array_equal(feature[:, 0, 0], np.arange(1, 101))


@pytest.mark.parametrize(("invariant", "width"), (("PH", 2), ("FPRC", 20)))
def test_adaptive_adapter_trims_legacy_endpoint_overflow(
    tmp_path: Path,
    invariant: str,
    width: int,
):
    _write_endpoint_overflow_legacy_feature(tmp_path / "legacy")
    spec = _adaptive_spec((("N", "O"), ("C", "C")))
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            representation_mode="dataset_adaptive",
        )
    )

    result = tool.compute_with_spec("10gs", invariant, spec)
    feature = np.load(result.output_path)

    assert feature.shape == (50, 2, width)
    assert np.array_equal(feature[:, 0, 0], np.arange(50))


def test_adaptive_eic_slices_channels_but_preserves_independent_tau_axis(tmp_path: Path):
    _write_fake_parameterized_legacy_feature(tmp_path / "legacy")
    spec = _adaptive_spec((("S", "H"),))
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            representation_mode="dataset_adaptive",
        )
    )

    result = tool.compute_with_spec("10gs", "EIC", spec)
    feature = np.load(result.output_path)

    assert feature.shape == (49, 1, 10)
    assert np.all(feature == 39)


def test_adaptive_adapter_rejects_pair_outside_audited_legacy_universe(tmp_path: Path):
    _write_fake_parameterized_legacy_feature(tmp_path / "legacy")
    spec = _adaptive_spec((("Zn", "C"),))
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            representation_mode="dataset_adaptive",
        )
    )

    with pytest.raises(ValueError, match="unsupported pairs"):
        tool.compute_with_spec("10gs", "PL", spec)


def test_adaptive_atom_loader_handles_standard_mol2_and_unsupported_protein_elements(
    tmp_path: Path,
):
    sample_root = tmp_path / "sample-a"
    sample_root.mkdir()
    (sample_root / "sample-a_pocket.pdb").write_text(
        "ATOM      1   CA ALA A   1       1.000   2.000   3.000  1.00 20.00           C\n"
        "HETATM    2   ZN  ZN A   2       4.000   5.000   6.000  1.00 20.00          Zn\n",
        encoding="utf-8",
    )
    (sample_root / "sample-a_ligand.mol2").write_text(
        "@<TRIPOS>MOLECULE\nsample-a\n2 0 0 0 0\nSMALL\nNO_CHARGES\n\n"
        "@<TRIPOS>ATOM\n"
        "      1 CL1          7.100000     8.200000     9.300000 Cl 1 LIG 0.0\n"
        "      2 BR1         10.100000    11.200000    12.300000 Br 1 LIG 0.0\n"
        "@<TRIPOS>BOND\n",
        encoding="utf-8",
    )

    class Atom:
        def __init__(self, *, atype, resname, chain, resid, coord):
            self.AType = atype
            self.Coord = coord

    class Module:
        pass

    Module.Atom = Atom

    class Model:
        pass

    model = Model()
    _load_adaptive_atoms(
        Module,
        model,
        pdb_folder=tmp_path,
        sample_id="sample-a",
    )

    assert [atom.AType for atom in model.Protein_Atoms] == ["C", "Zn"]
    assert [atom.AType for atom in model.Ligand_Atoms] == ["Cl", "Br"]
    assert model.Protein_AtomCoord.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert model.Ligand_AtomCoord.tolist() == [[7.1, 8.2, 9.3], [10.1, 11.2, 12.3]]


def test_adaptive_eic_rejects_unaudited_tau_grid(tmp_path: Path):
    _write_fake_parameterized_legacy_feature(tmp_path / "legacy")
    spec = _adaptive_spec(
        (("C", "C"),),
        eic_profile=FiltrationProfile("tau", 0.5, 5.0, 0.5, 10, "test"),
    )
    tool = PLBindFeatureTool(
        PLBindToolConfig(
            legacy_root=tmp_path / "legacy",
            pdb_folder=tmp_path / "all-pdbs",
            output_root=tmp_path / "features",
            representation_mode="dataset_adaptive",
        )
    )

    with pytest.raises(ValueError, match="audited tau grid"):
        tool.compute_with_spec("10gs", "EIC", spec)


def test_parity_validator_requires_exact_legacy_equivalence(tmp_path: Path):
    _write_fake_parameterized_legacy_feature(tmp_path / "legacy")

    result = validate_parity(
        sample_id="10gs",
        invariant="PL",
        legacy_root=tmp_path / "legacy",
        pdb_folder=tmp_path / "all-pdbs",
        output_root=tmp_path / "features",
    )

    assert result.exact_equal is True
    assert result.legacy_shape == result.adaptive_shape == (30, 40, 8)
    assert result.max_abs_difference == 0.0
