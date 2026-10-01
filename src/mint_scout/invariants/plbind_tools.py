from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from mint_scout.data.element_pairs import CASF_LIGAND_ELEMENTS_40, CASF_PROTEIN_ELEMENTS
from mint_scout.data.geometry import load_atom_cloud
from mint_scout.invariants.contracts import (
    FeatureToolResult,
    InvariantCapability,
    assert_feature_shape,
    validate_feature_file,
)
from mint_scout.invariants.manifest import stable_hash
from mint_scout.invariants.registry import INVARIANTS, get_capability, get_invariant
from mint_scout.representation import RepresentationSpec
from mint_scout.search.cost import measure_cost


DEFAULT_PLBIND_ROOT = Path("/path/to/legacy/embed_nn/plbind")
LEGACY_CASF_PAIR_ORDER = tuple(
    (protein, ligand)
    for protein in CASF_PROTEIN_ELEMENTS
    for ligand in CASF_LIGAND_ELEMENTS_40
)
TRAILING_FEATURE_DIMS = {"PH": 2, "PL": 8, "CA": 2, "FPRC": 20, "EIC": 10}


@dataclass(frozen=True)
class PLBindToolConfig:
    legacy_root: Path = DEFAULT_PLBIND_ROOT
    pdb_folder: Path | None = None
    output_root: Path = Path("cache/plbind_features")
    representation_mode: str = "legacy_casf"
    dataset_id: str = "dataset"
    adapter_version: str = "plbind_legacy_adapter_v3"
    validate_output_shape: bool = True

    @property
    def feature_py(self) -> Path:
        return self.legacy_root / "feature.py"


class PLBindFeatureTool:
    """Thin adapter around legacy plbind/feature.py.

    This treats each mathematical invariant as a separate tool call. The adapter
    imports the legacy module only when compute_one is invoked, so audit and dry
    runs do not need heavy scientific dependencies such as gudhi.
    """

    def __init__(self, config: PLBindToolConfig):
        self.config = config

    def adapter_for(self, invariant_name: str) -> "PLBindInvariantAdapter":
        return PLBindInvariantAdapter(tool=self, capability=get_capability(invariant_name))

    def adapters(self, invariant_names: tuple[str, ...] = INVARIANTS) -> tuple["PLBindInvariantAdapter", ...]:
        return tuple(self.adapter_for(name) for name in invariant_names)

    def capability_for(self, invariant_name: str) -> InvariantCapability:
        return get_capability(invariant_name)

    def compute_one(
        self,
        sample_id: str,
        invariant_name: str,
        *,
        dry_run: bool = False,
        representation_hash: str | None = None,
        input_hash: str | None = None,
    ) -> FeatureToolResult:
        return self._compute(
            sample_id,
            invariant_name,
            dry_run=dry_run,
            representation_hash=representation_hash,
            representation_spec=None,
            input_hash=input_hash,
        )

    def _compute(
        self,
        sample_id: str,
        invariant_name: str,
        *,
        dry_run: bool,
        representation_hash: str | None,
        representation_spec: RepresentationSpec | None,
        input_hash: str | None,
    ) -> FeatureToolResult:
        spec = get_invariant(invariant_name)
        self._assert_supported(spec.name, representation_spec=representation_spec)
        output_dir = self._output_dir(spec.name, representation_hash=representation_hash)
        if input_hash is not None:
            _assert_safe_cache_component("input_hash", input_hash)
            output_dir = output_dir / f"input-{input_hash}"
        output_path = output_dir / f"{sample_id}.npy"
        if dry_run:
            return FeatureToolResult(sample_id, spec.name, spec.legacy_name, output_path, None, "dry_run")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Legacy feature.py writes its intermediate result directly to the final
        # cache path. Serialize each cache key so another Slurm worker cannot read
        # that intermediate array before the adapter has normalized its channels.
        with _feature_cache_lock(output_path):
            if output_path.exists():
                if representation_spec is not None:
                    _normalize_adaptive_output(output_path, spec.name, representation_spec)
                self._validate_output(output_path, spec.name, representation_spec=representation_spec)
                return FeatureToolResult(
                    sample_id, spec.name, spec.legacy_name, output_path, None, "cached"
                )
            with measure_cost() as records:
                self._call_legacy(
                    sample_id,
                    spec.name,
                    spec.legacy_name,
                    output_dir,
                    representation_spec=representation_spec,
                )
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Legacy {spec.name} tool returned without creating expected output: {output_path}"
                )
            self._validate_output(output_path, spec.name, representation_spec=representation_spec)
            return FeatureToolResult(
                sample_id, spec.name, spec.legacy_name, output_path, records[0], "computed"
            )

    def compute_with_spec(
        self,
        sample_id: str,
        invariant_name: str,
        representation_spec: RepresentationSpec,
        *,
        dry_run: bool = False,
        input_hash: str | None = None,
    ) -> FeatureToolResult:
        representation_spec.assert_frozen()
        if representation_spec.mode.value != self.config.representation_mode:
            raise ValueError(
                f"Tool representation_mode={self.config.representation_mode!r} does not match "
                f"frozen spec mode={representation_spec.mode.value!r}"
            )
        return self._compute(
            sample_id,
            invariant_name,
            dry_run=dry_run,
            representation_hash=representation_spec.spec_hash,
            representation_spec=representation_spec,
            input_hash=input_hash,
        )

    def _output_dir(self, invariant_name: str, *, representation_hash: str | None) -> Path:
        if representation_hash is None:
            return self.config.output_root / invariant_name
        _assert_safe_cache_component("dataset_id", self.config.dataset_id)
        _assert_safe_cache_component("representation_hash", representation_hash)
        parameter_hash = stable_hash(
            {
                "adapter_version": self.config.adapter_version,
                "representation_mode": self.config.representation_mode,
                "invariant": invariant_name,
                "legacy_name": get_invariant(invariant_name).legacy_name,
                "expected_shape": get_invariant(invariant_name).expected_shapes.get(
                    self.config.representation_mode
                ),
            }
        )
        return (
            self.config.output_root
            / self.config.dataset_id
            / f"repr-{representation_hash}"
            / invariant_name
            / f"params-{parameter_hash}"
        )

    def _assert_supported(
        self,
        invariant_name: str,
        *,
        representation_spec: RepresentationSpec | None,
    ) -> None:
        capability = self.capability_for(invariant_name)
        if self.config.representation_mode not in capability.supported_representation_modes:
            raise ValueError(
                f"Legacy PLBind {invariant_name} does not support representation mode "
                f"{self.config.representation_mode!r}; supported modes are "
                f"{capability.supported_representation_modes}"
            )
        if self.config.representation_mode == "dataset_adaptive":
            if representation_spec is None:
                raise ValueError("dataset_adaptive computation requires a frozen RepresentationSpec")
            unsupported = tuple(
                pair for pair in representation_spec.pair_order if pair not in LEGACY_CASF_PAIR_ORDER
            )
            if unsupported:
                raise ValueError(
                    f"Legacy PLBind {invariant_name} cannot compute element pairs outside its audited "
                    f"CASF universe; unsupported pairs: {unsupported}"
                )

    def _validate_output(
        self,
        output_path: Path,
        invariant_name: str,
        *,
        representation_spec: RepresentationSpec | None,
    ) -> None:
        if self.config.validate_output_shape:
            if representation_spec is not None:
                validate_feature_file(
                    output_path,
                    expected_feature_shape(invariant_name, representation_spec),
                )
                return
            validate_feature_file(
                output_path,
                self.capability_for(invariant_name).expected_shape(self.config.representation_mode),
            )

    def _call_legacy(
        self,
        sample_id: str,
        invariant_name: str,
        legacy_name: str,
        output_dir: Path,
        *,
        representation_spec: RepresentationSpec | None,
    ) -> None:
        if not self.config.feature_py.exists():
            raise FileNotFoundError(f"Legacy feature.py not found: {self.config.feature_py}")
        pdb_folder = self.config.pdb_folder
        if pdb_folder is None:
            raise ValueError("pdb_folder is required for actual legacy feature computation")

        spec = importlib.util.spec_from_file_location("mint_scout_legacy_plbind_feature", self.config.feature_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import legacy feature.py from {self.config.feature_py}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(self.config.legacy_root))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(self.config.legacy_root))
        if representation_spec is None or representation_spec.mode.value == "legacy_casf":
            module.ProteinLigand(
                sample_id,
                legacy_name,
                pdb_folder=str(pdb_folder),
                pdb_feature_folder=str(output_dir),
            )
            return
        self._call_legacy_adaptive(
            module,
            sample_id,
            invariant_name,
            output_dir,
            representation_spec,
        )

    def _call_legacy_adaptive(
        self,
        module,
        sample_id: str,
        invariant_name: str,
        output_dir: Path,
        representation_spec: RepresentationSpec,
    ) -> None:
        import numpy as np

        profile = representation_spec.filtration_profiles[invariant_name]
        model = module.ProteinLigand(
            sample_id,
            "__mint_scout_deferred__",
            pdb_folder=str(self.config.pdb_folder),
            pdb_feature_folder=str(output_dir),
        )
        _load_adaptive_atoms(
            module,
            model,
            pdb_folder=self.config.pdb_folder,
            sample_id=sample_id,
        )
        model.set_euclidean_index_list()
        if invariant_name == "EIC":
            _assert_legacy_eic_profile(profile)
            model.get_euclidean_curvature()
        else:
            _assert_distance_profile(profile)
            if invariant_name == "CA":
                model.get_euclidean_facet(
                    min_edge=profile.start,
                    max_edge=profile.stop,
                    grid_step=profile.step,
                )
            else:
                grid_start_index = 0
                if invariant_name != "PL" and abs(profile.start) > 1e-10:
                    raise ValueError(f"Legacy {invariant_name} adaptive grid must start at 0")
                if invariant_name == "PL":
                    grid_start_index = int(round(profile.start / profile.step))
                    if (
                        grid_start_index < 0
                        or not np.isclose(
                            profile.start,
                            grid_start_index * profile.step,
                            rtol=0.0,
                            atol=1.0e-10,
                        )
                    ):
                        raise ValueError(
                            "Legacy PL adaptive grid start must align with its grid step"
                        )
                exclusive_grid_max = (
                    profile.num_points + grid_start_index
                ) * profile.step
                method_name = {
                    "PH": "get_persistent_homology",
                    "PL": "get_euclidean_L0",
                    "FPRC": "get_forman_graph_feature",
                }[invariant_name]
                getattr(model, method_name)(
                    grid_max=exclusive_grid_max,
                    grid_step=profile.step,
                )

        output_path = output_dir / f"{sample_id}.npy"
        _normalize_adaptive_output(output_path, invariant_name, representation_spec)

    def assert_output_shape(
        self,
        path: str | Path,
        invariant_name: str,
        *,
        representation_mode: str = "legacy_casf",
    ) -> tuple[int, ...]:
        capability = self.capability_for(invariant_name)
        return assert_feature_shape(path, capability.expected_shape(representation_mode))


def _load_adaptive_atoms(
    module,
    model,
    *,
    pdb_folder: Path | None,
    sample_id: str,
) -> None:
    if not hasattr(module, "Atom"):
        model.read_atom_from_pdb()
        return
    if pdb_folder is None:
        raise ValueError("pdb_folder is required for adaptive structure loading")

    sample_root = pdb_folder / sample_id
    protein = load_atom_cloud(sample_root / f"{sample_id}_pocket.pdb")
    ligand = load_atom_cloud(sample_root / f"{sample_id}_ligand.mol2")
    model.Protein_Atoms = [
        module.Atom(
            atype=element,
            resname="X",
            chain="protein",
            resid=index,
            coord=coordinate.tolist(),
        )
        for index, (element, coordinate) in enumerate(
            zip(protein.elements, protein.coordinates),
            start=1,
        )
    ]
    model.Ligand_Atoms = [
        module.Atom(
            atype=element,
            resname="ligand",
            chain="ligand",
            resid="ligand",
            coord=coordinate.tolist(),
        )
        for element, coordinate in zip(ligand.elements, ligand.coordinates)
    ]
    model.Protein_AtomCoord = protein.coordinates.copy()
    model.Ligand_AtomCoord = ligand.coordinates.copy()


@dataclass(frozen=True)
class PLBindInvariantAdapter:
    tool: PLBindFeatureTool
    capability: InvariantCapability

    def compute_one(self, sample_id: str, *, dry_run: bool = False) -> FeatureToolResult:
        return self.tool.compute_one(sample_id, self.capability.name, dry_run=dry_run)


def _assert_safe_cache_component(name: str, value: str) -> None:
    if not value or value in {".", ".."} or any(separator in value for separator in ("/", "\\")):
        raise ValueError(f"{name} must be a nonempty path-safe component")


@contextmanager
def _feature_cache_lock(output_path: Path) -> Iterator[None]:
    """Use an advisory per-feature lock shared by local and Slurm workers."""
    import fcntl

    lock_path = output_path.with_suffix(f"{output_path.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalize_adaptive_output(
    output_path: Path,
    invariant_name: str,
    representation_spec: RepresentationSpec,
) -> None:
    """Normalize legacy adaptive output to the frozen representation contract.

    The legacy PH/PL/FPRC tools build their axes with ``np.arange`` from a
    floating ``grid_max`` and ``grid_step``. For some adaptive step sizes this
    produces one extra endpoint, even though the frozen RepresentationSpec asks
    for a fixed number of points. We keep the requested prefix and channel order
    so every downstream feature matrix has the declared shape.
    """
    import numpy as np

    expected = expected_feature_shape(invariant_name, representation_spec)
    full = np.load(output_path)
    if full.ndim != 3:
        return
    if full.shape == expected:
        return
    if full.shape[2] != expected[2]:
        return

    profile = representation_spec.filtration_profiles[invariant_name]
    grid_start_index = 0
    if invariant_name == "PL":
        grid_start_index = int(round(profile.start / profile.step))
    grid_stop_index = grid_start_index + profile.num_points
    if full.shape[0] < grid_stop_index:
        return
    normalized = full[grid_start_index:grid_stop_index]

    if normalized.shape[1] == len(LEGACY_CASF_PAIR_ORDER):
        channel_indices = [
            LEGACY_CASF_PAIR_ORDER.index(pair) for pair in representation_spec.pair_order
        ]
        normalized = normalized[:, channel_indices, :]
    elif normalized.shape[1] != expected[1]:
        return

    if normalized.shape == expected:
        np.save(output_path, normalized)


def expected_feature_shape(
    invariant_name: str,
    representation_spec: RepresentationSpec,
) -> tuple[int, int, int]:
    profile = representation_spec.filtration_profiles[invariant_name]
    return (
        profile.num_points,
        len(representation_spec.pair_order),
        TRAILING_FEATURE_DIMS[invariant_name],
    )


def _assert_distance_profile(profile) -> None:
    if profile.scale_kind != "distance":
        raise ValueError(f"Expected distance filtration profile, got {profile.scale_kind!r}")
    expected_stop = profile.start + (profile.num_points - 1) * profile.step
    if abs(profile.stop - expected_stop) > 1e-8:
        raise ValueError("Filtration profile stop/step/num_points are inconsistent")


def _assert_legacy_eic_profile(profile) -> None:
    expected = ("tau", 0.2, 5.0, 0.1, 49)
    observed = (
        profile.scale_kind,
        round(profile.start, 10),
        round(profile.stop, 10),
        round(profile.step, 10),
        profile.num_points,
    )
    if observed != expected:
        raise ValueError(
            "Legacy EIC currently supports only its audited tau grid "
            "(0.2, 5.0, 0.1, 49); changing tau requires a legacy-code parameterization audit"
        )
