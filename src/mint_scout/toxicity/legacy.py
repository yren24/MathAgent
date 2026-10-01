from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from mint_scout.invariants.contracts import (
    FeatureToolResult,
    InvariantCapability,
    validate_feature_file,
)
from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationMode, RepresentationSpec
from mint_scout.search.cost import measure_cost
from mint_scout.toxicity.manifest import normalized_cas_key


TOXICITY_INVARIANTS = ("PH", "PL", "CA", "FPRC", "EIC")
TOXICITY_ELEMENT_UNIVERSE = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}
LEGACY_NAMES = {
    "PH": "homology",
    "PL": "lap",
    "CA": "facet",
    "FPRC": "forman",
    "EIC": "curvature",
}
LEGACY_TOXICITY_SHAPES = {
    "PH": (50, 30, 1),
    "PL": (50, 30, 8),
    "CA": (50, 30, 2),
    "FPRC": (50, 30, 20),
    "EIC": (25, 30, 10),
}
TRAILING_FEATURE_DIMS = {"PH": 1, "PL": 8, "CA": 2, "FPRC": 20, "EIC": 10}
DESCRIPTIONS = {
    "PH": "persistent homology on intramolecular element-pair networks",
    "PL": "persistent Laplacian on intramolecular element-pair networks",
    "CA": "commutative-algebraic facet descriptors for small molecules",
    "FPRC": "Forman persistent Ricci curvature for small molecules",
    "EIC": "element-interactive differential curvature for small molecules",
}


@dataclass(frozen=True)
class ToxicityToolConfig:
    legacy_root: Path
    molecule_dirs: tuple[Path, ...]
    output_root: Path = Path("cache/toxicity_features")
    dataset_id: str = "toxicity"
    adapter_version: str = "toxicity_legacy_adapter_v1"
    validate_output_shape: bool = True
    max_filtration: float = 10.0
    graph_step: float = 0.2
    homology_step: float = 0.2
    homology_dims: str = "0"
    facet_step: float = 0.2
    facet_min_edge: float = 0.0
    facet_dims: str = "0"
    curvature_tau_min: float = 0.2
    curvature_tau_max: float = 5.0
    curvature_tau_step: float = 0.2
    curvature_bidirectional: bool = False
    bond_delta: float = 0.45

    @property
    def feature_py(self) -> Path:
        return self.legacy_root / "toxicity_topology_features.py"

    def scientific_parameters(self) -> dict[str, object]:
        return {
            "max_filtration": self.max_filtration,
            "graph_step": self.graph_step,
            "homology_step": self.homology_step,
            "homology_dims": self.homology_dims,
            "facet_step": self.facet_step,
            "facet_min_edge": self.facet_min_edge,
            "facet_dims": self.facet_dims,
            "curvature_tau_min": self.curvature_tau_min,
            "curvature_tau_max": self.curvature_tau_max,
            "curvature_tau_step": self.curvature_tau_step,
            "curvature_bidirectional": self.curvature_bidirectional,
            "bond_delta": self.bond_delta,
        }


class ToxicityFeatureTool:
    """Thin, per-invariant adapter around the audited LD50 feature implementation."""

    def __init__(self, config: ToxicityToolConfig):
        self.config = config

    def adapter_for(self, invariant_name: str) -> "ToxicityInvariantAdapter":
        return ToxicityInvariantAdapter(self, self.capability_for(invariant_name))

    def adapters(
        self,
        invariant_names: tuple[str, ...] = TOXICITY_INVARIANTS,
    ) -> tuple["ToxicityInvariantAdapter", ...]:
        return tuple(self.adapter_for(name) for name in invariant_names)

    def capability_for(self, invariant_name: str) -> InvariantCapability:
        name = _invariant_name(invariant_name)
        return InvariantCapability(
            name=name,
            legacy_name=LEGACY_NAMES[name],
            description=DESCRIPTIONS[name],
            supported_systems=("small_molecule",),
            supported_representation_modes=("legacy_toxicity", "dataset_adaptive"),
            expected_shapes={
                "legacy_toxicity": LEGACY_TOXICITY_SHAPES[name],
                "dataset_adaptive": (-1, -1, TRAILING_FEATURE_DIMS[name]),
            },
            domain="single-molecule element-specific topology",
        )

    def compute_one(
        self,
        sample_id: str,
        invariant_name: str,
        *,
        dry_run: bool = False,
    ) -> FeatureToolResult:
        return self._compute(
            sample_id,
            invariant_name,
            dry_run=dry_run,
            representation_spec=None,
        )

    def compute_with_spec(
        self,
        sample_id: str,
        invariant_name: str,
        representation_spec: RepresentationSpec,
        *,
        dry_run: bool = False,
    ) -> FeatureToolResult:
        representation_spec.assert_frozen()
        if representation_spec.system_type != "small_molecule":
            raise ValueError("toxicity feature tools require system_type=small_molecule")
        if representation_spec.mode not in {
            RepresentationMode.LEGACY_TOXICITY,
            RepresentationMode.DATASET_ADAPTIVE,
        }:
            raise ValueError(
                f"Unsupported toxicity representation mode: {representation_spec.mode.value}"
            )
        unsupported = sorted(
            {
                element
                for pair in representation_spec.pair_order
                for element in pair
                if element not in TOXICITY_ELEMENT_UNIVERSE
            }
        )
        if unsupported:
            raise ValueError(
                "Legacy toxicity implementation cannot compute elements outside "
                f"its audited universe: {unsupported}"
            )
        return self._compute(
            sample_id,
            invariant_name,
            dry_run=dry_run,
            representation_spec=representation_spec,
        )

    def _compute(
        self,
        sample_id: str,
        invariant_name: str,
        *,
        dry_run: bool,
        representation_spec: RepresentationSpec | None,
    ) -> FeatureToolResult:
        name = _invariant_name(invariant_name)
        _assert_safe_component("sample_id", sample_id)
        _assert_safe_component("dataset_id", self.config.dataset_id)
        molecule_path = self._resolve_molecule(sample_id)
        parameter_hash = self._parameter_hash(name, representation_spec)
        input_hash = _sha256_file(molecule_path)[:16]
        representation_path = (
            "legacy_toxicity"
            if representation_spec is None
            else f"repr-{representation_spec.spec_hash}"
        )
        output_path = (
            self.config.output_root
            / self.config.dataset_id
            / representation_path
            / name
            / f"params-{parameter_hash}"
            / f"input-{input_hash}"
            / f"{sample_id}.npy"
        )
        if dry_run:
            return FeatureToolResult(
                sample_id, name, LEGACY_NAMES[name], output_path, None, "dry_run"
            )
        if output_path.exists():
            self._validate(output_path, name, representation_spec)
            return FeatureToolResult(
                sample_id, name, LEGACY_NAMES[name], output_path, None, "cached"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with measure_cost() as records:
            self._call_legacy(
                name,
                sample_id,
                molecule_path,
                output_path,
                representation_spec,
            )
        if not output_path.exists():
            raise FileNotFoundError(
                f"Legacy toxicity {name} returned without creating {output_path}"
            )
        self._validate(output_path, name, representation_spec)
        return FeatureToolResult(
            sample_id, name, LEGACY_NAMES[name], output_path, records[0], "computed"
        )

    def _resolve_molecule(self, sample_id: str) -> Path:
        matches = [directory / f"{sample_id}.mol2" for directory in self.config.molecule_dirs]
        exact = [path for path in matches if path.is_file()]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ValueError(f"Sample {sample_id!r} exists in multiple molecule directories")
        key = normalized_cas_key(sample_id)
        recovered: list[Path] = []
        if key is not None:
            for directory in self.config.molecule_dirs:
                for path in directory.glob("*.mol2"):
                    if normalized_cas_key(path.stem) == key:
                        recovered.append(path)
        if len(recovered) == 1:
            return recovered[0]
        if len(recovered) > 1:
            raise ValueError(f"Sample {sample_id!r} has ambiguous MOL2 matches: {recovered}")
        raise FileNotFoundError(f"No MOL2 file found for toxicity sample {sample_id!r}")

    def _parameter_hash(
        self,
        invariant_name: str,
        representation_spec: RepresentationSpec | None,
    ) -> str:
        if not self.config.feature_py.is_file():
            raise FileNotFoundError(
                f"Legacy toxicity feature script not found: {self.config.feature_py}"
            )
        return stable_hash(
            {
                "adapter_version": self.config.adapter_version,
                "exclusive_grid_stop_policy": "nextafter_toward_negative_infinity_v1",
                "invariant": invariant_name,
                "legacy_name": LEGACY_NAMES[invariant_name],
                "legacy_script_sha256": _sha256_file(self.config.feature_py),
                "parameters": self.config.scientific_parameters(),
                "representation": (
                    None
                    if representation_spec is None
                    else representation_spec.semantic_payload()
                ),
                "expected_shape": self._expected_shape(
                    invariant_name, representation_spec
                ),
            }
        )

    def _call_legacy(
        self,
        invariant_name: str,
        sample_id: str,
        molecule_path: Path,
        output_path: Path,
        representation_spec: RepresentationSpec | None,
    ) -> None:
        module_name = f"mint_scout_legacy_toxicity_{_sha256_file(self.config.feature_py)[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, self.config.feature_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import {self.config.feature_py}")
        module = importlib.util.module_from_spec(spec)
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = module
        sys.path.insert(0, str(self.config.legacy_root))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(self.config.legacy_root))
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
        if representation_spec is not None:
            module.PAIR_ELEMENT_NAMES = list(representation_spec.pair_order)
            module.PAIR_NAMES = [
                f"{left}-{right}" for left, right in representation_spec.pair_order
            ]
        args = self._legacy_args(invariant_name, representation_spec)
        wrote = module.generate_one(
            LEGACY_NAMES[invariant_name],
            sample_id,
            str(molecule_path),
            str(output_path),
            args,
        )
        if wrote is False and not output_path.exists():
            raise RuntimeError(
                f"Legacy toxicity generator skipped {sample_id} without an existing output"
            )

    def _legacy_args(
        self,
        invariant_name: str,
        representation_spec: RepresentationSpec | None,
    ) -> SimpleNamespace:
        if representation_spec is None:
            max_filtration = self.config.max_filtration
            graph_step = self.config.graph_step
            homology_step = self.config.homology_step
            facet_step = self.config.facet_step
            facet_min_edge = self.config.facet_min_edge
            tau_min = self.config.curvature_tau_min
            tau_max = self.config.curvature_tau_max
            tau_step = self.config.curvature_tau_step
        else:
            profile = representation_spec.filtration_profiles[invariant_name]
            if invariant_name == "EIC":
                max_filtration = self.config.max_filtration
                graph_step = self.config.graph_step
                homology_step = self.config.homology_step
                facet_step = self.config.facet_step
                facet_min_edge = self.config.facet_min_edge
                tau_min = profile.start
                tau_max = profile.stop
                tau_step = profile.step
            else:
                if profile.scale_kind != "distance":
                    raise ValueError(f"{invariant_name} requires a distance filtration")
                if not math.isclose(profile.start, 0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"Adaptive toxicity {invariant_name} filtration must start at zero"
                    )
                exclusive_stop = math.nextafter(
                    profile.step * profile.num_points,
                    -math.inf,
                )
                max_filtration = exclusive_stop
                graph_step = profile.step
                homology_step = profile.step
                facet_step = profile.step
                facet_min_edge = profile.start
                tau_min = self.config.curvature_tau_min
                tau_max = self.config.curvature_tau_max
                tau_step = self.config.curvature_tau_step
        return SimpleNamespace(
            overwrite=False,
            bond_delta=self.config.bond_delta,
            max_filtration=max_filtration,
            graph_step=graph_step,
            homology_step=homology_step,
            homology_dims=self.config.homology_dims,
            facet_step=facet_step,
            facet_min_edge=facet_min_edge,
            facet_dims=self.config.facet_dims,
            curvature_tau_min=tau_min,
            curvature_tau_max=tau_max,
            curvature_tau_step=tau_step,
            curvature_bidirectional=self.config.curvature_bidirectional,
        )

    def _expected_shape(
        self,
        invariant_name: str,
        representation_spec: RepresentationSpec | None,
    ) -> tuple[int, int, int]:
        if representation_spec is None:
            return LEGACY_TOXICITY_SHAPES[invariant_name]
        profile = representation_spec.filtration_profiles[invariant_name]
        return (
            profile.num_points,
            len(representation_spec.pair_order),
            TRAILING_FEATURE_DIMS[invariant_name],
        )

    def _validate(
        self,
        output_path: Path,
        invariant_name: str,
        representation_spec: RepresentationSpec | None,
    ) -> None:
        if self.config.validate_output_shape:
            validate_feature_file(
                output_path,
                self._expected_shape(invariant_name, representation_spec),
            )


@dataclass(frozen=True)
class ToxicityInvariantAdapter:
    tool: ToxicityFeatureTool
    capability: InvariantCapability

    def compute_one(self, sample_id: str, *, dry_run: bool = False) -> FeatureToolResult:
        return self.tool.compute_one(sample_id, self.capability.name, dry_run=dry_run)

    def compute_with_spec(
        self,
        sample_id: str,
        representation_spec: RepresentationSpec,
        *,
        dry_run: bool = False,
    ) -> FeatureToolResult:
        return self.tool.compute_with_spec(
            sample_id,
            self.capability.name,
            representation_spec,
            dry_run=dry_run,
        )


def _invariant_name(value: str) -> str:
    name = value.upper()
    if name not in TOXICITY_INVARIANTS:
        raise KeyError(
            f"Unknown toxicity invariant {value!r}; expected one of {TOXICITY_INVARIANTS}"
        )
    return name


def _assert_safe_component(name: str, value: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe {name}: {value!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one legacy LD50 mathematical feature as an isolated tool call."
    )
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--molecule-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="LD50")
    parser.add_argument("--invariant", required=True, choices=TOXICITY_INVARIANTS)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    tool = ToxicityFeatureTool(
        ToxicityToolConfig(
            legacy_root=args.legacy_root,
            molecule_dirs=tuple(args.molecule_dir),
            output_root=args.output_root,
            dataset_id=args.dataset_id,
        )
    )
    for sample_id in args.sample_id:
        result = tool.compute_one(sample_id, args.invariant, dry_run=args.dry_run)
        print(
            f"sample_id={result.sample_id} invariant={result.invariant_name} "
            f"legacy={result.legacy_name} status={result.status} output={result.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
