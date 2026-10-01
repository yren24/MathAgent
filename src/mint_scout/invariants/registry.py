from __future__ import annotations

from dataclasses import dataclass

from mint_scout.invariants.contracts import InvariantCapability


INVARIANTS = ("PH", "PL", "CA", "FPRC", "EIC")
LEGACY_CASF_MODE = "legacy_casf"
DATASET_ADAPTIVE_MODE = "dataset_adaptive"


@dataclass(frozen=True)
class InvariantSpec:
    name: str
    legacy_name: str
    description: str
    supported_systems: tuple[str, ...]
    supported_representation_modes: tuple[str, ...]
    expected_shapes: dict[str, tuple[int, ...]]
    domain: str

    def capability(self) -> InvariantCapability:
        return InvariantCapability(
            name=self.name,
            legacy_name=self.legacy_name,
            description=self.description,
            supported_systems=self.supported_systems,
            supported_representation_modes=self.supported_representation_modes,
            expected_shapes=self.expected_shapes,
            domain=self.domain,
        )


REGISTRY: dict[str, InvariantSpec] = {
    "PH": InvariantSpec(
        "PH",
        "homology",
        "persistent homology",
        ("protein_ligand",),
        (LEGACY_CASF_MODE, DATASET_ADAPTIVE_MODE),
        {LEGACY_CASF_MODE: (150, 40, 2), DATASET_ADAPTIVE_MODE: (-1, -1, 2)},
        "bipartite Rips dimension-0 plus alpha-complex higher-dimensional terms",
    ),
    "PL": InvariantSpec(
        "PL",
        "lap",
        "persistent Laplacian",
        ("protein_ligand",),
        (LEGACY_CASF_MODE, DATASET_ADAPTIVE_MODE),
        {LEGACY_CASF_MODE: (30, 40, 8), DATASET_ADAPTIVE_MODE: (-1, -1, 8)},
        "thresholded protein-ligand bipartite graph Laplacian eigenvalue statistics",
    ),
    "CA": InvariantSpec(
        "CA",
        "facet",
        "commutative-algebraic/facet descriptors",
        ("protein_ligand",),
        (LEGACY_CASF_MODE, DATASET_ADAPTIVE_MODE),
        {LEGACY_CASF_MODE: (141, 40, 2), DATASET_ADAPTIVE_MODE: (-1, -1, 2)},
        "legacy facet descriptor on bipartite distance matrix",
    ),
    "FPRC": InvariantSpec(
        "FPRC",
        "forman",
        "Forman persistent Ricci curvature",
        ("protein_ligand",),
        (LEGACY_CASF_MODE, DATASET_ADAPTIVE_MODE),
        {LEGACY_CASF_MODE: (30, 40, 20), DATASET_ADAPTIVE_MODE: (-1, -1, 20)},
        "unweighted Forman curvature statistics on thresholded bipartite graphs",
    ),
    "EIC": InvariantSpec(
        "EIC",
        "curvature",
        "element-interactive curvature",
        ("protein_ligand",),
        (LEGACY_CASF_MODE, DATASET_ADAPTIVE_MODE),
        {LEGACY_CASF_MODE: (49, 40, 10), DATASET_ADAPTIVE_MODE: (-1, -1, 10)},
        "element-interactive differential curvature across tau scales",
    ),
}


def get_invariant(name: str) -> InvariantSpec:
    key = name.upper()
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown invariant {name!r}; expected one of {sorted(REGISTRY)}") from exc


def get_capability(name: str) -> InvariantCapability:
    return get_invariant(name).capability()
