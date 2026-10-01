from __future__ import annotations

import gzip
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np

from mint_scout.data.element_inventory import normalize_element_token
from mint_scout.data.element_pairs import SystemType
from mint_scout.filtration import FiltrationConfig
from mint_scout.invariants.manifest import stable_hash


@dataclass(frozen=True)
class AtomCloud:
    elements: tuple[str, ...]
    coordinates: np.ndarray

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (n_atoms, 3)")
        if coordinates.shape[0] != len(self.elements):
            raise ValueError("elements and coordinates must contain the same number of atoms")
        if not self.elements:
            raise ValueError("atom cloud cannot be empty")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("atom coordinates must be finite")
        coordinates.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)

    def coordinates_for(self, element: str) -> np.ndarray:
        indices = [index for index, value in enumerate(self.elements) if value == element]
        return self.coordinates[np.asarray(indices, dtype=int)]

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update("\0".join(self.elements).encode("ascii"))
        digest.update(np.asarray(self.coordinates, dtype="<f8").tobytes())
        return digest.hexdigest()[:16]


@dataclass(frozen=True)
class GeometryProfile:
    system_type: SystemType
    modeling_dataset_fingerprint: str
    local_distance_quantile: float
    dataset_distance_quantile: float
    margin_factor: float
    max_filtration_angstrom: float
    sample_count: int
    local_summary_count: int
    pair_observation_counts: Mapping[str, int]
    anomalies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["pair_observation_counts"] = dict(sorted(self.pair_observation_counts.items()))
        payload["anomalies"] = list(self.anomalies)
        return payload


def load_atom_cloud(path: str | Path) -> AtomCloud:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".pdb", ".ent", ".pdbqt", ".pdb.gz", ".ent.gz", ".pdbqt.gz")):
        atoms = _iter_pdb_atoms(path)
    elif suffixes.endswith((".mol2", ".mol2.gz")):
        atoms = _iter_mol2_atoms(path)
    elif suffixes.endswith((".sdf", ".mol", ".sdf.gz", ".mol.gz")):
        atoms = _iter_molfile_atoms(path)
    elif suffixes.endswith((".xyz", ".xyz.gz")):
        atoms = _iter_xyz_atoms(path)
    else:
        raise ValueError(f"Unsupported molecular file type: {path}")
    rows = tuple(atoms)
    if not rows:
        raise ValueError(f"No valid atoms parsed from {path}")
    return AtomCloud(
        elements=tuple(row[0] for row in rows),
        coordinates=np.asarray([row[1] for row in rows], dtype=float),
    )


def load_modeling_clouds(
    role_paths_by_sample: Mapping[str, Mapping[str, str | Path]],
) -> dict[str, dict[str, AtomCloud]]:
    if not role_paths_by_sample:
        raise ValueError("modeling pool cannot be empty")
    clouds: dict[str, dict[str, AtomCloud]] = {}
    for sample_id in sorted(role_paths_by_sample):
        roles = role_paths_by_sample[sample_id]
        if not roles:
            raise ValueError(f"Sample {sample_id!r} has no structure roles")
        clouds[sample_id] = {role: load_atom_cloud(path) for role, path in sorted(roles.items())}
    return clouds


def modeling_dataset_fingerprint(clouds_by_sample: Mapping[str, Mapping[str, AtomCloud]]) -> str:
    payload = [
        {
            "sample_id": sample_id,
            "roles": {
                role: cloud.content_hash for role, cloud in sorted(clouds_by_sample[sample_id].items())
            },
        }
        for sample_id in sorted(clouds_by_sample)
    ]
    return stable_hash(payload)


def profile_protein_ligand_geometry(
    *,
    clouds_by_sample: Mapping[str, Mapping[str, AtomCloud]],
    retained_pairs: Iterable[tuple[str, str]],
    config: FiltrationConfig = FiltrationConfig(),
    protein_role: str = "protein",
    ligand_role: str = "ligand",
    distance_chunk_size: int = 2048,
) -> GeometryProfile:
    pairs = tuple(retained_pairs)
    if not pairs:
        raise ValueError("retained_pairs cannot be empty")
    local_summaries: list[float] = []
    observation_counts = {_pair_name(pair, protein_role, ligand_role): 0 for pair in pairs}
    for sample_id in sorted(clouds_by_sample):
        roles = clouds_by_sample[sample_id]
        if protein_role not in roles or ligand_role not in roles:
            raise ValueError(f"Sample {sample_id!r} is missing protein or ligand role")
        protein = roles[protein_role]
        ligand = roles[ligand_role]
        for pair in pairs:
            left = protein.coordinates_for(pair[0])
            right = ligand.coordinates_for(pair[1])
            if len(left) == 0 or len(right) == 0:
                continue
            nearest = _bidirectional_nearest_distances(left, right, chunk_size=distance_chunk_size)
            local_summaries.append(float(np.quantile(nearest, config.local_distance_quantile)))
            observation_counts[_pair_name(pair, protein_role, ligand_role)] += 1
    return _finish_geometry_profile(
        system_type="protein_ligand",
        clouds_by_sample=clouds_by_sample,
        local_summaries=local_summaries,
        observation_counts=observation_counts,
        config=config,
    )


def profile_protein_ligand_geometry_from_paths(
    *,
    role_paths_by_sample: Mapping[str, Mapping[str, str | Path]],
    retained_pairs: Iterable[tuple[str, str]],
    config: FiltrationConfig = FiltrationConfig(),
    protein_role: str = "protein",
    ligand_role: str = "ligand",
    distance_chunk_size: int = 2048,
) -> GeometryProfile:
    pairs = tuple(retained_pairs)
    if not pairs:
        raise ValueError("retained_pairs cannot be empty")
    local_summaries: list[float] = []
    observation_counts = {_pair_name(pair, protein_role, ligand_role): 0 for pair in pairs}
    fingerprint_rows = []
    for sample_id in sorted(role_paths_by_sample):
        paths = role_paths_by_sample[sample_id]
        if protein_role not in paths or ligand_role not in paths:
            raise ValueError(f"Sample {sample_id!r} is missing protein or ligand role")
        protein = load_atom_cloud(paths[protein_role])
        ligand = load_atom_cloud(paths[ligand_role])
        fingerprint_rows.append(
            {
                "sample_id": sample_id,
                "roles": {protein_role: protein.content_hash, ligand_role: ligand.content_hash},
            }
        )
        for pair in pairs:
            left = protein.coordinates_for(pair[0])
            right = ligand.coordinates_for(pair[1])
            if len(left) == 0 or len(right) == 0:
                continue
            nearest = _bidirectional_nearest_distances(left, right, chunk_size=distance_chunk_size)
            local_summaries.append(float(np.quantile(nearest, config.local_distance_quantile)))
            observation_counts[_pair_name(pair, protein_role, ligand_role)] += 1
    return _finish_geometry_summary(
        system_type="protein_ligand",
        modeling_dataset_fingerprint=stable_hash(fingerprint_rows),
        sample_count=len(role_paths_by_sample),
        local_summaries=local_summaries,
        observation_counts=observation_counts,
        config=config,
    )


def profile_small_molecule_geometry(
    *,
    clouds_by_sample: Mapping[str, Mapping[str, AtomCloud]],
    retained_pairs: Iterable[tuple[str, str]],
    config: FiltrationConfig = FiltrationConfig(),
    molecule_role: str = "molecule",
    distance_chunk_size: int = 2048,
) -> GeometryProfile:
    pairs = tuple(retained_pairs)
    if not pairs:
        raise ValueError("retained_pairs cannot be empty")
    local_summaries: list[float] = []
    observation_counts = {_pair_name(pair): 0 for pair in pairs}
    for sample_id in sorted(clouds_by_sample):
        roles = clouds_by_sample[sample_id]
        if molecule_role not in roles:
            raise ValueError(f"Sample {sample_id!r} is missing molecule role")
        cloud = roles[molecule_role]
        per_molecule_distances: list[np.ndarray] = []
        for pair in pairs:
            left = cloud.coordinates_for(pair[0])
            right = cloud.coordinates_for(pair[1])
            distances = _intramolecular_pair_distances(
                left,
                right,
                same_element=pair[0] == pair[1],
                chunk_size=distance_chunk_size,
            )
            if distances.size == 0:
                continue
            per_molecule_distances.append(distances)
            observation_counts[_pair_name(pair)] += 1
        if per_molecule_distances:
            combined = np.concatenate(per_molecule_distances)
            local_summaries.append(float(np.quantile(combined, config.local_distance_quantile)))
    return _finish_geometry_profile(
        system_type="small_molecule",
        clouds_by_sample=clouds_by_sample,
        local_summaries=local_summaries,
        observation_counts=observation_counts,
        config=config,
    )


def profile_small_molecule_geometry_from_paths(
    *,
    role_paths_by_sample: Mapping[str, Mapping[str, str | Path]],
    retained_pairs: Iterable[tuple[str, str]],
    config: FiltrationConfig = FiltrationConfig(),
    molecule_role: str = "molecule",
    distance_chunk_size: int = 2048,
) -> GeometryProfile:
    pairs = tuple(retained_pairs)
    if not pairs:
        raise ValueError("retained_pairs cannot be empty")
    local_summaries: list[float] = []
    observation_counts = {_pair_name(pair): 0 for pair in pairs}
    fingerprint_rows = []
    for sample_id in sorted(role_paths_by_sample):
        paths = role_paths_by_sample[sample_id]
        if molecule_role not in paths:
            raise ValueError(f"Sample {sample_id!r} is missing molecule role")
        cloud = load_atom_cloud(paths[molecule_role])
        fingerprint_rows.append(
            {"sample_id": sample_id, "roles": {molecule_role: cloud.content_hash}}
        )
        per_molecule_distances: list[np.ndarray] = []
        for pair in pairs:
            distances = _intramolecular_pair_distances(
                cloud.coordinates_for(pair[0]),
                cloud.coordinates_for(pair[1]),
                same_element=pair[0] == pair[1],
                chunk_size=distance_chunk_size,
            )
            if distances.size == 0:
                continue
            per_molecule_distances.append(distances)
            observation_counts[_pair_name(pair)] += 1
        if per_molecule_distances:
            local_summaries.append(
                float(np.quantile(np.concatenate(per_molecule_distances), config.local_distance_quantile))
            )
    return _finish_geometry_summary(
        system_type="small_molecule",
        modeling_dataset_fingerprint=stable_hash(fingerprint_rows),
        sample_count=len(role_paths_by_sample),
        local_summaries=local_summaries,
        observation_counts=observation_counts,
        config=config,
    )


def _finish_geometry_profile(
    *,
    system_type: SystemType,
    clouds_by_sample: Mapping[str, Mapping[str, AtomCloud]],
    local_summaries: list[float],
    observation_counts: Mapping[str, int],
    config: FiltrationConfig,
) -> GeometryProfile:
    return _finish_geometry_summary(
        system_type=system_type,
        modeling_dataset_fingerprint=modeling_dataset_fingerprint(clouds_by_sample),
        sample_count=len(clouds_by_sample),
        local_summaries=local_summaries,
        observation_counts=observation_counts,
        config=config,
    )


def _finish_geometry_summary(
    *,
    system_type: SystemType,
    modeling_dataset_fingerprint: str,
    sample_count: int,
    local_summaries: list[float],
    observation_counts: Mapping[str, int],
    config: FiltrationConfig,
) -> GeometryProfile:
    if not local_summaries:
        raise ValueError("No supported geometry observations were found for retained element pairs")
    rmax = config.margin_factor * float(np.quantile(local_summaries, config.dataset_distance_quantile))
    if not np.isfinite(rmax) or rmax <= 0.0:
        raise ValueError("Adaptive maximum filtration must be finite and positive")
    anomalies = tuple(
        f"GEOMETRY_PAIR_UNOBSERVED:{name}" for name, count in sorted(observation_counts.items()) if count == 0
    )
    return GeometryProfile(
        system_type=system_type,
        modeling_dataset_fingerprint=modeling_dataset_fingerprint,
        local_distance_quantile=config.local_distance_quantile,
        dataset_distance_quantile=config.dataset_distance_quantile,
        margin_factor=config.margin_factor,
        max_filtration_angstrom=rmax,
        sample_count=sample_count,
        local_summary_count=len(local_summaries),
        pair_observation_counts=dict(observation_counts),
        anomalies=anomalies,
    )


def _bidirectional_nearest_distances(left: np.ndarray, right: np.ndarray, *, chunk_size: int) -> np.ndarray:
    left_min = _nearest_distances(left, right, chunk_size=chunk_size)
    right_min = _nearest_distances(right, left, chunk_size=chunk_size)
    return np.concatenate((left_min, right_min))


def _nearest_distances(query: np.ndarray, reference: np.ndarray, *, chunk_size: int) -> np.ndarray:
    if chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive")
    minima = []
    for start in range(0, len(query), chunk_size):
        chunk = query[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        minima.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(minima)


def _intramolecular_pair_distances(
    left: np.ndarray,
    right: np.ndarray,
    *,
    same_element: bool,
    chunk_size: int,
) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.empty(0, dtype=float)
    if same_element:
        if len(left) < 2:
            return np.empty(0, dtype=float)
        squared = np.sum((left[:, None, :] - left[None, :, :]) ** 2, axis=2)
        return np.sqrt(squared[np.triu_indices(len(left), k=1)])
    chunks = []
    for start in range(0, len(left), chunk_size):
        chunk = left[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - right[None, :, :]) ** 2, axis=2)
        chunks.append(np.sqrt(squared).reshape(-1))
    return np.concatenate(chunks)


def _pair_name(
    pair: tuple[str, str],
    left_role: str | None = None,
    right_role: str | None = None,
) -> str:
    if left_role and right_role:
        return f"{left_role}:{pair[0]}|{right_role}:{pair[1]}"
    return f"{pair[0]}-{pair[1]}"


def _open_text(path: Path) -> Iterator[str]:
    opener = gzip.open if "".join(path.suffixes).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def _iter_pdb_atoms(path: Path) -> Iterator[tuple[str, tuple[float, float, float]]]:
    for line in _open_text(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = normalize_element_token(line[76:78]) if len(line) >= 78 else None
        if element is None:
            token = line[12:16].strip().lstrip("0123456789") if len(line) >= 16 else ""
            element = normalize_element_token(token[:2] if len(token) >= 2 and token[1].islower() else token[:1])
        try:
            coordinate = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except (ValueError, IndexError):
            continue
        if element:
            yield element, coordinate


def _iter_mol2_atoms(path: Path) -> Iterator[tuple[str, tuple[float, float, float]]]:
    in_atoms = False
    for line in _open_text(path):
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>"):
            in_atoms = stripped == "@<TRIPOS>ATOM"
            continue
        if not in_atoms or not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 6:
            continue
        element = normalize_element_token(parts[5].split(".", 1)[0])
        try:
            coordinate = (float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if element:
            yield element, coordinate


def _iter_molfile_atoms(path: Path) -> Iterator[tuple[str, tuple[float, float, float]]]:
    lines = list(_open_text(path))
    if len(lines) < 4:
        return
    try:
        atom_count = int(lines[3][0:3])
    except ValueError:
        return
    for line in lines[4 : 4 + atom_count]:
        if len(line) < 34:
            continue
        element = normalize_element_token(line[31:34])
        try:
            coordinate = (float(line[0:10]), float(line[10:20]), float(line[20:30]))
        except ValueError:
            continue
        if element:
            yield element, coordinate


def _iter_xyz_atoms(path: Path) -> Iterator[tuple[str, tuple[float, float, float]]]:
    for line in list(_open_text(path))[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        element = normalize_element_token(parts[0])
        try:
            coordinate = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            continue
        if element:
            yield element, coordinate
