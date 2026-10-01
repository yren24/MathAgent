from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping

from mint_scout.data.element_pairs import (
    AdaptivePairBuild,
    SupportThresholds,
    SystemType,
    build_dataset_adaptive_schema,
)
from mint_scout.data.geometry import (
    GeometryProfile,
    load_atom_cloud,
    profile_protein_ligand_geometry_from_paths,
    profile_small_molecule_geometry_from_paths,
)
from mint_scout.data.element_inventory import iter_elements_from_file
from mint_scout.filtration import (
    FiltrationConfig,
    make_distance_filtration_profile,
    make_tau_profile,
)
from mint_scout.representation import RepresentationMode, RepresentationSpec


DISTANCE_INVARIANTS = ("PH", "PL", "CA", "FPRC")
INVARIANT_LIBRARY = DISTANCE_INVARIANTS + ("EIC",)


@dataclass(frozen=True)
class HydrogenPolicyConfig:
    mode: Literal["auto", "include", "exclude"] = "auto"
    task_relevance: Literal["likely_relevant", "unlikely_relevant", "uncertain"] = (
        "uncertain"
    )
    min_explicit_sample_fraction: float = 0.95

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "include", "exclude"}:
            raise ValueError("hydrogen policy mode must be auto, include, or exclude")
        if self.task_relevance not in {
            "likely_relevant",
            "unlikely_relevant",
            "uncertain",
        }:
            raise ValueError("invalid hydrogen task relevance")
        if not 0.0 <= self.min_explicit_sample_fraction <= 1.0:
            raise ValueError("min_explicit_sample_fraction must be in [0, 1]")


@dataclass(frozen=True)
class MetalAwarenessConfig:
    enabled: bool = True
    protein_elements: tuple[str, ...] = (
        "Ca",
        "Cd",
        "Co",
        "Cs",
        "Cu",
        "Fe",
        "Hg",
        "K",
        "Mg",
        "Mn",
        "Na",
        "Ni",
        "Sr",
        "Zn",
    )
    min_sample_fraction: float = 0.005
    min_samples: int = 10
    max_ligand_distance_angstrom: float = 6.0
    min_proximal_samples: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_sample_fraction <= 1.0:
            raise ValueError("metal min_sample_fraction must be in [0, 1]")
        if self.min_samples < 0 or self.min_proximal_samples < 0:
            raise ValueError("metal sample thresholds must be non-negative")
        if self.max_ligand_distance_angstrom <= 0.0:
            raise ValueError("metal max_ligand_distance_angstrom must be positive")


@dataclass(frozen=True)
class RepresentationDesignConfig:
    support: SupportThresholds = field(default_factory=SupportThresholds)
    filtration: FiltrationConfig = field(default_factory=FiltrationConfig)
    eic_tau_start: float = 0.2
    eic_tau_stop: float = 5.0
    eic_tau_step: float = 0.1
    saturation_audit_enabled: bool = False
    distance_chunk_size: int = 2048
    adapter_supported_elements: Mapping[str, tuple[str, ...]] | None = None
    tolerated_unsupported_elements: Mapping[str, tuple[str, ...]] | None = None
    hydrogen_policy: HydrogenPolicyConfig = field(default_factory=HydrogenPolicyConfig)
    metal_awareness: MetalAwarenessConfig = field(default_factory=MetalAwarenessConfig)

    def __post_init__(self) -> None:
        if self.distance_chunk_size < 1:
            raise ValueError("distance_chunk_size must be positive")


@dataclass(frozen=True)
class RepresentationDesignResult:
    representation_spec: RepresentationSpec
    pair_build: AdaptivePairBuild
    geometry_profile: GeometryProfile


def design_adaptive_representation(
    *,
    system_type: SystemType,
    role_paths_by_sample: Mapping[str, Mapping[str, str | Path]],
    config: RepresentationDesignConfig = RepresentationDesignConfig(),
) -> RepresentationDesignResult:
    _assert_roles(system_type, role_paths_by_sample)
    elements_by_role = _elements_by_role(system_type, role_paths_by_sample)
    elements_by_role, hydrogen_audit = _apply_hydrogen_policy(
        elements_by_role,
        config.hydrogen_policy,
        config.adapter_supported_elements,
    )
    metal_audit = _audit_metals(
        system_type=system_type,
        role_paths_by_sample=role_paths_by_sample,
        config=config.metal_awareness,
        adapter_supported_elements=config.adapter_supported_elements,
    )
    elements_by_role = _apply_metal_policy(elements_by_role, metal_audit)
    elements_by_role, adapter_compatibility = _apply_adapter_element_support(
        elements_by_role,
        config.adapter_supported_elements,
        config.tolerated_unsupported_elements,
    )
    pair_build = build_dataset_adaptive_schema(
        system_type=system_type,
        elements_by_role=elements_by_role,
        thresholds=config.support,
    )
    if not pair_build.schema.pair_order:
        raise ValueError("No dataset-adaptive element pairs survived support filtering")
    if system_type == "protein_ligand":
        geometry = profile_protein_ligand_geometry_from_paths(
            role_paths_by_sample=role_paths_by_sample,
            retained_pairs=pair_build.schema.pair_order,
            config=config.filtration,
            distance_chunk_size=config.distance_chunk_size,
        )
    else:
        geometry = profile_small_molecule_geometry_from_paths(
            role_paths_by_sample=role_paths_by_sample,
            retained_pairs=pair_build.schema.pair_order,
            config=config.filtration,
            distance_chunk_size=config.distance_chunk_size,
        )
    common_profile = make_distance_filtration_profile(
        rmax=geometry.max_filtration_angstrom,
        config=config.filtration,
        source="modeling_pool_robust_geometry",
    )
    filtration_profiles = {name: common_profile for name in DISTANCE_INVARIANTS}
    filtration_profiles["EIC"] = make_tau_profile(
        start=config.eic_tau_start,
        stop=config.eic_tau_stop,
        step=config.eic_tau_step,
        source="configured_tau_not_distance_filtration",
    )
    saturation = _saturation_record(config.saturation_audit_enabled)
    anomalies = (
        tuple(pair_build.warnings)
        + tuple(geometry.anomalies)
        + tuple(adapter_compatibility["anomalies"])
        + tuple(saturation["anomalies"])
    )
    parameters = {
        "modeling_dataset_fingerprint": geometry.modeling_dataset_fingerprint,
        "support_thresholds": asdict(config.support),
        "element_support": [asdict(record) for record in pair_build.element_support],
        "pair_support": [asdict(record) for record in pair_build.pair_support],
        "max_element_pair_channels": config.support.max_element_pair_channels,
        "pair_cap_triggered": pair_build.cap_applied,
        "distance_summary": geometry.to_dict(),
        "max_filtration_points": config.filtration.max_points,
        "invariant_library": list(INVARIANT_LIBRARY),
        "invariant_adapter_settings": {
            name: {"filtration_profile": name} for name in INVARIANT_LIBRARY
        },
        "adapter_compatibility": adapter_compatibility,
        "hydrogen_policy": hydrogen_audit,
        "metal_awareness": metal_audit,
        "saturation_audit": saturation,
        "anomalies": list(anomalies),
    }
    spec = RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=pair_build.schema,
        filtration_profiles=filtration_profiles,
        parameters=parameters,
    ).freeze()
    return RepresentationDesignResult(
        representation_spec=spec,
        pair_build=pair_build,
        geometry_profile=geometry,
    )


def _elements_by_role(
    system_type: SystemType,
    role_paths_by_sample: Mapping[str, Mapping[str, str | Path]],
):
    if system_type == "protein_ligand":
        return {
            "protein": {
                sample_id: tuple(iter_elements_from_file(roles["protein"]))
                for sample_id, roles in role_paths_by_sample.items()
            },
            "ligand": {
                sample_id: tuple(iter_elements_from_file(roles["ligand"]))
                for sample_id, roles in role_paths_by_sample.items()
            },
        }
    return {
        "molecule": {
            sample_id: tuple(iter_elements_from_file(roles["molecule"]))
            for sample_id, roles in role_paths_by_sample.items()
        }
    }


def _assert_roles(system_type: SystemType, clouds: Mapping[str, Mapping[str, object]]) -> None:
    expected = {"protein", "ligand"} if system_type == "protein_ligand" else {"molecule"}
    for sample_id, roles in clouds.items():
        missing = expected - set(roles)
        if missing:
            raise ValueError(f"Sample {sample_id!r} is missing required roles: {sorted(missing)}")


def _apply_adapter_element_support(
    elements_by_role: Mapping[str, Mapping[str, tuple[str, ...]]],
    supported: Mapping[str, tuple[str, ...]] | None,
    tolerated: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, object]]:
    if supported is None:
        return (
            {role: dict(samples) for role, samples in elements_by_role.items()},
            {
                "enabled": False,
                "supported_elements": {},
                "tolerated_unsupported_elements": {},
                "unsupported_observed": {},
                "anomalies": [],
            },
        )
    filtered: dict[str, dict[str, tuple[str, ...]]] = {}
    unsupported_observed: dict[str, list[dict[str, object]]] = {}
    anomalies: list[str] = []
    for role, samples in elements_by_role.items():
        if role not in supported:
            raise ValueError(f"adapter_supported_elements is missing role {role!r}")
        allowed = frozenset(supported[role])
        tolerated_for_role = frozenset((tolerated or {}).get(role, ()))
        filtered[role] = {
            sample_id: tuple(element for element in elements if element in allowed)
            for sample_id, elements in samples.items()
        }
        observed = sorted({element for elements in samples.values() for element in elements} - allowed)
        unsupported_observed[role] = [
            {
                "element": element,
                "sample_presence": sum(element in set(elements) for elements in samples.values()),
                "tolerated": element in tolerated_for_role,
            }
            for element in observed
        ]
        if any(element not in tolerated_for_role for element in observed):
            anomalies.append(f"ADAPTER_UNSUPPORTED_ELEMENTS_OBSERVED_{role.upper()}")
    return (
        filtered,
        {
            "enabled": True,
            "supported_elements": {role: list(elements) for role, elements in supported.items()},
            "tolerated_unsupported_elements": {
                role: list(elements) for role, elements in (tolerated or {}).items()
            },
            "unsupported_observed": unsupported_observed,
            "anomalies": anomalies,
        },
    )


def _apply_hydrogen_policy(
    elements_by_role: Mapping[str, Mapping[str, tuple[str, ...]]],
    policy: HydrogenPolicyConfig,
    adapter_supported_elements: Mapping[str, tuple[str, ...]] | None,
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, object]]:
    sample_count = len(next(iter(elements_by_role.values()), {}))
    role_rows: dict[str, dict[str, object]] = {}
    include_requested = policy.mode == "include" or (
        policy.mode == "auto" and policy.task_relevance == "likely_relevant"
    )
    for role, samples in elements_by_role.items():
        presence = sum("H" in set(elements) for elements in samples.values())
        fraction = presence / sample_count if sample_count else 0.0
        adapter_supported = (
            adapter_supported_elements is None
            or "H" in adapter_supported_elements.get(role, ())
        )
        role_rows[role] = {
            "sample_presence": presence,
            "sample_fraction": fraction,
            "adapter_supported": adapter_supported,
        }
        if include_requested and fraction < policy.min_explicit_sample_fraction:
            raise ValueError(
                "Hydrogen-aware representation requires consistent explicit H coordinates: "
                f"role={role} observed_fraction={fraction:.6g} required_fraction="
                f"{policy.min_explicit_sample_fraction:.6g}. Standardize protonation first."
            )
        if include_requested and not adapter_supported:
            raise ValueError(
                "Hydrogen-aware representation was requested but the configured adapter "
                f"does not support explicit H for role={role}."
            )
    include = include_requested
    filtered = {
        role: {
            sample_id: (
                tuple(elements)
                if include
                else tuple(element for element in elements if element != "H")
            )
            for sample_id, elements in samples.items()
        }
        for role, samples in elements_by_role.items()
    }
    return filtered, {
        "requested_mode": policy.mode,
        "task_relevance": policy.task_relevance,
        "resolved_mode": "include" if include else "exclude",
        "min_explicit_sample_fraction": policy.min_explicit_sample_fraction,
        "roles": role_rows,
        "decision_source": (
            "explicit_request"
            if policy.mode != "auto"
            else "llm_task_relevance_then_deterministic_validation"
        ),
    }


def _audit_metals(
    *,
    system_type: SystemType,
    role_paths_by_sample: Mapping[str, Mapping[str, str | Path]],
    config: MetalAwarenessConfig,
    adapter_supported_elements: Mapping[str, tuple[str, ...]] | None,
) -> dict[str, object]:
    if not config.enabled or system_type != "protein_ligand":
        return {"enabled": config.enabled, "status": "NOT_APPLICABLE", "elements": []}
    sample_count = len(role_paths_by_sample)
    stats = {
        element: {"sample_presence": 0, "proximal_sample_presence": 0}
        for element in config.protein_elements
    }
    for sample_id in sorted(role_paths_by_sample):
        roles = role_paths_by_sample[sample_id]
        protein = load_atom_cloud(roles["protein"])
        ligand = load_atom_cloud(roles["ligand"])
        for element in config.protein_elements:
            metal_coordinates = protein.coordinates_for(element)
            if len(metal_coordinates) == 0:
                continue
            stats[element]["sample_presence"] += 1
            minimum = _minimum_cross_distance(metal_coordinates, ligand.coordinates)
            if minimum <= config.max_ligand_distance_angstrom:
                stats[element]["proximal_sample_presence"] += 1
    rows = []
    for element in config.protein_elements:
        presence = stats[element]["sample_presence"]
        proximal = stats[element]["proximal_sample_presence"]
        fraction = presence / sample_count if sample_count else 0.0
        eligible = (
            presence >= config.min_samples
            and fraction >= config.min_sample_fraction
            and proximal >= config.min_proximal_samples
        )
        adapter_supported = (
            adapter_supported_elements is None
            or element in adapter_supported_elements.get("protein", ())
        )
        if eligible and adapter_supported:
            status = "ELIGIBLE"
        elif eligible:
            status = "ELIGIBLE_ADAPTER_UNSUPPORTED"
        elif presence:
            status = "BELOW_SUPPORT_THRESHOLD"
        else:
            status = "NOT_OBSERVED"
        rows.append(
            {
                "element": element,
                "sample_presence": presence,
                "sample_fraction": fraction,
                "proximal_sample_presence": proximal,
                "adapter_supported": adapter_supported,
                "eligible": eligible,
                "status": status,
            }
        )
    return {
        "enabled": True,
        "status": "COMPLETE",
        "sample_count": sample_count,
        "max_ligand_distance_angstrom": config.max_ligand_distance_angstrom,
        "min_sample_fraction": config.min_sample_fraction,
        "min_samples": config.min_samples,
        "min_proximal_samples": config.min_proximal_samples,
        "elements": rows,
    }


def _apply_metal_policy(
    elements_by_role: Mapping[str, Mapping[str, tuple[str, ...]]],
    audit: Mapping[str, object],
) -> dict[str, dict[str, tuple[str, ...]]]:
    rows = audit.get("elements", ())
    rows = rows if isinstance(rows, (list, tuple)) else ()
    metal_elements = {
        str(row.get("element"))
        for row in rows
        if isinstance(row, Mapping)
    }
    allowed_metals = {
        str(row.get("element"))
        for row in rows
        if isinstance(row, Mapping)
        and bool(row.get("eligible"))
        and bool(row.get("adapter_supported"))
    }
    if not metal_elements:
        return {role: dict(samples) for role, samples in elements_by_role.items()}
    return {
        role: {
            sample_id: tuple(
                element
                for element in elements
                if element not in metal_elements or element in allowed_metals
            )
            for sample_id, elements in samples.items()
        }
        for role, samples in elements_by_role.items()
    }


def _minimum_cross_distance(left, right) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("inf")
    minimum = float("inf")
    for start in range(0, len(left), 256):
        delta = left[start : start + 256, None, :] - right[None, :, :]
        minimum = min(minimum, float((delta * delta).sum(axis=2).min()) ** 0.5)
    return minimum


def _saturation_record(enabled: bool) -> dict[str, object]:
    if not enabled:
        return {"enabled": False, "status": "NOT_REQUESTED", "anomalies": []}
    anomalies = [f"SATURATION_AUDIT_UNSUPPORTED_FOR_{name}" for name in INVARIANT_LIBRARY]
    return {"enabled": True, "status": "UNSUPPORTED", "anomalies": anomalies}
