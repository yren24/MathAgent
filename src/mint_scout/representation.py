from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mint_scout.data.element_pairs import (
    ElementPairSchema,
    SystemType,
    make_casf_protein_ligand_schema,
    make_toxicity_schema,
)
from mint_scout.filtration import FiltrationProfile
from mint_scout.invariants.manifest import stable_hash


class RepresentationMode(str, Enum):
    DATASET_ADAPTIVE = "dataset_adaptive"
    LEGACY_CASF = "legacy_casf"
    LEGACY_TOXICITY = "legacy_toxicity"


@dataclass(frozen=True)
class RepresentationSpec:
    representation_id: str
    mode: RepresentationMode
    system_type: SystemType
    schema_id: str
    pair_order: tuple[tuple[str, str], ...]
    pair_names: tuple[str, ...]
    filtration_profiles: Mapping[str, FiltrationProfile]
    parameters: Mapping[str, Any]
    frozen: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "filtration_profiles",
            MappingProxyType(dict(sorted(self.filtration_profiles.items()))),
        )
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))
        if len(self.pair_order) != len(self.pair_names):
            raise ValueError("pair_order and pair_names must have the same length")

    @classmethod
    def from_schema(
        cls,
        *,
        mode: RepresentationMode,
        schema: ElementPairSchema,
        filtration_profiles: dict[str, FiltrationProfile],
        parameters: dict[str, Any] | None = None,
        frozen: bool = False,
    ) -> "RepresentationSpec":
        payload = {
            "mode": mode.value,
            "schema_id": schema.schema_id,
            "pair_order": schema.pair_order,
            "filtration_profiles": {name: profile.to_dict() for name, profile in sorted(filtration_profiles.items())},
            "parameters": parameters or {},
        }
        representation_id = f"{mode.value}_{stable_hash(payload)}"
        return cls(
            representation_id=representation_id,
            mode=mode,
            system_type=schema.system_type,
            schema_id=schema.schema_id,
            pair_order=schema.pair_order,
            pair_names=schema.pair_names,
            filtration_profiles=filtration_profiles,
            parameters=parameters or {},
            frozen=frozen,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepresentationSpec":
        if "representation_spec" in data:
            nested = data["representation_spec"]
            if not isinstance(nested, Mapping):
                raise ValueError("representation_spec must be a mapping")
            data = nested
        required = {
            "representation_id",
            "mode",
            "system_type",
            "schema_id",
            "pair_order",
            "pair_names",
            "filtration_profiles",
            "parameters",
            "frozen",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"RepresentationSpec is missing fields: {sorted(missing)}")
        raw_profiles = data["filtration_profiles"]
        if not isinstance(raw_profiles, Mapping):
            raise ValueError("filtration_profiles must be a mapping")
        result = cls(
            representation_id=str(data["representation_id"]),
            mode=RepresentationMode(str(data["mode"])),
            system_type=str(data["system_type"]),
            schema_id=str(data["schema_id"]),
            pair_order=tuple(tuple(str(item) for item in pair) for pair in data["pair_order"]),
            pair_names=tuple(str(name) for name in data["pair_names"]),
            filtration_profiles={
                str(name): FiltrationProfile(**profile)
                for name, profile in raw_profiles.items()
            },
            parameters=dict(data["parameters"]),
            frozen=bool(data["frozen"]),
        )
        expected_hash = data.get("spec_hash")
        if expected_hash is not None and str(expected_hash) != result.spec_hash:
            raise ValueError(
                f"RepresentationSpec hash mismatch: expected {expected_hash}, computed {result.spec_hash}"
            )
        return result

    @classmethod
    def read(cls, path: str | Path) -> "RepresentationSpec":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("RepresentationSpec JSON must contain an object")
        return cls.from_dict(data)

    @property
    def spec_hash(self) -> str:
        return stable_hash(self.semantic_payload())

    def freeze(self) -> "RepresentationSpec":
        return replace(self, frozen=True)

    def assert_frozen(self) -> None:
        if not self.frozen:
            raise AssertionError("RepresentationSpec must be frozen before feature generation")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "representation_id": self.representation_id,
            "mode": self.mode.value,
            "system_type": self.system_type,
            "schema_id": self.schema_id,
            "pair_order": [list(pair) for pair in self.pair_order],
            "pair_names": list(self.pair_names),
            "filtration_profiles": {
                name: profile.to_dict() for name, profile in sorted(self.filtration_profiles.items())
            },
            "parameters": _to_plain(self.parameters),
            "frozen": self.frozen,
        }
        if include_hash:
            payload["spec_hash"] = self.spec_hash
        return payload

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "system_type": self.system_type,
            "schema_id": self.schema_id,
            "pair_order": [list(pair) for pair in self.pair_order],
            "pair_names": list(self.pair_names),
            "filtration_profiles": {
                name: profile.to_dict() for name, profile in sorted(self.filtration_profiles.items())
            },
            "parameters": _to_plain(self.parameters),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_to_plain(item) for item in value)
    return value


def make_legacy_casf_representation_spec() -> RepresentationSpec:
    """Return the audited fixed 40-channel PLBind representation."""
    profiles = {
        "PH": FiltrationProfile("distance", 0.0, 14.9, 0.1, 150, "legacy_np_arange_0_15_exclusive"),
        "PL": FiltrationProfile("distance", 0.0, 14.5, 0.5, 30, "legacy_np_arange_0_15_exclusive"),
        "CA": FiltrationProfile("distance", 1.0, 15.0, 0.1, 141, "legacy_inclusive_1_15"),
        "FPRC": FiltrationProfile("distance", 0.0, 14.5, 0.5, 30, "legacy_np_arange_0_15_exclusive"),
        "EIC": FiltrationProfile("tau", 0.2, 5.0, 0.1, 49, "legacy_tau_0_2_5_inclusive"),
    }
    return RepresentationSpec.from_schema(
        mode=RepresentationMode.LEGACY_CASF,
        schema=make_casf_protein_ligand_schema(),
        filtration_profiles=profiles,
        parameters={
            "modeling_dataset_fingerprint": "not_dataset_adaptive",
            "invariant_library": ["PH", "PL", "CA", "FPRC", "EIC"],
            "legacy_adapter": "plbind_legacy_adapter_v1",
            "anomalies": ["LEGACY_FIXED_REPRESENTATION"],
        },
    ).freeze()


def make_legacy_toxicity_representation_spec() -> RepresentationSpec:
    """Return the audited fixed LD50 representation from the legacy implementation."""
    distance = FiltrationProfile(
        "distance", 0.0, 9.8, 0.2, 50, "legacy_np_arange_0_10_exclusive"
    )
    profiles = {
        "PH": distance,
        "PL": distance,
        "CA": distance,
        "FPRC": distance,
        "EIC": FiltrationProfile(
            "tau", 0.2, 5.0, 0.2, 25, "legacy_tau_0_2_5_inclusive"
        ),
    }
    return RepresentationSpec.from_schema(
        mode=RepresentationMode.LEGACY_TOXICITY,
        schema=make_toxicity_schema(),
        filtration_profiles=profiles,
        parameters={
            "modeling_dataset_fingerprint": "not_dataset_adaptive",
            "invariant_library": ["PH", "PL", "CA", "FPRC", "EIC"],
            "legacy_adapter": "toxicity_legacy_adapter_v1",
            "bond_delta": 0.45,
            "homology_dims": [0],
            "facet_dims": [0],
            "curvature_bidirectional": False,
            "anomalies": ["LEGACY_FIXED_REPRESENTATION"],
        },
    ).freeze()
