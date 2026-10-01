"""Frozen category-specific schema used by the audited legacy MOF tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


LEGACY_CATEGORY_SCHEMA_ID = "mof_legacy_c0_c7_call_v1"
LEGACY_CATEGORY_ORDER = tuple([f"C{index}" for index in range(8)] + ["Call"])

_LEGACY_CATEGORY_MEMBERS: dict[str, frozenset[str]] = {
    "C0": frozenset({"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "Al", "Ga", "In", "Sn", "Pb", "Bi"}),
    "C1": frozenset({"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "W", "Re", "Ir", "Pt", "Au", "Hg", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}),
    "C2": frozenset({"B", "Si", "Ge", "As", "Sb", "Te"}),
    "C3": frozenset({"F", "Cl", "Br", "I"}),
    "C4": frozenset({"H"}),
    "C5": frozenset({"C"}),
    "C6": frozenset({"N", "P"}),
    "C7": frozenset({"O", "S", "Se"}),
}


def normalize_element(symbol: str) -> str:
    value = str(symbol).strip().strip("'\"")
    if not value:
        raise ValueError("Element symbol must not be empty")
    return value[0].upper() + value[1:].lower()


@dataclass(frozen=True)
class CategorySchema:
    schema_id: str
    category_order: tuple[str, ...]
    members: Mapping[str, frozenset[str]]
    all_atoms_category: str = "Call"
    unknown_policy: str = "report_only"

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("category schema_id is required")
        if not self.category_order or self.category_order[-1] != self.all_atoms_category:
            raise ValueError("all_atoms_category must be the final category")
        if len(set(self.category_order)) != len(self.category_order):
            raise ValueError("category names must be unique")
        if set(self.members) != set(self.category_order) - {self.all_atoms_category}:
            raise ValueError("members must define every non-all-atoms category exactly once")
        seen: set[str] = set()
        for category in self.category_order:
            if category == self.all_atoms_category:
                continue
            overlap = seen & set(self.members[category])
            if overlap:
                raise ValueError(f"elements assigned to multiple categories: {sorted(overlap)}")
            seen.update(self.members[category])

    def category_for(self, element: str | None) -> str | None:
        if element is None:
            return None
        normalized = normalize_element(element)
        for category in self.category_order:
            if category != self.all_atoms_category and normalized in self.members[category]:
                return category
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "category_order": list(self.category_order),
            "all_atoms_category": self.all_atoms_category,
            "unknown_policy": self.unknown_policy,
            "categories": {name: sorted(self.members[name]) for name in self.category_order if name != self.all_atoms_category},
        }


def legacy_category_schema() -> CategorySchema:
    return CategorySchema(
        schema_id=LEGACY_CATEGORY_SCHEMA_ID,
        category_order=LEGACY_CATEGORY_ORDER,
        members=_LEGACY_CATEGORY_MEMBERS,
    )


def load_category_schema(path: str | Path) -> CategorySchema:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("MOF category schema must be a mapping")
    categories = raw.get("categories")
    order = raw.get("category_order")
    if not isinstance(categories, Mapping) or not isinstance(order, list):
        raise ValueError("MOF category schema needs categories and category_order")
    all_atoms_category = str(raw.get("all_atoms_category", "Call"))
    members = {
        str(name): frozenset(normalize_element(value) for value in values)
        for name, values in categories.items()
        if str(name) != all_atoms_category
    }
    return CategorySchema(
        schema_id=str(raw.get("schema_id", "")),
        category_order=tuple(str(value) for value in order),
        members=members,
        all_atoms_category=all_atoms_category,
        unknown_policy=str(raw.get("unknown_policy", "report_only")),
    )
