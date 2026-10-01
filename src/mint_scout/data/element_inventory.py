from __future__ import annotations

import gzip
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from mint_scout.data.element_pairs import ElementPairSchema


KNOWN_ELEMENTS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
}


@dataclass(frozen=True)
class RoleElementSummary:
    role: str
    file_count: int
    atom_counts: dict[str, int]
    file_presence: dict[str, int]
    out_of_schema: tuple[str, ...]


@dataclass(frozen=True)
class ElementInventoryReport:
    schema_id: str
    roles: tuple[RoleElementSummary, ...]

    @property
    def schema_review_required(self) -> bool:
        return any(role.out_of_schema for role in self.roles)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_review_required": self.schema_review_required,
            "roles": [
                {
                    "role": role.role,
                    "file_count": role.file_count,
                    "atom_counts": role.atom_counts,
                    "file_presence": role.file_presence,
                    "out_of_schema": list(role.out_of_schema),
                }
                for role in self.roles
            ],
        }


def normalize_element_token(token: str) -> str | None:
    letters = "".join(ch for ch in token.strip() if ch.isalpha())
    if not letters:
        return None
    candidates = []
    if len(letters) >= 2:
        candidates.append(letters[:2].capitalize())
    candidates.append(letters[:1].capitalize())
    for candidate in candidates:
        if candidate in KNOWN_ELEMENTS:
            return candidate
    return None


def iter_elements_from_file(path: Path) -> Iterator[str]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".pdb", ".ent", ".pdbqt", ".pdb.gz", ".ent.gz", ".pdbqt.gz")):
        yield from _iter_pdb_elements(path)
    elif suffixes.endswith((".mol2", ".mol2.gz")):
        yield from _iter_mol2_elements(path)
    elif suffixes.endswith((".sdf", ".mol", ".sdf.gz", ".mol.gz")):
        yield from _iter_molfile_elements(path)
    elif suffixes.endswith((".xyz", ".xyz.gz")):
        yield from _iter_xyz_elements(path)
    else:
        raise ValueError(f"Unsupported molecular file type: {path}")


def summarize_role_elements(
    role: str,
    paths: Iterable[Path],
    *,
    allowed_elements: Iterable[str],
) -> RoleElementSummary:
    atom_counts: Counter[str] = Counter()
    file_presence: Counter[str] = Counter()
    file_count = 0
    for path in sorted(paths):
        file_count += 1
        elements = tuple(iter_elements_from_file(path))
        atom_counts.update(elements)
        file_presence.update(set(elements))
    out_of_schema = tuple(sorted(set(atom_counts) - set(allowed_elements)))
    return RoleElementSummary(
        role=role,
        file_count=file_count,
        atom_counts=dict(sorted(atom_counts.items())),
        file_presence=dict(sorted(file_presence.items())),
        out_of_schema=out_of_schema,
    )


def build_element_inventory_report(
    *,
    schema: ElementPairSchema,
    protein_paths: Iterable[Path] = (),
    ligand_paths: Iterable[Path] = (),
    tolerated_elements_by_role: dict[str, Iterable[str]] | None = None,
) -> ElementInventoryReport:
    tolerated = tolerated_elements_by_role or {}
    roles = []
    if schema.role_aware:
        roles.append(
            summarize_role_elements(
                "protein",
                protein_paths,
                allowed_elements=tuple(schema.left_elements) + tuple(tolerated.get("protein", ())),
            )
        )
        roles.append(
            summarize_role_elements(
                "ligand",
                ligand_paths,
                allowed_elements=tuple(schema.right_elements) + tuple(tolerated.get("ligand", ())),
            )
        )
    else:
        allowed = schema.global_element_order or tuple(set(schema.left_elements) | set(schema.right_elements))
        roles.append(
            summarize_role_elements(
                "molecule",
                tuple(protein_paths) + tuple(ligand_paths),
                allowed_elements=tuple(allowed) + tuple(tolerated.get("molecule", ())),
            )
        )
    return ElementInventoryReport(schema_id=schema.schema_id, roles=tuple(roles))


def render_markdown_report(report: ElementInventoryReport) -> str:
    lines = [
        "# Element Inventory Audit",
        "",
        f"- Frozen schema: `{report.schema_id}`",
        f"- Schema review required: `{str(report.schema_review_required).lower()}`",
        "- This audit reports dataset composition only; it does not mutate the feature schema.",
        "",
    ]
    for role in report.roles:
        lines.extend(
            [
                f"## {role.role.title()}",
                "",
                f"- Files scanned: `{role.file_count}`",
                f"- Out-of-schema elements: `{', '.join(role.out_of_schema) if role.out_of_schema else 'none'}`",
                "",
                "| Element | Atom count | File presence |",
                "| --- | ---: | ---: |",
            ]
        )
        for element, count in role.atom_counts.items():
            lines.append(f"| {element} | {count} | {role.file_presence.get(element, 0)} |")
        if not role.atom_counts:
            lines.append("| none | 0 | 0 |")
        lines.append("")
    return "\n".join(lines)


def _open_text(path: Path) -> Iterator[str]:
    opener = gzip.open if "".join(path.suffixes).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def _iter_pdb_elements(path: Path) -> Iterator[str]:
    for line in _open_text(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = normalize_element_token(line[76:78]) if len(line) >= 78 else None
        if element is None:
            element = _element_from_pdb_atom_name(line[12:16] if len(line) >= 16 else "")
        if element:
            yield element


def _element_from_pdb_atom_name(atom_name: str) -> str | None:
    token = atom_name.strip().lstrip("0123456789")
    if not token:
        return None
    if len(token) >= 2 and token[0].isalpha() and token[1].islower():
        return normalize_element_token(token[:2])
    return normalize_element_token(token[:1])


def _iter_mol2_elements(path: Path) -> Iterator[str]:
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
        if element:
            yield element


def _iter_molfile_elements(path: Path) -> Iterator[str]:
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
        if element:
            yield element


def _iter_xyz_elements(path: Path) -> Iterator[str]:
    for line in list(_open_text(path))[2:]:
        parts = line.split()
        if not parts:
            continue
        element = normalize_element_token(parts[0])
        if element:
            yield element
