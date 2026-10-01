from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal

from mint_scout.invariants.manifest import stable_hash


SystemType = Literal["protein_ligand", "small_molecule"]


CASF_PROTEIN_ELEMENTS = ("C", "N", "O", "S")
CASF_LIGAND_ELEMENTS_40 = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H")
CASF_LIGAND_ELEMENTS_20 = ("C", "N", "O", "S", "P")

TOX_GLOBAL_ORDER = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
TOX_FIRST_ELEMENTS = ("H", "C", "N", "O")
TOX_SECOND_ELEMENTS = TOX_GLOBAL_ORDER


@dataclass(frozen=True)
class ElementPairSchema:
    schema_id: str
    system_type: SystemType
    left_role: str | None
    right_role: str | None
    left_elements: tuple[str, ...]
    right_elements: tuple[str, ...]
    global_element_order: tuple[str, ...] | None
    pair_generation_rule: str
    role_aware: bool
    exclude_self_pairs: bool
    deduplicate_unordered_pairs: bool
    pair_order: tuple[tuple[str, str], ...]
    expected_pair_count: int

    def __post_init__(self) -> None:
        if len(self.pair_order) != self.expected_pair_count:
            raise ValueError(
                f"{self.schema_id}: expected {self.expected_pair_count} pairs, "
                f"got {len(self.pair_order)}"
            )
        if len(set(self.pair_order)) != len(self.pair_order):
            raise ValueError(f"{self.schema_id}: duplicate element-pair entries")

    @property
    def pair_names(self) -> tuple[str, ...]:
        if self.role_aware:
            left = self.left_role or "left"
            right = self.right_role or "right"
            return tuple(f"{left}:{a}|{right}:{b}" for a, b in self.pair_order)
        return tuple(f"{a}-{b}" for a, b in self.pair_order)


@dataclass(frozen=True)
class SupportThresholds:
    min_element_support_fraction: float = 0.005
    min_element_support_samples: int = 10
    min_pair_support_fraction: float = 0.005
    min_pair_support_samples: int = 10
    max_element_pair_channels: int = 50
    small_molecule_self_pairs: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("min_element_support_fraction", self.min_element_support_fraction),
            ("min_pair_support_fraction", self.min_pair_support_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_element_support_samples < 0 or self.min_pair_support_samples < 0:
            raise ValueError("support sample thresholds must be non-negative")
        if self.max_element_pair_channels < 1:
            raise ValueError("max_element_pair_channels must be positive")


@dataclass(frozen=True)
class ElementSupportRecord:
    role: str
    element: str
    atom_count: int
    sample_presence: int
    support_fraction: float
    retained: bool


@dataclass(frozen=True)
class PairSupportRecord:
    pair: tuple[str, str]
    sample_presence: int
    support_fraction: float
    retained: bool
    selected: bool = False


@dataclass(frozen=True)
class AdaptivePairBuild:
    schema: ElementPairSchema
    element_support: tuple[ElementSupportRecord, ...]
    pair_support: tuple[PairSupportRecord, ...]
    cap_applied: bool
    warnings: tuple[str, ...] = ()


def make_casf_protein_ligand_schema(
    *,
    schema_id: str = "casf_protein_ligand_40_v1",
    ligand_elements: Iterable[str] = CASF_LIGAND_ELEMENTS_40,
) -> ElementPairSchema:
    ligand_tuple = tuple(ligand_elements)
    pair_order = tuple((p, l) for p in CASF_PROTEIN_ELEMENTS for l in ligand_tuple)
    return ElementPairSchema(
        schema_id=schema_id,
        system_type="protein_ligand",
        left_role="protein",
        right_role="ligand",
        left_elements=CASF_PROTEIN_ELEMENTS,
        right_elements=ligand_tuple,
        global_element_order=None,
        pair_generation_rule="role_aware_cartesian_product",
        role_aware=True,
        exclude_self_pairs=False,
        deduplicate_unordered_pairs=False,
        pair_order=pair_order,
        expected_pair_count=len(pair_order),
    )


def make_toxicity_schema(schema_id: str = "toxicity_30_pair_v1") -> ElementPairSchema:
    pos = {element: idx for idx, element in enumerate(TOX_GLOBAL_ORDER)}
    pair_order = tuple(
        (a, b)
        for a in TOX_FIRST_ELEMENTS
        for b in TOX_SECOND_ELEMENTS
        if pos[a] < pos[b]
    )
    return ElementPairSchema(
        schema_id=schema_id,
        system_type="small_molecule",
        left_role=None,
        right_role=None,
        left_elements=TOX_FIRST_ELEMENTS,
        right_elements=TOX_SECOND_ELEMENTS,
        global_element_order=TOX_GLOBAL_ORDER,
        pair_generation_rule="first_precedes_second",
        role_aware=False,
        exclude_self_pairs=True,
        deduplicate_unordered_pairs=True,
        pair_order=pair_order,
        expected_pair_count=30,
    )


def build_dataset_adaptive_schema(
    *,
    system_type: SystemType,
    elements_by_role: dict[str, dict[str, Iterable[str]]],
    thresholds: SupportThresholds = SupportThresholds(),
    left_role: str = "protein",
    right_role: str = "ligand",
    schema_prefix: str = "dataset_adaptive",
) -> AdaptivePairBuild:
    if system_type == "protein_ligand":
        return _build_adaptive_protein_ligand_schema(
            elements_by_role=elements_by_role,
            thresholds=thresholds,
            left_role=left_role,
            right_role=right_role,
            schema_prefix=schema_prefix,
        )
    if system_type == "small_molecule":
        return _build_adaptive_small_molecule_schema(
            elements_by_sample=elements_by_role.get("molecule", {}),
            thresholds=thresholds,
            schema_prefix=schema_prefix,
        )
    raise ValueError(f"Unsupported system_type={system_type!r}")


def _build_adaptive_protein_ligand_schema(
    *,
    elements_by_role: dict[str, dict[str, Iterable[str]]],
    thresholds: SupportThresholds,
    left_role: str,
    right_role: str,
    schema_prefix: str,
) -> AdaptivePairBuild:
    left = _normalize_sample_elements(elements_by_role.get(left_role, {}))
    right = _normalize_sample_elements(elements_by_role.get(right_role, {}))
    _assert_matching_role_samples(left_role, left, right_role, right)
    modeling_ids = tuple(sorted(left))
    sample_count = len(modeling_ids)
    left_support, left_elements = _supported_elements(left_role, left, sample_count, thresholds)
    right_support, right_elements = _supported_elements(right_role, right, sample_count, thresholds)
    candidate_pairs = tuple((a, b) for a in left_elements for b in right_elements)
    pair_support = _pair_support_records(
        candidate_pairs,
        sample_count=sample_count,
        support_fn=lambda a, b: sum(a in set(left.get(sid, ())) and b in set(right.get(sid, ())) for sid in modeling_ids),
        thresholds=thresholds,
    )
    retained_pairs, cap_applied = _retained_pairs(pair_support, thresholds)
    pair_support = _mark_selected(pair_support, retained_pairs)
    schema = ElementPairSchema(
        schema_id=_adaptive_schema_id(schema_prefix, "protein_ligand", retained_pairs),
        system_type="protein_ligand",
        left_role=left_role,
        right_role=right_role,
        left_elements=left_elements,
        right_elements=right_elements,
        global_element_order=None,
        pair_generation_rule="dataset_adaptive_role_aware_support",
        role_aware=True,
        exclude_self_pairs=False,
        deduplicate_unordered_pairs=False,
        pair_order=retained_pairs,
        expected_pair_count=len(retained_pairs),
    )
    return AdaptivePairBuild(
        schema=schema,
        element_support=left_support + right_support,
        pair_support=pair_support,
        cap_applied=cap_applied,
        warnings=_build_warnings(retained_pairs, cap_applied, len(pair_support), len(retained_pairs)),
    )


def _build_adaptive_small_molecule_schema(
    *,
    elements_by_sample: dict[str, Iterable[str]],
    thresholds: SupportThresholds,
    schema_prefix: str,
) -> AdaptivePairBuild:
    elements_by_sample = _normalize_sample_elements(elements_by_sample)
    sample_count = len(elements_by_sample)
    element_support, supported = _supported_elements("molecule", elements_by_sample, sample_count, thresholds)
    pos = {element: idx for idx, element in enumerate(supported)}
    candidate_pairs = tuple(
        (a, b)
        for idx, a in enumerate(supported)
        for b in supported[idx if thresholds.small_molecule_self_pairs else idx + 1 :]
    )
    pair_support = _pair_support_records(
        candidate_pairs,
        sample_count=sample_count,
        support_fn=lambda a, b: sum(
            (a in set(elements) and b in set(elements)) if a != b else (a in set(elements))
            for elements in elements_by_sample.values()
        ),
        thresholds=thresholds,
    )
    retained_pairs, cap_applied = _retained_pairs(pair_support, thresholds)
    pair_support = _mark_selected(pair_support, retained_pairs)
    schema = ElementPairSchema(
        schema_id=_adaptive_schema_id(schema_prefix, "small_molecule", retained_pairs),
        system_type="small_molecule",
        left_role=None,
        right_role=None,
        left_elements=supported,
        right_elements=supported,
        global_element_order=supported,
        pair_generation_rule="dataset_adaptive_unordered_support",
        role_aware=False,
        exclude_self_pairs=not thresholds.small_molecule_self_pairs,
        deduplicate_unordered_pairs=True,
        pair_order=retained_pairs,
        expected_pair_count=len(retained_pairs),
    )
    if any(pos[a] > pos[b] for a, b in retained_pairs):
        raise AssertionError("small-molecule adaptive pairs must follow deterministic unordered order")
    return AdaptivePairBuild(
        schema=schema,
        element_support=element_support,
        pair_support=pair_support,
        cap_applied=cap_applied,
        warnings=_build_warnings(retained_pairs, cap_applied, len(pair_support), len(retained_pairs)),
    )


def summarize_element_frequency(elements_by_sample: dict[str, Iterable[str]]) -> dict[str, object]:
    sample_count = len(elements_by_sample)
    atom_counts: Counter[str] = Counter()
    sample_presence: Counter[str] = Counter()
    for elements in elements_by_sample.values():
        element_tuple = tuple(elements)
        atom_counts.update(element_tuple)
        sample_presence.update(set(element_tuple))
    return {
        "sample_count": sample_count,
        "atom_counts": dict(sorted(atom_counts.items())),
        "sample_presence": dict(sorted(sample_presence.items())),
    }


def out_of_schema_elements(schema: ElementPairSchema, observed_elements: Iterable[str]) -> tuple[str, ...]:
    allowed = set(schema.left_elements) | set(schema.right_elements)
    if schema.global_element_order:
        allowed |= set(schema.global_element_order)
    return tuple(sorted(set(observed_elements) - allowed))


def _normalize_sample_elements(elements_by_sample: dict[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    return {sample_id: tuple(elements) for sample_id, elements in elements_by_sample.items()}


def _supported_elements(
    role: str,
    elements_by_sample: dict[str, tuple[str, ...]],
    sample_count: int,
    thresholds: SupportThresholds,
) -> tuple[tuple[ElementSupportRecord, ...], tuple[str, ...]]:
    atom_counts: Counter[str] = Counter()
    sample_presence: Counter[str] = Counter()
    for elements in elements_by_sample.values():
        atom_counts.update(elements)
        sample_presence.update(set(elements))
    min_presence = max(
        thresholds.min_element_support_samples,
        _ceil_fraction(sample_count, thresholds.min_element_support_fraction),
    )
    retained = {
        element
        for element, presence in sample_presence.items()
        if presence >= min_presence
    }
    ordered_elements = tuple(
        element
        for element, _presence, _atoms in sorted(
            (
                (element, sample_presence[element], atom_counts[element])
                for element in sample_presence
                if element in retained
            ),
            key=lambda item: (-item[1], -item[2], item[0]),
        )
    )
    records = tuple(
        ElementSupportRecord(
            role=role,
            element=element,
            atom_count=atom_counts[element],
            sample_presence=sample_presence[element],
            support_fraction=sample_presence[element] / sample_count if sample_count else 0.0,
            retained=element in retained,
        )
        for element in sorted(sample_presence)
    )
    return records, ordered_elements


def _pair_support_records(
    candidate_pairs: tuple[tuple[str, str], ...],
    *,
    sample_count: int,
    support_fn: Callable[[str, str], int],
    thresholds: SupportThresholds,
) -> tuple[PairSupportRecord, ...]:
    min_presence = max(
        thresholds.min_pair_support_samples,
        _ceil_fraction(sample_count, thresholds.min_pair_support_fraction),
    )
    return tuple(
        PairSupportRecord(
            pair=pair,
            sample_presence=(presence := int(support_fn(*pair))),
            support_fraction=presence / sample_count if sample_count else 0.0,
            retained=presence >= min_presence,
        )
        for pair in candidate_pairs
    )


def _retained_pairs(
    pair_support: tuple[PairSupportRecord, ...],
    thresholds: SupportThresholds,
) -> tuple[tuple[tuple[str, str], ...], bool]:
    supported = tuple(record for record in pair_support if record.retained)
    ranked = tuple(
        record.pair
        for record in sorted(supported, key=lambda record: (-record.sample_presence, record.pair[0], record.pair[1]))
    )
    cap_applied = len(ranked) > thresholds.max_element_pair_channels
    return ranked[: thresholds.max_element_pair_channels], cap_applied


def _mark_selected(
    pair_support: tuple[PairSupportRecord, ...],
    retained_pairs: tuple[tuple[str, str], ...],
) -> tuple[PairSupportRecord, ...]:
    selected = set(retained_pairs)
    return tuple(replace(record, selected=record.pair in selected) for record in pair_support)


def _assert_matching_role_samples(
    left_role: str,
    left: dict[str, tuple[str, ...]],
    right_role: str,
    right: dict[str, tuple[str, ...]],
) -> None:
    left_only = sorted(set(left) - set(right))
    right_only = sorted(set(right) - set(left))
    if left_only or right_only:
        raise ValueError(
            "Protein-ligand adaptive schemas require aligned modeling samples; "
            f"only in {left_role}: {left_only[:10]}, only in {right_role}: {right_only[:10]}"
        )


def _adaptive_schema_id(schema_prefix: str, system_type: str, pairs: tuple[tuple[str, str], ...]) -> str:
    digest = stable_hash({"system_type": system_type, "pairs": pairs})
    return f"{schema_prefix}_{system_type}_{len(pairs)}_{digest}"


def _ceil_fraction(count: int, fraction: float) -> int:
    return int(math.ceil(count * fraction))


def _build_warnings(
    pairs: tuple[tuple[str, str], ...],
    cap_applied: bool,
    candidate_count: int,
    retained_count: int,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not pairs:
        warnings.append("No element pairs survived support filtering.")
    if cap_applied:
        warnings.append(
            f"PAIR_CAP_APPLIED: selected {retained_count} of {candidate_count} supported candidate pairs."
        )
    return tuple(warnings)
