from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence


def validate_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    if not fractions:
        raise ValueError("At least one fidelity fraction is required")
    result = tuple(float(f) for f in fractions)
    if result[-1] != 1.0:
        raise ValueError("The final fidelity fraction must be 1.0")
    if any(f <= 0.0 or f > 1.0 for f in result):
        raise ValueError(f"Fidelity fractions must be in (0, 1], got {result}")
    if any(a >= b for a, b in zip(result, result[1:])):
        raise ValueError(f"Fidelity fractions must be strictly increasing, got {result}")
    return result


def make_nested_fidelity_subsets(
    sample_ids: Sequence[str],
    fractions: Sequence[float],
    *,
    seed: int,
    protected_prefix: Iterable[str] = (),
) -> dict[float, tuple[str, ...]]:
    """Build deterministic nested subsets from a single shuffled sample order."""
    fraction_tuple = validate_fractions(fractions)
    ids = list(sample_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids contains duplicates")

    rng = random.Random(seed)
    rng.shuffle(ids)

    protected = [sid for sid in protected_prefix if sid in ids]
    remaining = [sid for sid in ids if sid not in set(protected)]
    ordered = protected + remaining

    n = len(ordered)
    subsets: dict[float, tuple[str, ...]] = {}
    for frac in fraction_tuple:
        size = n if frac == 1.0 else max(1, int(math.ceil(n * frac)))
        subsets[frac] = tuple(sorted(ordered[:size]))
    return subsets


def assert_nested(subsets: dict[float, Sequence[str]]) -> None:
    previous: set[str] | None = None
    for fraction in sorted(subsets):
        current = set(subsets[fraction])
        if previous is not None and not previous.issubset(current):
            raise AssertionError(f"Fidelity subset {fraction} is not nested")
        previous = current
