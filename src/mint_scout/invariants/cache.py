from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from mint_scout.search.cost import CostRecord, measure_cost


@dataclass(frozen=True)
class CachedFeatureStatus:
    sample_id: str
    path: Path
    status: str
    cost: CostRecord | None = None


class FeatureCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def sample_path(
        self,
        *,
        dataset_id: str,
        schema_id: str,
        invariant_name: str,
        parameter_hash: str,
        sample_id: str,
    ) -> Path:
        return (
            self.root
            / dataset_id
            / schema_id
            / invariant_name
            / parameter_hash
            / f"{sample_id}.npy"
        )

    def representation_sample_path(
        self,
        *,
        dataset_id: str,
        representation_hash: str,
        invariant_name: str,
        parameter_hash: str,
        sample_id: str,
    ) -> Path:
        return (
            self.root
            / dataset_id
            / f"repr-{representation_hash}"
            / invariant_name
            / parameter_hash
            / f"{sample_id}.npy"
        )

    def compute_missing(
        self,
        *,
        dataset_id: str,
        schema_id: str,
        invariant_name: str,
        parameter_hash: str,
        sample_ids: Iterable[str],
        compute_one: Callable[[str, Path], None],
    ) -> list[CachedFeatureStatus]:
        statuses: list[CachedFeatureStatus] = []
        for sample_id in sample_ids:
            out_path = self.sample_path(
                dataset_id=dataset_id,
                schema_id=schema_id,
                invariant_name=invariant_name,
                parameter_hash=parameter_hash,
                sample_id=sample_id,
            )
            if out_path.exists():
                statuses.append(CachedFeatureStatus(sample_id, out_path, "cached"))
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with measure_cost() as cost_records:
                compute_one(sample_id, out_path)
            statuses.append(CachedFeatureStatus(sample_id, out_path, "computed", cost_records[0]))
        return statuses
