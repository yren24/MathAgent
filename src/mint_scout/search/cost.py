from __future__ import annotations

import contextlib
import resource
import time
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class CostRecord:
    wall_seconds: float
    cpu_seconds: float

    @property
    def cpu_core_hours(self) -> float:
        return self.cpu_seconds / 3600.0


def _cpu_time() -> float:
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        self_usage.ru_utime
        + self_usage.ru_stime
        + child_usage.ru_utime
        + child_usage.ru_stime
    )


@contextlib.contextmanager
def measure_cost() -> Iterator[list[CostRecord]]:
    start_wall = time.perf_counter()
    start_cpu = _cpu_time()
    records: list[CostRecord] = []
    try:
        yield records
    finally:
        records.append(
            CostRecord(
                wall_seconds=time.perf_counter() - start_wall,
                cpu_seconds=max(0.0, _cpu_time() - start_cpu),
            )
        )
