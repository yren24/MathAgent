from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal


ScaleKind = Literal["distance", "tau"]


@dataclass(frozen=True)
class FiltrationConfig:
    local_distance_quantile: float = 0.95
    dataset_distance_quantile: float = 0.95
    margin_factor: float = 1.10
    default_step_angstrom: float = 0.1
    allowed_steps_angstrom: tuple[float, ...] = (0.1, 0.2, 0.25, 0.5, 1.0)
    max_points: int = 200
    include_zero: bool = True
    fixed_point_count: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.local_distance_quantile <= 1.0:
            raise ValueError("local_distance_quantile must be in (0, 1]")
        if not 0.0 < self.dataset_distance_quantile <= 1.0:
            raise ValueError("dataset_distance_quantile must be in (0, 1]")
        if self.margin_factor <= 0.0:
            raise ValueError("margin_factor must be positive")
        if self.default_step_angstrom <= 0.0:
            raise ValueError("default_step_angstrom must be positive")
        if self.max_points < 2:
            raise ValueError("max_points must be at least 2")
        if self.fixed_point_count is not None and self.fixed_point_count < 2:
            raise ValueError("fixed_point_count must be at least 2 when configured")
        if not self.allowed_steps_angstrom:
            raise ValueError("allowed_steps_angstrom cannot be empty")
        if any(step <= 0.0 for step in self.allowed_steps_angstrom):
            raise ValueError("allowed steps must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FiltrationProfile:
    scale_kind: ScaleKind
    start: float
    stop: float
    step: float
    num_points: int
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def choose_distance_step(rmax: float, config: FiltrationConfig = FiltrationConfig()) -> float:
    if rmax <= 0.0:
        raise ValueError("rmax must be positive")
    candidate_steps = tuple(
        step
        for step in sorted(set(config.allowed_steps_angstrom))
        if step >= config.default_step_angstrom
    )
    if not candidate_steps:
        raise ValueError("allowed_steps_angstrom must include the default step or a coarser step")
    for step in candidate_steps:
        if grid_point_count(0.0 if config.include_zero else step, rmax, step) <= config.max_points:
            return step
    raise ValueError(
        f"No allowed filtration step keeps rmax={rmax} within max_points={config.max_points}"
    )


def make_distance_filtration_profile(
    *,
    rmax: float,
    config: FiltrationConfig = FiltrationConfig(),
    source: str = "configured",
) -> FiltrationProfile:
    if config.fixed_point_count is not None:
        if config.include_zero:
            start = 0.0
            step = rmax / (config.fixed_point_count - 1)
        else:
            step = rmax / config.fixed_point_count
            start = step
        return FiltrationProfile(
            scale_kind="distance",
            start=float(start),
            stop=float(rmax),
            step=float(step),
            num_points=config.fixed_point_count,
            source=f"{source}_fixed_point_count",
        )
    step = choose_distance_step(rmax, config)
    start = 0.0 if config.include_zero else step
    num_points = grid_point_count(start, rmax, step)
    aligned_stop = start + (num_points - 1) * step
    return FiltrationProfile(
        scale_kind="distance",
        start=start,
        stop=float(aligned_stop),
        step=float(step),
        num_points=num_points,
        source=source,
    )


def make_tau_profile(*, start: float, stop: float, step: float, source: str = "configured") -> FiltrationProfile:
    if start <= 0.0 or stop <= start or step <= 0.0:
        raise ValueError("tau start/stop/step must be positive and ordered")
    return FiltrationProfile(
        scale_kind="tau",
        start=float(start),
        stop=float(stop),
        step=float(step),
        num_points=grid_point_count(start, stop, step),
        source=source,
    )


def grid_point_count(start: float, stop: float, step: float) -> int:
    if step <= 0.0 or stop < start:
        raise ValueError("invalid grid start/stop/step")
    return int(math.ceil((stop - start) / step - 1e-12)) + 1
