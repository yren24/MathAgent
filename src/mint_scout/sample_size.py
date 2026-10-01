from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


DEFAULT_MIN_LABELED_SAMPLES = 300
SMALL_DATA_WARNING_CODE = "SMALL_LABELED_MODELING_POOL"


@dataclass(frozen=True)
class LabeledSamplePolicy:
    min_labeled_samples: int = DEFAULT_MIN_LABELED_SAMPLES
    allow_small_data_override: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.min_labeled_samples, bool) or self.min_labeled_samples < 1:
            raise ValueError("labels.min_labeled_samples must be a positive integer")
        if not isinstance(self.allow_small_data_override, bool):
            raise ValueError("labels.allow_small_data_override must be true or false")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "LabeledSamplePolicy":
        values = dict(data or {})
        unknown = sorted(
            set(values) - {"min_labeled_samples", "allow_small_data_override"}
        )
        if unknown:
            raise ValueError(f"unknown labels fields: {unknown}")
        raw_minimum = values.get("min_labeled_samples", DEFAULT_MIN_LABELED_SAMPLES)
        if raw_minimum is None:
            raw_minimum = DEFAULT_MIN_LABELED_SAMPLES
        if isinstance(raw_minimum, bool):
            raise ValueError("labels.min_labeled_samples must be a positive integer")
        return cls(
            min_labeled_samples=int(raw_minimum),
            allow_small_data_override=values.get(
                "allow_small_data_override", False
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def warning(self, modeling_sample_count: int) -> str | None:
        if modeling_sample_count >= self.min_labeled_samples:
            return None
        disposition = (
            "explicit override permits this engineering run"
            if self.allow_small_data_override
            else "explicit confirmation is required before submission"
        )
        return (
            f"{SMALL_DATA_WARNING_CODE}: labeled modeling pool has "
            f"{modeling_sample_count} samples, below the configured provisional "
            f"minimum {self.min_labeled_samples}; {disposition}."
        )
