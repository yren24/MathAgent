from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple


ArtifactKey = Tuple[str, float]


@dataclass
class ReplayEnvironment:
    hidden_results: dict[ArtifactKey, Any]
    revealed_results: dict[ArtifactKey, Any] = field(default_factory=dict)

    def available_keys(self) -> tuple[ArtifactKey, ...]:
        return tuple(sorted(self.revealed_results))

    def reveal(self, invariant_name: str, fidelity_level: float) -> Any:
        key = (invariant_name, fidelity_level)
        if key not in self.hidden_results:
            raise KeyError(f"No hidden result for {key}")
        self.revealed_results[key] = self.hidden_results[key]
        return self.revealed_results[key]

    def get_revealed(self, invariant_name: str, fidelity_level: float) -> Any:
        key = (invariant_name, fidelity_level)
        if key not in self.revealed_results:
            raise PermissionError(f"Result {key} has not been revealed by an acquisition action")
        return self.revealed_results[key]
