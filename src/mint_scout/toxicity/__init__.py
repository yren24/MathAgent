"""Small-molecule toxicity dataset and legacy feature adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mint_scout.toxicity.legacy import (
        TOXICITY_INVARIANTS,
        ToxicityFeatureTool,
        ToxicityToolConfig,
    )

__all__ = [
    "TOXICITY_INVARIANTS",
    "ToxicityFeatureTool",
    "ToxicityToolConfig",
]


def __getattr__(name: str) -> Any:
    if name in {"TOXICITY_INVARIANTS", "ToxicityFeatureTool", "ToxicityToolConfig"}:
        from mint_scout.toxicity.legacy import (
            TOXICITY_INVARIANTS,
            ToxicityFeatureTool,
            ToxicityToolConfig,
        )

        exports = {
            "TOXICITY_INVARIANTS": TOXICITY_INVARIANTS,
            "ToxicityFeatureTool": ToxicityFeatureTool,
            "ToxicityToolConfig": ToxicityToolConfig,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
