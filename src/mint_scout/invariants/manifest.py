from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureManifest:
    invariant_name: str
    schema_id: str
    feature_dim: int
    ordered_blocks: tuple[str, ...]
    block_shapes: dict[str, tuple[int, ...]]
    block_names: tuple[str, ...]
    code_version: str
    parameter_hash: str

    def assert_compatible(self, other: "FeatureManifest") -> None:
        if self != other:
            raise AssertionError(f"Feature manifests are not compatible: {self} != {other}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
