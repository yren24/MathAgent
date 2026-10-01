"""MOF-specific legacy-parity utilities.

This namespace intentionally has no dependency on the protein-ligand pipeline.
Exports are lazy so a standard-library-only metrics postprocessor can run in
the minimal legacy GBT environment, where PyYAML is not installed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LEGACY_CATEGORY_SCHEMA_ID",
    "LEGACY_MOF_TOOLS",
    "CategorySchema",
    "LegacyMofPaths",
    "MofManifestRecord",
    "legacy_category_schema",
    "legacy_feature_command",
    "legacy_gbt_command",
    "write_manifest",
]


def __getattr__(name: str) -> Any:
    if name in {"LEGACY_CATEGORY_SCHEMA_ID", "CategorySchema", "legacy_category_schema"}:
        from . import categories

        return getattr(categories, name)
    if name in {"LEGACY_MOF_TOOLS", "LegacyMofPaths", "legacy_feature_command", "legacy_gbt_command"}:
        from . import legacy

        return getattr(legacy, name)
    if name in {"MofManifestRecord", "write_manifest"}:
        from . import manifest

        return getattr(manifest, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
