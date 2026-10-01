"""Adapters that invoke, but never reimplement, the legacy MOF programs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LEGACY_MOF_TOOLS = {
    "MOF_HOMOLOGY": "homology",
    "MOF_LAPLACIAN": "lap",
    "MOF_FACET": "facet",
    "MOF_FORMAN": "forman",
    "MOF_CURVATURE": "curvature",
}


@dataclass(frozen=True)
class LegacyMofPaths:
    legacy_root: Path
    cif_dir: Path
    data_dir: Path
    feature_dir: Path
    result_dir: Path
    feature_script: Path | None = None
    gbt_script: Path | None = None

    def resolved_feature_script(self) -> Path:
        return self.feature_script or self.legacy_root / "mof_topology_features_forman120_final.py"

    def resolved_gbt_script(self) -> Path:
        return self.gbt_script or self.legacy_root / "mof_topology_gbt.py"


def topology_for(tool_name: str) -> str:
    try:
        return LEGACY_MOF_TOOLS[tool_name.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported legacy MOF tool: {tool_name}") from exc


def legacy_feature_command(
    *, paths: LegacyMofPaths, tool_name: str, property_name: str, start: int = 0, end: int | None = None, python_bin: str = "python"
) -> tuple[str, ...]:
    command = [
        python_bin,
        str(paths.resolved_feature_script()),
        "--topology",
        topology_for(tool_name),
        "--property",
        property_name,
        "--all",
        "--start",
        str(start),
        "--cif-dir",
        str(paths.cif_dir),
        "--data-dir",
        str(paths.data_dir),
        "--output-dir",
        str(paths.feature_dir),
        "--xyz-dir",
        str(paths.result_dir / "xyz"),
    ]
    if end is not None:
        command.extend(("--end", str(end)))
    return tuple(command)


def legacy_gbt_command(
    *, paths: LegacyMofPaths, tool_name: str, property_name: str, folds: int = 5, repeat: int = 0, python_bin: str = "python"
) -> tuple[str, ...]:
    return (
        python_bin,
        str(paths.resolved_gbt_script()),
        "--topology",
        topology_for(tool_name),
        "--property",
        property_name,
        "--data-dir",
        str(paths.data_dir),
        "--feature-dir",
        str(paths.feature_dir),
        "--output-dir",
        str(paths.result_dir / "gbt"),
        "--folds",
        str(folds),
        "--repeat",
        str(repeat),
    )
