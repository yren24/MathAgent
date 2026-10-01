from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mint_scout.data.element_pairs import make_casf_protein_ligand_schema
from mint_scout.invariants.plbind_tools import PLBindFeatureTool, PLBindToolConfig
from mint_scout.invariants.registry import INVARIANTS
from mint_scout.representation import (
    RepresentationMode,
    RepresentationSpec,
    make_legacy_casf_representation_spec,
)


@dataclass(frozen=True)
class ParityResult:
    sample_id: str
    invariant: str
    legacy_shape: tuple[int, ...]
    adaptive_shape: tuple[int, ...]
    exact_equal: bool
    max_abs_difference: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "invariant": self.invariant,
            "legacy_shape": list(self.legacy_shape),
            "adaptive_shape": list(self.adaptive_shape),
            "exact_equal": self.exact_equal,
            "max_abs_difference": self.max_abs_difference,
        }


def validate_parity(
    *,
    sample_id: str,
    invariant: str,
    legacy_root: Path,
    pdb_folder: Path,
    output_root: Path,
) -> ParityResult:
    legacy_spec = make_legacy_casf_representation_spec()
    adaptive_spec = RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=make_casf_protein_ligand_schema(),
        filtration_profiles=dict(legacy_spec.filtration_profiles),
        parameters={"validation": "legacy_full_40_channel_parity"},
    ).freeze()
    common = {
        "legacy_root": legacy_root,
        "pdb_folder": pdb_folder,
        "output_root": output_root,
        "dataset_id": "plbind-parity",
    }
    legacy_tool = PLBindFeatureTool(PLBindToolConfig(**common, representation_mode="legacy_casf"))
    adaptive_tool = PLBindFeatureTool(
        PLBindToolConfig(**common, representation_mode="dataset_adaptive")
    )
    legacy_path = legacy_tool.compute_with_spec(sample_id, invariant, legacy_spec).output_path
    adaptive_path = adaptive_tool.compute_with_spec(sample_id, invariant, adaptive_spec).output_path
    legacy = np.load(legacy_path)
    adaptive = np.load(adaptive_path)
    exact_equal = bool(np.array_equal(legacy, adaptive, equal_nan=True))
    if legacy.shape == adaptive.shape and legacy.size:
        difference = np.abs(legacy.astype(np.float64) - adaptive.astype(np.float64))
        max_abs_difference = float(np.nanmax(difference))
    else:
        max_abs_difference = float("inf")
    return ParityResult(
        sample_id=sample_id,
        invariant=invariant,
        legacy_shape=tuple(legacy.shape),
        adaptive_shape=tuple(adaptive.shape),
        exact_equal=exact_equal,
        max_abs_difference=max_abs_difference,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate adaptive-adapter parity with legacy PLBind output.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--invariant", choices=INVARIANTS, action="append", required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--pdb-folder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    results = [
        validate_parity(
            sample_id=args.sample_id.lower(),
            invariant=invariant,
            legacy_root=args.legacy_root,
            pdb_folder=args.pdb_folder,
            output_root=args.output_root,
        )
        for invariant in args.invariant
    ]
    payload = {
        "report_schema": "mint-agent.plbind-parity.v1",
        "results": [result.to_dict() for result in results],
        "all_exact_equal": all(result.exact_equal for result in results),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["all_exact_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
