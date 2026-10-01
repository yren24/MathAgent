from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mint_scout.toxicity.legacy import (
    LEGACY_NAMES,
    TOXICITY_INVARIANTS,
    ToxicityFeatureTool,
    ToxicityToolConfig,
)


@dataclass(frozen=True)
class ToxicityParityResult:
    sample_id: str
    invariant: str
    reference_path: Path
    generated_path: Path
    reference_shape: tuple[int, ...]
    generated_shape: tuple[int, ...]
    exact_equal: bool
    max_abs_difference: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "invariant": self.invariant,
            "reference_path": str(self.reference_path),
            "generated_path": str(self.generated_path),
            "reference_shape": list(self.reference_shape),
            "generated_shape": list(self.generated_shape),
            "exact_equal": self.exact_equal,
            "max_abs_difference": self.max_abs_difference,
        }


def validate_toxicity_parity(
    *,
    sample_id: str,
    split: str,
    invariant: str,
    legacy_root: Path,
    dataset_dir: Path,
    reference_root: Path,
    output_root: Path,
) -> ToxicityParityResult:
    name = invariant.upper()
    if name not in TOXICITY_INVARIANTS:
        raise ValueError(f"Unknown toxicity invariant: {invariant}")
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    molecule_dir = dataset_dir / f"LD50_{split}_x"
    tool = ToxicityFeatureTool(
        ToxicityToolConfig(
            legacy_root=legacy_root,
            molecule_dirs=(molecule_dir,),
            output_root=output_root,
            dataset_id="LD50-parity",
        )
    )
    generated_path = tool.compute_one(sample_id, name).output_path
    reference_path = reference_root / LEGACY_NAMES[name] / split / f"{sample_id}.npy"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference feature does not exist: {reference_path}")
    reference = np.load(reference_path, allow_pickle=False)
    generated = np.load(generated_path, allow_pickle=False)
    shapes_match = reference.shape == generated.shape
    exact_equal = bool(shapes_match and np.array_equal(reference, generated, equal_nan=True))
    if shapes_match and reference.size:
        difference = np.abs(reference.astype(np.float64) - generated.astype(np.float64))
        max_abs_difference = float(np.nanmax(difference))
    elif shapes_match:
        max_abs_difference = 0.0
    else:
        max_abs_difference = float("inf")
    return ToxicityParityResult(
        sample_id=sample_id,
        invariant=name,
        reference_path=reference_path,
        generated_path=generated_path,
        reference_shape=tuple(int(value) for value in reference.shape),
        generated_shape=tuple(int(value) for value in generated.shape),
        exact_equal=exact_equal,
        max_abs_difference=max_abs_difference,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare regenerated LD50 features with frozen legacy arrays."
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument(
        "--invariant", choices=TOXICITY_INVARIANTS, action="append", required=True
    )
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    results = [
        validate_toxicity_parity(
            sample_id=args.sample_id,
            split=args.split,
            invariant=invariant,
            legacy_root=args.legacy_root,
            dataset_dir=args.dataset_dir,
            reference_root=args.reference_root,
            output_root=args.output_root,
        )
        for invariant in args.invariant
    ]
    payload = {
        "report_schema": "mint-agent.toxicity-legacy-parity.v1",
        "sample_id": args.sample_id,
        "split": args.split,
        "results": [result.to_dict() for result in results],
        "all_exact_equal": all(result.exact_equal for result in results),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["all_exact_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
