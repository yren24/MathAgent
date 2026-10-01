from __future__ import annotations

import argparse
from pathlib import Path

from mint_scout.config import load_yaml
from mint_scout.data.element_pairs import make_casf_protein_ligand_schema, make_toxicity_schema
from mint_scout.data.fidelity_sampler import assert_nested, make_nested_fidelity_subsets
from mint_scout.data.splits import assert_no_test_leakage, make_kfold_assignments
from mint_scout.invariants.registry import get_invariant


def _schema_from_config(config: dict):
    system_type = config.get("system_type")
    if system_type == "protein_ligand":
        return make_casf_protein_ligand_schema()
    if system_type == "small_molecule":
        return make_toxicity_schema()
    raise ValueError(f"Unsupported system_type={system_type!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic MathAgent smoke checks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--invariants", nargs="+", default=["PH", "PL"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    schema = _schema_from_config(config)
    invariants = [get_invariant(name).name for name in args.invariants]

    sample_ids = tuple(f"sample_{idx:03d}" for idx in range(40))
    test_sample_ids = sample_ids[-8:]
    dev_sample_ids = sample_ids[:-8]
    cv = config.get("cv", {})
    fidelity = config.get("fidelity", {})
    fold_ids = make_kfold_assignments(
        dev_sample_ids,
        n_splits=int(cv.get("n_splits", 5)),
        seed=int(cv.get("split_seed", 2026)),
    )
    subsets = make_nested_fidelity_subsets(
        dev_sample_ids,
        fidelity.get("fractions", [0.10, 0.25, 0.50, 1.00]),
        seed=int(cv.get("split_seed", 2026)),
    )
    assert_nested(subsets)
    assert_no_test_leakage(
        test_sample_ids=test_sample_ids,
        fold_ids=fold_ids,
        fidelity_sample_ids=subsets,
    )

    print(f"schema={schema.schema_id} pairs={schema.expected_pair_count}")
    print(f"invariants={','.join(invariants)}")
    print(f"fidelity_levels={','.join(str(k) for k in sorted(subsets))}")
    print("dry_run=true" if args.dry_run else "dry_run=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
