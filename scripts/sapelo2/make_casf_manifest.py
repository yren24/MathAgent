#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mint_scout.data.casf_index import load_casf_records, missing_structure_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a strict mint-agent manifest from PDBbind/CASF index files."
    )
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--structures-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--protein-template", default="{pdb}_pocket.pdb")
    parser.add_argument("--ligand-template", default="{pdb}_ligand.mol2")
    args = parser.parse_args()

    records = load_casf_records(
        index_root=args.index_root,
        structures_root=args.structures_root,
        year=args.year,
        protein_template=args.protein_template,
        ligand_template=args.ligand_template,
    )
    missing = missing_structure_paths(records)
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"missing {len(missing)} required structure files; first missing: {preview}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "target",
        "split",
        "protein_path",
        "ligand_path",
        "pdb_id",
        "benchmark_year",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.pdb_id,
                    "target": "" if record.label is None else f"{record.label:.15g}",
                    "split": record.split or "",
                    "protein_path": record.protein_path,
                    "ligand_path": record.ligand_path,
                    "pdb_id": record.pdb_id,
                    "benchmark_year": args.year,
                }
            )
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record.split or "unsplit"] = split_counts.get(record.split or "unsplit", 0) + 1
    print(f"wrote {len(records)} rows to {args.output}; splits={split_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
