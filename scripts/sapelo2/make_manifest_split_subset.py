#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic split-balanced manifest subset."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=0)
    parser.add_argument("--validation", type=int, default=0)
    parser.add_argument("--test", type=int, default=0)
    args = parser.parse_args()

    requested = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input manifest has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    selected = []
    for split, limit in requested.items():
        if limit < 0:
            raise ValueError(f"{split} count must be nonnegative")
        split_rows = [row for row in rows if row.get("split") == split]
        if len(split_rows) < limit:
            raise ValueError(
                f"not enough {split} rows: requested {limit}, available {len(split_rows)}"
            )
        selected.extend(split_rows[:limit])

    if not selected:
        raise ValueError("subset is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    print(
        f"wrote {len(selected)} rows to {args.output} "
        f"(train={args.train}, validation={args.validation}, test={args.test})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
