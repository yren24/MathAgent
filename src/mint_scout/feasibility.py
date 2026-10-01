from __future__ import annotations

import argparse
from pathlib import Path

from mint_scout.config import load_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or run MathAgent feasibility analysis.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    if args.dry_run:
        print(f"Feasibility dry run for task_id={config.get('task_id')}")
        print("Planned checks: feature cost, 31-subset landscape, low/high fidelity rank correlation, bootstrap interval behavior.")
        return 0
    raise SystemExit("Non-dry feasibility runs are intentionally blocked until Phase 0/1 audit checks are complete.")


if __name__ == "__main__":
    raise SystemExit(main())
