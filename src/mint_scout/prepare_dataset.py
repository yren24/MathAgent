from __future__ import annotations

import argparse
from pathlib import Path

from mint_scout.data.preparation import prepare_dataset_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and audit a configured dataset before agent execution."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = prepare_dataset_from_config(args.config, report_path=args.output)
    print(
        f"dataset={report['dataset_id']} provider={report['provider']} "
        f"samples={report['sample_count']} status={report['status']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
