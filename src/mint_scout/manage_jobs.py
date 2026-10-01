from __future__ import annotations

import argparse
import json
from pathlib import Path

from mint_scout.execution.jobs import (
    create_job_ledger,
    ledger_summary,
    load_job_ledger,
    merge_job_plans,
    reconcile_completed_jobs,
    refresh_job_ledger,
    resume_job_ledger,
    submit_ready_jobs,
    write_job_ledger,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit, inspect, and resume auditable mint-agent Slurm job plans."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument("--ledger", type=Path, required=True)
    submit_parser.add_argument("--execute", action="store_true")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--ledger", type=Path, required=True)
    status_parser.add_argument("--offline", action="store_true")

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--ledger", type=Path, required=True)
    resume_parser.add_argument("--execute", action="store_true")
    resume_parser.add_argument("--offline", action="store_true")
    resume_parser.add_argument("--resubmit-unknown", action="store_true")

    reconcile_parser = subparsers.add_parser("reconcile")
    reconcile_parser.add_argument("--ledger", type=Path, required=True)
    reconcile_parser.add_argument("--registry", type=Path, required=True)
    reconcile_parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "submit":
        report = _read_json_object(args.plan)
        if args.ledger.exists():
            ledger = load_job_ledger(args.ledger)
            merge_job_plans(ledger, report)
        else:
            ledger = create_job_ledger(report, source_plan=args.plan)
        submit_ready_jobs(ledger, execute=args.execute)
    else:
        ledger = load_job_ledger(args.ledger)
        refresh_job_ledger(ledger, query_scheduler=not args.offline)
        if args.command == "resume":
            resume_job_ledger(
                ledger,
                execute=args.execute,
                resubmit_unknown=args.resubmit_unknown,
            )
        elif args.command == "reconcile":
            reconcile_completed_jobs(ledger, registry=args.registry)

    write_job_ledger(ledger, args.ledger)
    print(json.dumps(ledger_summary(ledger), sort_keys=True))
    if args.command == "reconcile" and ledger.get("status") != "COMPLETE":
        return 2
    return 0


def _read_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
