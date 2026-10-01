from __future__ import annotations

import argparse
import json
from pathlib import Path

from mint_scout.agent.lifecycle import (
    advance_lifecycle,
    initialize_lifecycle,
    lifecycle_summary,
    load_lifecycle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or resume the auditable mint-agent Slurm lifecycle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--state", type=Path, required=True)
    start.add_argument("--task-config", required=True)
    start.add_argument("--registry", required=True)
    start.add_argument("--dataset-id", required=True)
    start.add_argument("--invariant", action="append", required=True)
    start.add_argument("--execution-profile", required=True)
    start.add_argument("--evidence-scope", default="full_train")
    start.add_argument("--representation-spec", default=None)
    start.add_argument("--sample-id-file", default=None)
    start.add_argument("--qc-config", default="configs/scout/v1.yaml")
    start.add_argument("--audit-config", default="configs/scout/v1.yaml")
    start.add_argument("--feature-diagnostic-top-k", type=int, default=10)
    start.add_argument("--scout-config", default="configs/scout/v1.yaml")
    start.add_argument("--gbt-config", default="configs/gbt/plbind_adaptive_gbt.yaml")
    start.add_argument("--user-target", type=float, default=None)
    start.add_argument("--lifecycle-id", default=None)
    start.add_argument(
        "--llm-scientific-mode",
        choices=("disabled", "shadow", "advisory"),
        default="disabled",
    )
    start.add_argument("--llm-model", default=None)
    start.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    start.add_argument("--llm-env-file", default=None)
    start.add_argument("--llm-cache-dir", default=None)
    start.add_argument("--llm-max-candidates", type=int, default=10)
    start.add_argument("--llm-timeout-seconds", type=float, default=180.0)
    _add_advance_options(start)

    advance = subparsers.add_parser("advance")
    advance.add_argument("--state", type=Path, required=True)
    _add_advance_options(advance)

    status = subparsers.add_parser("status")
    status.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "status":
        state = load_lifecycle(args.state)
    else:
        if args.command == "start":
            initialize_lifecycle(
                args.state,
                task_config=args.task_config,
                registry=args.registry,
                dataset_id=args.dataset_id,
                invariants=tuple(args.invariant),
                execution_profile=args.execution_profile,
                evidence_scope=args.evidence_scope,
                representation_spec=args.representation_spec,
                sample_id_file=args.sample_id_file,
                qc_config=args.qc_config,
                audit_config=args.audit_config,
                feature_diagnostic_top_k=args.feature_diagnostic_top_k,
                scout_config=args.scout_config,
                gbt_config=args.gbt_config,
                user_target=args.user_target,
                llm_scientific_mode=args.llm_scientific_mode,
                llm_model=args.llm_model,
                llm_api_key_env=args.llm_api_key_env,
                llm_env_file=args.llm_env_file,
                llm_cache_dir=args.llm_cache_dir,
                llm_max_candidates=args.llm_max_candidates,
                llm_timeout_seconds=args.llm_timeout_seconds,
                lifecycle_id=args.lifecycle_id,
            )
        state = advance_lifecycle(
            args.state,
            execute=args.execute,
            auto_continue=args.auto_continue,
            continuation_script=args.continuation_script,
            project_root=args.project_root,
            resume_failed=args.resume_failed,
            max_attempts=args.max_attempts,
            stop_after_stage=args.stop_after_stage,
        )

    print(json.dumps(lifecycle_summary(state), sort_keys=True))
    return (
        0
        if state["status"]
        in {
            "COMPLETE",
            "COMPLETE_TARGET_NOT_REACHED",
            "WAITING_FOR_JOBS",
            "PLANNED",
            "PAUSED_AFTER_STAGE",
        }
        else 2
    )


def _add_advance_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--auto-continue", action="store_true")
    parser.add_argument("--continuation-script", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        default=None,
        help="Pause after reconciling the zero-based Slurm job stage.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
