from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "start":
        from mint_scout.user_intake import main as intake_main

        return intake_main(args[1:])
    if args and args[0] == "explain":
        from mint_scout.agent.llm_explanation import main as explanation_main

        return explanation_main(args[1:])
    if args and args[0] == "experiment":
        from mint_scout.controlled_experiment import main as experiment_main

        return experiment_main(args[1:])
    if args and args[0] == "repair-filtration":
        from mint_scout.filtration_repair import main as repair_filtration_main

        return repair_filtration_main(args[1:])
    if args and args[0] == "lifecycle":
        from mint_scout.run_agent_lifecycle import main as lifecycle_main

        return lifecycle_main(args[1:])

    from mint_scout.run_pipeline import main as pipeline_main

    return pipeline_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
