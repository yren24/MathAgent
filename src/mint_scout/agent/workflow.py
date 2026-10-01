from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from mint_scout.config import load_yaml
from mint_scout.pipeline_reporting import write_pipeline_report
from mint_scout.run_pipeline import (
    PIPELINE_CONFIG_VERSION,
    PipelineLaunchPlan,
    build_pipeline_launch_plan,
    pipeline_status,
    submit_pipeline,
)
from mint_scout.agent.lifecycle import load_lifecycle
from mint_scout.toxicity.workflow import (
    TOXICITY_WORKFLOW_VERSION,
    build_toxicity_launch_plan,
    toxicity_workflow_report,
    toxicity_workflow_status,
    write_toxicity_workflow_state,
)


@dataclass(frozen=True)
class DomainWorkflow:
    pipeline_kind: str
    config_version: str
    build_launch_plan: Callable[[str | Path], PipelineLaunchPlan]
    status: Callable[[PipelineLaunchPlan], dict[str, Any]]
    report: Callable[[PipelineLaunchPlan], dict[str, Any]]
    write_submission_state: Callable[[PipelineLaunchPlan, Mapping[str, Any]], None]
    resumable: bool

    def submit(self, plan: PipelineLaunchPlan) -> dict[str, Any]:
        receipt = submit_pipeline(plan)
        self.write_submission_state(plan, receipt)
        return receipt


def _protein_ligand_report(plan: PipelineLaunchPlan) -> dict[str, Any]:
    state_path = Path(plan.state_path)
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    return write_pipeline_report(
        state=load_lifecycle(state_path), state_path=state_path,
    ).to_dict()


def _no_submission_state(
    _plan: PipelineLaunchPlan, _receipt: Mapping[str, Any]
) -> None:
    return None


def _toxicity_submission_state(
    plan: PipelineLaunchPlan, receipt: Mapping[str, Any]
) -> None:
    write_toxicity_workflow_state(launch_plan=plan, receipt=receipt)


WORKFLOWS = {
    PIPELINE_CONFIG_VERSION: DomainWorkflow(
        pipeline_kind="protein_ligand_agent_lifecycle_v1",
        config_version=PIPELINE_CONFIG_VERSION,
        build_launch_plan=build_pipeline_launch_plan,
        status=pipeline_status,
        report=_protein_ligand_report,
        write_submission_state=_no_submission_state,
        resumable=True,
    ),
    TOXICITY_WORKFLOW_VERSION: DomainWorkflow(
        pipeline_kind="small_molecule_toxicity_gbt_v1",
        config_version=TOXICITY_WORKFLOW_VERSION,
        build_launch_plan=build_toxicity_launch_plan,
        status=toxicity_workflow_status,
        report=toxicity_workflow_report,
        write_submission_state=_toxicity_submission_state,
        resumable=False,
    ),
}


def workflow_for_config(path: str | Path) -> DomainWorkflow:
    version = str(load_yaml(Path(path).expanduser().resolve()).get("version") or "")
    try:
        return WORKFLOWS[version]
    except KeyError as exc:
        supported = ", ".join(sorted(WORKFLOWS))
        raise ValueError(
            f"unsupported workflow config version {version!r}; expected one of {supported}"
        ) from exc


def submit_prepared_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    workflow = workflow_for_config(str(bundle["pipeline_config"]))
    declared_kind = bundle.get("pipeline_kind")
    if declared_kind is not None and declared_kind != workflow.pipeline_kind:
        raise ValueError("prepared bundle pipeline_kind does not match pipeline config")
    return workflow.submit(workflow.build_launch_plan(str(bundle["pipeline_config"])))
