# Small-Molecule Toxicity Agent Workflow

This document describes the first production-shaped toxicity workflow in
`MathAgent`. It is intentionally aligned with the protein-ligand workflow while
keeping a separate toxicity adapter so protein-ligand execution is not affected.

## Scope

The current workflow supports small-molecule regression tasks with structure
files listed in a manifest, such as LD50 toxicity prediction. The user can
choose `PCC2` (the default), `PCC`, or true `R2` as the primary metric. All
three, plus `RMSE` and `MAE`, are reported. The user can restrict the available
methods to a nonempty subset of `PH`, `PL`, `CA`, `FPRC`, and `EIC`.

This version requires labeled, explicit train/test splits. A separate
validation split is rejected during preflight rather than silently ignored.

## High-Level Flow

```text
Natural-language user request
  -> LLM intake converts the request to a strict UserRequest
  -> deterministic preflight validates paths, splits, labels, and roles
  -> toxicity workflow launcher submits a Slurm dependency chain
  -> deterministic train-only representation design
  -> optional LLM advisory proposes bounded representation candidates
  -> Python validates and materializes candidate representations
  -> train-only probe selection with hard representativeness gates
  -> isolated feature-tool execution for the requested methods
  -> feature QC and deterministic filtration repair if needed
  -> probe GBT scout ranks method subsets
  -> final full-train GBT and fixed-test evaluation
  -> JSON report with audit/protocol guards
```

## Shared Workflow Entry

Protein-ligand and toxicity requests use the same user intake and workflow
adapter interface. The config version selects the domain adapter. The adapter
builds the launch plan, submits it, reads status, and locates the final report.
Each domain keeps its own feature tools, representation policy, probe selector,
GBT configuration, and scientific execution code.

The common CLI accepts either generated `pipeline.yaml`:

```bash
python -m mint_scout.run_pipeline plan --config /path/to/pipeline.yaml
python -m mint_scout.run_pipeline status --config /path/to/pipeline.yaml
python -m mint_scout.run_pipeline report --config /path/to/pipeline.yaml
```

Structured YAML requests and natural-language requests now use the same
submission adapter. Toxicity submission records `toxicity.workflow.json`; the
launcher records `toxicity.jobs.json` with the final Slurm job ID so status can
distinguish an active run, a failed final job, and a completed report.

Protein-ligand retains its resumable lifecycle. Toxicity currently runs a
fixed Slurm dependency chain, so `resume` is not supported for that adapter.

## LLM Role

The LLM is advisory only.

It can:

- parse a natural-language user request into a strict request object;
- suggest one or two bounded toxicity representation candidates;
- provide rationale for probe/adaptive settings;
- help explain the final report.

It cannot:

- use test labels or test metrics;
- directly rank methods numerically;
- override feature QC, probe ranking, or final empirical results;
- remove deterministic baseline candidates.

Every LLM artifact records `used_for_numeric_decisions: false`.

## Deterministic Decisions

The final method and representation are selected by deterministic Python:

- representation candidates must pass train-only feature QC;
- probe candidates use hard gates for target distribution, molecule size,
  joint target-size strata, and pair coverage;
- method ranking uses the requested primary metric on probe OOF GBT evidence,
  bootstrap stability, and cost; the first `priority_order` candidate is the
  default, not necessarily the candidate with the highest point estimate;
- with `satisfy_target`, the first priority candidate meeting the explicit
  threshold on the train-only probe is frozen; if none qualifies, rank one is
  frozen and the unmet probe target is reported;
- the fixed test set is evaluated once after the selected subset is frozen.

For `maximize_rank1`, no threshold is needed. A user-supplied threshold, when
present, is checked in the final report. If the fixed-test primary metric is
below it, the final status is `TARGET_NOT_REACHED`. The workflow does not try
another candidate after reading test results.

## Key Outputs

For a generated toxicity run, the preflight report records the exact locations:

- `task.yaml`: frozen task configuration;
- `pipeline.yaml`: toxicity workflow launch configuration;
- `preflight.json`: user-facing audit of the request and protocol guards;
- `toxicity_representation_design.json`: train-only representation candidates;
- `toxicity_llm_advisory.json`: LLM representation advice, if enabled;
- `toxicity_probe_selection.json`: selected representative probe;
- `toxicity_probe_scout.json`: probe method/subset ranking;
- `toxicity_final_test_report.json`: full-train/fixed-test report.

The final report includes `PCC2`, `PCC`, true `R2`, `RMSE`, `MAE`, method costs,
selected subset, a bootstrap interval for the primary metric, and per-sample
predictions. The original `bootstrap_pcc2` field remains for comparison.

## Current Protocol Guard

The final test set is not used for representation selection, method ranking, or
repair. The workflow records:

```text
test_used_for_selection = false
full_train_cross_validation_used = false
final_test_query_budget = 1
```

This means the probe is used for selection, and the test set is used only once
for final evaluation.
