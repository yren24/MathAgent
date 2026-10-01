# MathAgent Architecture

MathAgent separates scientific computation from agent orchestration. Its optional LLM layer is an
outer interface around the deterministic lifecycle, not part of the numerical scientific engine.

## Domain Workflow Adapters

The user intake graph and `mint_scout.run_pipeline` CLI use one workflow
interface for protein-ligand and small-molecule toxicity tasks. The generated
pipeline config version selects a domain adapter. Each adapter builds a Slurm
launch plan, submits it, reports status, and locates the final report.

The protein-ligand adapter runs the existing resumable agent lifecycle. The
toxicity adapter runs its existing Slurm dependency chain and records the final
job ID for status queries. The two adapters retain their own representation,
probe, feature, and GBT implementations. Changing the common interface does
not change a domain's scientific settings or method ranking.

## Current Phase

The implemented path is deterministic and GBT-first through Scout, Phase 4 execution, grounded label
retrieval validation, and final artifact reporting:

When a task supplies `dataset_manifest`, `project_context_node` invokes the deterministic CSV/JSONL
manifest reader before consulting artifact memory. The reader constructs `DatasetManifest` and
`TaskCard`, validates explicit role paths and labels, and routes the evaluation mode. Unsupported task
types, missing roles, insufficient labels, and required label retrieval stop before representation or
feature work begins. Legacy CASF task configs without this field retain their audited loading path.

Manifest-backed PLBind batches stage arbitrary source paths as sample-specific symlinks with the exact
legacy pocket/ligand filenames. Both source-file SHA-256 values and a combined structure hash are
written to the feature manifest. The combined hash is part of the output cache path. Phase 4 loads
precomputed arrays from the QC-approved manifest itself rather than reconstructing their paths.

```text
natural-language request
-> LLM intent parser with strict structured output
-> explicit path/context merge
-> bounded manifest profiler with no raw values, sample IDs, or filesystem paths
-> optional LLM column mapper restricted to exact profiled column names
-> deterministic schema validation and preflight
-> NEEDS_USER_INPUT or immutable task/pipeline configs
-> optional Slurm submission

data manifest
-> data audit
-> feature tools
-> representative probe
-> shared 5-fold OOF GBT predictions
-> all nonempty prediction-level consensus subsets
-> stratified paired bootstrap
-> frozen top-tier candidate priority
-> explicit or probe-derived acceptance target
-> progressive full acquisition from the frozen queue
-> routed external-test or full-CV evaluation
-> target-aware stopping or TARGET_NOT_REACHED failure report
-> final all-training-data fit for inference predictions
-> grounded optional label-provider boundary with provenance validation
-> reproducible run artifacts and human-readable report

completed deterministic pipeline summary
-> redacted fact projection without sample IDs or filesystem paths
-> optional LLM explanation node
-> audited explanation artifact that cannot modify scientific outputs
```

The LLM intake graph and explanation graph may be unavailable without API credentials or network
access. The central scientific graph remains fully runnable from YAML and CLI in that case. API keys
are read only from the process environment and are not included in graph state, generated configs,
or artifacts.

The Phase 4 execution engine is implemented against a feature-provider interface and synthetic tests.
The production provider drives the legacy PLBind feature scripts as isolated calls, validates and loads
their cached arrays, and flattens each invariant tensor into the legacy GBT input vector. The Sapelo2
Phase 4 smoke command exercises this provider under Slurm and writes an auditable JSON report.

Dataset-adaptive representation design now streams the modeling-pool structures twice: element support
and pair construction first, robust geometry profiling second. It freezes the resulting pair order,
dataset fingerprint, common PH/PL/CA/FPRC distance grid, EIC tau grid, anomalies, and configuration in a
hashed RepresentationSpec. This design capability does not imply that the legacy mathematical adapter
can execute it; unsupported modes are rejected before feature computation.

Smoke runs are engineering checks only. They validate paths, imports, feature execution, cache writes, and report generation. Smoke metrics must not be used for feature selection, model selection, or final scientific claims.

The optional HiQBind preparation provider exercises the missing-manifest user path on an independent
protein-ligand dataset. It keeps endpoint and sign filters explicit, validates source log affinities
against their value/unit fields, refuses silent unit inference, records invalid-row exclusions, and
supports checksum-verified download plus interrupted-extraction recovery. A configured sample limit
is reported as an engineering subset rather than full-dataset evidence.

Manifest-backed CV can name an explicit group identifier. The probe selector then balances whole
groups across the configured number of folds and freezes that assignment for Scout and execution.
Preflight rejects identifier groups crossing protected user splits and reports repeated unsplit groups
when sample-level CV would risk structural leakage.

User intake attaches a labeled-sample policy to every generated task. A modeling pool below the
configurable provisional minimum stops at `NEEDS_SMALL_DATA_CONFIRMATION`; an explicitly approved
engineering override proceeds with a persistent warning. The graph rechecks the same task policy after
dataset preparation, so a missing-manifest workflow cannot bypass the guard.

## Probe Representativeness Audit

The frozen probe is audited against its full training/modeling pool without changing its membership.
The audit compares target values, protein-pocket atom counts, ligand atom counts, total atom counts,
joint target/total-size strata, and the independently recomputed support of every retained element
pair. Configurable KS-distance, standardized-mean-shift, and joint-cell-share thresholds produce review
warnings rather than automatic sample deletion or resampling.

## Artifact Memory

The artifact-memory registry is deterministic pipeline memory, not LLM conversational memory. It stores
searchable metadata and cryptographic content hashes for representation designs, frozen probes, feature
manifests, QC reports, Scout plans, model evaluations, and other artifacts. Evidence scope is explicit:
`smoke`, `probe`, `full_train`, `external_test`, `inference`, or `design`. The registry never promotes a
smoke artifact to scientific evidence and never treats a path alone as proof of compatibility.

Evaluation reuse requires matching sample order, representation, GBT parameter hash, target metric,
target value, and frozen folds. Current reports compare fold-assignment hashes. A historical report that
predates that field is reusable only when it records the exact Scout artifact as its fold source; a
missing top-level GBT hash may be derived from a complete embedded GBT config and is marked as derived
in registry metadata. Source result files remain unchanged.

## Search Unit

The search unit is a candidate pipeline, not a feature alone.

```text
candidate pipeline =
  feature set
  + feature schema
  + model family
  + normalization
  + training protocol
```

This keeps feature selection model-conditioned. For example, evidence from `PL + legacy_gbt` does not imply that `PL + mlp_v1` or `PL + gnn_v1` will be optimal.

## First Active Model Family

The first active model family is gradient boosting, using the legacy PLBind GBT configuration:

```text
StandardScaler
GradientBoostingRegressor
n_estimators = 4000
max_depth = 7
min_samples_split = 5
learning_rate = 0.01
subsample = 0.5
max_features = sqrt
n_runs = 3 (adaptive default; the frozen legacy reproduction profile remains 10)
```

Neural networks should be added later as new candidate pipelines with explicit input contracts, not as replacements for GBT assumptions.

The current `fixed_probe_ann` path is a bounded diagnostic toward that future model registry. It uses
the existing PLBind ANN pattern (flattened feature blocks, fold-local `StandardScaler`, BatchNorm/ReLU,
MSE, AdamW, and OneCycleLR) with one frozen configuration and shared probe folds. It separately reports
feature concatenation and prediction-level averaging, and evaluates each combination against its best
singleton component. Its output is not a Scout execution artifact and cannot change the V1 GBT search.

## Representation Boundary

`dataset_adaptive` schema construction can retain up to a configurable 50 element-pair channels. The
verified protein-ligand adapter can execute adaptive subsets and orderings only within its audited native
40 role-aware channel universe by calling the unchanged legacy implementation and slicing/reordering its
output. Pairs outside that universe are rejected. Capability metadata and hashed cache paths prevent
legacy and adaptive outputs from being confused.

## Label Boundary

Label retrieval is an optional provider interface, not a fifth numerical agent. A returned value is admitted
only after deterministic sample-ID, public-identifier, target, endpoint, match-status, source-provenance,
and configured-unit checks. Conflicting credible values remain quarantined; source-free or model-guessed
numeric labels cannot satisfy the retrieval schema.

## Reporting Boundary

The report writer keeps probe OOF evidence separate from full-CV or user-specified acceptance evidence.
It records acquisition/cache history, timing, anomalies, predictions and disagreement when applicable,
and marks a labeled external set as non-untouched once it participates in stop/continue decisions.

## Scout Boundary

Scout compares method subsets from held-out OOF predictions using one shared fold assignment, the
audited fixed GBT configuration, arithmetic-mean prediction consensus, and a paired stratified
bootstrap. Feature QC is a hard gate. The paired bootstrap defines the statistically competitive top
tier; ranking stability orders that tier, and measured feature-acquisition cost breaks remaining ties.
The historical normalized composite is retained only as a diagnostic field and never determines the
frozen priority order. LLM proposals are candidate generators and provenance only; they receive no
numerical ranking credit. Under `satisfy_target`, acceptance evidence determines whether the
acquisition queue stops but target-reaching subsets retain their frozen Probe priority. Under an
explicit maximization objective, acceptance scores choose the best subset only after the required
method set has been acquired.

Representation candidates use a separate hierarchical train-only, label-free decision. They share the
exact Probe samples for comparable feature audits. The selector uses element-pair and adapter coverage,
filtration adequacy, feature degeneracy and warning burden, and a configurable feature-dimension budget.
It does not use GBT performance, bootstrap ranking stability, runtime cost, test evidence, or an LLM
prior. The baseline wins only if it remains tied after all representation-suitability stages.

## Execution Boundary

Execution consumes Scout's frozen priority order. It acquires one missing invariant family at a time,
reuses cached features when available, and after every acquisition reevaluates every nonempty consensus
subset available for free. If one or more subsets meet the configured target, it selects the smallest
passing subset, then the best score, then lower acquisition cost, then canonical name order. If no subset
meets the target after the queue is exhausted, it returns `TARGET_NOT_REACHED` with the best achieved
subset and score.

For a labeled train/validation/test dataset, the same objective is evaluated on validation; the selected
subset is then frozen before test features are loaded, and test is queried once. For a labeled train/test
dataset, execution derives an invariant-acquisition queue from the frozen Probe
priority order. It fits the configured three-run GBT ensemble only for the newly acquired invariant,
retains row-aligned predictions from earlier stages, and scores every nonempty subset constructible from
the acquired methods. No full-training CV is performed. `satisfy_target` stops at the first stage with a
target-reaching frozen-ranked subset. `maximize_rank1` continues until every method in Probe Rank 1 has
been acquired, then selects the best acceptance score among all subsets of those methods, using frozen
Probe priority to break numerical ties. `maximize_all` continues through every requested method. Because
these decisions query the labeled acceptance split, it is not independent test evidence. For full
labeled CV, execution uses full shared out-of-fold predictions.
For unlabeled inference samples, execution fits final models on all modeling samples and returns
predictions plus disagreement `std` and `range` only when the selected subset has multiple invariants.

Execution retains row-aligned per-invariant evaluation predictions and the selected consensus
prediction. Full-CV reports label these as `full_oof_predictions`; explicit labeled-test reports label
them as `external_eval_predictions`. PCC, RMSE, MAE, R2, and selected-consensus fold metrics are derived
from those saved vectors. A final all-modeling-data fit is performed only when inference samples are
actually requested.

Scout and Execution exchange a frozen artifact containing modeling sample order, shared full-data folds,
priority order, acceptance target, RepresentationSpec hash, GBT parameter hash, and probe hash. Execution
rejects any mismatch instead of regenerating those decisions.

## LangGraph Orchestration

The first executable graph is cache-first and orchestrates verified deterministic artifacts:

```text
project_context_node
-> artifact_memory_node
-> data_audit_node (dataset composition and target summary)
-> representation_design_node (train-only frozen RepresentationSpec)
-> probe_setup_node (frozen probe selection, then representativeness audit)
-> optional controlled_probe_node (validated LLM Probe variants, deterministic comparison)
-> scout_node
-> scientific_critic_node
-> acceptance_selection_node (train/test) OR feature_planning_node (other routed modes)
-> cached feature reuse OR feature_tool_node
-> feature_qc_node
-> filtration_audit_node
-> feature_outlier_diagnostic jobs when sample-level norm outliers are present
-> evaluation_node
-> report_node
```

The graph first freezes Scout's priority over the complete requested invariant universe. For train/test
data, the acceptance node examines cumulative invariant-acquisition prefixes, computes only the next
missing full-train and acceptance feature family, and then scores all subsets available at no additional
feature cost. A compatible `TARGET_REACHED` result stops `satisfy_target`; a
`MAXIMIZATION_COMPLETE` result stops either maximization mode. Otherwise the next prefix is planned,
followed by prefix-scoped QC, filtration audit, and fixed-GBT evaluation. For three-way splits, the
validation selector remains separate and the frozen selected subset is queried on test once.
This prevents later invariant families from being computed before the preceding stage has shown that
they are needed.

For a new dataset, the first three preparation nodes are executable lifecycle stages. The graph emits
exactly one missing setup job at a time: dataset audit, representation design, probe selection, then
probe audit. Each continuation validates and registers the output before the next dependency is
planned. Dataset-audit `REVIEW` stops for an explicit schema-policy decision; probe-audit warnings are
recorded but do not silently change the frozen probe.

When scientific LLM planning is enabled in advisory mode, the graph may materialize a controlled
experiment matrix. The LLM can propose at most two bounded Probe-setting variants and at most two
bounded representation-setting variants from hard-coded allowlists. The deterministic validator injects
the baseline, rejects unsupported values, records hashes, and keeps test evidence unavailable. Probe
variants are compared first using train-only representativeness audits, not GBT performance. Only after a
single Probe is frozen may representation variants be compared using shared-Probe, label-free coverage,
filtration, feature-health, and dimensional evidence. The LLM therefore proposes hypotheses and bounded
experiment variants, but receives no numerical ranking
weight. Deterministic Python still performs validation, budget checks, artifact registration, QC, and
ranking. Scientific LLM calls use a configurable `llm_scientific.timeout_seconds`; an API or validation
failure produces an audited deterministic fallback without blocking the lifecycle. Only validated LLM
outputs are cached, so a transient timeout can be retried by a later lifecycle continuation.

After preparation, `scout_node` emits one independent probe feature job for every missing requested
invariant. Once all exact probe feature artifacts are registered, it emits one joint feature-QC job,
then one joint filtration-audit job. Both analysis reports must reproduce the frozen probe's selection
hash and sample order as well as the dataset, evidence scope, representation hash, invariant set, and
input-manifest paths. Only after both gates certify the current probe artifacts may the graph emit
per-invariant Scout OOF jobs. A sample-level robust norm warning adds a read-only diagnostic job before
Scout. Reuse requires the exact QC hash, probe selection, sample order, representation, and invariant;
the diagnostic neither removes the sample nor changes the representation.

The graph is deliberately read-only with respect to scientific artifacts. If compatible probe
features are missing, it returns `NEEDS_SCOUT_FEATURES` with auditable job plans; the following states
are `NEEDS_SCOUT_QC`, `NEEDS_SCOUT_FILTRATION_AUDIT`, optional `NEEDS_FEATURE_DIAGNOSTIC`, and
`NEEDS_SCOUT_OOF`. Heavy work is submitted to
compute nodes by the lifecycle controller rather than performed on a login node. LangGraph's
in-process checkpointer holds workflow state for one run, while the SQLite artifact registry is the
persistent cross-run scientific memory. No LLM participates in numeric decisions in this phase.

## Persistent Lifecycle Controller

`mint_scout.run_agent_lifecycle` connects the read-only graph to the existing audited Slurm job
manager without moving numerical work into the graph. A lifecycle JSON file is the durable state
machine. Each stage records its graph report, immutable job-plan signature, ledger, scheduler attempts,
registration result, and optional continuation job. The lifecycle configuration and thread identity
are frozen at creation.

With execution disabled, the controller writes a graph report and `PLANNED` ledger only. With explicit
execution and auto-continuation enabled, it submits the stage jobs and a short controller job depending
on all of them via Slurm `afterany`. The continuation refreshes scheduler state, verifies outputs,
registers compatible artifacts, and invokes the graph again. It exits after scheduling the next stage,
so no allocation waits idly for a long feature or GBT job.

`stop_after_stage` is an optional zero-based safety boundary for bounded integration runs. After the
selected job batch is complete and reconciled, the controller clears the active ledger and records
`PAUSED_AFTER_STAGE` before another graph evaluation can plan expensive work. Advancing later without
that boundary resumes from the persisted state.

Scheduler failures, missing outputs, reconciliation errors, exhausted retry limits, and repeated
already-registered plans stop in `NEEDS_REVIEW`. Failure resubmission is opt-in and the lifecycle checks
the configured attempt bound before every retry; a failed job is never handled by the initial-submit
path. A graph `COMPLETE` or exhausted `TARGET_NOT_REACHED` state is terminal. The generic Slurm wrapper
receives paths and module initialization through environment variables; a cluster-specific execution
profile may override the lifecycle script, as the Sapelo2 profile does.

`feature_tool_node` remains generic. It calls registered feature tools such as `PH`, `PL`, `FPRC`, `EIC`, and `CA`. New mathematical methods should add a wrapper, registry entry, config schema, and tests without changing the graph shape.

With an execution profile, the feature tool node emits scheduler-neutral job metadata plus an
auditable Slurm command. Submission is a separate explicit side effect managed by
`mint_scout.manage_jobs`. Its persistent JSON ledger supports dry-run review, submission, scheduler
status refresh, manifest validation, and selective resume. Scheduler completion alone is insufficient:
the expected feature manifest or QC report must exist and parse cleanly before the job is treated as
complete. Feature reconciliation checks the invariant against the immutable job-plan identity and
verifies every declared feature output path. QC reconciliation checks dataset, evidence scope,
representation hash, invariant set, input manifests, and sample order. Only then does the job manager
register the artifact in persistent memory. The graph remains read-only; after this explicit registry
write, a new graph invocation can continue at the next node.

The Scout node uses the same lifecycle in two stages. It emits one independent OOF job for each
missing invariant, reuses compatible registered OOF reports, and emits one prediction-level combine
job only after the requested set is complete. Final Scout execution artifacts require an exact
invariant-set match, because a queue frozen for five methods is not the same candidate pipeline as a
PL-only queue.

Feature QC and filtration audit must match the active feature plan's evidence scope. In particular, a
probe-only filtration audit cannot satisfy a full-train graph run, even when the invariant and
representation hash match.

Feature QC answers whether generated arrays are structurally and numerically usable: file presence,
shape, finite values, sample-level sparsity and magnitude, robust norm outliers, and population-level
all-zero or constant coordinates. The following filtration audit separately evaluates behavior along
the frozen scale axis, including sparse/saturated tails, active boundaries, and the number and last
location of effective transitions. Thresholds are configuration values and warnings never mutate a
frozen representation automatically.

`feature_health` is a deterministic reporting layer over those two artifacts. It validates their
scientific identity, classifies errors as blocking, and translates warnings into sample-review or
separately hashed filtration-comparison actions. It never removes a sample and never edits the active
`RepresentationSpec`; `EXTEND` or `SHORTEN` means design a new comparison spec, not rewrite the current
run. This summary is explanation and control metadata, not an LLM decision.

`feature_outlier_diagnostic` is the executable explanation behind a sample-review action. It compares
the flagged array with the complete frozen population along filtration, element-pair, and component
axes. FPRC component labels come from the audited legacy implementation. When protein/ligand source
paths exist, the report also records the dominant pair's graph size, edge density, degrees, and Forman
edge-curvature range. Its output is persistent scientific memory, keyed to the certifying QC hash.

The acceptance node emits a registered no-CV GBT job for each required acquisition stage when compatible
evidence is absent. Each later stage names the immediately preceding report and reuses its per-invariant
predictions. Reconcilers require exact feature manifests, a certifying QC report, the frozen Scout
artifact, representation hash, GBT hash, sample order, and selection objective. The general evaluation
node continues to emit the routed full-CV job where that protocol applies. Additional runners such as
`mlp_v1`, `gnn_v1`, small-molecule GBT, or future ensembles remain separate candidate pipelines and
must declare their own input contracts rather than inheriting GBT feature rankings.
