# MathAgent

MathAgent is an agentic scientific workflow for adaptive mathematical
representation selection in molecular modeling. It combines deterministic Python
pipelines, Slurm execution, cache-aware artifact tracking, and optional LLM
advisory nodes.

The current implementation supports:

- protein-ligand binding affinity workflows;
- quantitative small-molecule toxicity workflows;
- legacy mathematical feature tools exposed as isolated feature generators;
- GBT-based probe screening and final evaluation;
- optional LangGraph orchestration and LLM advisory output.

The Python package currently keeps the internal module name `mint_scout` for
backward compatibility, while the public project and CLI are named
`MathAgent`/`mathagent`.

## Core Idea

MathAgent does not ask the LLM to directly choose scientific results. The LLM is
used as a bounded advisor: it can parse a user request, suggest candidate
representation strategies, and explain the final run. The numerical choices are
validated by deterministic Python code.

The workflow keeps a persistent audit trail:

```text
user request
  -> structured request / task config
  -> data and split validation
  -> train-only adaptive representation design
  -> representative probe selection
  -> feature generation and QC
  -> probe GBT ranking and stability checks
  -> progressive full-scale candidate evaluation
  -> final report and artifact manifest
```

The main mathematical feature families are:

- `PH`: persistent homology summaries;
- `PL`: persistent Laplacian summaries;
- `CA`: commutative algebra / facet-vector summaries;
- `FPRC`: Forman-Ricci curvature summaries;
- `EIC`: element-interactive curvature summaries.

## Repository Contents

```text
src/mint_scout/        Core workflow, data, feature, model, and execution code
configs/              Template task, pipeline, execution, GBT, and request configs
scripts/slurm/        Generic Slurm wrappers
scripts/sapelo2/      Sapelo2-oriented wrappers with user-editable paths
scripts/diagnostics/  Offline diagnostic and comparison helpers
tests/                Unit and workflow tests
docs/                 Architecture notes
```

Large data files, generated feature matrices, run artifacts, Slurm logs, caches,
and API keys are intentionally not included.

## Installation

Use Python 3.10 or newer.

```bash
git clone https://github.com/yren24/MathAgent.git
cd MathAgent
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[dev,agent]"     # LangGraph workflow support
python -m pip install -e ".[dev,datasets]"  # optional dataset readers
python -m pip install -e ".[dev,ann]"       # optional ANN diagnostics
```

On HPC systems, prefer the site-provided Python modules when available. The
Sapelo2 helper scripts source `scripts/sapelo2/load_modules.sh`; edit that file
and `configs/execution/sapelo2.yaml` for your own account and scratch paths.

## External Inputs Required

MathAgent expects datasets and legacy feature implementations to be supplied
outside the repository.

For protein-ligand workflows, provide:

- a manifest or dataset provider configuration;
- protein and ligand structure paths;
- labels and train/test split columns when available;
- the legacy `embed_nn/plbind` feature tools, configured through an execution
  profile or `configs/invariants/plbind_tools.yaml`.

For toxicity workflows, provide:

- a molecule manifest;
- molecule structure directories;
- labels and train/test split columns;
- the legacy toxicity feature implementation path if using the wrapped legacy
  generator.

## API Key Setup

LLM features are optional. Copy the example file and fill it locally:

```bash
cp .env.example .env
```

or create a user-level file:

```bash
mkdir -p ~/.config/mathagent
cp .env.example ~/.config/mathagent/openai.env
```

Do not commit real API keys.

## Running From a Structured YAML Request

Start from an editable request file:

```bash
mathagent start \
  --request configs/requests/generic_protein_ligand.example.yaml \
  --output-dir runs/intake/my-run
```

To submit the generated pipeline through the configured Slurm profile:

```bash
mathagent start \
  --request configs/requests/generic_protein_ligand.example.yaml \
  --output-dir runs/intake/my-run \
  --execute
```

The command writes a resolved request, task config, pipeline config, preflight
report, and launch plan in the output directory.

## Running From a Natural-Language Prompt

Natural-language intake requires an OpenAI-compatible API key in the environment
and a model name.

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.6-terra"

mathagent start \
  --prompt-file /path/to/user_prompt.txt \
  --model "$OPENAI_MODEL" \
  --execution-profile configs/execution/template_hpc.yaml \
  --manifest-path /path/to/data/manifest.csv \
  --output-dir runs/intake/natural-request
```

To also allow LLM scientific advisory for representation/probe strategy:

```bash
mathagent start \
  --prompt-file /path/to/user_prompt.txt \
  --model "$OPENAI_MODEL" \
  --llm-scientific-mode advisory \
  --llm-scientific-model "$OPENAI_MODEL" \
  --llm-scientific-env-file ~/.config/mathagent/openai.env \
  --execution-profile configs/execution/template_hpc.yaml \
  --manifest-path /path/to/data/manifest.csv \
  --output-dir runs/intake/natural-request \
  --execute
```

The advisory output is treated as hypothesis generation. Deterministic
validation still enforces split boundaries, data availability, feature QC,
allowed methods, candidate ranking rules, and stopping policy.

## Running a Pipeline Directly

Use `mathagent` or the module entry point:

```bash
mathagent plan --config configs/pipelines/generic_protein_ligand.example.yaml
mathagent submit --config configs/pipelines/generic_protein_ligand.example.yaml --execute
mathagent status --config configs/pipelines/generic_protein_ligand.example.yaml
mathagent resume --config configs/pipelines/generic_protein_ligand.example.yaml --execute
mathagent report --config configs/pipelines/generic_protein_ligand.example.yaml
```

Equivalent module form:

```bash
python -m mint_scout.run_pipeline status \
  --config configs/pipelines/generic_protein_ligand.example.yaml
```

## Sapelo2 Launcher Example

Edit `configs/execution/sapelo2.yaml` first:

- `paths.project_root`;
- `paths.scratch_root`;
- `tools.plbind_root`;
- module setup in `scripts/sapelo2/load_modules.sh`.

Then submit only the lightweight launcher from the login/submit node:

```bash
env PIPELINE_CONFIG=configs/pipelines/generic_protein_ligand.example.yaml \
    PIPELINE_COMMAND=submit \
    EXECUTE=1 \
  sbatch scripts/sapelo2/run_pipeline_launcher.sbatch
```

Feature generation, QC, probe evaluation, and final GBT evaluation are submitted
as compute jobs. The login node should only launch and inspect workflows.

## Candidate Ranking Policy

The ranking policy is deterministic:

1. Candidate features must pass hard QC gates: valid shape, no invalid numeric
   values, sufficient retained channel support, and acceptable filtration audit
   status.
2. Probe performance is the primary evidence under a fixed probe and fixed CV
   protocol.
3. Closely matched probe candidates are compared using bootstrap/rerun ranking
   stability.
4. If candidates remain effectively tied, lower computational cost is preferred.
5. User constraints such as allowed methods, target metric, and stopping mode are
   enforced before final evaluation.

If the user provides a target threshold, MathAgent can stop after the first
full-scale candidate satisfying the target. If the user requests
best-available mode, MathAgent evaluates the frozen promoted candidate pool and
returns the best validation/test result allowed by that protocol.

## Tests

```bash
python -m compileall src
pytest
```

Some tests use synthetic fixtures only. End-to-end HPC tests require a configured
execution profile, dataset manifests, and external feature tools.

## Security and Reproducibility

- This repository excludes raw data, generated features, caches, logs, and run
  artifacts.
- `.env`, `*.env`, and credential files are ignored.
- Example paths are placeholders and must be edited for each machine.
- Generated reports include hashes, frozen config paths, and provenance records
  so runs can be audited without mutating earlier artifacts.
