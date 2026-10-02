# MathAgent

**Multi-Agentic Optimization of Mathematical Invariants for Molecular Prediction**

MathAgent is a scientific workflow framework for selecting mathematical
invariant representations for molecular prediction tasks. It combines optional
LLM advisory, deterministic Python validation, cached feature artifacts, and
full-scale model evaluation behind an auditable command-line interface.

The current public release focuses on two task families:

- protein-ligand binding affinity regression;
- quantitative small-molecule toxicity regression.

Raw datasets, generated feature matrices, Slurm logs, private paths, and API
credentials are intentionally excluded from this repository.

## What MathAgent Does

MathAgent is designed for settings where the best mathematical representation
depends on the dataset, task, split, and metric. A user can specify the
prediction objective, allowed invariant families, desired metric, execution
mode, and cluster profile. The workflow then:

1. parses the request into a structured task configuration;
2. audits the dataset and split without using held-out labels for design;
3. proposes and validates adaptive representation settings;
4. selects a representative training subsample for low-cost candidate
   screening;
5. computes requested mathematical invariant blocks through external tool
   adapters;
6. ranks method combinations with GBT-based screening evidence;
7. evaluates promoted full-scale candidates under the requested stopping mode;
8. writes reports, hashes, configs, and decision traces for reproducibility.

LLM output is treated as a proposal, not as scientific evidence. Python
validators enforce schema constraints, leakage guards, feature-quality checks,
and deterministic ranking rules before any full-scale evaluation is launched.

## Agentic Workflow

MathAgent follows the terminology used in the manuscript:

| Agent | Role | Main output |
|-------|------|-------------|
| **DataAgent** | Understand the user request, task type, metric, labels, split, and dataset manifest. | Structured request, dataset audit, run configuration. |
| **DesignAgent** | Generate and validate representation settings using train-only structural statistics and optional LLM proposals. | Frozen representation specification and advisory record. |
| **ScreenAgent** | Choose a representative training subsample, compute low-cost evidence, and rank invariant combinations. | Candidate ranking, stability evidence, promoted candidates. |
| **TestAgent** | Reuse cached full-scale features, evaluate frozen candidates, and stop according to the selected policy. | Final metrics, selected combination, stop reason, report. |

The implementation is agentic in the workflow sense: specialized stages share a
structured state, exchange validated artifacts, and can resume after jobs finish
on an HPC system. The public code uses LangGraph-compatible organization, while
the scientific decisions remain reproducible Python logic.

## Mathematical Tool Layer

MathAgent wraps existing mathematical feature implementations instead of
rewriting them inside the workflow.

| Code | Representation family | Scientific signal |
|------|------------------------|-------------------|
| `PH` | Persistent homology | Multiscale topological summaries. |
| `PL` | Persistent Laplacian | Topological-spectral summaries. |
| `CA` | Commutative algebra / facet-vector descriptors | Algebraic and combinatorial structure. |
| `FPRC` | Forman persistent Ricci curvature | Multiscale discrete curvature summaries. |
| `EIC` | Element-interactive curvature | Element-resolved geometric and curvature summaries. |

The adapters record feature schema, sample order, representation hash, external
tool path, and provenance metadata so cached artifacts are reused only when they
match the active configuration.

## Execution Modes

MathAgent supports two main user-guided modes.

| Mode | Behavior |
|------|----------|
| **Target-constrained** | Evaluate full-scale candidates in frozen screening order and stop when the user-specified metric threshold is reached. |
| **Best-available** | Evaluate a fixed top-k candidate pool selected before full-scale evaluation and return the best result in that pool. |

Both modes keep the candidate order fixed before full-scale evaluation. For
protected benchmark comparisons, the held-out test set is evaluated once after
selection. For user-facing acceptance workflows, a user-provided acceptance set
can be used to decide whether the requested target has been met.

## Installation

Use Python 3.10 or newer.

```bash
git clone https://github.com/yren24/MathAgent.git
cd MathAgent
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[dev,agent]"     # LangGraph support
python -m pip install -e ".[dev,datasets]"  # optional dataset readers
```

## API Keys

LLM features are optional. Copy the example file and fill it locally:

```bash
cp .env.example .env
```

or use a user-level credential file:

```bash
mkdir -p ~/.config/mathagent
cp .env.example ~/.config/mathagent/openai.env
```

Do not commit real credentials.

`OPENAI_MODEL` is user configurable. Set it to any OpenAI model available to
your account and compatible with the API endpoint used by your environment.

## Quick Start

### Structured request

```bash
mathagent start \
  --request configs/requests/generic_protein_ligand.example.yaml \
  --output-dir runs/intake/my-run
```

Add `--execute` to submit the resolved workflow through the configured Slurm
profile:

```bash
mathagent start \
  --request configs/requests/generic_protein_ligand.example.yaml \
  --output-dir runs/intake/my-run \
  --execute
```

### Natural-language request

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="<your-openai-model>"

mathagent start \
  --prompt-file /path/to/user_prompt.txt \
  --model "$OPENAI_MODEL" \
  --execution-profile configs/execution/template_hpc.yaml \
  --manifest-path /path/to/data/manifest.csv \
  --output-dir runs/intake/natural-request
```

Enable LLM scientific advisory:

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

### Direct pipeline launcher

```bash
mathagent plan --config configs/pipelines/generic_protein_ligand.example.yaml
mathagent submit --config configs/pipelines/generic_protein_ligand.example.yaml --execute
mathagent status --config configs/pipelines/generic_protein_ligand.example.yaml
mathagent resume --config configs/pipelines/generic_protein_ligand.example.yaml --execute
mathagent report --config configs/pipelines/generic_protein_ligand.example.yaml
```

Equivalent module entry point:

```bash
python -m mint_scout.run_pipeline status \
  --config configs/pipelines/generic_protein_ligand.example.yaml
```

## HPC Usage

MathAgent is cluster agnostic. Start from the generic profile and adapt it to
your own scheduler environment:

```text
configs/execution/template_hpc.yaml
scripts/slurm/
```

Edit at least:

- `paths.project_root`;
- `paths.scratch_root`;
- external mathematical tool paths;
- module setup;
- partition, memory, CPU, and wall-time limits.

The `scripts/sapelo2/` directory is a site-specific example for UGA Sapelo2.
It is useful as a template, but users on other clusters should copy the generic
profile and keep machine-specific edits local.

## Reproduction Notes

For a clone-to-run checklist, see [REPRODUCE.md](REPRODUCE.md).

The public repository supports:

- package import and CLI entry points;
- request validation and config generation;
- deterministic pipeline planning;
- Slurm job-plan construction;
- unit tests using synthetic fixtures.

Full scientific runs require user-supplied inputs:

- molecular structures or molecule files;
- labels and split files;
- external mathematical feature-tool directories;
- a configured execution profile;
- optional LLM credentials.

## Repository Layout

```text
MathAgent/
├── README.md
├── REPRODUCE.md
├── pyproject.toml
├── figures/                  Manuscript-style framework and user-guide figures
├── configs/                  Request, task, representation, GBT, and HPC configs
├── scripts/
│   ├── slurm/                Generic Slurm wrappers
│   ├── sapelo2/              Optional site-specific launcher examples
│   └── diagnostics/          Offline comparison helpers
├── src/mint_scout/           Core Python package
│   ├── agent/                LLM intake, advisory, explanation, graph helpers
│   ├── data/                 Dataset manifests, splits, geometry, preparation
│   ├── execution/            Artifact registry, Slurm jobs, lifecycle engine
│   ├── invariants/           Legacy mathematical feature tool adapters
│   ├── models/               GBT model utilities
│   ├── scout/                Candidate screening, ranking, and target policy
│   ├── toxicity/             Small-molecule toxicity workflow
│   └── mof/                  Experimental MOF workflow utilities
└── tests/                    Unit and workflow tests
```

The package name `mint_scout` is retained for backward compatibility with the
development code. The public CLI is `mathagent`.

## Important Modules

| Workflow concept | Representative code |
|------------------|---------------------|
| User request parsing | `src/mint_scout/user_intake.py`, `src/mint_scout/agent/llm_intake.py` |
| LLM scientific advisory | `src/mint_scout/agent/llm_scientific.py`, `src/mint_scout/toxicity/advisory.py` |
| Representation design | `src/mint_scout/design_representation.py`, `src/mint_scout/representation_design.py` |
| Training-subsample screening | internal selection and audit modules |
| Feature tools and QC | `src/mint_scout/invariants/`, `src/mint_scout/feature_qc.py`, `src/mint_scout/filtration_audit.py` |
| Progressive evaluation | `src/mint_scout/evaluate_acceptance_gbt.py`, `src/mint_scout/evaluate_frozen_test_gbt.py` |
| Slurm orchestration | `src/mint_scout/run_pipeline.py`, `src/mint_scout/execution/` |

## Tests

```bash
python -m compileall src
pytest
```

Some tests use synthetic fixtures only. Full HPC runs require configured data,
external feature tools, and a Slurm execution profile.

## Security and Data Policy

- Raw datasets, generated features, caches, logs, and run artifacts are excluded.
- `.env`, `*.env`, and local credential files are ignored.
- Example paths are placeholders and must be edited for each machine.
- External mathematical feature tools are referenced through user-configurable paths.
- Generated reports preserve hashes, frozen config paths, and provenance records
  for auditability.
