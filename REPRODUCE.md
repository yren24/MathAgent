# Reproducing MathAgent

This page gives a clone-to-run checklist. There are three tiers:

- **A - Inspect and test the code**: no datasets, no API key.
- **B - Run request intake and planning**: optional LLM API key.
- **C - Run full scientific workflows**: external datasets, legacy feature tools,
  and a configured Slurm/HPC profile.

---

## A. Inspect and test the code

```bash
git clone https://github.com/yren24/MathAgent.git
cd MathAgent
python -m pip install -e ".[dev]"
python -m compileall src
pytest
```

The full test suite includes workflow and Slurm-planning tests. End-to-end
feature generation requires user-provided external tools and data.

---

## B. Run request intake and pipeline planning

### B1. Structured YAML request

```bash
mathagent start \
  --request configs/requests/generic_protein_ligand.example.yaml \
  --output-dir runs/intake/example
```

Expected outputs under `runs/intake/example`:

- resolved request YAML;
- generated task config;
- generated pipeline config;
- preflight report;
- launch plan.

### B2. Natural-language request

Natural-language intake requires an LLM API key.

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY and OPENAI_MODEL
set -a
. ./.env
set +a

mathagent start \
  --prompt "Use this dataset to predict binding affinity and identify the best invariant combination." \
  --model "$OPENAI_MODEL" \
  --execution-profile configs/execution/template_hpc.yaml \
  --manifest-path /path/to/data/manifest.csv \
  --output-dir runs/intake/natural-example
```

This creates the same structured artifacts as B1, but the initial request is
translated from natural language.

---

## C. Run full scientific workflows

Full workflows require data and external feature generators that are not shipped
in this repository.

### C1. Prepare external inputs

For protein-ligand workflows:

1. Prepare a manifest with sample IDs, labels, split labels, protein paths, and
   ligand paths.
2. Place protein and ligand structure files on disk.
3. Install or copy the legacy `embed_nn/plbind` feature implementation.
4. Edit `configs/execution/template_hpc.yaml` or `configs/execution/sapelo2.yaml`.
5. Set `tools.plbind_root` to the legacy feature-tool directory.

For toxicity workflows:

1. Prepare a molecule manifest with sample IDs, labels, split labels, and
   molecule identifiers.
2. Place molecule structure files on disk.
3. Configure the legacy toxicity feature-tool root if using the wrapped legacy
   implementation.
4. Edit the execution profile and task/request config.

### C2. Dry-run the pipeline plan

```bash
mathagent plan --config configs/pipelines/generic_protein_ligand.example.yaml
```

Inspect the planned paths, invariant list, execution profile, and run ID before
submitting.

### C3. Submit to Slurm

```bash
mathagent submit \
  --config configs/pipelines/generic_protein_ligand.example.yaml \
  --execute
```

or submit the lightweight Sapelo2 launcher:

```bash
env PIPELINE_CONFIG=configs/pipelines/generic_protein_ligand.example.yaml \
    PIPELINE_COMMAND=submit \
    EXECUTE=1 \
  sbatch scripts/sapelo2/run_pipeline_launcher.sbatch
```

### C4. Monitor and resume

```bash
mathagent status --config configs/pipelines/generic_protein_ligand.example.yaml
mathagent resume --config configs/pipelines/generic_protein_ligand.example.yaml --execute
mathagent report --config configs/pipelines/generic_protein_ligand.example.yaml
```

The report command reads registered immutable artifacts and renders the final
summary without recomputing feature tools.

---

## Reproducibility Notes

- The repository does not include benchmark data, generated features, caches, or
  result tables.
- Run IDs should be unique. If you intentionally repeat a run, use a new run ID
  or archive previous controlled artifacts.
- LLM advisory output is not treated as numerical evidence. Deterministic
  validators enforce allowed methods, split boundaries, representation bounds,
  feature QC, candidate ranking, and stopping policy.
- Exact end-to-end wall time depends on the cluster, feature-tool speed, probe
  size, and number of promoted candidates.
