#!/bin/bash
#SBATCH --job-name=mint-agent-feature-batch
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export TASK_CONFIG=${TASK_CONFIG:-configs/tasks/casf2016_sapelo2.yaml}
export RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}
export CACHE_ROOT=${CACHE_ROOT:-/path/to/workdir/mathagent/cache}
export LOG_ROOT=${LOG_ROOT:-/path/to/workdir/mathagent/logs}
export PYTHON_BIN=${PYTHON_BIN:-python}
export MINT_MODULES=${MINT_MODULES:-"Python-bundle-PyPI/2025.04-GCCcore-14.2.0:scikit-learn/1.7.0-gfbf-2025a:PyYAML/6.0.2-GCCcore-14.2.0"}
export INVARIANT=${INVARIANT:-PL}
export SPLIT=${SPLIT:-all}
export LIMIT=${LIMIT:-16}

bash "${MINT_AGENT_ROOT}/scripts/slurm/run_feature_batch.sbatch"
