#!/bin/bash
#SBATCH --job-name=mint-agent-bdb-prepare
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
SOURCE_ROOT=${SOURCE_ROOT:-/path/to/workdir/mathagent/external/bdb2020plus/BDB2020+}
EXTERNAL_ROOT=${EXTERNAL_ROOT:-/path/to/workdir/mathagent/external/bdb2020plus}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "${EXTERNAL_ROOT}/prepared" "${EXTERNAL_ROOT}/runs" /path/to/workdir/mathagent/logs
cd "${MINT_AGENT_ROOT}"
module purge
module load OpenBabel/3.1.1-gompi-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
export PYTHONPATH=${MINT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

"${PYTHON_BIN}" -m mint_scout.data.bdb2020plus \
  --source-root "${SOURCE_ROOT}" \
  --source-archive "${EXTERNAL_ROOT}/BDB2020plus.tgz" \
  --output-root "${EXTERNAL_ROOT}/prepared/structures" \
  --manifest "${EXTERNAL_ROOT}/prepared/manifest.csv" \
  --audit-report "${EXTERNAL_ROOT}/runs/preparation_${SLURM_JOB_ID}.json" \
  --pocket-cutoff-angstrom 10.0
