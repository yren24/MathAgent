#!/bin/bash
#SBATCH --job-name=mint-agent-probe-ann
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:P100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH=${MINT_AGENT_ROOT}/src
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ANN_CONFIG=${ANN_CONFIG:-configs/ann/probe_mlp_v1.yaml}
REPRESENTATION_SPEC=${REPRESENTATION_SPEC:-/path/to/workdir/mathagent/runs/representation_train_offset-0_limit-all_47823391.json}
PROBE_SELECTION=${PROBE_SELECTION:-/path/to/workdir/mathagent/runs/probe_selection_47979548.json}
FEATURE_QC_REPORT=${FEATURE_QC_REPORT:-/path/to/workdir/mathagent/runs/feature_qc_47983592.json}
RUN_ROOT=${RUN_ROOT:-/path/to/workdir/mathagent/runs}

PH_MANIFEST=${PH_MANIFEST:-/path/to/workdir/mathagent/runs/feature_batch_PH_train_offset-0_limit-all_47979665.jsonl}
PL_MANIFEST=${PL_MANIFEST:-/path/to/workdir/mathagent/runs/feature_batch_PL_train_offset-0_limit-all_47979666.jsonl}
CA_MANIFEST=${CA_MANIFEST:-/path/to/workdir/mathagent/runs/feature_batch_CA_train_offset-0_limit-all_47979668.jsonl}
FPRC_MANIFEST=${FPRC_MANIFEST:-/path/to/workdir/mathagent/runs/feature_batch_FPRC_train_offset-0_limit-all_47979669.jsonl}
EIC_MANIFEST=${EIC_MANIFEST:-/path/to/workdir/mathagent/runs/feature_batch_EIC_train_offset-0_limit-all_47979670.jsonl}

mkdir -p /path/to/workdir/mathagent/logs "${RUN_ROOT}"
cd "${MINT_AGENT_ROOT}"

module purge
module load PyTorch/2.9.1-foss-2024a-CUDA-12.6.0
module load Python-bundle-PyPI/2024.06-GCCcore-13.3.0
module load scikit-learn/1.6.1-gfbf-2024a
module load PyYAML/6.0.2-GCCcore-13.3.0

"${PYTHON_BIN}" -m mint_scout.run_casf_ann_probe \
  --task-config configs/tasks/casf2016_sapelo2.yaml \
  --ann-config "${ANN_CONFIG}" \
  --representation-spec "${REPRESENTATION_SPEC}" \
  --probe-selection "${PROBE_SELECTION}" \
  --feature-qc-report "${FEATURE_QC_REPORT}" \
  --feature-manifest "PH=${PH_MANIFEST}" \
  --feature-manifest "PL=${PL_MANIFEST}" \
  --feature-manifest "CA=${CA_MANIFEST}" \
  --feature-manifest "FPRC=${FPRC_MANIFEST}" \
  --feature-manifest "EIC=${EIC_MANIFEST}" \
  --output "${RUN_ROOT}/probe_ann_${SLURM_JOB_ID}.json"
