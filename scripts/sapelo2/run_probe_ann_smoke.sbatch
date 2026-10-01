#!/bin/bash
#SBATCH --job-name=mint-agent-ann-smoke
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:P100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/workdir/mathagent/logs/%x-%j.out
#SBATCH --error=/path/to/workdir/mathagent/logs/%x-%j.err

set -euo pipefail

export MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
export PYTHONPATH=${MINT_AGENT_ROOT}/src
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p /path/to/workdir/mathagent/logs
cd "${MINT_AGENT_ROOT}"

module purge
module load PyTorch/2.9.1-foss-2024a-CUDA-12.6.0
module load Python-bundle-PyPI/2024.06-GCCcore-13.3.0
module load scikit-learn/1.6.1-gfbf-2024a
module load PyYAML/6.0.2-GCCcore-13.3.0

python -c "import torch; assert torch.cuda.is_available(); print('torch=' + torch.__version__ + ' cuda=' + str(torch.version.cuda) + ' device=' + torch.cuda.get_device_name(0))"
python -m pytest -q tests/test_ann.py
