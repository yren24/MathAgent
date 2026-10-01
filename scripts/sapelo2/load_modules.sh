#!/bin/bash
set -euo pipefail

module purge

SAPELO2_MODULES=${SAPELO2_MODULES:-"Python-bundle-PyPI/2025.04-GCCcore-14.2.0 scikit-learn/1.7.0-gfbf-2025a PyYAML/6.0.2-GCCcore-14.2.0"}

for module_name in ${SAPELO2_MODULES}; do
  module load "${module_name}"
done
