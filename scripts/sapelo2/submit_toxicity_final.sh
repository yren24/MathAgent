#!/bin/bash

set -euo pipefail

MINT_AGENT_ROOT=${MINT_AGENT_ROOT:-/path/to/MathAgent}
cd "${MINT_AGENT_ROOT}"

test_job=$(sbatch --parsable scripts/slurm/run_toxicity_final_tests.sbatch)
plan_job=$(sbatch --parsable --dependency="afterok:${test_job}" \
  scripts/slurm/run_toxicity_final_plan.sbatch)
feature_job=$(sbatch --parsable --dependency="afterok:${plan_job}" \
  --array=0-9%10 scripts/slurm/run_toxicity_final_feature_array.sbatch)
gbt_job=$(sbatch --parsable --dependency="afterok:${feature_job}" \
  --array=0-4%5 scripts/slurm/run_toxicity_final_gbt_array.sbatch)
combine_job=$(sbatch --parsable --dependency="afterok:${gbt_job}" \
  scripts/slurm/run_toxicity_final_gbt_combine.sbatch)

printf 'test_job=%s\n' "${test_job}"
printf 'plan_job=%s\n' "${plan_job}"
printf 'feature_job=%s\n' "${feature_job}"
printf 'gbt_job=%s\n' "${gbt_job}"
printf 'combine_job=%s\n' "${combine_job}"
printf 'final_report=%s\n' \
  '/path/to/workdir/mathagent/toxicity/final-v1/gbt/toxicity_final_test_report.json'
