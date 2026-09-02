#!/bin/bash
# Calibrated-vs-calibrated tuner: two-leaf +/- P-hydro at the 5 dry EC sites.
# Protocol: docs/land/phydro_calibrated_comparison_prereg.md
#
# Array task = one (arm, init-id): 0-5 beta_theta init 0..5, 6-11 phydro init 0..5.
#   TUNE:  sbatch scripts/cluster/ec_site/run_tune_stress_arms.sbatch
#   EVAL:  MODE=evaluate sbatch --array=0-13 scripts/cluster/ec_site/run_tune_stress_arms.sbatch
#          (evaluate array = 7 cells per arm: init -1 (untuned defaults) .. 5)
#   Then:  python scripts/run/tune_ec_stress_arms.py --verdict   (cheap, login OK)
# Nelder-Mead budgets (prereg deviation 2): EVALS_A=200 (4 params) EVALS_B=400
# (8 params), overridable through the environment.
#SBATCH --account=glab --partition=glab1 --time=72:00:00 --cpus-per-task=16 --mem=64G
#SBATCH --job-name=tune_stress --array=0-11
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land/scripts/tmp/_tune_stress_%a.log
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-land
cd "$WT"
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
MODE=${MODE:-tune}
ARMS=(beta_theta phydro)
EVALS_BY_ARM=(${EVALS_A:-200} ${EVALS_B:-400})

if [ "$MODE" = evaluate ]; then
  ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 7))]}
  INIT=$((SLURM_ARRAY_TASK_ID % 7 - 1))
  echo "=== evaluate $ARM init $INIT  HEAD=$(git rev-parse --short HEAD) ==="
  $PY scripts/run/tune_ec_stress_arms.py --arm "$ARM" --init-id "$INIT" --evaluate
else
  ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 6))]}
  EVALS=${EVALS_BY_ARM[$((SLURM_ARRAY_TASK_ID / 6))]}
  INIT=$((SLURM_ARRAY_TASK_ID % 6))
  echo "=== tune $ARM init $INIT evals $EVALS  HEAD=$(git rev-parse --short HEAD) ==="
  $PY scripts/run/tune_ec_stress_arms.py --arm "$ARM" --init-id "$INIT" --evals "$EVALS"
fi
echo "rc=$? TUNE-STRESS-DONE"
