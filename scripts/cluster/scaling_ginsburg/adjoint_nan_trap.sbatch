#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=adj_nantrap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/adjoint_horizon_sweep/logs/%x_%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$REPO"
mkdir -p results/adjoint_horizon_sweep/logs

# Name the op that creates the NaN, at the horizon where it appears.
#
# The June 2026 attempt at this (jobs 8533824 / 8533838) piped its output
# through `| tail -70`, which discarded exactly the part that matters: JAX
# prints the PRE-TRANSFORMATION stack — the line in OUR source that built the
# offending operation — ABOVE the long internal traceback, so tailing the log
# kept 70 lines of jax internals and threw the answer away.  Both surviving
# logs begin mid-JAX-frame for that reason and the op was never identified.
# NOTHING here is piped or tailed.
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_DEBUG_NANS=1
export JAX_TRACEBACK_FILTERING=off

echo "=== git SHA: $(git -C "$REPO" rev-parse HEAD)"
for cfg in receipt today; do
  case "$cfg" in
    receipt) EXTRA="--convection sbm --radiation gray --microphysics none" ;;
    today)   EXTRA="" ;;
  esac
  echo "############ TRAP config=$cfg ############"
  # 144 steps = 24 h, the horizon the original receipt names.  The trap raises
  # on the first NaN, so this job is EXPECTED to exit non-zero; the traceback
  # is the deliverable, not the exit code.
  "$PY_GPU" -u scripts/validate/adjoint_horizon_sweep.py \
    --variants classical --n-lat 32 --n-lev 8 \
    --wrt ic --horizons 144 --cli-args "$EXTRA" \
    --out "results/adjoint_horizon_sweep/trap_${SLURM_JOB_ID}_${cfg}"
  echo "---- config=$cfg exit=$? ----"
done
