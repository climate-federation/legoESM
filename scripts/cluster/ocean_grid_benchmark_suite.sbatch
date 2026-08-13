#!/bin/bash
# Standard ocean-grid benchmark suite: one CASE per array task, all grids
# that case supports (latlon / mpas / fesom / tripole and the legacy
# cubed_sphere / spectral where registered).
#
# Submit:  sbatch --array=0-9 scripts/cluster/ocean_grid_benchmark_suite.sbatch
#SBATCH --job-name=ocn-bench
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/ocnbench_%A_%a.out
#SBATCH --error=logs/ocnbench_%A_%a.out
set -euo pipefail
# REPO defaults to the primary checkout but is overridable, so the suite can
# be run from a git worktree without editing this file. A worktree does not
# import itself: PYTHONPATH must point at ITS packages, or the run silently
# executes the main checkout's code.
REPO=${REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM}
VENV=${VENV:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv}
cd "$REPO"; mkdir -p logs
export PYTHONPATH="$(ls -d "$REPO"/packages/*/ | tr '\n' ':')$REPO/src"
CASES=(rest_state_stratified_with_land rest_state_uniform_with_land \
       rest_state_stratified_no_land rest_state_uniform_no_land \
       geostrophic_adjustment phillips_two_layer inertia_gravity_wave \
       barotropic_wave lock_exchange inertia_gravity_wave_channel)
CASE=${CASES[${SLURM_ARRAY_TASK_ID:-0}]}
# Optional single-arm rerun: GRID=<name> reruns just that grid for the case,
# leaving the other arms' artifacts in place. Used when a fix touches one
# dycore and rerunning the whole suite would change more than one variable.
GRID_ARG=()
if [[ -n "${GRID:-}" ]]; then GRID_ARG=(--grid "$GRID"); fi
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "=== ocean benchmark suite | $CASE | $(date -Is) ==="
srun "$VENV"/bin/python scripts/matrix/run_ocean_test_matrix.py \
    --only "$CASE" "${GRID_ARG[@]}" \
    --output "results/ocean_grid_benchmark/$CASE"
echo "=== COMPLETED $CASE rc=$? $(date -Is) ==="
