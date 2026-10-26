#!/bin/bash
# THE WHOLE ocean test matrix: every registered case on every grid it is
# registered for. 65 (case, grid, resolution) entries across 27 case names
# and 8 grid types -- latlon, mpas, tripole, fesom, and the regional and
# channel variants of latlon and mpas.
#
# One array task per CASE NAME, not per entry, so every grid of a case runs
# in the same job and its cross-grid comparison is written together; and so
# that one long case (neverworld2_lite and isomip_plus are 365-day runs)
# cannot hold up the rest.
#
# Submit:  sbatch --array=0-26 scripts/cluster/ocean_full_matrix.sbatch
# Quick:   QUICK=1 sbatch --array=0-26 scripts/cluster/ocean_full_matrix.sbatch
#
#SBATCH --job-name=ocn-matrix
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/ocnmatrix_%A_%a.out
#SBATCH --error=logs/ocnmatrix_%A_%a.out
set -uo pipefail
# NOT -e: the matrix exits non-zero whenever any case fails its gate, and
# the point of a full sweep is to see WHICH fail. -e would abort the task
# and report one row as if it were the answer.

# Overridable so the sweep can run from a git worktree. A worktree does not
# import itself: without PYTHONPATH the job silently exercises the main
# checkout's code.
REPO=${REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM}
VENV=${VENV:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv}
cd "$REPO"; mkdir -p logs
export PYTHONPATH="$(ls -d "$REPO"/packages/*/ | tr '\n' ':')$REPO/src"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Case names in _build_test_matrix() order. Kept as a literal list rather
# than derived at submit time so the array index is reproducible across
# runs; tests/ocean/unit/test_full_matrix_case_list.py fails if the
# registry gains or loses a case name, which is what stops this drifting.
CASES=(rest_state_stratified_with_land rest_state_uniform_with_land \
       rest_state_stratified_no_land rest_state_uniform_no_land \
       barotropic_wave barotropic_double_gyre barotropic_double_gyre_sin2 \
       baroclinic_gyre baroclinic_gyre_cos global_barotropic_wind \
       global_barotropic_wind_1lev geostrophic_adjustment \
       phillips_two_layer inertia_gravity_wave \
       inertia_gravity_wave_channel lock_exchange overflow \
       stommel_gyre_tracer eady_uniform eady_instability acc_channel \
       global_overturning dino munk_gyre held_larichev neverworld2_lite \
       isomip_plus)
CASE=${CASES[${SLURM_ARRAY_TASK_ID:-0}]}

# EXACT match. Plain --only is a substring test, and this registry has
# names that are prefixes of others (barotropic_double_gyre is a prefix of
# barotropic_double_gyre_sin2, global_barotropic_wind of
# global_barotropic_wind_1lev, inertia_gravity_wave of
# inertia_gravity_wave_channel), so a substring filter would silently run
# more cases than the array task is accounted for.
QUICK_ARG=()
if [[ -n "${QUICK:-}" ]]; then QUICK_ARG=(--quick); fi

OUT=${OUT:-results/ocean_full_matrix}
echo "=== ocean FULL matrix | $CASE | $(date -Is) ==="
srun "$VENV"/bin/python scripts/matrix/run_ocean_test_matrix.py \
    --only "=$CASE" "${QUICK_ARG[@]}" --output "$OUT/$CASE"
rc=$?
echo "=== COMPLETED $CASE rc=$rc $(date -Is) ==="
