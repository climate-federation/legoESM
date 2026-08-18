#!/bin/bash
# Standard ocean-grid benchmark suite: one CASE per array task, all grids
# that case supports (latlon / mpas / fesom / tripole and the legacy
# cubed_sphere / spectral where registered).
#
# Submit:  sbatch --array=0-8 scripts/cluster/ocean_grid_benchmark_suite.sbatch
#
# Then:    .venv/bin/python scripts/plot/plot_ocean_grid_benchmark_maps.py
#          -> results/ocean_grid_benchmark/maps/{maps_<case>,maps_all_cases}.{png,pdf}
#
# Runs from the checkout the SCRIPT lives in, not a hardcoded path, and pins
# PYTHONPATH to that same checkout. Both matter: the venv is an editable
# install pointing at ONE working copy, so a job launched from a worktree
# without this ran that worktree's scripts against a DIFFERENT copy's library
# code. When that other copy had uncommitted work in progress, seven arms
# failed with an error that belonged to somebody else's branch.
#
# The tripole arm needs data/grids/eORCA1.2_mesh_mask.nc (~480 MB). It is not
# in git; symlink it from a checkout that has it, or those arms error and the
# figure shows an empty column.
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
# The checkout this script belongs to, so a worktree runs ITS OWN code.
# Override with REPO=... only when you mean to.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"; mkdir -p logs
# Pin the library to the same checkout; see the note at the top.
PYTHONPATH="$(printf '%s:' "$REPO"/packages/*/ )$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
# FAIL LOUDLY if REPO resolved somewhere unexpected. Some SLURM sites copy the
# submitted script to a spool directory, which makes BASH_SOURCE point away
# from the checkout; the glob would then match nothing, a literal-star path
# would sit in PYTHONPATH, Python would ignore it in silence, and the job would
# quietly run the editable install's copy again -- i.e. the exact incident this
# guard exists to prevent, reappearing with no symptom.
[[ -d "$REPO/packages" && -n "$(echo "$REPO"/packages/*/)" ]] || {
    echo "FATAL: REPO=$REPO has no packages/ (BASH_SOURCE=${BASH_SOURCE[0]})" >&2
    exit 2; }
# And PROVE the pin won: print where the library actually resolved from. The
# string being right does not mean the import obeys it -- an editable install
# that used a meta-path finder would beat PYTHONPATH silently.
python_bin="${PYTHON_BIN:-$REPO/.venv/bin/python}"
"$python_bin" - <<'PYCHK' || exit 2
import sys, legoesm.ocean.dynamics.barotropic_mpas as m
import os
want = os.environ["REPO"]
print(f"[provenance] legoesm.ocean resolves to {m.__file__}", flush=True)
if not m.__file__.startswith(want):
    sys.exit(f"FATAL: library came from OUTSIDE {want}; the PYTHONPATH pin "
             f"did not win. Refusing to run against another checkout.")
PYCHK
CASES=(rest_state_stratified_with_land rest_state_uniform_with_land \
       rest_state_stratified_no_land rest_state_uniform_no_land \
       geostrophic_adjustment phillips_two_layer inertia_gravity_wave \
       barotropic_wave lock_exchange)
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
srun .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
    --only "$CASE" "${GRID_ARG[@]}" \
    --output "results/ocean_grid_benchmark/$CASE"
echo "=== COMPLETED $CASE rc=$? $(date -Is) ==="
