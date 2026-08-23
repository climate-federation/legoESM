#!/bin/bash
# Cross-grid consistency for the ocean benchmark suite, on a GPU node.
#
# The default (artifact-reading) mode of scripts/validate/ocean_grid_consistency.py
# runs no model and belongs on a login node. Its --self-error, --refinement
# and --refined-budget modes DO run the model -- two runs per arm per case,
# and four at 4x resolution -- so they need a compute allocation. This
# wrapper exists so nobody reaches for the login node again.
#
# Submit:
#   sbatch scripts/cluster/ocean_grid_consistency.sbatch
#   MODE="--self-error --refinement --refined-budget" \
#     CASES=phillips_two_layer sbatch scripts/cluster/ocean_grid_consistency.sbatch
#
#SBATCH --job-name=ocn-consist
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/ocnconsist_%j.out
#SBATCH --error=logs/ocnconsist_%j.out
set -euo pipefail
# Overridable so the validator can be run from a git worktree. A worktree
# does not import itself: without PYTHONPATH the job silently exercises the
# main checkout's code.
REPO=${REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM}
VENV=${VENV:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv}
cd "$REPO"; mkdir -p logs
export PYTHONPATH="$(ls -d "$REPO"/packages/*/ | tr '\n' ':')$REPO/src"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODE=${MODE:---self-error --refinement}
RUNS_ROOT=${RUNS_ROOT:-$REPO/results/ocean_grid_benchmark}
SELF_ROOT=${SELF_ROOT:-$REPO/results/ocean_self_error}
CASE_ARG=()
if [[ -n "${CASES:-}" ]]; then CASE_ARG=(--self-error-cases "$CASES"); fi

echo "=== ocean cross-grid consistency | MODE=$MODE | $(date -Is) ==="
srun "$VENV"/bin/python scripts/validate/ocean_grid_consistency.py \
    --runs-root "$RUNS_ROOT" \
    --self-error-root "$SELF_ROOT" \
    --out "$RUNS_ROOT/cross_grid_consistency.json" \
    $MODE "${CASE_ARG[@]}" \
    | tee "$RUNS_ROOT/consistency_report.txt"
echo "=== COMPLETED rc=$? $(date -Is) ==="
