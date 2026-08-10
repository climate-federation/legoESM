#!/bin/bash
# Lock exchange (Petersen et al. 2015) across three ocean dycores, ONE arm per
# array task so each grid gets its own log and its own failure mode.
#
#   0 = fesom   (FESOM2 unstructured triangles, fesom_jax)
#   1 = mpas    (Voronoi / TRiSK)
#   2 = latlon  (structured C-grid)
#   3 = tripole (NEMO eORCA1 curvilinear C-grid + north fold)
#
# Protocol is held FIXED across arms (the controlled-comparison rule): same
# experiment, same LockExchangeConfig (H_max=20 m, nlev=20,
# land_lat_threshold=80), same DEFAULT_DT, same duration. Only the grid varies.
# The comparable number is the RPE drift (spurious diapycnal mixing), which is
# normalised per unit area and so survives the differing cell counts.
#
# Submit:  sbatch --array=0-3 scripts/cluster/lock_exchange_dycore_comparison.sbatch
#
#SBATCH --job-name=lockex-dycore
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/lockex_%A_%a.out
#SBATCH --error=logs/lockex_%A_%a.out

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
cd "$REPO"
mkdir -p logs results/lock_exchange_dycore

GRIDS=(fesom mpas latlon tripole)
GRID=${GRIDS[${SLURM_ARRAY_TASK_ID:-0}]}

# GPU (user directive 2026-08-09): A100 on the gpu partition.
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=1
# XLA spawns one thread per core and trips the per-user process cap on Levante;
# pin to the allocated cores (see the levante-xla-thread-abort note).
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

echo "=== lock_exchange | grid=$GRID | $(date -Is) ==="
git -c safe.directory='*' rev-parse HEAD || true

srun .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
    --only lock_exchange \
    --grid "$GRID" \
    --days 5 \
    --output "results/lock_exchange_dycore/$GRID"

echo "=== COMPLETED grid=$GRID rc=$? $(date -Is) ==="
