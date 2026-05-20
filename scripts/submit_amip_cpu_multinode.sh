#!/bin/bash
#SBATCH --job-name=lego-cpu-amip
#SBATCH --account=bd1083
#SBATCH --partition=compute
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/work/bd1083/b309178/logs/lego-cpu-amip-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/lego-cpu-amip-%j.err
#SBATCH --mail-type=FAIL,END

# 30-day AMIP run on C16 cubed-sphere with RRTMG radiation, CPU compute nodes.
# Uses all CMIP6 ICON forcing channels (SST/SIC, GHG, ozone, aerosol, solar, volcanic)
# and ERA5 initial conditions.
#
# MPI topology: 6 ranks, one per cubed-sphere face.  Each rank runs on a
# dedicated compute node with 1 virtual JAX CPU device.
#
# Timing reference: C16 with RRTMG (rad-update-steps=6, so RRTMG every 3600s)
# on 6 CPU nodes ≈ 2–5 min/simulated day → 30 days in ~1–2.5 h wall time.
# After timing is known, adjust --days and resubmit (use --restart-from to chain).
#
# Grid choice vs GPU benchmarks:
#   C48 (production, 2°)  — too slow on CPU, ~50× more work/rank than C16
#   C24 (~2.7°)           — feasible but ~6× slower than C16 on CPU
#   C16 (~4°)             — good balance for CPU; comparable to a fast GPU C48 run
#   C8  (~8°)             — fallback if C16 is still too slow
#
# Usage:
#   sbatch scripts/submit_amip_cpu_multinode.sh

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
OUTPUT=/scratch/b/b309178/amip_cpu_c16_rrtmg_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM CPU AMIP C16 RRTMG"
echo "Job ID:   $SLURM_JOB_ID"
echo "Nodes:    $SLURM_JOB_NODELIST"
echo "NTASKS:   $SLURM_NTASKS"
echo "Output:   $OUTPUT"
echo "Started:  $(date)"
echo "=============================="

source /work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh
conda activate diffesm

cd "$REPO"

# CPU-specific JAX settings.
#   JAX_PLATFORMS=cpu   — force CPU backend; avoid failed CUDA initialisation.
#   XLA_FLAGS           — give each MPI rank exactly 1 virtual CPU device so
#                         the cubed-sphere face decomposition maps cleanly
#                         (1 face per rank, 1 device per rank).
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JAX_ENABLE_X64=1

$PYTHON scripts/run_amip_levante.py \
    --year 1979 \
    --days 30 \
    --dt 600 \
    --resolution 16 \
    --nlev 40 \
    --ic-zarr "$IC_ZARR" \
    --radiation rrtmg \
    --rad-update-steps 6 \
    --diag-days 5 \
    --checkpoint-days 30 \
    --cmip-output \
    --monthly-means \
    --distributed \
    --n-ranks 6 \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
