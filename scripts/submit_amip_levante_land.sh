#!/bin/bash
#SBATCH --job-name=legoesm-amip-land
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/work/bd1083/b309178/logs/legoesm-amip-land-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/legoesm-amip-land-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# legoESM AMIP — 30-day land-tile VALIDATION run (Levante GPU node)
#
# First run with the interactive slab-land surface tile active.  Uses the
# CMIP6 MPI-ESM1-2-LR land-sea mask (sftlf) regridded to the C16 grid, so
# continents get realistic land albedo + a prognostic land skin
# temperature instead of being treated as ocean.
#
# Purpose: confirm the slab-land tile runs cleanly end-to-end (compile,
# scan, checkpoint) and that T_mean stays sane.  Compare against the
# ocean-only radiation-fix validation (job 25058615: day 30 = 273.8 K).
#
# Usage:
#   sbatch scripts/submit_amip_levante_land.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python

# --- run parameters ---
YEAR=1979
DAYS=30
DT=150
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
OUTPUT=/scratch/b/b309178/amip_${YEAR}_land_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM AMIP 30-day LAND validation"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Year:      $YEAR  Days: $DAYS  dt: ${DT}s"
echo "Land mask: $LAND_MASK"
echo "Output:    $OUTPUT"
echo "Started:   $(date)"
echo "=============================="

# --- environment ---
source /work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh
conda activate diffesm

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo ""
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo ""

cd "$REPO"

$PYTHON scripts/run_amip_levante.py \
    --year "$YEAR" \
    --days "$DAYS" \
    --dt "$DT" \
    --ic-zarr "$IC_ZARR" \
    --land-mask-file "$LAND_MASK" \
    --diag-days 5 \
    --checkpoint-days 30 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
