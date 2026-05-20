#!/bin/bash
#SBATCH --job-name=legoesm-amip
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/work/bd1083/b309178/logs/legoesm-amip-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/legoesm-amip-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# legoESM AMIP run — Levante GPU node
#
# Timing reference (C48/L40, dt=150s, RRTMG, 4×A100-80GB, --production-profile):
#   JIT first segment:  ~2650s  (warm XLA cache)
#   Per simulated day:  TBD     (SPMD halo + rad_update_steps=24 vs old 6)
#   75 days total:      est. <12h — safe within 12h limit
#
# Usage:
#   sbatch scripts/submit_amip_levante.sh
#
# To change run length, edit --days below.  80 days is the practical
# maximum for a 12h job; reduce to 30 for a quick validation run.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python

# --- run parameters ---
YEAR=1979
DAYS=75
DT=150
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
OUTPUT=/scratch/b/b309178/amip_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM AMIP run"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Year:    $YEAR  Days: $DAYS  dt: ${DT}s"
echo "Output:  $OUTPUT"
echo "Started: $(date)"
echo "=============================="

# Activate conda env so that CUDA libs are on LD_LIBRARY_PATH
source /work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh
conda activate diffesm

cd "$REPO"

$PYTHON scripts/run_amip_levante.py \
    --year "$YEAR" \
    --days "$DAYS" \
    --dt "$DT" \
    --ic-zarr "$IC_ZARR" \
    --diag-days 5 \
    --checkpoint-days 30 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
