#!/bin/bash
#SBATCH --job-name=legoesm-c96
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-c96-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-c96-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# C96 RESOLUTION PUSH — headline test of the resolution hypothesis.
# The dry Held-Suarez test (jobs 25804484/85) showed the eddy-driven jet is
# RESOLUTION-STARVED: C32 dycore makes only 2.9 m/s (vs 28 benchmark), C96 makes
# 15+ (still rising).  So a stronger circulation at C96 should improve the
# extratropical jets/transport and — via a stronger Brewer-Dobson cell — the
# tropical cold point.
#
# PHYSICS + VERTICAL GRID IDENTICAL to the C32/L45 cold-point baseline 25747041
# (hines-only GWD, NOT the orographic composite) so RESOLUTION is the only change.
# dt=75 (dry-HS proved C96 dynamics stable at dt=200; moist physics needs margin).
# Frequent checkpoints (every 3 d) for restart-chaining — one 8 h job reaches
# ~10-13 simulated days; chain to ~30 d for the jet spin-up + early cold point.
#
# Usage:  sbatch scripts/submit_amip_c96l45_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

YEAR=1979
DAYS=30
DT=75
RES=96
NLEV=45
PTOP_PA=100
STRETCH=1.2
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_c96l45_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  C${RES} / L${NLEV}  [resolution push, hines-only, dt=${DT}]"
echo "Job:    ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}   Days: $DAYS"
echo "Output: $OUTPUT   Started: $(date)"
echo "=============================="

source $CONDA/etc/profile.d/conda.sh
conda activate $ENV

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
$PYTHON -c "import jax; print(f'JAX {jax.__version__}: {jax.devices()}')" || true

cd "$REPO"

$PYTHON scripts/run_amip_levante.py \
    --year "$YEAR" \
    --days "$DAYS" \
    --dt "$DT" \
    --resolution "$RES" \
    --nlev "$NLEV" \
    --radiation rrtmg \
    --convection tiedtke \
    --turbulence holtslag_boville \
    --gravity-wave-drag hines \
    --clouds sundqvist \
    --microphysics morrison \
    --ic-zarr "$IC_ZARR" \
    --land-mask-file "$LAND_MASK" \
    --albedo-land-file "$LAND_ALBEDO" \
    --diag-days 2 \
    --checkpoint-days 3 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT" \
    --extra --aerosol-ccn --convective-precip-efficiency 0.6 \
            --vertical-coord hybrid --p-top "$PTOP_PA" --stretching "$STRETCH"

echo "Finished: $(date)"
echo "Output: $OUTPUT"
