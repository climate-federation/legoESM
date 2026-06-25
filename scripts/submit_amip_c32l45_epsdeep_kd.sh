#!/bin/bash
#SBATCH --job-name=legoesm-eps
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-eps-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-eps-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# RAISED-LID / VERTICAL-RESOLUTION SWEEP, member 1: C32 / L45.
#
# Diagnosis (90d job 25739722): the SW-gpoint + convective-precip-efficiency
# fixes SOLVED the free-trop warm bias (now +3-4K). The RESIDUAL bias is
# UPPER-LEVEL: cold point +17K @100hPa, top layer runs away +0.2 K/day, no
# equilibrium. ROOT CAUSE (level audit): the CURRENT L30 hybrid grid (p_top=2hPa,
# stretching=2) puts the ENTIRE stratosphere/UTLS above 100 hPa in ONE level at
# 35 hPa (top layer spans 2->70 hPa) and only 2 levels in the TTL (200-70 hPa).
# The cold point cannot form (65-hPa gap above the 100-hPa level) and the single
# thick top layer collects the ozone SW-heating peak (~30 hPa) with no resolution
# to cool -> warm runaway. This is a vertical-resolution + level-distribution
# problem (NOT a sponge / NOT a Newtonian-T patch; the sponge is dynamical and
# CESM/ICON get a stable stratosphere from radiation alone — fix the grid+radiation).
#
# This member: nlev=45, p_top=1 hPa (100 Pa), stretching=1.2 (less surface-
# concentrated than the L30 default of 2.0 -> redistributes levels into the UTLS).
# Gives ~3 levels 100-2 hPa and ~5 levels in the TTL (200-70 hPa), top full
# level ~17 hPa. PHYSICS IDENTICAL to the 90d baseline (tiedtke + aerosol-ccn +
# convective-precip-efficiency 0.6) and ozone held at "standard" so ONLY the
# vertical grid changes vs job 25739722. (Ozone "standard" over-estimates
# lower-strat O3 ~3x — the obvious NEXT lever if the strat still runs warm once
# resolved; kept fixed here for a clean resolution comparison.)
#
# Success: tropical cold point forms toward 192K @100hPa (was +17K); top-layer
# warming rate drops toward 0; free trop stays ~+3-4K. Compare d30/d60 tropical
# T(plev) vs the 90d L30 baseline.
#
# Usage:  sbatch scripts/submit_amip_c32l45_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

# --- run parameters ---
YEAR=1979
DAYS=60
DT=200
RES=32
NLEV=45
PTOP_PA=100          # 1 hPa lid (default L30 used 200 Pa = 2 hPa)
STRETCH=1.2          # less surface-concentrated than L30 default 2.0 -> more UTLS levels
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_c32l45eps_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  C${RES} / L${NLEV}  [raised-lid sweep: p_top=${PTOP_PA}Pa, stretch=${STRETCH}]"
echo "Job:    ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}"
echo "Year:   $YEAR  Days: $DAYS  dt: ${DT}s"
echo "Output: $OUTPUT"
echo "Started: $(date)"
echo "=============================="

# --- environment ---
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

echo "--- GPUs ---"
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
    --checkpoint-days 30 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT" \
    --extra --aerosol-ccn --convective-precip-efficiency 0.6 \
            --vertical-coord hybrid --p-top "$PTOP_PA" --stretching "$STRETCH"

echo "Finished: $(date)"
echo "Output: $OUTPUT"
