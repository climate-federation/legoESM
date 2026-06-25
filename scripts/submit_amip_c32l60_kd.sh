#!/bin/bash
#SBATCH --job-name=legoesm-l60
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-l60-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-l60-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# RAISED-LID / VERTICAL-RESOLUTION SWEEP, member 2: C32 / L60.
#
# See submit_amip_c32l45_kd.sh for the full diagnosis. Short version: the
# residual warm bias after the convprecip fix is UPPER-LEVEL (cold point +17K
# @100hPa, top layer +0.2 K/day runaway), caused by the L30 grid placing the
# whole stratosphere/UTLS in ONE 35-hPa level + only 2 TTL levels. Fix =
# add levels + redistribute to the UTLS (vertical resolution, NOT a sponge or
# Newtonian-T patch).
#
# This member: nlev=60, p_top=1 hPa (100 Pa), stretching=1.0 (near-uniform in
# eta -> maximal UTLS/strat resolution). Gives ~5 levels 100-2 hPa and ~7 in the
# TTL (200-70 hPa), top full level ~12 hPa. dt=150 (more/thinner layers -> safety
# margin vs the L45/L30 dt=200; dt affects stability, not the equilibrium climate
# we are comparing). PHYSICS IDENTICAL to the 90d baseline (job 25739722):
# tiedtke + aerosol-ccn + convective-precip-efficiency 0.6, ozone "standard".
#
# Success (strongest test of the resolution hypothesis): cold point forms toward
# 192K @100hPa; top-layer warming rate -> 0; free trop stays ~+3-4K. If L60 STILL
# runs warm aloft with the strat resolved, the next lever is the ozone profile
# (standard over-estimates lower-strat O3 ~3x) or upper-level LW cooling — NOT a
# patch. Compare d30/d60 tropical T(plev) vs the L30 baseline and the L45 member.
#
# Usage:  sbatch scripts/submit_amip_c32l60_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

# --- run parameters ---
YEAR=1979
DAYS=60
DT=150
RES=32
NLEV=60
PTOP_PA=100          # 1 hPa lid
STRETCH=1.0          # near-uniform-in-eta -> maximal UTLS/strat resolution
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_c32l60_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

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
