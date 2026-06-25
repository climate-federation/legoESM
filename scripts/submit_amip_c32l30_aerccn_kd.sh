#!/bin/bash
#SBATCH --job-name=legoesm-ac
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-ac-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-ac-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# AEROSOL-CCN PROBE (route 1) — C32/L30, single A100, Tiedtke convection.
#
# Diagnosis (Tiedtke d60 checkpoint): the marine boundary-layer liquid deck
# (93% of LWP, peak q_c=0.245 g/kg at sig=0.96/T=284K) will not rain. Morrison
# warm rain uses KK2000 (PRC = 1350*q_c^2.47*Nc[cm^-3]^-1.79) with a UNIFORM
# continental Nc=Nc_0=100 cm^-3 everywhere (predict_Nc=False, N_c carry zero).
# Reconstructed autoconversion timescale at the deck peak = 159 h (~6.6 days);
# BL-integrated warm-rain source only 0.165 mm/day -> LWP 0.24 (3-8x high) +
# precip 0.67 (3x low).
#
# Fix (physics, not a patch): --aerosol-ccn derives the SPECIFIED cloud-droplet
# number from the column Kinne AOD already loaded for the aerosol direct effect
# (Andreae 2009 inverted: N_CCN=(AOD/0.0027)^(1/0.640), clipped 10-1e4 cm^-3).
# Remote oceans -> low AOD -> low Nc -> fast drizzle; polluted land -> high Nc.
# This gives an EMERGENT land/ocean CDNC contrast from the aerosol climatology
# (Twomey 1st + KK2000 2nd indirect effects) instead of a hardcoded constant.
# The wrapper already sets --aerosol-forcing external (prerequisite); the flag
# is forwarded verbatim via --extra. Everything else identical to the Tiedtke
# baseline. 30 days resolves the LWP/precip response (warm-rain tau ~hours-days).
#
# Usage:  sbatch scripts/submit_amip_c32l30_aerccn_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

# --- run parameters ---
YEAR=1979
DAYS=30
DT=200
RES=32
NLEV=30
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_c32l30_aerccn_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  C${RES} / L${NLEV}  [aerosol-CCN probe, tiedtke]"
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
    --extra --aerosol-ccn

echo "Finished: $(date)"
echo "Output: $OUTPUT"
