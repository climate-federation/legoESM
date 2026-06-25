#!/bin/bash
#SBATCH --job-name=legoesm-cp
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-cp-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-cp-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# CONVECTIVE-PRECIP-EFFICIENCY PROBE — C32/L30, A100, Tiedtke + aerosol-CCN.
#
# Keystone (validated): the +26-30K tropical warm bias is driven by excess
# SUSPENDED convective condensate trapping LW and shutting down tropical
# free-trop radiative cooling (clouds cut column LW cooling 137 W/m^2;
# clear-sky -2.5 K/day -> cloudy ~0). The subgrid-autoconversion probe
# (25733392) proved the microphysics SINK is not rate-limiting (free-trop
# cloud changed <2%): the cloud is SOURCE-buffered. Tiedtke detrains 100% of
# its updraft condensate as suspended grid cloud (tiedtke.py:289) with NO
# in-updraft precipitation -- the plume docstring delegates precip to the
# caller, which never did it.
#
# Fix (missing physics, not a knob): --convective-precip-efficiency 0.6
# diverts 60% of the detrained convective condensate to RAIN (a precipitating
# species that sediments via microphysics and is invisible to radiation,
# which sees only q_c/q_i), leaving 40% as anvil cloud. Observed deep-conv
# CPE ~0.5-0.9; same convention as SBM/Emanuel precip efficiency. Expected:
# grid q_c/q_i source cut -> LWP/IWP fall -> tropical mid-trop LW cooling
# restored toward -2 K/day -> warm bias erodes -> meridional gradient steepens
# -> winds/evaporation recover; convective surface precip rises (NOT E-capped:
# generated in-plume). Only change vs swfix baseline is the precip efficiency.
#
# Success: tropical mid-trop (sig 0.3-0.5) radiative cooling restored; T_atm /
# upper-trop toward ERA5; R_TOA toward 0. Compare vs swfix (job 25717303:
# T_atm 267, LWP 0.16, IWP 0.084, precip 0.75, CWV 34, R_TOA -58).
#
# Usage:  sbatch scripts/submit_amip_c32l30_convprecip_kd.sh
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
OUTPUT=/scratch/b/b309165/amip_c32l30_convprecip_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  C${RES} / L${NLEV}  [conv-precip-efficiency 0.6, tiedtke]"
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
    --extra --aerosol-ccn --convective-precip-efficiency 0.6

echo "Finished: $(date)"
echo "Output: $OUTPUT"
