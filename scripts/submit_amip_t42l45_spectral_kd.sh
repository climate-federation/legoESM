#!/bin/bash
#SBATCH --job-name=legoesm-t42sp
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-t42sp-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-t42sp-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# GRID-TYPE EXPLORATION: SPECTRAL (gaussian) T42 / L45, mirroring the C32/L45
# cubed-sphere baseline (job 25747041) as closely as possible.
#
# Motivation: the per-process tendency decomposition (job 25795037) showed the
# residual tropical cold-point / UTLS warm bias is a 3-way balance — radiative
# cooling vs convective heating (dominant) + a DYNAMICAL heating term that warms
# the 137-194 hPa layer where the real ascending TTL should cool it. The
# cubed-sphere dynamics is contributing spurious UTLS warming. Test whether the
# SPECTRAL primitive-equation dycore (no grid imprinting, clean wave / vertical-
# motion representation) gives a better tropical UTLS circulation.
#
# T42 (~2.8 deg, 128x64 gaussian) ~ C32. dt=150 (spectral stability; cross-grid
# table value). PHYSICS + VERTICAL GRID IDENTICAL to the C32/L45 baseline.
#
# SMOKE = 15 days first (the earlier lat-lon grid probe 25741395 blew up day 2);
# if stable + clean, extend to 60 d for the UTLS comparison.
#
# Usage:  sbatch scripts/submit_amip_t42l45_spectral_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

# --- run parameters ---
YEAR=1979
DAYS=20              # rest-IC stability smoke (ERA5->spectral unwired)
DT=150               # spectral T42 stability (cross-grid table); baseline cube used 200
TRUNC=42             # T42 ~ C32
NLEV=45
PTOP_PA=100
STRETCH=1.2
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_t42l45sp_rest_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  SPECTRAL T${TRUNC} / L${NLEV}  [grid-type exploration]"
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
    --resolution "$TRUNC" \
    --nlev "$NLEV" \
    --radiation rrtmg \
    --convection tiedtke \
    --turbulence holtslag_boville \
    --gravity-wave-drag hines \
    --clouds sundqvist \
    --microphysics morrison \
    --land-mask-file "$LAND_MASK" \
    --albedo-land-file "$LAND_ALBEDO" \
    --diag-days 2 \
    --checkpoint-days 15 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT" \
    --extra --truncation "$TRUNC" --aerosol-ccn --convective-precip-efficiency 0.6 \
            --vertical-coord hybrid --p-top "$PTOP_PA" --stretching "$STRETCH"

echo "Finished: $(date)"
echo "Output: $OUTPUT"
