#!/bin/bash
#SBATCH --job-name=legoesm-ll45
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-ll45-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-ll45-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# LAT-LON GRID-TYPE PROBE, RETRY at conservative dt + matched L45 grid.
#
# The cold-point +13K bias decomposition (job 25795037) pinned ~30% of the warm
# UTLS on SPURIOUS DYNAMICAL heating where the real ascending TTL should cool ->
# test whether a DIFFERENT dycore (lat-lon C-grid vs cubed-sphere cdgrid) gives
# a better tropical UTLS circulation. Spectral + MPAS are wiring-blocked; lat-lon
# is the only OTHER fully-wired AMIP dycore (ERA5 IC + stateful Tiedtke/HB/Hines
# both threaded, aerosol-ccn wired).
#
# PRIOR ATTEMPT 25741395 (latlon56 / L30 / dt=75 = ladder value) BLEW UP day 2
# (non-finite winds, CFL reported fine). RETRY changes: (1) dt 75->50 (conservative,
# below the ladder, to rule out a marginal-CFL / polar-filter dt sensitivity);
# (2) NLEV 30->45 with the L45 hybrid grid (p_top=100Pa, stretch=1.2) IDENTICAL to
# the C32/L45 cold-point baseline 25747041 so the UTLS comparison is clean; (3) 15d
# SMOKE first — survive past day 2 => dt helped, run on for the cold-point compare;
# blow up again => the lat-lon instability is structural (polar filter), not dt.
#
# Physics IDENTICAL to the C32/L45 baseline (tiedtke + aerosol-ccn +
# convective-precip-efficiency 0.6) so ONLY grid/dycore differs.
#
# Usage:  sbatch scripts/submit_amip_latlon56l45_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

# --- run parameters ---
YEAR=1979
DAYS=15              # SMOKE: prior latlon blew up day 2; survive that => continue
DT=50               # conservative; prior 25741395 died at ladder dt=75
GRID=latlon
NLAT=56             # n_lon = 2*NLAT = 112 -> 6272 cells (~C32 6144)
DISC=latlon_cgrid
NLEV=45
PTOP_PA=100         # match C32/L45 baseline 25747041
STRETCH=1.2         # match C32/L45 baseline 25747041
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_latlon56l45_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  ${GRID} nlat=${NLAT} (${DISC}) / L${NLEV}  [grid-type probe retry, dt=${DT}]"
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
    --grid-type "$GRID" \
    --resolution "$NLAT" \
    --discretization "$DISC" \
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
    --checkpoint-days 15 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT" \
    --extra --aerosol-ccn --convective-precip-efficiency 0.6 --use-polar-filter \
            --vertical-coord hybrid --p-top "$PTOP_PA" --stretching "$STRETCH"

echo "Finished: $(date)"
echo "Output: $OUTPUT"
