#!/bin/bash
#SBATCH --job-name=legoesm-ll
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-ll-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-ll-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# LAT-LON GRID-ARTIFACT PROBE — same physics as the cubed-sphere convprecip
# run, a DIFFERENT grid + dynamical core (regular lat-lon C-grid).
#
# Hypothesis (user): the +24-30K tropical warm bias may be a CUBED-SPHERE
# dycore artifact. It is SCHEME-INDEPENDENT (SBM/Tiedtke/Zhang all warm) and
# radiation is clean (deep-trop clear-sky cooling -138 W/m2). Diagnosis:
# convective heating +12 K/day is ~balanced by dynamics -11, but a small
# persistent imbalance accumulates because the dycore circulation is 4-5x too
# weak (jet 6-9 vs 30-40 m/s) to export the convective heat adiabatically.
# TEST: run the SAME physics on a lat-lon C-grid (different grid + dynamical
# core). Bias PERSISTS -> not a cubed-sphere artifact (physics: cloud-radiation
# coupling / convective balance). Bias CHANGES -> grid/dycore-specific.
# Icosahedral/MPAS is NOT wired for the AMIP path (wrapper: cubed_sphere/latlon
# only); lat-lon is the wired different-grid test.
#
# Grid: lat-lon n_lat=56 (n_lon=112) = 6272 cells ~ C32 (6144). C-grid
# discretization + Fourier polar filter (pole-CFL relief). dt=75s = stability-
# ladder value (auto_dt_rce latlon/56); ERA5 IC is native lat-lon (180x360) so
# regrids cleanly. Physics IDENTICAL to the cubed-sphere convprecip run
# (tiedtke + aerosol-ccn + convective-precip-efficiency 0.6) so ONLY grid/dycore
# differs. (aerosol-ccn IS wired for latlon, unlike MPAS.)
#
# Usage:  sbatch scripts/submit_amip_latlon56_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

# --- run parameters ---
YEAR=1979
DAYS=30
DT=75
GRID=latlon
NLAT=56            # n_lon = 2*NLAT = 112 -> 6272 cells (~C32)
DISC=latlon_cgrid
NLEV=30
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_latlon56_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  ${GRID} nlat=${NLAT} (${DISC}) / L${NLEV}  [grid-artifact probe]"
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
    --checkpoint-days 30 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT" \
    --extra --aerosol-ccn --convective-precip-efficiency 0.6 --use-polar-filter

echo "Finished: $(date)"
echo "Output: $OUTPUT"
