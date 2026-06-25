#!/bin/bash
#SBATCH --job-name=legoesm-ico
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-ico-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-ico-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# ICOSAHEDRAL (MPAS Voronoi) GRID-ARTIFACT PROBE — same physics as the
# cubed-sphere convprecip run, different dynamical core/grid.
#
# Purpose: the +24-30K tropical warm bias is SCHEME-INDEPENDENT (SBM, Tiedtke,
# Zhang all show it) and radiation is clean (deep-trop clear-sky cooling -138
# W/m2, textbook). Diagnosis: convective heating (+12 K/day) is ~balanced by
# dynamics (-11) but a small persistent imbalance (+0.8 K/day) accumulates to a
# too-warm equilibrium, because the dycore circulation is 4-5x too weak (jet
# 6-9 vs 30-40 m/s) to export the convective heat adiabatically. If that weak
# circulation is a CUBED-SPHERE artifact, a fundamentally different grid (MPAS
# centroidal Voronoi / icosahedral) should export the heat and settle cooler.
# Bias persists -> cloud-radiation-coupling, not the grid. Decisive either way.
#
# Resolution: MPAS level 5 = 10*4^5+2 = 10242 cells (C32 = 6144); closest
# C32-equivalent (slightly finer). "icosahedral" normalises to "mpas".
# NOTE: --aerosol-ccn is NOT wired for MPAS (run_amip rejects it) -> omitted;
# keep --convective-precip-efficiency 0.6 (grid-agnostic) so physics matches
# the cubed-sphere convprecip run as closely as possible.
#
# RISKS (could fail-fast, which is itself informative):
#  (1) the ERA5 IC zarr / CMIP SST/forcing must regrid onto the Voronoi mesh;
#      if the AMIP IC path is cubed-sphere/lat-lon-only it will error on load.
#  (2) dt=200s stability on the Voronoi mesh (similar spacing to C32) — reduce
#      to ~150s if the dycore blows up.
#
# Usage:  sbatch scripts/submit_amip_ico_l5_kd.sh
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
GRID=icosahedral
LEVEL=5          # MPAS subdivision level -> 10242 cells (~C32-equivalent)
NLEV=30
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_ico_l5_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  ${GRID} L${LEVEL} (~10242 cells) / L${NLEV}  [grid-artifact probe]"
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
    --resolution "$LEVEL" \
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
    --extra --convective-precip-efficiency 0.6

echo "Finished: $(date)"
echo "Output: $OUTPUT"
