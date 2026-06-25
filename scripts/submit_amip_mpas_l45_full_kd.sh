#!/bin/bash
#SBATCH --job-name=legoesm-mpasfull
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-mpasfull-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-mpasfull-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# FULL MPAS-Voronoi AMIP — the cube C32/L45 setup ported to MPAS.  Combines all
# of the MPAS forcing wiring:
#   * ERA5 initial conditions          (era5_to_mpas_carry: scalars->cells,
#                                        winds->edge-normal, orography via phis)
#   * land-sea mask + land albedo      (external NetCDF regridded onto cells)
#   * solar / ozone / Kinne aerosol-OD / volcanic / GHG (auto-resolved CMIP6
#                                        forcing by run_amip_levante.py)
#   * AMIP SST surface forcing
#   * aerosol-CCN specified droplet number (Andreae 2009 AOD->CCN; Phase C —
#                                        now wired on the MPAS combined-physics
#                                        microphysics + radiation factories)
#   * RRTMGP radiation sub-cycle       (1-hour cadence; --production-profile)
#   * hybrid sigma-pressure vertical   (raised lid p_top=1 hPa, stretch=1.2)
#
# DAYS / LEVEL are env-overridable for a quick validation, e.g.
#   DAYS=3 LEVEL=5 sbatch scripts/submit_amip_mpas_l45_full_kd.sh
# Defaults: 60 days, mesh level 5 (~10242 cells, ~comparable to C32).
#
# Usage:  sbatch scripts/submit_amip_mpas_l45_full_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

YEAR=1979
DAYS=${DAYS:-60}
DT=${DT:-60}             # MPAS hydrostatic stability (cube used 200)
LEVEL=${LEVEL:-5}        # voronoi refinement (~10242 cells); level 4 ~ 2562
NLEV=45
PTOP_PA=100             # 1 hPa lid
STRETCH=1.2             # UTLS-resolving level distribution (matches C32/L45)
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_mpas_full_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM FULL AMIP  MPAS-Voronoi L${LEVEL} / L${NLEV}  [ERA5 IC + land + albedo + aerosol-CCN + subcycle]"
echo "Job:    ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME:-?}   dt: ${DT}s  Days: $DAYS"
echo "Output: $OUTPUT"
echo "Started: $(date)"
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
    --checkpoint-days 15 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT" \
    --extra --grid-type voronoi --discretization mpas \
            --vertical-coord hybrid --p-top "$PTOP_PA" --stretching "$STRETCH" \
            --aerosol-ccn \
            --convective-precip-efficiency 0.6 \
            --no-rrtmgp-gpoint-checkpoint

echo "Finished: $(date)"
echo "Output: $OUTPUT"
