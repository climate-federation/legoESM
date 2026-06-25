#!/bin/bash
#SBATCH --job-name=legoesm-mpasera5
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-mpasera5-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-mpasera5-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# Phase B validation: ERA5 initial conditions on the MPAS Voronoi path
# (era5_to_mpas_carry: scalars->cells, winds->edge-normal via angleEdge,
# vertical pressure->hybrid).  L4 + ERA5 IC + land mask + albedo + hybrid
# vertical coord — the near-full AMIP setup at low resolution.  Orography
# arrives via the ERA5 IC phis (not --topography), matching the cube setup.
#
# Usage:  sbatch scripts/submit_amip_mpas_L4_era5_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

YEAR=1979
DAYS=1
DT=60
LEVEL=4
NLEV=30
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_mpas_era5_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs
echo "MPAS AMIP ERA5-IC test  level${LEVEL}/L${NLEV}  dt=${DT} days=${DAYS}  Job ${SLURM_JOB_ID}  $(date)"
echo "Output: $OUTPUT"

source $CONDA/etc/profile.d/conda.sh
conda activate $ENV
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader || true
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
    --diag-days 1 \
    --cmip-output \
    --production-profile \
    --output "$OUTPUT" \
    --extra --grid-type voronoi --discretization mpas \
            --vertical-coord hybrid \
            --convective-precip-efficiency 0.6 \
            --no-rrtmgp-gpoint-checkpoint

echo "Finished $(date)  Output $OUTPUT"
