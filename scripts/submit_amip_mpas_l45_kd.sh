#!/bin/bash
#SBATCH --job-name=legoesm-mpas
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-mpas-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-mpas-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# GRID-TYPE EXPLORATION: MPAS Voronoi / L45. Test whether the MPAS dycore gives
# a better tropical UTLS circulation than the cubed sphere (per the per-process
# decomposition, cube dynamics adds spurious +0.2-0.5 K/day UTLS warming where
# the real ascending TTL cools).
#
# MPAS threads the stateful physics (Tiedtke conv-prog / HB TKE / Hines GWD)
# unlike the spectral run loop. ERA5 IC is NOT wired for MPAS (era5_to_mpas_carry
# not landed) so this uses the DEFAULT rest IC -> needs spinup; first goal is
# STABILITY + wiring (does MPAS run our moist AMIP physics at all).
#
# Level 5 (~2562 cells, coarse/fast) for the first smoke; dt=60 (MPAS hydrostatic
# is unstable at larger dt even when CFL reports tiny). PHYSICS + VERTICAL GRID
# else identical to the C32/L45 baseline. SMOKE 20 d; scale to level 6 + 60 d if
# stable + sane.
#
# Usage:  sbatch scripts/submit_amip_mpas_l45_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

YEAR=1979
DAYS=20
DT=60                # MPAS hydrostatic stability (cross-grid table); cube used 200
LEVEL=5              # voronoi refinement level (~2562 cells); C32 ~ level 6 (10242)
NLEV=45
PTOP_PA=100
STRETCH=1.2
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_mpasl45_rest_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  MPAS-Voronoi L${LEVEL} / L${NLEV}  [grid-type exploration, rest IC]"
echo "Job:    ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}   dt: ${DT}s  Days: $DAYS"
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
    --diag-days 2 \
    --checkpoint-days 15 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT" \
    --extra --grid-type voronoi --discretization mpas \
            --convective-precip-efficiency 0.6 \
            --vertical-coord hybrid --p-top "$PTOP_PA" --stretching "$STRETCH"

echo "Finished: $(date)"
echo "Output: $OUTPUT"
