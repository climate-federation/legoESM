#!/bin/bash
#SBATCH --job-name=legoesm-mpassmk
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-mpassmk-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-mpassmk-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# MINIMAL MPAS-Voronoi AMIP smoke — validate the _run_mpas path runs the full
# moist production physics (Tiedtke/HB/Hines/Morrison/Sundqvist, which MPAS
# already THREADS as PhysicsState — no #405).  MPAS dry-HS proved the dycore
# makes a correct midlat jet (32 m/s @38, eddies ~45); now confirm the AMIP
# pipeline runs.
#
# OCEAN-ONLY (no --land-mask-file => _create_topography takes the flat branch,
# avoids the VoronoiMesh grid.n crash), DEFAULT IC (no --ic-zarr => avoids the
# unlanded era5_to_mpas_carry), NO --aerosol-ccn (cube/latlon-only guard).
# SIGMA vertical coord (MPAS uses sigma, not hybrid).  dt=60 (MPAS moist CFL;
# ssp_rk54 auto).  10-day smoke.  Dycore diffusion from compute_diffusion:
# nu_del2=A_h (now the reduced default), nu_del4=dx^4/86400 (stable per dry-HS).
#
# Usage:  sbatch scripts/submit_amip_mpas_smoke_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

YEAR=1979
DAYS=10
DT=60
LEVEL=5              # ~10242 cells ~ C32
NLEV=30
OUTPUT=/scratch/b/b309165/amip_mpas_smoke_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs
echo "MPAS AMIP smoke  level${LEVEL}/L${NLEV}  dt=${DT} days=${DAYS}  Job ${SLURM_JOB_ID}  $(date)"
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
    --diag-days 2 \
    --cmip-output \
    --production-profile \
    --output "$OUTPUT" \
    --extra --grid-type voronoi --discretization mpas \
            --convective-precip-efficiency 0.6 \
            --no-rrtmgp-gpoint-checkpoint

echo "Finished $(date)  Output $OUTPUT"
