#!/bin/bash
#SBATCH --job-name=legoesm-tk60
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-tk60-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-tk60-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# Convective-detrainment PROBE — C32/L30, single A100.
#
# Hypothesis: the SBM (Simplified Betts-Miller) run desiccates the upper
# troposphere (RH_ice~0) because relaxation convection has NO mass flux /
# detrainment, so it neither moistens the upper trop nor detrains condensate
# into an anvil -> IWP~0 + a +22K warm bias (no cirrus LW cooling).
#
# Probe 1 (mass_flux, job 25688227) made it WORSE: simple bulk plume with
# UNIFORM detrainment + weak mass flux -> compensating subsidence dominated ->
# hot (290-296K) dry collapse. This probe uses TIEDTKE (ECMWF/ICON operational
# mass-flux: deep/mid/shallow branches, moisture-convergence closure,
# detrainment concentrated at cloud top = anvil) so dq_c_conv_dt detrains
# updraft condensate ALOFT, which microphysics freezes to ice -> cirrus.
# Everything else identical to submit_amip_c32l30_kd.sh. 60 days (equilibration) shows the
# upper-trop moistening / IWP / T(z) trend vs the SBM baseline (job 25670704).
#
# Usage:  sbatch scripts/submit_amip_c32l30_tiedtke60_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

# --- run parameters ---
YEAR=1979
DAYS=60
DT=200
RES=32
NLEV=30
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_c32l30_tiedtke60_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  C${RES} / L${NLEV}  [tiedtke probe]"
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
    --output "$OUTPUT"

echo "Finished: $(date)"
echo "Output: $OUTPUT"
