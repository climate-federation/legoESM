#!/bin/bash
#SBATCH --job-name=legoesm-orogwd
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-orogwd-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-orogwd-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# OROGRAPHIC GWD PROBE: add McFarlane (orographic) to the baseline Hines
# (non-orographic) GWD on the cubed sphere.  GWD scheme = "hines+mcfarlane"
# (a NEW composite that SUMS the two source tendencies).
#
# Motivation: the cold-point decomposition (job 25795037) showed ~30% of the
# tropical UTLS warm bias is SPURIOUS DYNAMICAL heating where the real
# ascending TTL should cool.  Orographic GWD deposits extratropical
# stratospheric momentum that drives the Brewer-Dobson cell, whose tropical
# branch upwells and adiabatically cools the TTL — the missing process by
# elimination.  This is the physically-correct missing source (NOT a patch):
# CMIP-class GCMs run orographic + non-orographic GWD together.
#
# Real subgrid orography: McFarlane's launch stress ∝ h_topo²; here h_topo is
# the ICON-extpar SSO_STDH (standard deviation of subgrid orography) remapped
# to 0.25° lat-lon then to the model columns — real mountains (max ~1240 m on
# C32 over the Tibetan Plateau), ~0 over ocean (NO spurious ocean drag).
#
# PHYSICS + GRID else IDENTICAL to the C32/L45 cold-point baseline 25747041
# (cold point +13K) so ONLY the added orographic GWD differs.  Compare tropical
# T(plev) @100hPa and the UTLS dynamical heating vs that baseline.
#
# Usage:  sbatch scripts/submit_amip_c32l45_orogwd_kd.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python

YEAR=1979
DAYS=60
DT=200
RES=32
NLEV=45
PTOP_PA=100
STRETCH=1.2
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
LAND_MASK=/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc
LAND_ALBEDO=/work/bd1083/b309178/diffESM/land_data/extpar_albedo_latlon_0p25deg.nc
SSO_FILE=/work/bd1083/b309178/diffESM/land_data/extpar_sso_latlon_0p25deg.nc
OUTPUT=/scratch/b/b309165/amip_c32l45_orogwd_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "legoESM AMIP  C${RES} / L${NLEV}  [orographic GWD: hines+mcfarlane, real SSO_STDH]"
echo "Job:    ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}"
echo "Year:   $YEAR  Days: $DAYS  dt: ${DT}s"
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
    --resolution "$RES" \
    --nlev "$NLEV" \
    --radiation rrtmg \
    --convection tiedtke \
    --turbulence holtslag_boville \
    --gravity-wave-drag hines+mcfarlane \
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
    --extra --aerosol-ccn --convective-precip-efficiency 0.6 \
            --vertical-coord hybrid --p-top "$PTOP_PA" --stretching "$STRETCH" \
            --subgrid-orography-file "$SSO_FILE"

echo "Finished: $(date)"
echo "Output: $OUTPUT"
