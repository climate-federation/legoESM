#!/usr/bin/env bash
#SBATCH --job-name=amip_canonical_30d
#SBATCH --account=bd1179
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/b/b309178/amip_canonical_30d/slurm-%j.out
#SBATCH --error=/scratch/b/b309178/amip_canonical_30d/slurm-%j.err
#
# Longer test: 30 days, C48/L40, canonical physics stack
# (rrtmg + sundqvist + kessler + bechtold + louis), synthetic forcing deck.
# Uses forcing_amip/ pre-generated files. Validates full physics pipeline
# on 4 GPUs.
# (2026-07-08: convection sbm -> bechtold; AMIP convection is
# bechtold/tiedtke only by directive.)

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PY=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
OUTPUT=/scratch/b/b309178/amip_canonical_30d
FORCING="${REPO}/forcing_amip"
PKG="${REPO}/packages"
mkdir -p "${OUTPUT}"

export PYTHONPATH="${REPO}/src:${PKG}/core:${PKG}/atmosphere:${PKG}/coupler:${PKG}/tools:${PKG}/ml:${PKG}/ocean:${PKG}/land:${PKG}/ice"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

echo "========================================"
echo "  AMIP canonical 30d  job=${SLURM_JOB_ID:-local}  host=$(hostname)"
echo "  C48/L40  rrtmg+sundqvist+kessler+bechtold+louis  synthetic forcing  4 GPU"
echo "  $(date -u)"
echo "========================================"
nvidia-smi -L 2>&1 | head -4 || true

cd "${REPO}"
"${PY}" -u scripts/run/run_amip.py \
    --dataset custom \
    --forcing-path "${FORCING}/sst_sic_amip_1979-2014.nc" \
    --sst-var sst \
    --sst-offset 273.15 \
    --sic-var sic \
    --grid-type cubed_sphere \
    --discretization cdgrid \
    --resolution 48 \
    --nlev 40 \
    --radiation rrtmg \
    --rad-update-steps 12 \
    --diurnal-cycle \
    --clouds sundqvist \
    --microphysics kessler \
    --convection bechtold \
    --turbulence louis \
    --ozone-source standard \
    --ghg-forcing constant \
    --co2-ppmv 415 \
    --ch4-ppbv 1900 \
    --n2o-ppbv 332 \
    --days 30 \
    --dt 300 \
    --diag-days 5 \
    --checkpoint-days 30 \
    --monthly-means \
    --output "${OUTPUT}"

echo "[$(date -u)] Done. Exit 0."
