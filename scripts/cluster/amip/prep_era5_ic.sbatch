#!/usr/bin/env bash
#SBATCH --job-name=era5_ic_prep
#SBATCH --account=bd1179
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b/b309178/era5_ic_prep/slurm-%j.out
#SBATCH --error=/scratch/b/b309178/era5_ic_prep/slurm-%j.err
#
# Prepare ERA5 initial-condition Zarr store from Levante pool GRIB files.
# Runs CDO remapnn on 6 variables (T, u, v, q, sp, skt, z_sfc) for a single
# date and writes a Zarr compatible with --ic era5 --ic-path.
# CPU-only — no GPU needed. Use compute partition.
#
# Usage:
#   sbatch scripts/cluster/amip/prep_era5_ic.sbatch
# or with a custom date/output:
#   sbatch --export=YEAR=1979,MONTH=1,DAY=1,HOUR=0,OUT=/scratch/b/b309178/era5_ic_1979-01-01.zarr \
#       scripts/cluster/amip/prep_era5_ic.sbatch

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PY=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
PKG="${REPO}/packages"

# Allow override via --export
YEAR="${YEAR:-1979}"
MONTH="${MONTH:-1}"
DAY="${DAY:-1}"
HOUR="${HOUR:-0}"
OUT="${OUT:-/scratch/b/b309178/era5_ic_${YEAR}-$(printf '%02d' ${MONTH})-$(printf '%02d' ${DAY}).zarr}"

mkdir -p /scratch/b/b309178/era5_ic_prep
mkdir -p "$(dirname "${OUT}")" 2>/dev/null || true

export PYTHONPATH="${REPO}/src:${PKG}/core:${PKG}/atmosphere:${PKG}/coupler:${PKG}/tools:${PKG}/ml:${PKG}/ocean:${PKG}/land:${PKG}/ice"

echo "========================================"
echo "  ERA5 IC prep  job=${SLURM_JOB_ID:-local}  host=$(hostname)"
echo "  Date: ${YEAR}-$(printf '%02d' ${MONTH})-$(printf '%02d' ${DAY}) ${HOUR}:00 UTC"
echo "  Output: ${OUT}"
echo "  $(date -u)"
echo "========================================"

cd "${REPO}"
"${PY}" -u scripts/data/prep_era5_ic.py \
    --year "${YEAR}" \
    --month "${MONTH}" \
    --day "${DAY}" \
    --hour "${HOUR}" \
    --out "${OUT}"

echo "[$(date -u)] Done. Zarr written to: ${OUT}"
