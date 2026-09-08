#!/bin/bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/slurm_logs/%x-%j.out
#
# Offline land spin-up (run_lmip_biophys.py) for a coupled AMIP campaign.
# CONFIG and OUTPUT come from the environment so one wrapper serves every mesh.
set -euo pipefail
: "${REPO:?set REPO}"; : "${CONFIG:?set CONFIG}"; : "${OUTDIR:?set OUTDIR}"
PY="${PY:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
PKG="${REPO}/packages"
export PYTHONPATH="${REPO}/src:${PKG}/core:${PKG}/atmosphere:${PKG}/coupler:${PKG}/tools:${PKG}/ml:${PKG}/ocean:${PKG}/land:${PKG}/ice"
export JAX_ENABLE_X64=1
mkdir -p "${OUTDIR}"
cd "${REPO}"
echo "[spinup] config=${CONFIG} out=${OUTDIR} $(date -u)"
# --restart-from lets a walltime kill resume from the last finished YEAR: the
# driver flushes a restart per year, so a killed run loses at most one.
srun "${PY}" -u scripts/run/run_lmip_biophys.py \
     --config "${CONFIG}" --output "${OUTDIR}" ${EXTRA:-}
echo "[spinup] done $(date -u)"
ls -la "${OUTDIR}" | tail -5
