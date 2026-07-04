#!/usr/bin/env bash
#SBATCH --job-name=amip_c48_sota
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#
# =============================================================================
# legoESM AMIP — SOTA config, single-GPU, proper CMIP6 protocol
#   C48/L40 cubed-sphere + MULTILAYER (Richards) land + CLM PFT/texture map
# =============================================================================
# The full state-of-the-art AMIP stack (config/amip/amip_sota.yaml):
#   - C48/L40, cd-grid, dt 150 s
#   - RRTMG correlated-k radiation, Morrison 2-moment micro, SBM convection,
#     Sundqvist clouds, aerosol_ccn (Twomey + lifetime indirect effects)
#   - use_multilayer_land: 10-layer Richards soil (3 m) + CLM PFT/texture surface
#     map + prognostic snow + Jarvis stomata  (REPLACES the Manabe slab bucket)
#   - real ERA5 IC + real ETOPO topo + full transient CMIP6 forcing
#   - runs on the diurnal-cycle-fixed compiled driver (#720/#721)
#
# SINGLE GPU by necessity: the multilayer land state has no partition spec and
# raises NotImplementedError under any distributed (SPMD / multi-rank MPI) run
# (model_driver.py:6060). So this is 1 node / 1 GPU / 1 rank.
#
# Wall-clock (C48/L40 ~11 min/sim-day on 1 GPU): 10 d ~2 h, 30 d ~6 h — both fit
# the 12 h GPU cap in ONE job (no restart => the deep-soil spin-up is preserved;
# land_ml is NOT yet in the checkpoint, so a chained/restarted multilayer run
# would reset the soil — do NOT chain this until land_ml checkpointing lands).
#
# Overridable via --export: DAYS (default 10), DIAG_DAYS, OUTDIR.
# Submit under bb1596, e.g.:
#   sbatch --account=bb1596 --time=04:00:00 --job-name=amip_c48_sota_10d \
#     --export=ALL,DAYS=10,OUTDIR=/scratch/b/b309178/amip_c48_sota_10d \
#     scripts/cluster/amip/amip_c48_sota_land.sbatch
# =============================================================================

set -uo pipefail

: "${REPO:=/work/bd1083/b309178/diffESM/legoesm_ap}"
: "${DAYS:=10}"
: "${DIAG_DAYS:=1}"
: "${OUTDIR:=/scratch/b/b309178/amip_c48_sota_${DAYS}d}"
: "${CLM_SURFDATA:=${REPO}/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
PY=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
PKG="${REPO}/packages"
CONFIG="${REPO}/config/amip/amip_sota.yaml"

# shellcheck source=/dev/null
source "${REPO}/config/amip/amip_production.sh"

mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/slurm-${SLURM_JOB_ID:-local}.out") \
     2> >(tee -a "${OUTDIR}/slurm-${SLURM_JOB_ID:-local}.err" >&2)

for p in "${ERA5_IC}" "${ETOPO}" "${SST}" "${SIC}" "${SOLAR}" "${GHG}" "${OZONE}" "${AEROSOL}" "${VOLCANIC}" "${CONFIG}" "${CLM_SURFDATA}"; do
    [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done

export PYTHONPATH="${REPO}/src:${PKG}/core:${PKG}/atmosphere:${PKG}/coupler:${PKG}/tools:${PKG}/ml:${PKG}/ocean:${PKG}/land:${PKG}/ice"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

BRANCH=$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "AMIP C48/L40 SOTA (multilayer land + CLM)  DAYS=${DAYS}  job=${SLURM_JOB_ID:-local}  branch=${BRANCH}  $(date -u)"
echo "  CONFIG: ${CONFIG}"
echo "  CLM surfdata: ${CLM_SURFDATA}"
echo "  OUTDIR: ${OUTDIR}  DIAG_DAYS: ${DIAG_DAYS}"

cd "${REPO}"
START=$(date +%s)
"${PY}" -u scripts/run/run_amip.py \
    --config "${CONFIG}" \
    "${AMIP_PATH_FLAGS[@]}" \
    --clm-surfdata-path "${CLM_SURFDATA}" \
    --days "${DAYS}" \
    --diag-days "${DIAG_DAYS}" \
    --cmip-output \
    --checkpoint-days 0 \
    --output "${OUTDIR}"
RC=$?
echo "[c48-sota] run_amip exit=${RC}  wall=$(( $(date +%s) - START ))s  $(date -u)"
exit "${RC}"
