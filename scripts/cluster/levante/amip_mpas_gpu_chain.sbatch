#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/slurm_logs/%x-%j.out
#
# MPAS-lane AMIP CMIP chain on DKRZ Levante GPU (2026-07-23), adapted from the
# proven cube-lane /scratch/b/b309178/amip_decade/amip365_chain.sbatch with
# these deliberate deltas:
#   1. Runs from THIS checkout (legoesm_pg, clean main) with its own uv venv
#      (CUDA jax installed 2026-07-23) — fully independent of the legoesm_ap
#      checkout + diffesm conda env the cube lane mutates mid-campaign.
#   2. MPAS grid overrides arrive via EXTRA (grid/resolution/dt/convection);
#      base physics + forcing selectors stay amip_production.yaml + the
#      Levante paths in amip_production.sh (single source of truth).
#   3. Chain link = 12 h GPU job; run_amip checkpoints + exits cleanly at
#      MAX_WALLCLOCK_SECONDS (11 h) and the link resubmits itself afterok
#      until TARGET_DAYS is reached.
# Env: NAME (outdir under ROOT, required), EXTRA (flag overrides), TARGET_DAYS
#      (default 365), PREV_CKPT_DIR (optional warm start).
set -uo pipefail
: "${REPO:=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM}"
: "${ROOT:=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs}"
: "${TARGET_DAYS:=365}"
: "${MAX_WALLCLOCK_SECONDS:=39600}"   # 11 h: checkpoint+exit before the 12 h kill
OUTDIR="${ROOT}/${NAME}"
# PY overridable so a git-worktree bisect can run an older REPO tree with the
# main checkout's venv (PATH-style editable .pth entries; PYTHONPATH wins).
: "${PY:=${REPO}/.venv/bin/python}"
PKG="${REPO}/packages"

# --- Century transient deck (opt-in: CENTURY_DECK=1) ------------------------
# Select the era-correct OZONE (50-yr CMIP6 chunks) and VOLCANIC (per-year
# CMIP6 files) for THIS link's simulated year, computed from the newest
# checkpoint day + START_YEAR, BEFORE sourcing amip_production.sh (whose
# `: "${VAR:=default}"` respects pre-set env).  Approximations, documented:
# the file is pinned per LINK (<= ~1 sim-year timing error at link
# boundaries), volcanic clamps to the available 1850-2014 range (post-2014
# holds 2014 background), ozone holds the ssp-era file after 2014 if
# present else the last historical chunk.  Off (default) = byte-identical.
if [[ "${CENTURY_DECK:-0}" == "1" ]]; then
    : "${START_YEAR:=1923}"
    _ICONP=/pool/data/ICON/grids/public/mpim
    _PRE_CKPT=$(ls -1 "${OUTDIR}"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1 || true)
    _PRE_DAY=0
    [[ -n "${_PRE_CKPT}" ]] && _PRE_DAY=$(basename "${_PRE_CKPT}" | sed -E 's/checkpoint_day_0*([0-9]+)\.npz/\1/')
    SIM_YEAR=$(( START_YEAR + _PRE_DAY / 365 ))
    _VOLC_YEAR=${SIM_YEAR}
    (( _VOLC_YEAR < 1850 )) && _VOLC_YEAR=1850
    (( _VOLC_YEAR > 2014 )) && _VOLC_YEAR=2014
    export VOLCANIC="${_ICONP}/common/aerosol_volcanic_cmip6/bc_aeropt_cmip6_volc_lw_b16_sw_b14_${_VOLC_YEAR}.nc"
    _OZDIR="${_ICONP}/common/ozone_cmip6_forcing"
    if   (( SIM_YEAR < 1900 )); then _OZF="historical/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_185001-189912.nc"
    elif (( SIM_YEAR < 1950 )); then _OZF="historical/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_190001-194912.nc"
    elif (( SIM_YEAR < 2000 )); then _OZF="historical/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_195001-199912.nc"
    else                             _OZF="historical/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_200001-201412.nc"
    fi
    if (( SIM_YEAR > 2014 )); then
        _SSP=$(ls -1 "${_OZDIR}/ssp370"/vmro3_*.nc 2>/dev/null | head -1 || true)
        [[ -n "${_SSP}" ]] && _OZF="ssp370/$(basename "${_SSP}")"
    fi
    export OZONE="${_OZDIR}/${_OZF}"
    # SST/SIC must span the century too.  amip_production.sh defaults to the
    # ICON pool bc_sst_1979_2016.nc, which (a) does not cover 1923 and (b)
    # stores KELVIN under a units='degC' attribute — so a CENTURY_DECK run
    # left on the default got era-correct ozone+volcanic bolted onto
    # 1979-2016 SST, a silently mismatched deck.  Default to the PCMDI
    # 1870-2022 AMIP boundary condition (Celsius, hence SST_OFFSET 273.15)
    # and keep them env-overridable like every other path here.
    _PCMDI=/pool/data/INPUT4MIP/data/input4MIPs/CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10
    _V=v20250807
    : "${SST:=${_PCMDI}/ocean/mon/tosbcs/gn/${_V}/tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc}"
    : "${SIC:=${_PCMDI}/seaIce/mon/siconcbcs/gn/${_V}/siconcbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc}"
    export SST SIC
    [[ -f "${VOLCANIC}" ]] || { echo "CENTURY_DECK: missing volcanic file ${VOLCANIC}" >&2; exit 3; }
    [[ -f "${OZONE}"    ]] || { echo "CENTURY_DECK: missing ozone file ${OZONE}"    >&2; exit 3; }
    [[ -f "${SST}"      ]] || { echo "CENTURY_DECK: missing SST file ${SST}"        >&2; exit 3; }
    [[ -f "${SIC}"      ]] || { echo "CENTURY_DECK: missing SIC file ${SIC}"        >&2; exit 3; }
    echo "CENTURY_DECK: sim year ${SIM_YEAR} -> volcanic ${_VOLC_YEAR}, ozone $(basename "${_OZF}"), sst $(basename "${SST}")"
fi
# shellcheck source=/dev/null
source "${REPO}/config/amip/amip_production.sh"
mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/slurm-${SLURM_JOB_ID:-local}.out") \
     2> >(tee -a "${OUTDIR}/slurm-${SLURM_JOB_ID:-local}.err" >&2)

export PYTHONPATH="${REPO}/src:${PKG}/core:${PKG}/atmosphere:${PKG}/coupler:${PKG}/tools:${PKG}/ml:${PKG}/ocean:${PKG}/land:${PKG}/ice"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# Persistent JIT cache on scratch: later chain links skip the cold compile.
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-/scratch/b/b381103/legoesm_jit_cache}"
mkdir -p "${LEGOESM_JIT_CACHE_DIR}" 2>/dev/null || true

export PATH=/sw/spack-levante/git-2.43.7-2ofazl/bin:$PATH
BRANCH=$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "AMIP MPAS-lane chain  NAME=${NAME}  job=${SLURM_JOB_ID:-local}  branch=${BRANCH}  $(date -u)"
echo "  TARGET_DAYS=${TARGET_DAYS}  EXTRA='${EXTRA:-}'"
nvidia-smi -L 2>&1 | head -2 || true
"${PY}" -c "import jax; print('  jax', jax.__version__, jax.devices())" || exit 90
"${PY}" -c "import legoesm.atmosphere as m; print('  legoesm.atmosphere ->', list(m.__path__))" || exit 91

LATEST_CKPT=$(ls -1 "${OUTDIR}"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1 || true)
if [[ -z "${LATEST_CKPT}" && -n "${PREV_CKPT_DIR:-}" ]]; then
    LATEST_CKPT=$(ls -1 "${PREV_CKPT_DIR}"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "${LATEST_CKPT}" ]] && echo "  Warm-starting from external ${LATEST_CKPT}"
fi
LATEST_DAY=0
if [[ -n "${LATEST_CKPT}" ]]; then
    LATEST_DAY=$(basename "${LATEST_CKPT}" | sed -E 's/checkpoint_day_0*([0-9]+)\.npz/\1/')
    echo "  Resuming from ${LATEST_CKPT} (day ${LATEST_DAY}/${TARGET_DAYS})"
else
    echo "  First link — starting from IC"
fi
if (( LATEST_DAY >= TARGET_DAYS )); then
    echo "  Already at/past target. Nothing to do."; exit 0
fi
# MPAS restart contract (model_driver._run_mpas): --days is the REMAINING
# days for this link, NOT the absolute target (cube/lat-lon convention).
# Do NOT pass --start-day; the restart checkpoint carries the time origin.
DAYS_THIS_JOB=$(( TARGET_DAYS - LATEST_DAY ))
echo "  This link runs ${DAYS_THIS_JOB} remaining days (of ${TARGET_DAYS})"
# The run-manifest guard hashes the FULL config incl. --days, so the
# remaining-days convention makes every link a "different config" in the
# same directory.  Archive the prior link's manifest (provenance kept
# per-link) so the guard can mint a fresh one — the guard's own documented
# remedy.  Proper fix (follow-up): a continuation-aware config hash that
# exempts days/restart_from.
if [[ -n "${LATEST_CKPT}" && -f "${OUTDIR}/run_manifest.json" ]]; then
    mv "${OUTDIR}/run_manifest.json" "${OUTDIR}/run_manifest.day${LATEST_DAY}.json"
    echo "  Archived prior link manifest -> run_manifest.day${LATEST_DAY}.json"
fi

cd "${REPO}"
# CONFIG_YAML override: debugging configs (e.g. sub-daily diag_days for NaN
# localization) without touching the tracked production YAML.
: "${CONFIG_YAML:=${REPO}/config/amip/amip_production.yaml}"
CMD=( "${PY}" -u scripts/run/run_amip.py
      --config "${CONFIG_YAML}"
      "${AMIP_PATH_FLAGS[@]}"
      --days "${DAYS_THIS_JOB}"
      --checkpoint-days 10
      --max-wallclock-seconds "${MAX_WALLCLOCK_SECONDS}"
      --output "${OUTDIR}" )
[[ -n "${LATEST_CKPT}" ]] && CMD+=( --restart-from "${LATEST_CKPT}" )
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && CMD+=( ${EXTRA} )

echo "[chain] launching: ${CMD[*]}"
"${CMD[@]}"
RC=$?
echo "[chain] run_amip exit=${RC}  $(date -u)"

if (( RC == 0 )); then
    NEW_CKPT=$(ls -1 "${OUTDIR}"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1 || true)
    NEW_DAY=0
    [[ -n "${NEW_CKPT}" ]] && NEW_DAY=$(basename "${NEW_CKPT}" | sed -E 's/checkpoint_day_0*([0-9]+)\.npz/\1/')
    if (( NEW_DAY < TARGET_DAYS )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        NEXT=$(sbatch --account=bb1596 \
                 --dependency="afterok:${SLURM_JOB_ID}" \
                 --job-name="${SLURM_JOB_NAME:-amip_mpas}" \
                 --time=12:00:00 --gres=gpu:1 --constraint=a100_80 --mem=64G \
                 --export="ALL,NAME=${NAME},ROOT=${ROOT},TARGET_DAYS=${TARGET_DAYS},MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS},EXTRA=${EXTRA:-}" \
                 "$0" | awk '{print $NF}')
        echo "[chain] ${NEW_DAY}/${TARGET_DAYS} d done; queued next link: ${NEXT}"
    else
        echo "[chain] reached ${NEW_DAY}/${TARGET_DAYS} d — chain complete."
    fi
fi
exit "${RC}"
