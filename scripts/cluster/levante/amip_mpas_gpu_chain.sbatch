#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/slurm_logs/%x-%j.out
# FOLLOWS PRODUCTION, DELIBERATELY (2026-09-23): the production deck became
# the CAM6 suite on that date and this script keeps naming it, because its
# job is to run whatever the authoritative production configuration IS, not
# to reproduce one past run.  Runs started before that date continued from a
# checkpoint must name config/amip/amip_sundqvist_l36.yaml instead.
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
# LAT-LON restart contract is the OPPOSITE: the compiled lane resumes at the
# checkpoint's absolute day and --days is the ABSOLUTE target, so passing
# remaining days computes NEGATIVE steps and the link exits after 0 steps
# (observed 2026-08-02: "Starting compiled run: -51840 steps" looped the llF
# chain forever). Detect the lane from EXTRA.
# Detect the lat-lon lane from EXTRA *or* from a CONFIG_YAML that sets the
# grid (codex: submit_fullphys.sh selects lat-lon via YAML only, which the
# EXTRA-only detector missed — reintroducing the negative-step loop there).
_IS_LATLON=0
[[ " ${EXTRA:-} " == *"--grid-type latlon"* ]] && _IS_LATLON=1
if [[ -f "${CONFIG_YAML:-}" ]] && grep -Eq '^[[:space:]]*grid_type:[[:space:]]*"?latlon' "${CONFIG_YAML}"; then
    # EXTRA overrides YAML, so only trust the YAML when EXTRA has no grid-type.
    [[ " ${EXTRA:-} " != *"--grid-type "* ]] && _IS_LATLON=1
fi
if (( _IS_LATLON )); then
    DAYS_THIS_JOB=${TARGET_DAYS}
else
    DAYS_THIS_JOB=$(( TARGET_DAYS - LATEST_DAY ))
fi
# Lane-confirmed NCCL recipe (measured @64 GPUs; comm-only, no numerics change). Any manual NCCL knob wins.
if [[ -z "${NCCL_MIN_NCHANNELS:-}" && -z "${NCCL_MAX_NCHANNELS:-}" && -z "${NCCL_P2P_NET_CHUNKSIZE:-}" ]]; then
    if (( _IS_LATLON )); then
        export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
        echo "[nccl-recipe] lat-lon lane: NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16 (chunk not isolated; left unset)"
    else
        export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
        echo "[nccl-recipe] mpas lane: NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072"
    fi
else
    echo "[nccl-recipe] manual override kept: NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-} NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-}"
fi

# Lane-confirmed halo comm-wins (measured, opt-in; adopted alongside the NCCL
# recipe of PR #1671). Dormant at gpu:1 (no halo exchange); they engage only
# when a run is scaled to multiple GPUs. Each respects a manual override.
if (( _IS_LATLON )); then
    # ${VAR-1}: unset -> on; set '0' (the sanctioned off-value for BOTH flags;
    # fused-halo honours only '0', packed-exchange also '') stays off.
    export LEGOESM_LATLON_PACKED_EXCHANGE="${LEGOESM_LATLON_PACKED_EXCHANGE-1}"
    export LEGOESM_LATLON_SPMD_FUSED_HALO="${LEGOESM_LATLON_SPMD_FUSED_HALO-1}"
    echo "[comm-recipe] lat-lon: packed stage-entry exchange (25->13 CPs) + fused halo on"
else
    # MPAS wide-halo (-14% @64) is deliberately NOT defaulted: it RAISES on the
    # *_scan integrators (packages/core/.../sharded_dynamics.py: wide-halo needs
    # the unrolled spelling to thread its shrinking masks), and production MPAS
    # runs a scan integrator (--time-integrator auto). Adopting it first needs a
    # switch to an unrolled integrator (e.g. ssp_rk3), a separate decision with
    # its own local-step tradeoff. Set LEGOESM_MPAS_WIDE_HALO=1 manually only
    # with an unrolled integrator.
    :
fi
echo "  This link runs to/for ${DAYS_THIS_JOB} days (target ${TARGET_DAYS}, resume day ${LATEST_DAY})"
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
# ---------------------------------------------------------------------------
# DEVICE COUNT.  Standing user directive 2026-09-23: a production AMIP arm runs
# on AT LEAST FOUR GPUs, never one.  The CAM6 run-1 arm ran on a single device
# at about four simulated days per hour, so a 60-day arm cost roughly fifteen
# hours of wall clock that the sharded mesh did not require.
#
# EXPLICIT, not inherited: the count is named here and in the SBATCH request
# above and in the self-resubmit below, because a default is not a record.
# The MPAS lane federates with the mpi4jax halo backend -- distributed_mode
# 'spmd' is cubed-sphere only (driver/config.py refuses it on a Voronoi mesh)
# -- so this is ONE RANK PER GPU under srun, with --distributed.
# CUDA-aware MPI stays OFF: MPI4JAX_USE_CUDA_MPI=1 segfaults on this cluster.
: "${AMIP_N_GPUS:=4}"
if (( AMIP_N_GPUS < 4 )); then
    echo "[chain] REFUSING: AMIP_N_GPUS=${AMIP_N_GPUS} is below the floor of 4." >&2
    echo "[chain] A production AMIP arm runs on at least four GPUs (user" >&2
    echo "[chain] directive 2026-09-23).  Set AMIP_N_GPUS=4 or higher." >&2
    exit 64
fi

# PAIR CONSISTENCY.  Both arms of a comparison MUST use the same device count:
# the ocean solver converges to a tolerance at one device and runs a fixed
# iteration count when sharded, and the halo path has had device-count
# dependent defects, so a mismatch is a confound rather than a speed
# difference.  The count is written into the run directory on the first link
# and checked on every later one, which also catches a chain that is resumed
# with a different request.  AMIP_PAIR_DIR names the partner arm when there is
# one; give both arms of a pair the same value and neither can drift.
_DEVCOUNT_FILE="${OUTDIR}/.device_count"
if [[ -f "${_DEVCOUNT_FILE}" ]]; then
    _PREV=$(cat "${_DEVCOUNT_FILE}")
    if [[ "${_PREV}" != "${AMIP_N_GPUS}" ]]; then
        echo "[chain] REFUSING: this run started on ${_PREV} GPU(s) and this" >&2
        echo "[chain] link requests ${AMIP_N_GPUS}.  Continuing a run at a" >&2
        echo "[chain] different device count changes the ocean solver and the" >&2
        echo "[chain] halo path mid-stream.  Start a new run directory." >&2
        exit 65
    fi
else
    echo "${AMIP_N_GPUS}" > "${_DEVCOUNT_FILE}"
fi
if [[ -n "${AMIP_PAIR_DIR:-}" && -f "${AMIP_PAIR_DIR}/.device_count" ]]; then
    _PARTNER=$(cat "${AMIP_PAIR_DIR}/.device_count")
    if [[ "${_PARTNER}" != "${AMIP_N_GPUS}" ]]; then
        echo "[chain] REFUSING: partner arm ${AMIP_PAIR_DIR} runs on" >&2
        echo "[chain] ${_PARTNER} GPU(s), this arm requests ${AMIP_N_GPUS}." >&2
        echo "[chain] A pair scored against itself must hold the device count" >&2
        echo "[chain] fixed, or the difference is a confound." >&2
        exit 66
    fi
fi
echo "[chain] device count: ${AMIP_N_GPUS} GPU(s), one rank each${AMIP_PAIR_DIR:+ (pair ${AMIP_PAIR_DIR})}"

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

CMD+=( --distributed )
LAUNCH=( srun --ntasks="${AMIP_N_GPUS}" --ntasks-per-node="${AMIP_N_GPUS}"
         --gpus-per-node="${AMIP_N_GPUS}" --gpu-bind=none --kill-on-bad-exit=1 )
echo "[chain] launching on ${AMIP_N_GPUS} GPU(s): ${LAUNCH[*]} ${CMD[*]}"
"${LAUNCH[@]}" "${CMD[@]}"
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
                 --time=12:00:00 --constraint=a100_80 --mem=0 \
                 --ntasks="${AMIP_N_GPUS}" --ntasks-per-node="${AMIP_N_GPUS}" \
                 --gpus-per-node="${AMIP_N_GPUS}" \
                 --export="ALL,NAME=${NAME},ROOT=${ROOT},TARGET_DAYS=${TARGET_DAYS},MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS},EXTRA=${EXTRA:-},AMIP_N_GPUS=${AMIP_N_GPUS},AMIP_PAIR_DIR=${AMIP_PAIR_DIR:-}" \
                 "$0" | awk '{print $NF}')
        echo "[chain] ${NEW_DAY}/${TARGET_DAYS} d done; queued next link: ${NEXT}"
    else
        echo "[chain] reached ${NEW_DAY}/${TARGET_DAYS} d — chain complete."
    fi
fi
exit "${RC}"
