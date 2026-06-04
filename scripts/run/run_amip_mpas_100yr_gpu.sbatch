#!/usr/bin/env bash
#SBATCH --job-name=amip_mpas_100yr
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=results/amip_mpas_100yr/slurm-%j.out
#SBATCH --error=results/amip_mpas_100yr/slurm-%j.err
#
# 100-year AMIP (CMIP6-like) on the MPAS SCVT Voronoi mesh, single GPU.
#
# Enabled on this branch:
#   * _run_mpas now applies the prescribed SST as the radiative surface
#     boundary (fixed annual-mean, sea-ice-blended per-cell anchor via
#     set_T_sfc_override).  Without it the MPAS radiation used the lowest
#     air level T[...,-1] as the surface temperature, so the column had no
#     external thermal anchor and cold-drifted (<T_atm> 290->257 K in 30 d,
#     decoupled from the 292.6 K SST) — i.e. it was not actually AMIP-forced.
#   * _run_mpas now checkpoints (checkpoint_day_NNNN.npz) + restarts —
#     previously it never wrote checkpoints, so no chain was possible.
#   * Integrator default is ssp_rk54.  dt=240 s is stable for BOTH ssp_rk3
#     and ssp_rk54 here (zero dissipation; GPU job 8088578); ssp_rk54 is
#     kept as the safe default because it tolerates ∇⁴ hyperdiffusion (which
#     ssp_rk3 cannot at large dt) and matches the spectral PE.
#
# Resolution: level-5 SCVT (10242 cells, ~210 km).  dt=240 s sits well below
# the ~450 s cold-start ceiling (a radiative startup transient, NOT a clean
# CFL — the advective/GW CFL diagnostic over-reports dt_max; see core/cfl.py
# mpas advisory) and below the ~300 s pure-dynamics gravity-wave limit, with
# margin.  Validated: 30-day L5 dt=240 run finite, mass-conserved (job 8087144).
#
# Restart chain: --checkpoint-days 365 writes one checkpoint per simulated
# year.  THIS launcher resumes the next SLURM link from the latest
# checkpoint and submits the next link with --dependency=afterany, so the
# 100-yr integration spans as many 7-day jobs as needed.  Filenames are in
# ABSOLUTE simulated days; TARGET_DAYS is absolute.
#
# MPAS-path caveats (documented limits, not bugs):
#   * SST coupling is RADIATIVE ONLY and FIXED (annual-mean).  The prescribed
#     SST anchors the surface longwave (set_T_sfc_override); but there is no
#     seasonal SST cycle (the JIT-static MPAS step bakes the override at first
#     compile — a per-step update would need a traced forcing arg) and no
#     turbulent surface sensible/latent heat flux (needs the edge->cell wind
#     interp, #3).  So this is an SST-anchored DRY radiative-dynamical AMIP.
#   * The MPAS hydrostatic state is dry (u,T,p_s,phis; no moisture), so
#     convection/microphysics are off — no hydrological cycle.
#   * Turbulence is off (MPAS edge->cell wind interp unimplemented, #3).
#   * The MPAS run path writes only lightweight scalar diagnostics
#     (DiagnosticCollector is bypassed, AMIP.md #4).

set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"

: "${RES:=5}"                 # SCVT bisection level (5 = 10242 cells)
: "${NLEV:=40}"
: "${DT:=240.0}"
: "${TARGET_DAYS:=36500}"     # 100 yr, absolute simulated days
: "${RAD:=gray}"              # rrtmgp on the MPAS path is INFEASIBLE at 100 yr:
                              # a 2-day L5 run took 2.5 h+ at 100% GPU (vs 18 s
                              # gray) — ~500x slower, compute-bound (no retrace;
                              # job 8087144).  100 yr with rrtmgp would be
                              # years of walltime.  gray runs ~0.11 sim-day/s ->
                              # ~4 wall-days/100 yr.  Caveat: gray has no CMIP6
                              # GHG/ozone/aerosol forcing (idealized greenhouse).
: "${RAD_UPDATE_STEPS:=15}"   # ~1 h radiation cadence at dt=240 (3600 s)
: "${OUTPUT_DIR:=${REPO}/results/amip_mpas_100yr}"
mkdir -p "${OUTPUT_DIR}"

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HOME=${HOME:-/burg-archive/home/pg2328}

# -------- restart-chain bookkeeping (absolute simulated days) ----------
LATEST_CKPT=$(ls -1 "${OUTPUT_DIR}"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1 || true)
LATEST_DAY=0
if [[ -n "${LATEST_CKPT}" ]]; then
    LATEST_DAY=$(basename "${LATEST_CKPT}" | sed -E 's/checkpoint_day_0*([0-9]+)\.npz/\1/')
    [[ -z "${LATEST_DAY}" ]] && LATEST_DAY=0
fi
DAYS_THIS_JOB=$(( TARGET_DAYS - LATEST_DAY ))

echo "========================================"
echo "  AMIP MPAS 100-yr  job=${SLURM_JOB_ID:-<none>}  host=$(hostname)"
echo "  level=${RES} (10242 cells)  nlev=${NLEV}  dt=${DT}s  rad=${RAD}"
echo "  TARGET=${TARGET_DAYS} d  latest_ckpt_day=${LATEST_DAY}  this_job=${DAYS_THIS_JOB} d"
echo "  restart_from=${LATEST_CKPT:-<fresh>}"
echo "  $(date -u)"
echo "========================================"
nvidia-smi -L 2>&1 | head -2 || true

if (( DAYS_THIS_JOB <= 0 )); then
    echo "[chain] TARGET ${TARGET_DAYS} d already reached (latest ${LATEST_DAY}). Done."
    exit 0
fi

CMD=( "${PY}" -u scripts/run/run_amip.py
    --dataset analytical
    --grid-type voronoi --discretization mpas
    --resolution "${RES}" --nlev "${NLEV}"
    --radiation "${RAD}" --rad-update-steps "${RAD_UPDATE_STEPS}"
    --convection none --microphysics none --turbulence none --gravity-wave-drag none
    --co2-ppmv 415 --ch4-ppbv 1900 --n2o-ppbv 332 --ozone-source standard
    --days "${DAYS_THIS_JOB}" --dt "${DT}"
    --diag-days 30 --checkpoint-days 365
    --output "${OUTPUT_DIR}"
)
# IMPORTANT: do NOT pass --start-day — the chain relies on config.start_day
# staying 0 and on absolute-day checkpoint filenames.  --days is the
# REMAINING days for this link (see ModelDriver._run_mpas restart contract).
if [[ -n "${LATEST_CKPT}" ]]; then
    CMD+=( --restart-from "${LATEST_CKPT}" )
fi

echo "[chain] launching: ${CMD[*]}"
"${CMD[@]}"
RC=$?
echo "[chain] run_amip exit=${RC}  $(date -u)"

# -------- auto-chain the next link if not done and this link succeeded ----
if (( RC == 0 )); then
    NEW_CKPT=$(ls -1 "${OUTPUT_DIR}"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1 || true)
    NEW_DAY=0
    [[ -n "${NEW_CKPT}" ]] && NEW_DAY=$(basename "${NEW_CKPT}" | sed -E 's/checkpoint_day_0*([0-9]+)\.npz/\1/')
    if (( NEW_DAY < TARGET_DAYS )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        NEXT=$(sbatch --dependency=afterany:"${SLURM_JOB_ID}" "$0" | awk '{print $NF}')
        echo "[chain] ${NEW_DAY}/${TARGET_DAYS} d done; queued next link job ${NEXT}"
    else
        echo "[chain] reached ${NEW_DAY}/${TARGET_DAYS} d — chain complete."
    fi
fi
exit "${RC}"
