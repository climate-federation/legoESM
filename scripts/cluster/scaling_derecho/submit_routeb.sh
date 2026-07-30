#!/bin/bash
# ===========================================================================
# Submit the route-B (SPMD/NCCL-fabric) THROUGHPUT campaign into ONE outdir so a
# single finalize_scaling.sh aggregates + plots everything (per-grid Mcells/s):
#   routeb_gpu_sweep.pbs  -- FANS OUT one GPU job per grid (GRIDS, default all 3):
#                            latlon | icosahedral | cubed-sphere (1 proc/GPU,
#                            legoesm-gpu + aws-ofi-nccl).  Each grid uses its OWN
#                            resolution axis + device ladder, so they land as
#                            SEPARATE curves (never overlaid).
#   routeb_cpu_sweep.pbs  -- FANS OUT one CPU-node job per grid (same GRIDS), the
#                            CPU-node half of the CPU-node-vs-A100 comparison
#                            (1 proc/node, gloo).  RUN_CPU=0 skips it.
#
# Usage:
#   LEGOESM_NCCL_OFI_LIB=/glade/work/$USER/nccl-ofi/<tag>/lib \
#     scripts/cluster/scaling_derecho/submit_routeb.sh <outdir> [res1 res2 res3]
#   GRIDS="icosahedral" ... submit_routeb.sh <outdir> 7 8   # one grid + custom res
#   PRECISIONS="float32 float64" ... submit_routeb.sh <outdir>  # both precisions
#     (each precision DOUBLES a lane's runtime -- raise -l walltime to match)
#   RUN_CPU=0 ... submit_routeb.sh <outdir>                 # GPU curves only
#   RUN_GPU=0 ... submit_routeb.sh <outdir>                 # CPU curves only
#   DRYRUN=1 ... submit_routeb.sh <outdir>                  # preview, no jobs
# A shared RES_LIST is passed to the GPU sweep ONLY for a single-grid run (the
# resolution axes differ per grid); multi-grid runs use each grid's own default.
# Accounts: PBS_ACCOUNT sets both; PBS_ACCOUNT_CPU / PBS_ACCOUNT_GPU override per
# backend (CPU and GPU often bill to different Derecho allocations), e.g.
#   PBS_ACCOUNT_CPU=UABC0001 PBS_ACCOUNT_GPU=UABC0002 LEGOESM_NCCL_OFI_LIB=... \
#     submit_routeb.sh <outdir>
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
QSUB="${QSUB:-qsub}"
# CPU and GPU often bill to DIFFERENT Derecho allocations.  PBS_ACCOUNT is the
# default for both; PBS_ACCOUNT_CPU / PBS_ACCOUNT_GPU override per backend.
PBS_ACCOUNT="${PBS_ACCOUNT:-P08010000}"
ACCT_CPU="${PBS_ACCOUNT_CPU:-$PBS_ACCOUNT}"
ACCT_GPU="${PBS_ACCOUNT_GPU:-$PBS_ACCOUNT}"

OUT="${1:-}"
if [ -z "$OUT" ]; then
    echo "usage: $0 <outdir> [res1 res2 res3]   (outdir REQUIRED)" >&2
    exit 2
fi
shift || true
# Resolutions: CLI args win, else $RES_LIST, else the default 3.  Passed to PBS
# COLON-joined -- a comma is PBS's `-v` list delimiter, so a comma-joined value
# would be misparsed as extra variable names ("cannot send environment").  The
# PBS scripts translate ':' back to spaces.
# Track whether the user gave explicit positional RES args (only meaningful for
# a single-grid run -- per-grid default resolution axes differ).
_res_explicit=0; if [ "$#" -gt 0 ]; then _res_explicit=1; fi
RES_LIST="${*:-${RES_LIST:-128 256 512}}"
RES_JOINED="$(echo "$RES_LIST" | tr ' ' ':')"
# PRECISIONS must be FORWARDED EXPLICITLY: `qsub -v` passes only the variables
# it lists (unlike `-V`), so an exported PRECISIONS in the submitting shell is
# silently DROPPED and the job runs the float32 default while the user believes
# they asked for float64.  Same trap class as LEGOESM_REPO.  Colon-joined for
# the same reason as RES_LIST (comma is PBS's `-v` delimiter); the PBS scripts
# translate ':' back to spaces.  Empty unless the user set it, so the lanes keep
# their float32 default and existing submissions are unchanged.
PREC_JOINED="$(echo "${PRECISIONS:-}" | tr ' ' ':')"
# Grids to fan the GPU sweep across (each -> its own job + subdir + curve).
GRIDS="${GRIDS:-latlon icosahedral cubed-sphere}"
_n_grids="$(echo $GRIDS | wc -w | tr -d ' ')"
[ "${DRYRUN:-0}" = "1" ] || mkdir -p "$OUT"

if [ "${RUN_GPU:-1}" = "1" ] && [ -z "${LEGOESM_NCCL_OFI_LIB:-}" ]; then
    echo "WARNING: LEGOESM_NCCL_OFI_LIB unset -> the GPU lane falls back to TCP" >&2
    echo "         sockets (2-3x slower). Build it first:" >&2
    echo "         scripts/cluster/scaling_derecho/build_nccl_ofi.sh" >&2
fi

_sub() {  # _sub <acct> <label> <pbs> <-v vars>
    local acct="$1" label="$2" pbs="$3" vars="$4" jid
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "[DRYRUN] ${label}: ${QSUB} -A ${acct} -v ${vars} ${pbs}"
        return 0
    fi
    # Capture the job id; an empty result means qsub errored (e.g. a bad -v) --
    # surface it LOUDLY instead of printing "submitted: <blank>".
    if jid="$("${QSUB}" -A "${acct}" -v "${vars}" "${pbs}")" && [ -n "$jid" ]; then
        echo "  submitted ${label}: ${jid}"
    else
        echo "  !!! FAILED to submit ${label} (qsub error above; -v was: ${vars})" >&2
    fi
}

echo "=== route-B throughput campaign: GPU grids=[${GRIDS}] -> ${OUT} ==="
echo "    accounts: CPU=${ACCT_CPU}  GPU=${ACCT_GPU}"

# --- GPU: one job per grid (each grid its own resolution axis + device ladder).
#     Skip the whole GPU side with RUN_GPU=0 (CPU-only campaign). --------------
if [ "${RUN_GPU:-1}" = "1" ]; then
    for g in $GRIDS; do
        case "$g" in
          latlon|icosahedral|cubed-sphere|cubed_sphere) ;;
          *) echo "  !!! SKIP unknown GRID='$g' (latlon|icosahedral|cubed-sphere)" >&2; continue ;;
        esac
        vars="OUTDIR=${OUT}/routeb_gpu_${g},LEGOESM_NCCL_OFI_LIB=${LEGOESM_NCCL_OFI_LIB:-},GRID=${g}"
        # `if`, not `[ ... ] &&`: under `set -e` a bare failing AND-list
        # can abort the script, and the empty case is the DEFAULT path.
        if [ -n "$PREC_JOINED" ]; then vars="PRECISIONS=${PREC_JOINED},${vars}"; fi
        # Shared RES override is only meaningful for a single-grid run (the axes
        # differ across grids); multi-grid uses each grid's per-grid default.
        if [ "$_res_explicit" = 1 ] && [ "$_n_grids" -eq 1 ]; then
            vars="RES_LIST=${RES_JOINED},${vars}"
        fi
        _sub "${ACCT_GPU}" "route-B GPU ${g}" "${SCRIPT_DIR}/routeb_gpu_sweep.pbs" "$vars"
    done
else
    echo "    (GPU lanes skipped: RUN_GPU=0)"
fi

# --- CPU: one CPU-node job per grid (the CPU half of the CPU-vs-A100 comparison,
#     1 proc/node gloo).  Skip the whole CPU side with RUN_CPU=0. --------------
if [ "${RUN_CPU:-1}" = "1" ]; then
    for g in $GRIDS; do
        case "$g" in
          latlon|icosahedral|cubed-sphere|cubed_sphere) ;;
          *) continue ;;   # unknown grid already warned on the GPU pass
        esac
        vars="OUTDIR=${OUT}/routeb_cpu_${g},GRID=${g}"
        # `if`, not `[ ... ] &&`: under `set -e` a bare failing AND-list
        # can abort the script, and the empty case is the DEFAULT path.
        if [ -n "$PREC_JOINED" ]; then vars="PRECISIONS=${PREC_JOINED},${vars}"; fi
        if [ "$_res_explicit" = 1 ] && [ "$_n_grids" -eq 1 ]; then
            vars="RES_LIST=${RES_JOINED},${vars}"
        fi
        _sub "${ACCT_CPU}" "route-B CPU ${g}" "${SCRIPT_DIR}/routeb_cpu_sweep.pbs" "$vars"
    done
else
    echo "    (CPU lanes skipped: RUN_CPU=0)"
fi

echo "=== when the jobs finish:  scripts/cluster/scaling_derecho/finalize_scaling.sh ${OUT} ==="
echo "    -> ${OUT}/all_tidy.csv + ${OUT}/plots/  (per-grid Mcells/s panels)"
