#!/bin/bash
# ===========================================================================
# Submit the route-B (SPMD) CPU-node-vs-A100 THROUGHPUT comparison, latlon.
# Fires BOTH sweeps into ONE outdir so a single finalize_scaling.sh aggregates +
# plots them together (CPU column vs GPU column, Mcells/s):
#   routeb_cpu_sweep.pbs  -- 1..16 NODES (1 proc/node), legoesm-mpi
#   routeb_gpu_sweep.pbs  -- 1..16 A100  (1 proc/GPU),  legoesm-gpu + aws-ofi-nccl
# Both run the identical SPMD lat-band code path, so it is a like-for-like
# CPU-vs-GPU comparison.  (Route B has no icosahedral path; MPAS ico uses route
# A -> submit_scaling.sh.)
#
# Usage:
#   LEGOESM_NCCL_OFI_LIB=/glade/work/$USER/nccl-ofi/<tag>/lib \
#     scripts/cluster/scaling_derecho/submit_routeb.sh <outdir> [res1 res2 res3]
#   DRYRUN=1 ... submit_routeb.sh <outdir>      # preview, no jobs
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
QSUB="${QSUB:-qsub}"
PBS_ACCOUNT="${PBS_ACCOUNT:-P08010000}"

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
RES_LIST="${*:-${RES_LIST:-128 256 512}}"
RES_JOINED="$(echo "$RES_LIST" | tr ' ' ':')"
[ "${DRYRUN:-0}" = "1" ] || mkdir -p "$OUT"

if [ -z "${LEGOESM_NCCL_OFI_LIB:-}" ]; then
    echo "WARNING: LEGOESM_NCCL_OFI_LIB unset -> the GPU lane falls back to TCP" >&2
    echo "         sockets (2-3x slower). Build it first:" >&2
    echo "         scripts/cluster/scaling_derecho/build_nccl_ofi.sh" >&2
fi

_sub() {  # _sub <label> <pbs> <-v vars>
    local label="$1" pbs="$2" vars="$3" jid
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "[DRYRUN] ${label}: ${QSUB} -A ${PBS_ACCOUNT} -v ${vars} ${pbs}"
        return 0
    fi
    # Capture the job id; an empty result means qsub errored (e.g. a bad -v) --
    # surface it LOUDLY instead of printing "submitted: <blank>".
    if jid="$("${QSUB}" -A "${PBS_ACCOUNT}" -v "${vars}" "${pbs}")" && [ -n "$jid" ]; then
        echo "  submitted ${label}: ${jid}"
    else
        echo "  !!! FAILED to submit ${label} (qsub error above; -v was: ${vars})" >&2
    fi
}

echo "=== route-B CPU-vs-A100 throughput (latlon), res=[${RES_LIST}] -> ${OUT} ==="
_sub "route-B CPU (1..16 nodes)" "${SCRIPT_DIR}/routeb_cpu_sweep.pbs" \
     "RES_LIST=${RES_JOINED},OUTDIR=${OUT}/routeb_cpu"
_sub "route-B GPU (1..16 A100)"  "${SCRIPT_DIR}/routeb_gpu_sweep.pbs" \
     "RES_LIST=${RES_JOINED},OUTDIR=${OUT}/routeb_gpu,LEGOESM_NCCL_OFI_LIB=${LEGOESM_NCCL_OFI_LIB:-}"

echo "=== when both finish:  scripts/cluster/scaling_derecho/finalize_scaling.sh ${OUT} ==="
echo "    -> ${OUT}/all_tidy.csv + ${OUT}/plots/cpu_vs_gpu_scaling_latlon.png (Mcells/s panels)"
