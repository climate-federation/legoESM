#!/bin/bash
# ===========================================================================
# Top-level submitter for the "full CPU node vs 1 A100" comparison, ONE GRID at
# a time, covering its resolutions.  Each resolution is fanned out into its own
# CPU job + GPU job (so a slow high-res case gets its own walltime and runs in
# parallel with the others).
#
# Dispatch (the two cube scripts differ from the rest -- cube needs the mpi4jax
# face-scatter path, the others use the native MPI driver):
#   cubed-sphere -> cube_scaling_cpu.sh   + cube_scaling_gpu.sh (RANKS=1)
#   latlon/ico/spectral -> scaling_cpu.sh + scaling_gpu.sh
#
# THE single supported way to run scaling on Derecho.  Everything else in this
# directory is a building block this script drives.
#
# Usage:
#   scripts/cluster/scaling_derecho/submit_scaling.sh <outdir> <grid> [res ...]
#     <outdir> : REQUIRED scratch dir for all results, e.g.
#                $SCRATCH/legoesm_scaling/cmp01 (each job writes a unique subdir)
#     <grid>   : cubed-sphere | latlon | icosahedral | spectral
#     [res]    : resolutions to cover (default per-grid list if omitted)
#
#   # default resolutions for the grid:
#   submit_scaling.sh $SCRATCH/legoesm_scaling/run1 latlon
#   # explicit resolutions:
#   submit_scaling.sh $SCRATCH/legoesm_scaling/run1 cubed-sphere 48 96 192
#   # knobs (pass through to the jobs) + dry run / one tier:
#   PHYSICS=moist PRECISION=float32 DRYRUN=1 submit_scaling.sh $SCRATCH/x icosahedral
#   CPU_ONLY=1 submit_scaling.sh $SCRATCH/x spectral
# ===========================================================================
set -euo pipefail

# Resolve repo root from this script's location so each job's PBS_O_WORKDIR (=
# the dir we qsub from) resolves _env.sh and the relative job-script paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
SD="scripts/cluster/scaling_derecho"
QSUB="${QSUB:-qsub}"
# PBS account: passed to every qsub via -A so a new user sets it ONCE (here or in
# _env.sh / their shell) instead of editing the #PBS -A header in every job
# script.  Default matches _env.sh's PBS_ACCOUNT.
PBS_ACCOUNT="${PBS_ACCOUNT:-P08010000}"

OUTDIR="${1:-}"
GRID="${2:-}"
if [ -z "$OUTDIR" ] || [ -z "$GRID" ]; then
    echo "usage: $0 <outdir> <cubed-sphere|latlon|icosahedral|spectral> [res ...]" >&2
    echo "  <outdir> is REQUIRED (scratch dir for all results)." >&2
    exit 2
fi
shift 2
RES=("$@")
# Create the results root now (skip on a dry run -- it must not touch the FS).
[ "${DRYRUN:-0}" = "1" ] || mkdir -p "$OUTDIR"

# Default per-grid resolutions (match the underlying scripts) + reject unknowns.
if [ "${#RES[@]}" -eq 0 ]; then
    case "$GRID" in
        cubed-sphere) RES=(48 96 192) ;;
        latlon)       RES=(128 256) ;;
        icosahedral)  RES=(6 7 8) ;;
        spectral)     RES=(85 170) ;;
        *) echo "ERROR: unknown grid '$GRID'" >&2; exit 2 ;;
    esac
else
    case "$GRID" in
        cubed-sphere|latlon|icosahedral|spectral) ;;
        *) echo "ERROR: unknown grid '$GRID'" >&2; exit 2 ;;
    esac
fi

# Optional pass-through knobs (each prefixed with a comma so they append cleanly
# to the per-job -v var string).
EXTRA_VARS=""
if [ -n "${PHYSICS:-}" ];   then EXTRA_VARS="${EXTRA_VARS},PHYSICS=${PHYSICS}"; fi

# Precision is FANNED OUT one value per job (so f32 and f64 run as separate queue
# jobs -- parallelism, and a slow f64 case can't starve f32).  Default = the full
# matrix; a legacy singular PRECISION still selects one.  NON-cube grids only;
# the cube pair (Phase 2) keeps its current single-precision behaviour.
PRECISIONS="${PRECISIONS:-${PRECISION:-float32 float64}}"

# Multi-node GPU sweep (icosahedral + latlon).  NODES>1 overrides the GPU job's
# `select=` to span N nodes (4 A100/node); scaling_gpu.sh's default GPU_RANKS
# (1 2 4 8 16 ...) then self-caps to the granted TOTAL_GPUS = NODES*4, so the
# A100 curve runs past one node.  Allowed for the two grids with a genuine
# domain decomposition: icosahedral (MPAS cell partition) and latlon (lat-band /
# 2-D pencil make_latlon_mpi_step, wired in #659).  cubed-sphere is a <=6-GPU
# single-node face shard and spectral has no MPI path, so both stay rejected
# (#641/#660).  CPU side is unchanged (full-node 1..128 rank sweep, one node).
# NOTE: multi-node latlon (>4 GPU) is newly enabled and NOT yet validated on real
# hardware (#660) -- confirm MPI==serial (cells/rank halves, matched SYPD) on the
# first run before trusting the curve.
NODES="${NODES:-1}"
GPU_SELECT=()
if [ "${NODES}" -gt 1 ]; then
    case "$GRID" in
        icosahedral|latlon) ;;   # genuine domain decomposition -> multi-node OK
        *)
            echo "ERROR: NODES>1 (multi-node GPU) supported only for icosahedral|latlon; got '$GRID' (cubed-sphere=<=6-GPU single-node face shard, spectral=no MPI; see #641/#660)." >&2
            exit 2 ;;
    esac
    GPU_SELECT=(-l "select=${NODES}:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB")
    echo "=== multi-node GPU: ${NODES} nodes x 4 A100 = $((NODES * 4)) GPUs (${GRID}) ==="
fi

submit() {  # submit "<label>" <qsub args...>
    local label="$1"; shift
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "[DRYRUN] ${label}: ${QSUB} -A ${PBS_ACCOUNT} $*"
        return 0
    fi
    local jobid
    jobid="$(${QSUB} -A "${PBS_ACCOUNT}" "$@")"
    echo "  submitted ${label}: ${jobid}"
}

echo "=== repo: ${REPO_ROOT} ==="
echo "=== CPU-vs-A100 scaling: grid=${GRID}  resolutions=[${RES[*]}] ==="
echo "=== outdir: ${OUTDIR} ==="

# --- Non-cube fan-out: one WEAK job per precision (no res sweep -- weak derives
#     its own per-rank resolution) + one STRONG job per (resolution, precision).
#     Each job gets a UNIQUE CAMP so the per-job aggregate never races; the final
#     finalize_scaling.sh re-aggregates the whole $OUTDIR tree.  $2 = backend
#     ("cpu"|"gpu"); GPU jobs carry the (possibly multi-node) GPU_SELECT. -------
emit_noncube() {
    local bk="$1" script tag
    if [ "$bk" = gpu ]; then script="$SD/scaling_gpu.sh"; tag="a100"
    else                       script="$SD/scaling_cpu.sh"; tag="cpu"; fi
    local prec camp vars R
    for prec in $PRECISIONS; do
        # WEAK (spectral has no MPI -> no weak curve; skip it).  STRONG_ONLY=1
        # skips the weak jobs entirely (the cpu-vs-gpu plot uses strong only).
        if [ "$GRID" != "spectral" ] && [ "${STRONG_ONLY:-0}" != "1" ]; then
            camp="${OUTDIR}/${GRID}_${tag}_weak_${prec}"
            vars="GRID=${GRID},MODES=weak,PRECISIONS=${prec},CAMP=${camp}${EXTRA_VARS}"
            if [ "$bk" = gpu ]; then
                submit "gpu ${GRID} weak prec=${prec}" ${GPU_SELECT[@]+"${GPU_SELECT[@]}"} -v "$vars" "$script"
            else
                submit "cpu ${GRID} weak prec=${prec}" -v "$vars" "$script"
            fi
        fi
        # STRONG: one job per (resolution, precision)
        for R in "${RES[@]}"; do
            camp="${OUTDIR}/${GRID}_${tag}_strong_res${R}_${prec}"
            vars="GRID=${GRID},MODES=strong,RESOLUTIONS=${R},PRECISIONS=${prec},CAMP=${camp}${EXTRA_VARS}"
            if [ "$bk" = gpu ]; then
                submit "gpu ${GRID} strong res=${R} prec=${prec}" ${GPU_SELECT[@]+"${GPU_SELECT[@]}"} -v "$vars" "$script"
            else
                submit "cpu ${GRID} strong res=${R} prec=${prec}" -v "$vars" "$script"
            fi
        done
    done
}

if [ "$GRID" = "cubed-sphere" ]; then
    # Cube keeps its existing single-precision strong-only pair (Phase 2 will
    # add precision/mode via the face-scatter driver).
    for R in "${RES[@]}"; do
        CPU_CAMP="${OUTDIR}/${GRID}_cpu_res${R}"     # unique per (grid,backend,res)
        GPU_CAMP="${OUTDIR}/${GRID}_a100_res${R}"
        CPU_VARS="STRONG_RES=${R},CAMP=${CPU_CAMP}${EXTRA_VARS}"
        GPU_VARS="STRONG_RES=${R},CAMP=${GPU_CAMP}${EXTRA_VARS}"
        if [ "${GPU_ONLY:-0}" != "1" ]; then
            submit "cpu ${GRID} res=${R} (rank sweep)" -v "$CPU_VARS" "$SD/cube_scaling_cpu.sh"
        fi
        if [ "${CPU_ONLY:-0}" != "1" ]; then
            submit "gpu ${GRID} res=${R} (A100 sweep)" -v "$GPU_VARS" "$SD/cube_scaling_gpu.sh"
        fi
        # Opt-in route-B cube CPU lane (#764): jax.distributed/gloo cs-spmd,
        # a SEPARATE curve from the route-A face-scatter above (different
        # stack; extends past 6 via 6*kt^2).  Off by default so the standard
        # campaign is unchanged; CUBE_ROUTEB=1 adds it into the same $OUTDIR
        # (finalize re-aggregates the whole tree, keying on grid+n_devices).
        if [ "${CUBE_ROUTEB:-0}" = "1" ] && [ "${GPU_ONLY:-0}" != "1" ]; then
            RB_CAMP="${OUTDIR}/${GRID}_cpu_routeb_res${R}"
            RB_VARS="STRONG_RES=${R},CAMP=${RB_CAMP}${EXTRA_VARS}"
            submit "cpu ${GRID} res=${R} (route-B cs-spmd)" \
                -v "$RB_VARS" "$SD/cube_scaling_cpu_routeb.sh"
        fi
    done
else
    [ "${GPU_ONLY:-0}" != "1" ] && emit_noncube cpu
    [ "${CPU_ONLY:-0}" != "1" ] && emit_noncube gpu
fi

echo "=== done ===  watch the queue with: qstat -u \$USER"
