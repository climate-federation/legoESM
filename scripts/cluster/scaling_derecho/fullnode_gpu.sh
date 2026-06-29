#!/bin/bash -l
#PBS -N fullnode_gpu
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# GPU STRONG-SCALING sweep for latlon / icosahedral / spectral -- the GPU half
# of the CPU-vs-A100 comparison.  One rank per GPU (route-A mpi4jax) AT EACH
# resolution, so the GPU side is a real scaling curve -- not a single A100.
# icosahedral and latlon can both go MULTI-NODE (1->2->4->8->16 ...): icosahedral
# via the MPAS cell partition, latlon via the lat-band / 2-D-pencil
# make_latlon_mpi_step wired in #659 -- both genuine domain decompositions with
# no face/divisor cap.  cubed-sphere is a <=6-GPU single-node face shard and
# spectral has no MPI path (1 GPU only); see #641/#660.  NOTE: multi-node latlon
# (>4 GPU) is newly enabled and NOT yet validated on real hardware (#660) --
# verify MPI==serial (cells/rank halves, matched SYPD) on the first run.
# Pair with fullnode_cpu.sh; SAME driver (run_cpu_mpi_scaling.py) + resolutions.
#
# REQUIRES the route-A overlay env (legoesm-gpu with a CUDA-built mpi4jax; see
# README Step 1b) -- multi-GPU mpi4jax halos.  Cubed-sphere is handled by
# cube_strong_gpu.sh (face-scatter, RANKS=1 2 3).
#
# Multi-node: submit_fullnode.sh sets NODES>1 and overrides the qsub `select=`
# to span nodes; this script then sweeps GPU_RANKS up to NODES*4.  Direct:
#   GRID=icosahedral RESOLUTIONS="7 8" GPU_RANKS="1 2 4 8" ./fullnode_gpu.sh
#
# Driven by submit_fullnode.sh (one job per resolution).  Direct:
#   GRID=latlon RESOLUTIONS=256 ./fullnode_gpu.sh
# ===========================================================================
set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ] \
        && [ -f "${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho/_env.sh" ]; then
    SCRIPT_DIR="${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

# Route-A runtime stack (README Step 1b): GNU cray-mpich the bindings were built
# against (default 8.1.32) + cuda.  We deliberately do NOT load
# craype-accel-nvidia80: that module matters only at BUILD time to put the GTL
# on the link line, and mpi4py already has libmpi_gtl_cuda linked in (verified
# with `ldd` -> /opt/cray/pe/mpich/8.1.32/gtl/lib/libmpi_gtl_cuda.so.0).  Loading
# it was an unnecessary env delta vs the validated interactive stack; at runtime
# GPU-direct only needs the GTL findable (it is) + MPICH_GPU_SUPPORT_ENABLED=1.
module load gcc cray-mpich cuda 2>/dev/null || true
export MPICH_GPU_SUPPORT_ENABLED=1          # MPICH side: GPU-aware transfers on
# mpi4jax side: WITHOUT this, mpi4jax stages every halo GPU->host->MPI->host->GPU
# (it prints "Not using CUDA-enabled MPI" and the scaling curve measures the
# host-staging path, not GPU-direct -- misleading multi-node numbers).  Override
# with MPI4JAX_USE_CUDA_MPI=0 only if the overlay's mpi4jax is NOT a CUDA build.
export MPI4JAX_USE_CUDA_MPI="${MPI4JAX_USE_CUDA_MPI:-1}"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

# --- Per-grid sweep (raise on cube / unknown: dispatch hardening) ------------
GRID="${GRID:-${1:-latlon}}"
PHYSICS="${PHYSICS:-none}"
# Sweep precision AND scaling mode (see fullnode_cpu.sh).  The submitter fans
# these out one value per job; back-compat singular PRECISION still works.
PRECISIONS="${PRECISIONS:-${PRECISION:-float32 float64}}"
MODES="${MODES:-strong weak}"
EXTRA=""
# Total GPUs across ALL allocated nodes = (#unique hosts) x (GPUs per node).
# Counting by UNIQUE hosts is correct regardless of $PBS_NODEFILE format -- one
# line per MPI rank (16 lines / 4 unique) OR one line per node (4 lines / 4
# unique) both give 4 nodes -> x4 = 16.  The OLD `wc -l` only worked for the
# per-rank format; if Derecho's GPU queue writes one line per node it under-
# counted to 4 and SKIPped every >4-GPU point.  We do NOT yet know which format
# Derecho uses -- so we LOG both the raw line count and the unique-host count
# below; the next multi-node run reveals it (lines==nodes -> per-node format).
# GPUs/node from the LOCAL nvidia-smi (Derecho A100 nodes = 4); only ever a
# multiplier, never the total (it is blind to the other nodes).  These vars are
# also reused for rank placement (--ppn) further down, so compute them always.
_gpus_per_node="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0)"
[ "${_gpus_per_node:-0}" -ge 1 ] 2>/dev/null || _gpus_per_node=4
_raw_lines=0; _n_nodes=1
if [ -n "${PBS_NODEFILE:-}" ] && [ -r "${PBS_NODEFILE}" ]; then
    _raw_lines="$(grep -c . "$PBS_NODEFILE" 2>/dev/null || echo 0)"
    _n_nodes="$(sort -u "$PBS_NODEFILE" 2>/dev/null | grep -c . )"
fi
[ "${_n_nodes:-0}" -ge 1 ] 2>/dev/null || _n_nodes=1
# Explicit NGPUS override still wins (off-PBS / manual partial runs).
TOTAL_GPUS="${NGPUS:-$(( _n_nodes * _gpus_per_node ))}"
[ "${TOTAL_GPUS:-0}" -ge 1 ] 2>/dev/null || TOTAL_GPUS=4
echo "    GPU allocation: nodes=${_n_nodes} (PBS_NODEFILE lines=${_raw_lines}) x ${_gpus_per_node} GPU/node -> TOTAL_GPUS=${TOTAL_GPUS}"
case "$GRID" in
  cubed-sphere)
    echo "ERROR: cubed-sphere GPU uses cube_strong_gpu.sh (face-scatter)." >&2
    exit 2 ;;
  latlon)
    # Multi-node ladder (powers of 2 -> 2-D pencil decomposition); self-caps to
    # TOTAL_GPUS below, so default is 1 2 4 on one node, up to 1..16 on >=4 nodes.
    # >4-GPU latlon is newly enabled (#659 wiring) and NOT yet validated on real
    # hardware past 4 GPU (#660) -- verify MPI==serial before trusting the curve.
    GPU_RANKS="${GPU_RANKS:-1 2 4 8 16}"     # route-A, one rank per A100
    EXTRA="--latlon-2d"
    RESOLUTIONS="${RESOLUTIONS:-128 256}" ;;
  icosahedral)
    # Multi-node ladder (powers of 2 -> MPAS cell partition); capped below to
    # TOTAL_GPUS so the same default is correct on 1 node (1 2 4) or N nodes.
    GPU_RANKS="${GPU_RANKS:-1 2 4 8 16}"
    RESOLUTIONS="${RESOLUTIONS:-6 7 8}" ;;
  spectral)
    GPU_RANKS="1"                            # no MPI -> 1 GPU only
    RESOLUTIONS="${RESOLUTIONS:-85 170}" ;;
  *)
    echo "ERROR: unknown GRID='$GRID' (latlon | icosahedral | spectral)" >&2
    exit 2 ;;
esac
# Spectral has no MPI path: weak == strong == a single 1-device point (and weak
# ignores RESOLUTIONS), so restrict it to strong only.
[ "$GRID" = spectral ] && MODES="strong"

# rank -> local GPU pin (Cray PALS; run_cpu_mpi_scaling auto-pin does not read
# PALS, so without this every rank grabs GPU 0 -> the eff=0.5 self-blind bug).
PIN='export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-0}}}; exec "$@"'

STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/${GRID}_gpu_${STAMP}}"
mkdir -p "$CAMP"
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

echo "=== $GRID GPU scaling: modes=[$MODES] prec=[$PRECISIONS] gpus=[$GPU_RANKS] (TOTAL_GPUS=$TOTAL_GPUS) res=[$RESOLUTIONS] ==="
echo "    physics=$PHYSICS  outdir=$CAMP"

# Multi-node rank placement: pin _gpus_per_node ranks/node so PALS_LOCAL_RANKID
# resets 0..(gpus/node-1) ON EACH node and the CVD pin maps rank->local GPU.
# Without it PALS may pack all N ranks onto the first node (CVD 0..N-1 -> ranks
# >3 reference absent GPUs, or every rank shares GPU 0 = the eff=0.5 bug) -- the
# multi-node analog of the self-blind pin.  Single-node sweeps (_n_nodes=1) add
# NO flag, so the proven single-node path is byte-identical.  Override with
# PPN=<n> (force a value) or PPN="" (disable, e.g. if PALS rejects --ppn).
PPN_ARG=()
if [ "${PPN:-auto}" = auto ]; then
    [ "${_n_nodes:-1}" -gt 1 ] && PPN_ARG=(--ppn "${_gpus_per_node}")
elif [ -n "${PPN:-}" ]; then
    PPN_ARG=(--ppn "${PPN}")
fi
[ "${#PPN_ARG[@]}" -gt 0 ] && echo "    rank placement: mpiexec ${PPN_ARG[*]} (multi-node, ${_n_nodes} nodes)"

rc_all=0
skipped=""        # multi-node points dropped because the allocation was too small
for PREC in $PRECISIONS; do
 for MODE in $MODES; do
  # weak: resolution auto-derived per rank count (--resolution 0); no res sweep.
  if [ "$MODE" = weak ]; then RES_LIST="0"; else RES_LIST="$RESOLUTIONS"; fi
  for R in $RES_LIST; do
    for N in $GPU_RANKS; do
      if [ "$N" -gt "$TOTAL_GPUS" ]; then
        echo "!!! WARNING: $GRID $MODE $PREC res=$R N=$N > TOTAL_GPUS=$TOTAL_GPUS -- SKIP" >&2
        echo "    (this point needs a larger allocation: NODES=$(( (N + 3) / 4 )) -> select spanning $(( (N + 3) / 4 )) node(s); see submit_fullnode.sh)" >&2
        skipped="${skipped} ${MODE}/${PREC}/res${R}/N${N}"
        continue
      fi
      echo "--- $GRID $MODE $PREC res=$R gpus=$N ---"
      mpiexec ${PPN_ARG[@]+"${PPN_ARG[@]}"} -n "$N" bash -c "$PIN" _ \
          "$PY" scripts/bench/run_cpu_mpi_scaling.py \
          --grid "$GRID" --mode "$MODE" --resolution "$R" \
          --physics "$PHYSICS" --precision "$PREC" \
          --device gpu $EXTRA \
          --output-dir "$CAMP" < /dev/null \
        || { echo "  $MODE $PREC res=$R gpus=$N FAILED rc=$?"; rc_all=1; }
    done
  done
 done
done

"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/${GRID}_gpu_tidy.csv"

if [ -n "$skipped" ]; then
    echo "=== INCOMPLETE SWEEP: TOTAL_GPUS=$TOTAL_GPUS capped these points ===" >&2
    for s in $skipped; do echo "    SKIPPED $s" >&2; done
    echo "    Resubmit with more nodes (NODES>=2 in submit_fullnode.sh) to fill them." >&2
fi

echo "=== DONE rc=$rc_all ===   CSV: $CAMP/${GRID}_gpu_tidy.csv"
exit $rc_all
