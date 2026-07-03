#!/bin/bash -l
#PBS -N scaling_cpu
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=premium
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# CPU STRONG-SCALING sweep for latlon / icosahedral / spectral -- the CPU half
# of the CPU-vs-A100 comparison.  Runs as a BATCH job or interactively.  Pair
# with scaling_gpu.sh (the GPU strong-scaling sweep).
#
# latlon and icosahedral are genuinely domain-decomposed.  STRONG scaling sweeps
# by FULL NODE -- ranks 128, 256, 512, 1024, 2048 = 1,2,4,8,16 nodes (one rank
# per core, all cores of each node) -- so the CPU curve is a full-node-vs-full-
# A100-node comparison; sub-node core points are NOT swept.  WEAK scaling keeps
# the fine per-core ladder (1,2,4,...) so the constant-cells-per-rank efficiency
# curve stays resolved.  Spectral has no MPI path (rank-1 only); it fills the
# node with XLA threads as a single point.  Driver: run_cpu_mpi_scaling.py
# (mpi4jax, PALS-safe).  Cubed-sphere is handled by cube_scaling_cpu.sh.
#
# Driven by submit_scaling.sh (one job per resolution).  Direct:
#   GRID=latlon RESOLUTIONS=256 ./scaling_cpu.sh
#   qsub -v GRID=icosahedral,RESOLUTIONS=7 scaling_cpu.sh
# ===========================================================================
set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ] \
        && [ -f "${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho/_env.sh" ]; then
    SCRIPT_DIR="${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-mpi}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"
module load gcc cray-mpich 2>/dev/null || true   # match the mpi4py/mpi4jax build ABI

# --- Per-grid sweep (raise on cube / unknown: dispatch hardening) ------------
GRID="${GRID:-${1:-latlon}}"
PHYSICS="${PHYSICS:-none}"
# Sweep precision AND scaling mode.  Defaults cover the full matrix; the
# submitter fans these out one value per job (PRECISIONS=<one>, MODES=<one>) so
# f32/f64 and weak/strong run as separate queue jobs.  Back-compat: a legacy
# singular PRECISION still selects one precision.
PRECISIONS="${PRECISIONS:-${PRECISION:-float32 float64}}"
MODES="${MODES:-strong weak}"
# Rank cap = the MPI slots PBS granted (one line per slot in $PBS_NODEFILE).
# Do NOT use $NCPUS: PBS EXPORTS its own $NCPUS = cores-PER-RANK (=1 for
# mpiprocs=128), so reading it silently SKIPs the whole rank ladder.  Override
# with $LEGOESM_NCPUS for an off-PBS / partial run.
_CORES="$( { [ -n "${PBS_NODEFILE:-}" ] && wc -l < "$PBS_NODEFILE"; } 2>/dev/null | tr -d '[:space:]' )"
_CORES="${LEGOESM_NCPUS:-${_CORES:-128}}"   # off-PBS fallback: a full Derecho node
if [ "${_CORES:-0}" -lt 2 ] 2>/dev/null; then
    echo "WARNING: core count=${_CORES:-?} (<2) -- only the 1-rank case will run." >&2
    echo "         Submit via submit_scaling.sh (qsub), or set LEGOESM_NCPUS." >&2
    _CORES=128
fi
EXTRA=""
THREADS=1                              # pure-MPI default: one thread per rank
# Derecho main-node CPU count (EPYC 128 cores/node).  Used to spread MPI ranks
# across nodes via --ppn on multi-node jobs; `_CORES` already covers ALL nodes
# because it's read from PBS_NODEFILE, so cores_per_node stays a compile-time
# constant tied to the queue's hardware.
_CORES_PER_NODE="${LEGOESM_CORES_PER_NODE:-128}"
# Power-of-two rank ladder capped at `_CORES` = TOTAL cores across all requested
# nodes.  On a single node (_CORES=128) this reproduces the historical
# "1 2 4 8 16 32 64 128" ladder; NODES=16 auto-extends to "... 256 512 1024 2048"
# without any hand-edit here or in submit_scaling.sh.
_rank_ladder() {
    local max="$1" out=1 n=1
    while [ "$n" -lt "$max" ]; do
        n=$(( n * 2 ))
        [ "$n" -le "$max" ] && out="$out $n"
    done
    echo "$out"
}
# Full-NODE ladder for STRONG scaling: rank counts _CORES_PER_NODE, 2x, 4x, ...
# (= 1,2,4,8,16 nodes) capped at the granted _CORES.  On a 16-node alloc this is
# "128 256 512 1024 2048"; on a single node it is just "128" (the one full-node
# point).  Sub-node core counts are deliberately omitted -- the CPU strong curve
# is full-node granularity to mirror the full-A100-node GPU comparison.
_node_ladder() {
    local total="$1" ppn="$2" out="" n="$2"
    while [ "$n" -le "$total" ]; do
        out="${out:+$out }$n"
        n=$(( n * 2 ))
    done
    [ -z "$out" ] && out="$total"   # alloc smaller than one node: use what we have
    echo "$out"
}
case "$GRID" in
  cubed-sphere)
    echo "ERROR: cubed-sphere uses cube_scaling_cpu.sh (6-face scatter)." >&2
    exit 2 ;;
  latlon)
    # strong: full-node ladder; weak: fine per-core ladder.  RANKS=... overrides
    # strong; WEAK_RANKS=... overrides weak.
    STRONG_RANKS="${RANKS:-$(_node_ladder "$_CORES" "$_CORES_PER_NODE")}"
    WEAK_RANKS="${WEAK_RANKS:-$(_rank_ladder "$_CORES")}"
    EXTRA="--latlon-2d"                            # 2-D pencil: low halo at high ranks
    RESOLUTIONS="${RESOLUTIONS:-128 256}" ;;
  icosahedral)
    STRONG_RANKS="${RANKS:-$(_node_ladder "$_CORES" "$_CORES_PER_NODE")}"  # MPAS partition, full nodes
    WEAK_RANKS="${WEAK_RANKS:-$(_rank_ladder "$_CORES")}"
    RESOLUTIONS="${RESOLUTIONS:-6 7 8}" ;;
  spectral)
    STRONG_RANKS="1"; WEAK_RANKS="1"               # no MPI -> single point...
    THREADS="${THREADS_SPECTRAL:-$_CORES_PER_NODE}"  # ...fill ONE node with XLA threads (multi-node NA)
    RESOLUTIONS="${RESOLUTIONS:-85 170}" ;;
  *)
    echo "ERROR: unknown GRID='$GRID' (latlon | icosahedral | spectral)" >&2
    exit 2 ;;
esac
# Spectral has no MPI path: weak == strong == a single 1-device point, and the
# weak path ignores RESOLUTIONS (auto-derives), so restrict it to strong only.
[ "$GRID" = spectral ] && MODES="strong"

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
if [ "$THREADS" -le 1 ]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/${GRID}_cpu_${STAMP}}"
mkdir -p "$CAMP"
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

_NODES_ALLOC=$(( _CORES / _CORES_PER_NODE ))
[ "$_NODES_ALLOC" -lt 1 ] && _NODES_ALLOC=1
echo "=== $GRID CPU scaling: modes=[$MODES] prec=[$PRECISIONS] strong_ranks=[$STRONG_RANKS] weak_ranks=[$WEAK_RANKS] x ${THREADS} thr  res=[$RESOLUTIONS] ==="
echo "    physics=$PHYSICS cores=$_CORES nodes~=$_NODES_ALLOC (ppn=$_CORES_PER_NODE)  outdir=$CAMP"

rc_all=0
for PREC in $PRECISIONS; do
 for MODE in $MODES; do
  # weak: resolution is auto-derived per rank count (--resolution 0 -> constant
  # cells/rank), so there is NO resolution sweep.  strong: explicit fixed sizes.
  if [ "$MODE" = weak ]; then RES_LIST="0"; RANK_LIST="$WEAK_RANKS"
  else RES_LIST="$RESOLUTIONS"; RANK_LIST="$STRONG_RANKS"; fi
  for R in $RES_LIST; do
    for N in $RANK_LIST; do
      if [ "$(( N * THREADS ))" -gt "$_CORES" ]; then
        echo "--- $GRID $MODE $PREC res=$R N=$N: needs $(( N * THREADS )) > $_CORES cores -- SKIP ---"
        continue
      fi
      # Spread ranks across nodes: --ppn = min(N, cores/node).  Single-node
      # runs are unaffected (mpiexec ignores --ppn > N).  Multi-node runs
      # (NODES>1 in submit_scaling.sh) NEED it — without --ppn cray-mpich packs
      # every rank onto the first node, gives the queue a 128-rank ceiling, and
      # silently discards the extra nodes' cores.
      _MPI_PPN=$(( N < _CORES_PER_NODE ? N : _CORES_PER_NODE ))
      echo "--- $GRID $MODE $PREC res=$R ranks=$N ppn=$_MPI_PPN ---"
      mpiexec -n "$N" --ppn "$_MPI_PPN" \
          "$PY" scripts/bench/run_cpu_mpi_scaling.py \
          --grid "$GRID" --mode "$MODE" --resolution "$R" \
          --physics "$PHYSICS" --precision "$PREC" \
          --device cpu $EXTRA \
          --output-dir "$CAMP" < /dev/null \
        || { echo "  $MODE $PREC res=$R ranks=$N FAILED rc=$?"; rc_all=1; }
    done
  done
 done
done

"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/${GRID}_cpu_tidy.csv"

echo "=== DONE rc=$rc_all ===   CSV: $CAMP/${GRID}_cpu_tidy.csv"
exit $rc_all
