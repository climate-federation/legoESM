#!/bin/bash -l
#SBATCH --job-name=share_sz
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=share_sz_%j.log
# ===========================================================================
# DOES THE BEST CORE SHARE DEPEND ON HOW MUCH WORK A RANK HOLDS?
#
# This is the cheap half of the CPU regression question, and it can REFUTE
# the explanation before a 32-node sweep runs.
#
# THE REGRESSION: the CPU atmosphere gets SLOWER from 64 to 128 ranks
# (822 -> 1240 ms, one revision, one job). The partition does degrade at 128,
# but smoothly -- one extra exchange round and 9% more imbalance -- which
# cannot produce a 51% jump and cannot explain 64 being FASTER than 32.
#
# THE REMAINING EXPLANATION: each rank takes its full node share, 64 hardware
# threads at 4 ranks per node, at EVERY rank count. That share was picked on
# one node with a LARGE local problem. Strong scaling shrinks the local
# problem as ranks are added, so the claim is that the best share falls with
# it -- not that 64 is wrong everywhere.
#
# That claim is about LOCAL PROBLEM SIZE, and local problem size can be
# varied on ONE NODE by changing the mesh instead of the rank count. Four
# ranks throughout, so the partition, the halo pattern and the rank layout
# are identical across every arm and the only thing moving is how many cells
# a rank holds and how many threads it has to chew them.
#
# FALSIFIABILITY, WRITTEN BEFORE SUBMIT.
#   SUPPORTS the explanation: the best share at the SMALLEST mesh is smaller
#     than at the largest, by more than the repeat spread. Then a share fixed
#     at 64 is wrong precisely where strong scaling ends up, and the 32-node
#     sweep is worth running to find where the crossover sits at real rank
#     counts.
#   REFUTES it: the penalty for keeping 64 does not rise as the mesh shrinks.
#     Then the LOCAL-SIZE mechanism is dead -- but not the thread pool, which
#     this node cannot see. The multi-node sweep is then TRUNCATED, not
#     cancelled: 128 ranks at 64 against 32, plus a 64-rank control.
#   INCONCLUSIVE: the change is inside the combined repeat uncertainty. The
#     verdict is a penalty TREND with carried uncertainty, never a winner --
#     picking the fastest share makes a verdict out of one millisecond.
#
# This cannot replace the multi-node sweep even if it supports the claim: one
# node has no inter-node fabric, and at 128 ranks the local problem is small
# AND the exchange is expensive at once. It can only refute cheaply.
#
# ARMS INTERLEAVED, TWO REPEATS, and the thread-pool census recorded per
# share -- a sweep over a dead knob varies pinning alone and still moves
# timings.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cpu_share_vs_local_size.sbatch
# ===========================================================================
set -uo pipefail
trap 'echo "SHARE_SZ_INCOMPLETE: killed at $(date +%T)"' TERM
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

ND=4
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/share_sz_j${SLURM_JOB_ID}}"
[ -e "$OUTDIR" ] && { echo "REFUSING: $OUTDIR exists" >&2; exit 1; }
mkdir -p "$OUTDIR"
echo "provenance: $($PY -c 'import legoesm.parallel.sharded_dynamics as m; print(m.__file__)')"
echo "sha=$(git -C "$REPO" rev-parse --short HEAD) outdir=$OUTDIR"

# Subdivision 5 through 8 at 4 ranks gives about 2.6k, 10k, 41k and 164k
# cells per rank -- spanning, and then some, the 20k a rank holds at 128
# ranks on subdivision 9.
MESHES="${MESHES:-5 6 7 8}"
SHARES="${SHARES:-64 32 16 8}"
want=0; ok=0

for REP in 1 2; do
  for SUB in $MESHES; do
    for CPT in $SHARES; do
      want=$(( want + 1 ))
      TAG="sub${SUB}_cpt${CPT}_r${REP}"
      if [ "$REP" = 1 ] && [ "$SUB" = "${MESHES%% *}" ]; then
        srun --export=ALL --ntasks=1 --cpu-bind=cores --cpus-per-task="$CPT" \
          "$PY" "$REPO/scripts/bench/thread_pool_census.py" \
          "$OUTDIR/threads_cpt${CPT}.json" >/dev/null 2>&1
      fi
      echo "=== $TAG $(date +%T) ==="
      if timeout --kill-after=30 600 srun --export=ALL --ntasks="$ND" \
          --ntasks-per-node="$ND" --cpu-bind=cores --cpus-per-task="$CPT" \
          --kill-on-bad-exit=1 \
          "$PY" scripts/bench/bench_mpas_spmd_scaling.py --multicontroller \
          --n-devices "$ND" --subdivision "$SUB" --nlev 26 --lloyd 0 \
          --wide-halo --partition-method sfc --steps 12 --warmup 3 \
          --out "$OUTDIR/${TAG}.jsonl"; then
        ok=$(( ok + 1 ))
      else
        echo "ARM FAILED $TAG rc=$?"
      fi
    done
  done
done

echo "=== done: $ok of $want arms ==="
[ "$ok" -lt "$want" ] && { echo "SHARE_SZ_INCOMPLETE: $(( want - ok )) missing"; exit 1; }
$PY "$SUBMIT_DIR/scripts/bench/_summarise_share_vs_size.py" "$OUTDIR"
