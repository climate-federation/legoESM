#!/bin/bash -l
#SBATCH --job-name=ranks_nodes
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8   # max; each arm sets its own
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=ranks_nodes_%j.log
# ===========================================================================
# IS THE CPU KINK ABOUT RANK COUNT OR ABOUT NODE COUNT?
#
# The CPU atmosphere costs 51% MORE time going 64 -> 128 ranks (822 -> 1240
# ms at 4 ranks/node, one revision, one job). Three explanations are already
# dead: the partition degrades only smoothly, the per-rank core share is
# refuted (more cores wins at every local problem size), and the wide halo is
# worth two points. What is left is the inter-node exchange -- and at 4 ranks
# per node, going 64 -> 128 ranks ALSO goes 16 -> 32 nodes. The two have never
# been separated.
#
# THIS SEPARATES THEM by holding the NODE count at 16 and putting 8 ranks on
# each, so 64 and 128 ranks both fit inside 16 nodes:
#
#     ranks   nodes at 4/node   nodes at 8/node
#       64           16                8
#      128           32               16
#
# If the 64 -> 128 slowdown is about crossing more nodes, it should shrink or
# vanish here. If it is about rank count itself, it should survive.
#
# A CONFOUND, NAMED, NOT HIDDEN: 8 ranks per node also halves the cores per
# rank, 64 -> 32. That axis was measured on 2026-09-23 and more cores always
# won, so this layout is handicapped on it. That makes the test CONSERVATIVE
# in one direction only: if the kink SHRINKS despite the handicap, the
# inter-node term is real and large. If the kink GROWS, the two effects are
# tangled and this job cannot separate them -- which is a reportable outcome,
# not a failure.
#
# FALSIFIABILITY, WRITTEN BEFORE SUBMIT. The reference is the measured 4-per-
# node ratio, 1240/822 = 1.51.
#   INTER-NODE: the 64 -> 128 ratio here is at or below 1.15. Crossing nodes
#     is what costs, and the fix is in the exchange, not the rank count.
#   RANK COUNT: the ratio is 1.4 or above. More ranks costs regardless of how
#     they are placed, which points at the collective's participant count.
#   TANGLED: anything between, or a ratio above 1.51. Report as unseparated.
#
# Two repeats, arms interleaved, and both rank counts run inside ONE
# allocation so the machine is the same for both.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cpu_ranks_vs_nodes.sbatch
# ===========================================================================
set -uo pipefail
trap 'echo "RANKS_NODES_INCOMPLETE: killed at $(date +%T)"' TERM
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

# The node count is FIXED for every arm; the ranks-per-node varies to reach
# the rank count. Deriving nodes from ranks is what broke the first attempt:
# it put 64 ranks on 8 nodes and 128 on 16, so node count doubled alongside
# rank count and the confound survived untouched.
NN=${SLURM_JOB_NUM_NODES:?}
SUB=9
# The receipts under investigation are 26-level. This stays at 26 on purpose
# even though the ladder moved to 40 today: it exists to explain THOSE rows.
NLEV=26
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/ranks_nodes_j${SLURM_JOB_ID}}"
[ -e "$OUTDIR" ] && { echo "REFUSING: $OUTDIR exists" >&2; exit 1; }
mkdir -p "$OUTDIR"
echo "provenance: $($PY -c 'import legoesm.parallel.sharded_dynamics as m; print(m.__file__)')"
echo "sha=$(git -C "$REPO" rev-parse --short HEAD) nodes=$NN nlev=$NLEV outdir=$OUTDIR"

want=0; ok=0
for REP in 1 2; do
  for ND in 64 128; do
    PPN=$(( ND / NN ))
    want=$(( want + 1 ))
    TAG="nd${ND}_r${REP}"
    echo "=== $TAG nodes=$NN ppn=$PPN $(date +%T) ==="
    if timeout --kill-after=30 900 srun --export=ALL --nodes="$NN" \
        --ntasks="$ND" --ntasks-per-node="$PPN" --cpu-bind=cores \
        --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))" \
        --kill-on-bad-exit=1 \
        "$PY" scripts/bench/bench_mpas_spmd_scaling.py --multicontroller \
        --n-devices "$ND" --subdivision "$SUB" --nlev "$NLEV" --lloyd 0 \
        --wide-halo --partition-method sfc --steps 12 --warmup 3 \
        --out "$OUTDIR/${TAG}.jsonl"; then
      ok=$(( ok + 1 ))
    else
      echo "ARM FAILED $TAG rc=$?"
    fi
  done
done

echo "=== done: $ok of $want arms ==="
[ "$ok" -lt "$want" ] && { echo "RANKS_NODES_INCOMPLETE: $(( want - ok )) missing"; exit 1; }
$PY - "$OUTDIR" "$NN" <<'PYEOF'
import glob, json, re, sys, os
from collections import defaultdict
out_dir, nn = sys.argv[1], int(sys.argv[2])
best = defaultdict(list)
for f in sorted(glob.glob(os.path.join(out_dir, "nd*_r*.jsonl"))):
    nd = int(re.search(r"nd(\d+)_", f).group(1))
    for line in open(f):
        ms = json.loads(line).get("steady_median_ms")
        if ms:
            best[nd].append(float(ms))
for nd in sorted(best):
    v = best[nd]
    print(f"{nd:>4} ranks on {nn} nodes, {nd // nn} per node, "
          f"{256 // (nd // nn)} cores each: {min(v):8.1f} ms "
          f"(spread {max(v) - min(v):.1f})")

# KNOWN ANSWER: at 64 ranks on 16 nodes this is exactly the production
# layout, 4 ranks per node with the full core share, which measured 822 ms.
# If this arm is far from that, the harness is wrong before any ratio is read.
if 64 in best:
    got, want = min(best[64]), 822.0
    off = abs(got - want) / want
    print(f"\nknown answer: the 64-rank arm IS the production layout and "
          f"measured {got:.0f} ms against 822 ms recorded, {off * 100:.0f}% off.")
    if off > 0.15:
        print("HARNESS SUSPECT: that is too far from the recorded number to "
              "read a ratio off this job. Resolve before interpreting.")
        print("RANKS_NODES_DONE")
        raise SystemExit(0)

if 64 in best and 128 in best:
    r = min(best[128]) / min(best[64])
    spread = ((max(best[128]) - min(best[128])) / min(best[128])
              + (max(best[64]) - min(best[64])) / min(best[64]))
    print(f"\n64 -> 128 ranks at a FIXED {nn} nodes: x{r:.2f} "
          f"(+/- {r * spread:.2f}); when node count doubled too it was x1.51")
    print("The confound that remains, named: holding nodes fixed forces the "
          "ranks per node to double, which halves the cores per rank. That "
          "axis was measured and more cores always won, so this layout is "
          "handicapped for the 128-rank arm.")
    if r <= 1.15:
        print("INTER-NODE: the slowdown largely disappears once both rank "
              "counts sit on the same nodes, DESPITE the 128-rank arm having "
              "half the cores per rank. Crossing nodes is what costs.")
    elif r >= 1.40:
        print("NOT INTER-NODE ALONE: the slowdown survives at constant node "
              "count. It follows the rank count -- or the cores-per-rank "
              "halving that came with it, which this job cannot separate.")
    else:
        print("BETWEEN: the ratio sits between the registered bounds. "
              "Unseparated.")
print("RANKS_NODES_DONE")
PYEOF
