#!/bin/bash -l
#SBATCH --job-name=cube_place_ab
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=cube_place_ab.%j.log
# ===========================================================================
# IS THE CUBE'S WEAK-SCALING COST PLACEMENT? One variable: ranks per node.
#
# WHAT THE TRACE FOUND (job 27630818, and it refuted the previous suspect).
# At an identical 96 x 96 x 26 local tile, the collective families occupy,
# per step, on the four traced ranks:
#
#   tiles/face  devices  nodes   exchange      reduction     gather
#            2       24      6   50-79 ms      146-175 ms    absent
#            3       54     14   693-820 ms    228-331 ms    93-224 ms
#
# So the ALL-GATHERS, which the compiled census made the obvious suspect,
# are the SMALLEST of the three at 3 tiles per face. The neighbour
# exchange is the term that explodes: roughly TWELVE times the occupancy
# for five per cent more messages, at identical local work. Each message
# therefore waits about twelve times longer.
#
# THE HYPOTHESIS THIS JOB TESTS. Both reviewers named node count as the
# unexcluded confound in that comparison: 6 nodes versus 14. Every tile in
# a kt x kt tiling has four neighbours whatever kt is, so the partner
# COUNT does not change -- what changes is how many of those partners are
# off-node, and how many ranks share a node's network interface.
#
# (a) NUMBER: exchange occupancy in ms/step for the SAME executable --
#     2 tiles per face, C192, 24 ranks, float32 -- packed 4 ranks per node
#     on 6 nodes, and 2 ranks per node on 12 nodes. Nothing else differs:
#     same resolution, same tile, same code, same allocation, same step
#     and warmup counts.
# (b) CONFIRMS placement as the driver if the sparser packing moves
#     exchange occupancy by more than about 2x. REFUTES it if the two
#     agree within roughly 20%, in which case node count is not what the
#     3-tile arm was measuring and the next suspect is the transport's
#     own scaling with rank count.
# (c) No cheaper test exists: this is a runtime wait, the compiled program
#     is identical in both arms by construction, and the offline schedule
#     estimator has no notion of placement at all.
#
# WHAT MOVES BESIDES PLACEMENT, named rather than claimed away: at 2 ranks
# per node each rank has twice the memory bandwidth and twice the cores
# available to it, so local compute may speed up and shift the balance.
# That biases toward finding a LARGER exchange share in the sparse arm,
# so a null result is the stronger of the two outcomes here.
#
# KNOWN-ANSWER GATE. Both arms must reproduce the compiled census exactly
# through --expect-census, and must hold four rank directories. The 4-per
# -node arm must also reproduce the 259.7 ms/step of the trace job's
# matching arm to within a few per cent; if it does not, the allocation is
# not comparable and neither number may be quoted.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cube_placement_ab.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR" || exit 2
PP="$SUBMIT_DIR/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$SUBMIT_DIR/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=0
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
BENCH=scripts/bench/bench_cube_tiled_step_scaling.py
OUT="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/cube_place_j${SLURM_JOB_ID}}"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists — a stale capture would be analysed"; exit 2
fi
mkdir -p "$OUT" || exit 2
echo "outdir=$OUT"
FAILED=0

arm () {   # ranks_per_node nodes label
  local NPN=$1 NN=$2 LBL=$3
  local TD="$OUT/$LBL"
  local CPT=$((SLURM_CPUS_ON_NODE / NPN))
  mkdir -p "$TD"
  echo "=== arm $LBL: 24 ranks, $NPN per node on $NN nodes, $CPT cpus/task"
  srun --nodes="$NN" --ntasks=24 --ntasks-per-node="$NPN" \
       --cpus-per-task="$CPT" --cpu-bind=cores --overlap \
       "$PY" "$BENCH" --kt 2 --resolution 192 --nlev 26 \
       --steps 16 --warmup 8 --multicontroller \
       --profile-dir "$TD" --out "$OUT/$LBL.jsonl" \
       > "$OUT/$LBL.bench.log" 2>&1
  local rc=$?
  grep -oE "steady_median=[0-9.]+ms/step" "$OUT/$LBL.bench.log" | tail -1
  if [ $rc -ne 0 ] || [ ! -s "$OUT/$LBL.jsonl" ]; then
    echo "ARM FAILED $LBL rc=$rc"; FAILED=1; return 1
  fi
  local CENSUS
  CENSUS=$("$PY" - "$OUT/$LBL.jsonl" <<'EOF'
import json, sys
row = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
c = row["metadata"]["extra"]["hlo_collectives"]
name = {"collective_permute": "permute", "all_gather": "all-gather",
        "all_to_all": "all-to-all", "all_reduce": "all-reduce",
        "reduce_scatter": "reduce-scatter"}
unknown = sorted(k for k, v in c.items() if v and k not in name
                 and k != "total")
if unknown:
    sys.exit(f"census has families this probe cannot map: {unknown}")
print(json.dumps({name[k]: v for k, v in c.items() if k in name and v}))
EOF
)
  [ -z "$CENSUS" ] && { echo "no census for $LBL"; FAILED=1; return 1; }
  echo "expected census: $CENSUS"
  "$PY" scripts/bench/analyze_jax_trace_gaps.py \
      --trace-root "$TD" --steps 4 --time-by-family \
      --devices-per-rank 1 --expect-ranks 4 --expect-census "$CENSUS" \
      --out "$OUT/${LBL}_family.json" || { FAILED=1; return 1; }
}

arm 4 6  "packed4"
arm 2 12 "sparse2"

[ $FAILED -ne 0 ] && { echo "CUBE_PLACE_AB_INCOMPLETE"; exit 1; }
echo "CUBE_PLACE_AB_DONE"
