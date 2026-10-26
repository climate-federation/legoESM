#!/bin/bash -l
#SBATCH --job-name=cube_kt_trace
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=14
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=cube_kt_trace.%j.log
# ===========================================================================
# WHERE DOES THE CUBE'S WEAK-SCALING COST GO? Timed trace, matched pair.
#
# THE MEASUREMENT BEING EXPLAINED. Three CPU rows give every device the
# SAME local tile (96 x 96 columns x 26 levels), so they are a controlled
# weak triple and ideal scaling is a flat wall clock. Measured instead,
# in BOTH precisions:
#     2 tiles/face, 24 devices, C192:  258.4 ms/step (f32)
#     3 tiles/face, 54 devices, C288: 1282.2 ms/step (f32)
#     4 tiles/face, 96 devices, C384:  924.6 ms/step (f32)
# The excess to explain at 3 tiles/face is 1282.2 - 258.4 = 1023.8 ms/step.
#
# THE SUSPECT, AND WHY IT IS ONLY A SUSPECT. The compiled census shows
# twelve all-to-alls at 2 tiles/face becoming twelve all-gathers at 3 and
# sixteen at 4, while the neighbour exchanges move only 396 -> 440. BOTH
# reviewers refused that attribution: a count correlation is not a cost,
# 54 devices being WORSE than 96 does not fit a gather-volume story, and
# the node count and rank placement are unexcluded.
#
# (a) NUMBER: wall time per step in which each collective family is
#     OCCUPIED -- the union of its intervals on the traced ranks -- at 2
#     and 3 tiles per face, from the real multi-process executable.
# (b) WHAT IT CAN AND CANNOT DECIDE. Both reviewers were explicit and an
#     earlier version of this header got this wrong, so it is spelled out.
#     Occupancy CANNOT CONFIRM a cause: a collective's duration includes
#     waiting for late participants, so a gather that is cheap on the wire
#     occupies the whole step it spends waiting for someone else's
#     arithmetic. A large gather number therefore means "the step waits
#     HERE", not "the gather costs this".
#     Occupancy CAN RULE OUT ONE THING, narrowly: if the gather family
#     occupies far less than the 1023.8 ms/step excess on every traced
#     rank, then gathers are not where the TRACED RANKS' step goes. That
#     is not the same as dropping the suspect, because a gather on an
#     untraced rank can delay a later operation that IS traced, and would
#     then show up here as permute or as idle rather than as gather.
#     Whatever survives goes to a controlled intervention, which is the
#     only thing that confirms.
# (c) A trace is the only way to see which family waits; the compiled HLO
#     is already known and settled nothing. Both arms are minutes on the
#     free CPU partition and neither needs a GPU.
#
# RANK LOCALITY. The bench traces ranks 0-3 only, so a collective whose
# participant group excludes them is invisible here. A small number is
# therefore evidence about the traced ranks, not about the executable.
# Reported as such.
#
# KNOWN-ANSWER GATE, ENFORCED not printed. The analyzer is handed each
# arm's own compiled census -- read back from the row the bench writes,
# never hardcoded -- and refuses with exit 3 unless every family appears
# exactly (HLO count) x (devices per rank) x (traced steps) times. One
# device per rank under this launch. The same gate passed on a
# virtual-CPU capture at 396 x 24 x 3 = 28512 permutes.
#
# TRACE WINDOW. The bench opens the trace at iteration == warmup and
# closes it at iteration == min(warmup + 4, steps - 1). With
# --steps 16 --warmup 8 that is iterations 8, 9, 10, 11: FOUR steps. The
# window is NOT independent of --steps -- at --steps 12 it would be three
# -- so the analyzer's divisor is tied to this launch's numbers.
#
# WHAT IS HELD AND WHAT IS NOT. Held: code, precision (pinned float32
# explicitly, not inherited), local tile, step and warmup counts, physics
# off, one allocation. NOT held, and named as confounds rather than
# claimed away: the two arms use 6 and 14 nodes, so the off-node fraction
# and the rank placement differ, and 54 ranks do not fill 14 nodes evenly.
# Resolution differs by construction -- that is what holds the tile fixed.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cube_kt_trace_ab.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR" || exit 2
REPO="$SUBMIT_DIR"
PP="$REPO/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$REPO/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=0          # pinned: the timed rows above are float32
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
BENCH=scripts/bench/bench_cube_tiled_step_scaling.py
OUT="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/cube_kt_trace_j${SLURM_JOB_ID}}"
# A reused output directory would let a stale capture be analysed as if it
# were this job's (codex P1). Refuse rather than mix.
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT already exists — a stale capture would be analysed"
  exit 2
fi
mkdir -p "$OUT" || exit 2
echo "outdir=$OUT"
NPN=4
CPT=$((SLURM_CPUS_ON_NODE / NPN))
TRACED_STEPS=4                   # see TRACE WINDOW above
FAILED=0

run_arm () {   # kt res n_devices
  local KT=$1 RES=$2 ND=$3
  local NN=$(( (ND + NPN - 1) / NPN ))
  local TD="$OUT/kt${KT}"
  mkdir -p "$TD"
  echo "=== arm kt=$KT res=$RES devices=$ND nodes=$NN $(date -Is)"
  srun --nodes="$NN" --ntasks="$ND" --ntasks-per-node="$NPN" \
       --cpus-per-task="$CPT" --cpu-bind=cores --overlap \
       "$PY" "$BENCH" --kt "$KT" --resolution "$RES" --nlev 26 \
       --steps 16 --warmup 8 --multicontroller \
       --profile-dir "$TD" --out "$OUT/kt${KT}.jsonl" \
       > "$OUT/kt${KT}.bench.log" 2>&1
  local rc=$?
  tail -5 "$OUT/kt${KT}.bench.log"
  if [ $rc -ne 0 ]; then
    echo "ARM FAILED kt=$KT rc=$rc — no attribution will be made for it"
    FAILED=1
    return 1
  fi
  if [ ! -s "$OUT/kt${KT}.jsonl" ]; then
    echo "ARM PRODUCED NO ROW kt=$KT — refusing to attribute"
    FAILED=1
    return 1
  fi
  echo "arm kt=$KT ok"
}

attribute () {   # kt
  local KT=$1
  local TD="$OUT/kt${KT}"
  echo "--- attribution kt=$KT"
  # The expectation is READ BACK from this arm's own row, so it can never
  # drift from the executable that was traced.
  local CENSUS
  CENSUS=$("$PY" - "$OUT/kt${KT}.jsonl" <<'EOF'
import json, sys
row = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
c = row["metadata"]["extra"]["hlo_collectives"]
name = {"collective_permute": "permute", "all_gather": "all-gather",
        "all_to_all": "all-to-all", "all_reduce": "all-reduce",
        "reduce_scatter": "reduce-scatter"}
# A census key this mapping does not know would be silently dropped from
# the expectation, and the gate would then pass with a whole family
# unaccounted for (GLM). Refuse instead.
unknown = sorted(k for k, v in c.items() if v and k not in name
                 and k != "total")
if unknown:
    sys.exit(f"census has families this probe cannot map: {unknown}")
print(json.dumps({name[k]: v for k, v in c.items()
                  if k in name and v}))
EOF
)
  local crc=$?
  if [ $crc -ne 0 ] || [ -z "$CENSUS" ]; then
    echo "could not read the compiled census for kt=$KT — REFUSING"
    FAILED=1
    return 1
  fi
  echo "expected census: $CENSUS"
  "$PY" scripts/bench/analyze_jax_trace_gaps.py \
      --trace-root "$TD" --steps "$TRACED_STEPS" --time-by-family \
      --devices-per-rank 1 --expect-ranks 4 --expect-census "$CENSUS" \
      --out "$OUT/kt${KT}_family.json"
  local arc=$?
  if [ $arc -ne 0 ]; then
    echo "ATTRIBUTION REFUSED kt=$KT rc=$arc"
    FAILED=1
    return 1
  fi
}

run_arm 2 192 24 && attribute 2
run_arm 3 288 54 && attribute 3

if [ $FAILED -ne 0 ]; then
  echo "CUBE_KT_TRACE_INCOMPLETE — at least one arm or attribution failed"
  exit 1
fi
echo "CUBE_KT_TRACE_DONE"
