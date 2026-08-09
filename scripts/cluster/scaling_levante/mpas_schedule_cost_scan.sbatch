#!/bin/bash -l
#SBATCH --job-name=mpas_sched_cost
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=mpas_schedule_cost_scan.%j.log
# ===========================================================================
# Does RECOLOURING still have room on the MPAS halo schedule?  CPU-only,
# ZERO GPU hours.
#
# Note this is NOT a two-way choice between colouring and ownership: a
# redesigned DIRECTED schedule (see the SCOPE note below) is a third path,
# and it is not bounded by the quantity measured here.  What this run
# settles is only whether recolouring THIS undirected graph is exhausted.
#
# WHY THIS RUN EXISTS
# MPAS GPU is the worst-scaling lane we have: measured/modelled-bound 3.16x
# (s8@16) to 4.47x (s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute
# rounds.  Both independent reviews (codex + GLM, 2026-08-07) ranked cutting
# that round count as the top structural lever, at 300-800 LOC and 7-14 days
# for a partitioner with a new objective.  That estimate is only worth
# spending if recolouring is genuinely exhausted, and the colourer's own
# lower bound decides it.
#
# The schedule is a proper EDGE colouring of the device communication graph
# (one colour = one ppermute round) and max_degree is that graph's maximum
# vertex degree, so VIZING bounds the chromatic index: Delta <= chi' <=
# Delta + 1.  Read coloring_gap = n_rounds - max_degree through that:
#     gap == 0 -> PROVABLY OPTIMAL; recolouring headroom is exactly zero.
#     gap == 1 -> indistinguishable from optimal (a Class 2 graph really
#                 needs Delta+1, and deciding Class 1/2 is NP-complete);
#                 nothing provable to win.
#     gap >= 2 -> at least gap-1 rounds of genuine recolouring headroom.
# So recolouring is capped at ~1 round out of 12-14 (<= ~7%) wherever the
# multi-start search lands on Delta or Delta+1 — which it does on every
# configuration probed so far.  This run measures whether the production
# working points are in that regime.
#
# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
#     (method x n_dev) at the production working points.
# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row
#     (recolouring then removes at least gap-1 rounds, guaranteed).
#     gap == 0: recolouring is provably worthless AT THIS OWNERSHIP.
#     gap == 1: INCONCLUSIVE — Class 1 vs Class 2 is NP-complete, so this
#     neither confirms nor refutes a one-round win.
#     Note what a gap of 0 does NOT prove: Delta bounds only a proper
#     UNDIRECTED edge colouring of this graph, and the builder forces BOTH
#     ppermute directions per pair even when one send map is empty, so a
#     redesigned DIRECTED schedule is not bounded by Delta.  "Ownership is
#     the only path" would overclaim; the honest statement is "recolouring
#     this undirected graph is exhausted".
# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
#     step.  It cannot be shrunk further onto small meshes: measured
#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
#     but EVERY one of those rows auto-selects the ALLGATHER strategy
#     (cells/device below the threshold), so production runs no ppermute
#     schedule there and the number is counterfactual.  Only meshes big
#     enough to keep cells/device above the threshold answer the question.
#     s6 lloyd=0 also gave gap == 0 on all three methods — rounds 7/7/6 at
#     np8 and 13/10/10 at np16 for geometric/sfc/metis.  Both are genuinely
#     ppermute (L6 = 40,962 cells; padded, np8 gives 40,968/8 = 5,121 and
#     np16 gives 40,976/16 = 2,561 cells/device, each above the 2,000
#     threshold).  But np8's 7 == n_dev-1 is complete-graph saturation and
#     says nothing; only np16 (13/10/10 < 15) escapes it.  Still a SMALL
#     working point; s8-s10 are the production ones.
#
# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
# A scan that misses the known answer is a broken instrument, and its s10
# row must not be believed.
#
# lloyd=0 throughout: it is what the reference census used AND the cache key
# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
# synthetic scaling mesh, recorded in every row's metadata so it can never be
# read back as a production SCVT receipt.
#
# Arms run cheapest-first and each writes its own JSON, so a later arm that
# runs out of time or memory cannot lose an earlier arm's result.  s10@128 is
# last and is the one genuinely at risk: the scorer is known not to have
# finished at subdiv-8@128 on a laptop, which is why this asks for 24 h.
#
# SUBMIT (from the repo root):
#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

OUT=results/a1/mpas_schedule_cost
mkdir -p "$OUT"
# NOT overridable, deliberately.  An earlier revision made this settable
# from the environment so the exit-status logic could be regression-tested;
# that reintroduced exactly the failure class this script exists to prevent
# (a stray exported variable redirects a real scan to something else, every
# arm exits 0, SLURM files it COMPLETED with no results).  The shell test
# drives the interpreter instead — see
# tests/bench/test_mpas_schedule_cost_scan_sbatch.py.
BENCH=scripts/bench/bench_voronoi_partition_methods.py

# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
# missing partitioner is reported as "unavailable" rather than substituted.
# Say which python and whether metis is really there, so a two-method table
# cannot be misread as a three-method one.
echo "[scan] python=$PY"
"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"

run_arm () {  # $1=level  $2=rank-counts  $3=label  $4=expect-rounds (may be "")
  local json="$OUT/schedule_cost_s$1.json"
  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
  # Delete any previous artifact FIRST. A failed rerun that leaves the
  # earlier run's JSON in place produces a stale file with a current-looking
  # mtime story, and a stale receipt read as current is worse than a missing
  # one (codex round 5).
  rm -f "$json"
  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
  "$PY" "$BENCH" \
      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
      --methods geometric,sfc,metis --schedule-cost \
      ${4:+--expect-rounds "$4"} \
      --out "$json"
  local rc=$?   # capture BEFORE anything else runs, or `date` overwrites it
  echo "[scan] arm $3 exit=$rc"
  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
  # An exit code alone is not evidence the arm produced anything (CLAUDE.md:
  # "tool status is not evidence"). Require the artifact to exist and carry
  # rows, so a $PY that silently does nothing cannot yield SCAN_DONE with no
  # results. This does not make the launcher unspoofable — $PY and $REPO come
  # from _env.sh and remain configurable, as every launcher here is — but it
  # does make "succeeded without measuring" DETECTABLE.
  # PARSE the JSON — a substring grep for "n_ranks" passes on
  # {"rows": [], "note": "n_ranks"}, which is exactly the empty-but-
  # successful artifact this is supposed to catch (codex round 6).
  if [ "$rc" -eq 0 ] && ! "$PY" -c '
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
rows = doc.get("rows")
# `rows` MUST be a list of mappings, and a row only counts as SCORED if it
# carries a schedule with a round count.  Checking `"n_ranks" in r` alone
# passes on {"rows": {"n_ranks": 1}} (iterating a dict yields its keys) and
# on a quality-only row with no schedule at all (codex round 7).
if not isinstance(rows, list):
    sys.exit(1)
ok = any(
    isinstance(r, dict)
    and "n_ranks" in r
    and isinstance(r.get("schedule"), dict)
    and "n_rounds" in r["schedule"]
    for r in rows
)
sys.exit(0 if ok else 1)
' "$json"; then
    echo "[scan] arm $3 exited 0 but $json is missing, unparseable, or has"
    echo "[scan] no scored rows — refusing to call that a result"
    return 90
  fi
  return $rc
}

# Arms 1-2 are a MECHANICAL instrument check, not a comment: --expect-rounds
# makes the bench exit non-zero unless it reproduces the census below, and a
# listed pair that never got scored (e.g. pymetis missing) counts as a
# FAILURE, not a skip.  Arm 3's unknown s10 number is only quoted if both
# pass — an instrument that misses the known answer cannot be trusted on the
# unknown one.
S8_EXPECT="sfc:64=12,sfc:128=14,metis:64=13,metis:128=19,geometric:64=16,geometric:128=21"
S9_EXPECT="sfc:64=11,sfc:128=13,metis:64=14,metis:128=18,geometric:64=14,geometric:128=18"

run_arm 8 64,128 "1/3 VALIDATION s8" "$S8_EXPECT"; RC8=$?
run_arm 9 64,128 "2/3 VALIDATION s9" "$S9_EXPECT"; RC9=$?

if [ "$RC8" -ne 0 ] || [ "$RC9" -ne 0 ]; then
  echo "[scan] INSTRUMENT CHECK FAILED (s8 rc=$RC8, s9 rc=$RC9)."
  echo "[scan] The scorer does not reproduce the reference census, so an s10"
  echo "[scan] number from it would be untrustworthy. NOT running arm 3."
  echo "SCAN_ABORTED_VALIDATION"
  exit 1
fi

# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
run_arm 10 128 "3/3 UNKNOWN s10" ""; RC10=$?
echo "[scan] arm 3 rc=$RC10"

if [ "$RC10" -ne 0 ]; then
  # Do NOT let SLURM record this as COMPLETED: there is no valid s10 row,
  # and an exit-0 job with a missing result reads as a successful scan.
  echo "SCAN_FAILED_ARM3"
  exit "$RC10"
fi

echo "SCAN_DONE"
