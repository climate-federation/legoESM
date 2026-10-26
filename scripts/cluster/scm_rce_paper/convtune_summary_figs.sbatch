#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=convtune_sum
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_sum_%j.out
set +e
# ---------------------------------------------------------------------------
# Render the paired before/after tuning summary for one convection arm, in the
# same figure family as the SCM turbulence campaign
# (scripts/plot/plot_scm_les_tuning_summary.py).
#
# Runs the plotter's own unit tests FIRST: the figure encodes two claims that
# a silent bug would invert -- which schemes are flat BY CONSTRUCTION (dca has
# no tunable parameters; kuo is inactive in a single column and its row IS the
# no-convection baseline) and which merely failed their search.
#
# NO_CONV is the MEASURED combined score of convection=none in this exact
# configuration, not a guess: job 9379420 printed 5.736693549527354 and the
# arm's own kuo checkpoint carries the same value bit-for-bit, which is the
# cross-check that the baseline belongs to this arm's protocol.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
OUT=${OUT:?export OUT=<arm results dir>}
NO_CONV=${NO_CONV:-5.736693549527354}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}
echo "OUT=$OUT"
NCK=$(ls $OUT/scheme_*.json 2>/dev/null | wc -l)
echo "checkpoints present: $NCK / 10"
[ "$NCK" -eq 10 ] || { echo "ABORT: incomplete arm ($NCK/10)"; exit 92; }

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
export MPLBACKEND=Agg
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_tmp}
mkdir -p "$TMPDIR"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo; echo "############ plotter unit tests ############"
JAX_ENABLE_X64=1 $PY -m pytest -q \
  $WT/tests/unit/test_plot_scm_rce_convection_tuning_summary.py 2>&1 | tail -15
TEST_RC=${PIPESTATUS[0]}
echo "TEST_RC=$TEST_RC"
[ $TEST_RC -eq 0 ] || { echo "ABORT: plotter gates red"; exit $TEST_RC; }

echo; echo "############ merge (ranking table) ############"
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF --outdir $OUT --merge-only 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -25
MERGE_RC=${PIPESTATUS[0]}
echo "MERGE_RC=$MERGE_RC"
[ $MERGE_RC -eq 0 ] || { echo "ABORT: merge refused or failed"; exit $MERGE_RC; }

echo; echo "############ paired before/after + profile overlay ############"
JAX_ENABLE_X64=1 $PY $WT/scripts/plot/plot_scm_rce_convection_tuning_summary.py \
  $OUT --reference-dir $REF --last-reference-files 5 \
  --no-convection-score $NO_CONV 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -25
SUM_RC=${PIPESTATUS[0]}
echo "SUM_RC=$SUM_RC"
[ $SUM_RC -eq 0 ] || { echo "ABORT: summary figure failed"; exit $SUM_RC; }

echo; echo "############ per-scheme paper figures ############"
JAX_ENABLE_X64=1 $PY $WT/scripts/plot/plot_scm_rce_convection_paper.py \
  --results-dir $OUT --reference-dir $REF --last-reference-files 5 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -15
FIG_RC=${PIPESTATUS[0]}
echo "FIG_RC=$FIG_RC"

echo; echo "############ ranking by EACH component ############"
$PY - "$OUT" <<'PYEOF'
import json, glob, sys, os
recs = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "scheme_*.json"))):
    j = json.load(open(f))
    recs.append((j["scheme"], j["prior"], j["tuned"], len(j.get("records") or ())))
NO_CONV = 5.736693549527354
def ff(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")
for key, label in (("score", "COMBINED"), ("T_rmse", "TEMPERATURE"),
                   ("qv_rmse", "WATER VAPOUR"), ("cloud_rmse", "CONDENSATE"),
                   ("precip_rmse", "PRECIPITATION")):
    print(f"\n--- ranked by {label} ({key}) ---")
    order = sorted(recs, key=lambda r: ff(r[2].get(key)))
    for s, p, t, n in order:
        flat = "  [flat: no params]" if n == 0 else (
            "  [flat: inactive]" if ff(p.get("score")) == ff(t.get("score")) and s == "kuo" else "")
        worse = "  WORSE than no convection" if key == "score" and ff(t.get(key)) > NO_CONV else ""
        print(f"  {s:16s} {ff(p.get(key)):9.4f} -> {ff(t.get(key)):8.4f}  "
              f"({n:2d} par){flat}{worse}")
PYEOF

echo; echo "artifacts:"; ls -1 $OUT/*.png $OUT/*.csv 2>/dev/null | tail -20
echo ALLDONE
