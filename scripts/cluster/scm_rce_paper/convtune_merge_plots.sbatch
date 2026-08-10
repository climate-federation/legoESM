#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=convtune_fig
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_fig_%j.out
set +e
# ---------------------------------------------------------------------------
# Merge the per-scheme checkpoints into the ranking table and render the
# figures: equilibrium T and q_v profiles per scheme (CRM black, a priori
# dashed, tuned solid) and the a-priori-vs-tuned RMSE bars in PHYSICAL units
# (K and g/kg).
#
# The merge stage REFUSES to publish a ranking computed over a subset of the
# schemes, so this is also the gate that says whether the arm is complete.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
ARM=${ARM:-implicit_flux}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
OUT=$REPO/results/scm_rce_convtune/arm_${ARM}_morrison

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
echo "ARM=$ARM  OUT=$OUT"
echo "checkpoints present: $(ls $OUT/scheme_*.json 2>/dev/null | wc -l) / 10"

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
export MPLBACKEND=Agg
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo; echo "############ merge ############"
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF --outdir $OUT --merge-only 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -20
echo "MERGE_RC=${PIPESTATUS[0]}"

echo; echo "############ paper figures ############"
JAX_ENABLE_X64=1 $PY $WT/scripts/plot/plot_scm_rce_convection_paper.py \
  --results-dir $OUT --reference-dir $REF --last-reference-files 5 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -20
echo "FIG_RC=${PIPESTATUS[0]}"

echo; echo "############ ranking (physical units) ############"
$PY - "$OUT/summary_table.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")
rows.sort(key=lambda r: f(r["tuned_T_rmse_K"]))
hdr = f"{'scheme':18s} {'T RMSE [K] a->t':>22s} {'qv RMSE [g/kg] a->t':>24s} {'#par':>5s} {'verdict':>12s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['scheme']:18s} "
          f"{f(r['apriori_T_rmse_K']):9.3f} -> {f(r['tuned_T_rmse_K']):8.3f} "
          f"{f(r['apriori_qv_rmse_g_kg']):11.4f} -> {f(r['tuned_qv_rmse_g_kg']):9.4f} "
          f"{r['n_tuned_params']:>5s} {r['tuned_verdict']:>12s}")
PYEOF

echo; echo "artifacts:"; ls -1 $OUT/*.png $OUT/*.csv $OUT/*.md $OUT/paper/* 2>/dev/null | tail -20
echo ALLDONE
