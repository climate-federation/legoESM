#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tqtune_smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_smoke_%j.out
set +e
# ---------------------------------------------------------------------------
# End-to-end smoke of the NEW path (thermo objective + physical parameter set +
# local refinement + the T/q ranking figure) on two schemes and a short
# integration.  Its job is to catch wiring, not physics: a mis-threaded flag
# would otherwise surface eleven hours into a production arm.
#
# It also checks the two things a passing exit code would NOT establish:
#   * the tuner minimised the thermo score (tuned <= prior on THAT column);
#   * the CSV's `objective` column says `thermo` and its rows are ordered by it.
#
# --allow-partial is REQUIRED here and only here: the merge gate refuses a table
# missing 8 of 10 schemes, which is correct for a campaign and wrong for a
# two-scheme wiring check.  The production arms never pass it.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=${REF:-$REPO/results/rcemip_ref_sam300}
OUT=${OUT:-$REPO/results/scm_rce_tqtune/_smoke}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
echo "host: $(hostname)  date: $(date -Is)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
[ -d "$REF/snapshots3d" ] || { echo "PREFLIGHT FAIL: no reference at $REF"; exit 94; }

export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR" || { echo "PREFLIGHT FAIL: cannot create TMPDIR"; exit 89; }
trap 'rm -rf /burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}' EXIT

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

rm -rf "$OUT"
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF \
  --outdir $OUT \
  --schemes mass_flux,dca \
  --objective thermo \
  --param-set physical \
  --thermo-humidity logq \
  --tune-refine-frac 0.35 \
  --subsidence-solve implicit_flux \
  --microphysics morrison \
  --hard-saturation-adjustment \
  --days 3 \
  --dt 600 \
  --analysis-days 1 \
  --last-reference-files 5 \
  --radiation rrtmgp \
  --scm-convection-substeps 10 \
  --scm-microphysics-substeps 30 \
  --bl-anchor-top-m -1 \
  --tune-evals 8 \
  --tune-seed 20260815 \
  --allow-partial 2>&1 | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
RC=${PIPESTATUS[0]}
echo "RUN_RC=$RC"
[ $RC -eq 0 ] || { echo "SMOKE FAILED in the run"; exit $RC; }

CSV=$(ls $OUT/*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] || { echo "SMOKE FAILED: no CSV written"; exit 96; }
echo "CSV=$CSV"

echo "=== ASSERTIONS ==="
JAX_ENABLE_X64=1 $PY - "$CSV" <<'PYEOF'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
assert rows, "empty CSV"
bad = []
for r in rows:
    if r["objective"] != "thermo":
        bad.append(f"{r['scheme']}: objective={r['objective']!r}, expected 'thermo'")
    prior = float(r["prior_objective"]); tuned = float(r["tuned_objective"])
    if math.isfinite(prior) and math.isfinite(tuned) and tuned > prior:
        bad.append(f"{r['scheme']}: tuned objective {tuned} WORSE than prior {prior}")
    if not math.isfinite(float(r["tuned_thermo_score"])):
        bad.append(f"{r['scheme']}: non-finite thermo score")
    for col in ("tuned_trop_rh_rmse", "tuned_thermo_logq_term",
                "tuned_thermo_rh_term", "tuned_thermo_window_std"):
        if not math.isfinite(float(r[col])):
            bad.append(f"{r['scheme']}: non-finite {col}")
order = [float(r["tuned_objective"]) for r in rows]
if order != sorted(order):
    bad.append(f"rows not ordered by the tuned objective: {order}")
# dca exposed NO parameter under the extended tier; the physical set admits its
# CAPE trigger, so it must now report exactly one. If this reads 0 the tier-0
# opt-in never reached the tuner and every scheme's trigger is untouched.
for r in rows:
    if r["scheme"] == "dca" and int(r["n_tuned_params"]) != 1:
        bad.append(f"dca reported {r['n_tuned_params']} tuned params, expected 1 "
                   "(its CAPE trigger, admitted by the physical set)")
    if r["thermo_humidity_variable"] != "logq":
        bad.append(f"{r['scheme']}: humidity variable "
                   f"{r['thermo_humidity_variable']!r}, expected 'logq'")
    if int(r["unique_evals"]) < 2:
        bad.append(f"{r['scheme']}: unique_evals={r['unique_evals']} - the "
                   "search did not actually integrate distinct columns")
    if r["scheme"] != "dca" and int(r["refine_proposals"]) < 1:
        bad.append(f"{r['scheme']}: refinement never proposed anything")
for r in rows:
    print(f"{r['scheme']:<12} obj {float(r['prior_objective']):.4f} -> "
          f"{float(r['tuned_objective']):.4f}   T {float(r['apriori_T_rmse_K']):.3f} -> "
          f"{float(r['tuned_T_rmse_K']):.3f} K   RH {float(r['apriori_trop_rh_rmse']):.4f} -> "
          f"{float(r['tuned_trop_rh_rmse']):.4f}   n_par {r['n_tuned_params']}   "
          f"noise+-{float(r['tuned_thermo_window_std']):.4f}   "
          f"evals req {r['requested_evals']} uniq {r['unique_evals']}")
if bad:
    print("SMOKE ASSERTIONS FAILED:")
    for b in bad:
        print("  -", b)
    raise SystemExit(1)
print("SMOKE ASSERTIONS OK")
PYEOF
ARC=$?
[ $ARC -eq 0 ] || { echo "SMOKE FAILED in the assertions"; exit $ARC; }

echo "=== FIGURE ==="
JAX_ENABLE_X64=1 $PY $WT/scripts/plot/plot_scm_rce_tq_ranking.py "$CSV" \
  --out $OUT/tq_ranking_smoke.png --humidity logq --suptitle "smoke" 2>&1 | tail -3
FRC=${PIPESTATUS[0]}
echo "FIG_RC=$FRC"
echo "SMOKE_STATUS=$([ $FRC -eq 0 ] && echo OK || echo FAILED)"
echo ALLDONE
exit $FRC
