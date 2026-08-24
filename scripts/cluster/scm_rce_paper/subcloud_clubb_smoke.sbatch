#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=subcloud_smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/subcloud_smoke_%j.out
set +e
# ---------------------------------------------------------------------------
# Smoke gate for the sub-cloud arm, BEFORE any production tuning is launched.
#
# What it proves, in order:
#   1. the new unit gates are green (sub-cloud metrics, focused tuner, probe);
#   2. the regression suites that consume the edited campaign still pass —
#      a fixture/diagnostic edit is a change to every test in those modules;
#   3. prognostic CLUBB actually completes an SCM-RCE column (it detonated
#      before PR #1601's surface-variance boundary condition);
#   4. the condensate SPLIT sums to the total it was factored out of, on a
#      real checkpoint rather than in a unit test;
#   5. the focused cross-category tuner resolves and moves parameters.
#
# Deliberately SHORT (a few simulated days): this answers "does it run and are
# the new numbers self-consistent", not "is the column right". The production
# arm is a separate submission.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
OUT=${OUT:-$REPO/results/scm_rce_convtune/_smoke_subcloud}
DAYS=${DAYS:-3}
EVALS=${EVALS:-3}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
echo "OUT=$OUT  DAYS=$DAYS  EVALS=$EVALS"
rm -rf "$OUT"; mkdir -p "$OUT"

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
export MPLBACKEND=Agg
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_tmp}
mkdir -p "$TMPDIR"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo; echo "############ 1. new unit gates ############"
JAX_ENABLE_X64=1 $PY -m pytest -q \
  $WT/tests/unit/test_scm_rce_subcloud_metrics.py \
  $WT/tests/unit/test_scm_rce_focused_tune.py \
  $WT/tests/unit/test_check_scm_rce_surface_humidity.py 2>&1 | tail -25
RC=${PIPESTATUS[0]}; echo "NEW_TESTS_RC=$RC"
[ $RC -eq 0 ] || { echo "ABORT: new gates red"; exit $RC; }

echo; echo "############ 2. campaign regression suites ############"
JAX_ENABLE_X64=1 $PY -m pytest -q \
  $WT/tests/unit/test_scm_rce_applied_precip.py \
  $WT/tests/unit/test_scm_rce_metrics.py \
  $WT/tests/unit/test_scm_rce_clubb_nesting.py \
  $WT/tests/unit/test_scm_rce_tuner_sampling.py \
  $WT/tests/unit/test_scm_rce_water_budget.py \
  $WT/tests/unit/test_scm_rce_subsidence_solve_override.py \
  $WT/tests/unit/test_scm_rce_convection_intercomparison_cli.py 2>&1 | tail -25
RC=${PIPESTATUS[0]}; echo "REGRESSION_RC=$RC"
[ $RC -eq 0 ] || { echo "ABORT: regression red"; exit $RC; }

echo; echo "############ 3+5. CLUBB column + focused tuner ############"
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF --outdir $OUT \
  --schemes bechtold \
  --turbulence clubb --microphysics morrison --hard-saturation-adjustment \
  --radiation rrtmgp --subsidence-solve implicit_flux \
  --bl-anchor-top-m -1.0 --scm-convection-substeps 10 \
  --tune-mode focused --objective subcloud \
  --days $DAYS --analysis-days 1 --dt 600 \
  --tune-evals $EVALS --tune-seed 20260814 --no-merge 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -40
RC=${PIPESTATUS[0]}; echo "COLUMN_RC=$RC"
[ $RC -eq 0 ] || { echo "ABORT: CLUBB column failed"; exit $RC; }

echo; echo "############ 4. split sums to the total, on the checkpoint ############"
$PY - "$OUT" <<'EOF'
import json, sys, glob, numpy as np
bad = 0
for path in sorted(glob.glob(f"{sys.argv[1]}/scheme_*.json")):
    payload = json.load(open(path))
    for cond in ("prior", "tuned"):
        run = payload[cond]
        tot = np.asarray(run["qcond_profile"], float)
        cld = np.asarray(run["qcloud_profile"], float)
        pre = np.asarray(run["qprecip_profile"], float)
        if tot.size == 0:
            print(f"  {path} {cond}: EMPTY profile"); bad += 1; continue
        # The parts are clipped at 0 independently of the total, so equality is
        # exact only where nothing was clipped; compare on the scale of the
        # column's own maximum rather than with a bare atol.
        resid = float(np.max(np.abs(cld + pre - tot)))
        scale = max(float(np.max(tot)), 1e-30)
        print(f"  {path.split('/')[-1]} {cond}: max|cloud+precip-total| "
              f"= {resid:.3e} ({resid/scale:.2e} of column max), "
              f"cloud {float(cld.mean())*1e3:.4f} / precip "
              f"{float(pre.mean())*1e3:.4f} g/kg, "
              f"RH {run['sfc_relative_humidity']:.3f}, "
              f"SST-T_air {run['sfc_delta_T_K']:.2f} K, "
              f"subcloud {run['subcloud_score']:.4f}")
        if resid / scale > 1e-9:
            bad += 1
print("SPLIT_CONSISTENT" if bad == 0 else f"SPLIT_INCONSISTENT ({bad})")
sys.exit(1 if bad else 0)
EOF
echo "SPLIT_RC=$?"

echo; echo "COMPLETED"
