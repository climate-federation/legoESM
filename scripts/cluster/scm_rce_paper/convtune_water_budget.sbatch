#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=convtune_h2o
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_h2o_%j.out
set +e
# ---------------------------------------------------------------------------
# Why does the SCM-RCE column report ~0 surface precipitation while sitting in
# a steady state?  d(CWV+CWC)/dt = E - P separates the two candidate answers:
# a closed column (no surface moisture flux) from a live one whose water
# leaves uncounted.  Run for two convection schemes with DIFFERENT closures --
# sbm relaxes toward a reference profile, mass_flux detrains condensate -- so
# the answer is not a property of one scheme's plumbing.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

for SCHEME in sbm mass_flux; do
  echo; echo "############ convection=$SCHEME ############"
  JAX_ENABLE_X64=1 $PY $WT/scripts/validate/scm_rce_water_budget.py \
    --reference-dir $REF --convection $SCHEME --microphysics morrison \
    --days-short 20 --days-long 25 --analysis-days 2 2>&1 \
    | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
  echo "RC_${SCHEME}=${PIPESTATUS[0]}"
done

# Control: the SAME column with convection OFF. If E is live, a column with no
# convection still evaporates and must accumulate water, which pins the
# surface flux independently of any convection scheme's plumbing.
echo; echo "############ CONTROL: convection=none ############"
JAX_ENABLE_X64=1 $PY $WT/scripts/validate/scm_rce_water_budget.py \
  --reference-dir $REF --convection none --microphysics morrison \
  --days-short 5 --days-long 10 --analysis-days 2 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
echo "RC_none=${PIPESTATUS[0]}"
echo ALLDONE
