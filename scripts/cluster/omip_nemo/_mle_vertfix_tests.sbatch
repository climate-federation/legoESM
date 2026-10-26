#!/bin/bash
#SBATCH --job-name=mlevfix
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mlevfix_%j.out
#
# Gate the MLE vertical-branch fix:
#  1. full MLE unit suites (latlon + MPAS) at the FIXED code -> must pass;
#  2. the NEW regression tests against the PRE-FIX code (a throwaway worktree
#     at the parent commit, new test files copied in) -> must FAIL, proving
#     the tests are non-vacuous ("shown to FAIL when the feature is removed").
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
FIXSHA=$(git rev-parse HEAD)
echo "=== HEAD $FIXSHA"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

NFAIL=0
echo; echo "########## fixed code: full MLE suites"
$PY -m pytest tests/ocean/unit/test_mle.py tests/ocean/unit/test_mle_mpas.py -q
rc=$?; echo "=== fixed suites rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1))

echo; echo "########## pre-fix code: new regression tests must FAIL"
# PINNED: parent of the vertical-branch fix commit 465823b83 -- NOT HEAD~1,
# which drifts as test-only commits land on top (the first rerun of this gate
# pointed HEAD~1 at the fix itself and reported the tests vacuous).
PREFIX_SHA=d6e8a1063
WT=$(mktemp -d /tmp/mle_prefix_XXXX)
git worktree add --detach "$WT" "$PREFIX_SHA" >/dev/null 2>&1 || { echo "FATAL: worktree"; exit 1; }
cp tests/ocean/unit/test_mle.py tests/ocean/unit/test_mle_mpas.py "$WT/tests/ocean/unit/"
( cd "$WT" && \
  PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT" \
  $PY -m pytest tests/ocean/unit/test_mle.py tests/ocean/unit/test_mle_mpas.py \
     -q -k "uniform" )
rc=$?
if [ $rc -eq 0 ]; then
  echo "=== VACUOUS: new tests PASSED on pre-fix code (rc=0) -- they prove nothing"
  NFAIL=$((NFAIL+1))
else
  echo "=== GOOD: new tests FAIL on pre-fix code (rc=$rc) -- non-vacuous"
fi
git worktree remove --force "$WT" >/dev/null 2>&1 || true

echo; echo "=== ALL DONE (failed gates: $NFAIL) ==="
exit $((NFAIL > 0))
