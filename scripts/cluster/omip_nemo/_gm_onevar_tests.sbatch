#!/bin/bash
#SBATCH --job-name=gmonevar
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/gmonevar_%j.out
#
# Gate for the --gm-treguier ONE-VARIABLE fix (2026-08-12).
#
# THE DEFECT: build_tripole's --gm-treguier branch built its GM/Redi block from
# run_omip._DEFAULT_BATHY_GM_REDI, which is the LAT-LON bathymetry default
# (kappa_GM=kappa_Redi=800, Visbeck ON, slope_scheme="triads"), NOT the tripole
# NEMO-match recipe's block (600/600, Visbeck OFF, slope_scheme="centered").
# Every --gm-treguier arm therefore moved FOUR fields at once against its
# control, so the day-5 blow-up of job 9379514 was unattributable.
#
# NON-VACUITY: the new test must FAIL at the pre-fix commit.  PREFIX_SHA is
# PINNED (not HEAD~1) because a test-only commit can slide HEAD~1 onto the fix
# -- that exact drift produced a false "VACUOUS" verdict on the MLE gate.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
PREFIX_SHA=b9334f9b8
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

run_stage "gm_treguier CLI + one-variable + operator flags" \
  $PY -u -m pytest -q tests/unit/test_run_omip_core2_gm_treguier.py

run_stage "nemo_match_recipe (aei0 900 correction)" \
  $PY -u -m pytest -q tests/ocean/unit/test_nemo_match_recipe.py

run_stage "run_omip CLI (treguier param plumbing)" \
  $PY -u -m pytest -q tests/unit/test_run_omip_cli.py

run_stage "treguier kappa unit suite (unchanged physics)" \
  $PY -u -m pytest -q tests/ocean/unit/test_treguier_gm.py

# --- non-vacuity: the new class must fail at the pinned pre-fix commit -------
echo; echo "########## NON-VACUITY at $PREFIX_SHA"
WT=$(mktemp -d /burg-archive/glab/users/pg2328/wt_gmonevar_XXXX)
git -C $REPO worktree add --detach "$WT" $PREFIX_SHA >/dev/null 2>&1 || {
  echo "FATAL: could not create pinned worktree"; exit 1; }
echo "prefix worktree HEAD: $(git -C $WT rev-parse --short HEAD)"
# copy ONLY the test file forward; the source stays at the pre-fix commit
cp -f $REPO/tests/unit/test_run_omip_core2_gm_treguier.py \
      $WT/tests/unit/test_run_omip_core2_gm_treguier.py
PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT" \
  $PY -u -m pytest -q \
  $WT/tests/unit/test_run_omip_core2_gm_treguier.py::TestTreguierBlockIsOneVariable
PRERC=$?
echo "=== pre-fix rc=$PRERC (EXPECT NON-ZERO; 0 => the test is VACUOUS)"
[ $PRERC -ne 0 ] || NFAIL=$((NFAIL + 1))
git -C $REPO worktree remove --force "$WT" >/dev/null 2>&1

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
