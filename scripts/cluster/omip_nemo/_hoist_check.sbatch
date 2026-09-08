#!/bin/bash
#SBATCH --job-name=hoistchk
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/hoistchk_%j.out
#
# The SSS-restoring input assembly moved above the ocean step.  The claim is
# VALUE-IDENTICAL: the ice is updated earlier in the same iteration, the runoff
# already fed fw, the target selection is pure, and _sss_ice reads only
# land_mask (static) + ice_state (untouched by _ocean_step).
#
# This gate runs the suites that exercise the restoring call path and the OMIP
# driver's CLI/plumbing.  It does NOT prove trajectory bit-identity -- that
# needs a short paired run, which is the next step, not this one.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
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

run_stage "sss restoring suite" \
  $PY -u -m pytest -q tests/ocean/unit/test_sss_restoring.py
run_stage "sss appliers" \
  $PY -u -m pytest -q tests/unit/test_sss_apply.py
run_stage "OMIP core2 CLI (driver imports + arg wiring)" \
  $PY -u -m pytest -q tests/unit/test_run_omip_core2_gm_treguier.py tests/unit/test_run_omip_cli.py
run_stage "driver imports cleanly" \
  $PY -u -c "import scripts.run.run_omip_core2 as R; print('import OK', hasattr(R,'main'))"

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
