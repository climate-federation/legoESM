#!/bin/bash
#SBATCH --job-name=sssnorm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sssnorm_%j.out
#
# Gate for the NEMO-form SSS restoring (normalization="live_s") and the
# narrowed real_freshwater x --sss-restore guard.
#
# WHY: all three scored arms ran freshwater_closure=virtual_salt_flux (the CLI
# default is None -> the config default stands), i.e. the known spurious Arctic
# salt source is still on, which is why the +1.4 psu Arctic SSS bias is
# COMMON-MODE across tripolar/MPAS/FESOM2.  real_freshwater was blocked because
# our restoring divided by S_target (a virtual-salt convention).  NEMO ORCA1
# runs nn_sssr=2 and divides by the LIVE salinity, which IS a water flux.
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

run_stage "sss_restoring unit suite (incl. the NEMO-formula comparison)" \
  $PY -u -m pytest -q tests/ocean/unit/test_sss_restoring.py

run_stage "sss appliers (consume dS_dt_top)" \
  $PY -u -m pytest -q tests/unit/test_sss_apply.py

run_stage "restoring extension" \
  $PY -u -m pytest -q tests/ocean/unit/test_restoring_extension.py

run_stage "OMIP CLI round-trip" \
  $PY -u -m pytest -q tests/unit/test_run_omip_cli.py

# The guard is exercised by TestRealFreshwaterRestoringGuard in the first
# stage.  An earlier version of this gate drove it through main() instead and
# never reached it: main applies the rule only AFTER the model is built, so the
# run died constructing the tripole grid and the assertion read "PASSED_GUARD"
# for the wrong reason.  The predicate is now split out and tested directly.

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
