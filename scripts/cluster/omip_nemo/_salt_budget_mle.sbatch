#!/bin/bash
#SBATCH --job-name=saltbdg
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/saltbdg_%j.out
#
# DISCRIMINATOR for the MLE d90 surface freshening (34.25 -> 29.71 psu at the
# SURFACE over 90 days, job 9337800):
#   conservation defect  -> total salt content falls in the MLE arm;
#   redistribution       -> total fixed, depth bins trade (surface loses,
#                           deep gains), i.e. MLE over-restratifies.
# Control (fwreal, no MLE) is scored with the SAME probe at the SAME days: if
# the CONTROL's total also drifts, the drift is not attributable to --mle.
#
# Probe self-test runs FIRST: quoting a probe's number before its tests pass
# is the exact failure mode the validate-instrument-first rule exists for.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
MESH=$REPO/data/grids/eORCA1.2_mesh_mask.nc
[ -r "$MESH" ] || { echo "FATAL: mesh mask unreadable"; exit 1; }

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

run_stage "probe + comparator unit tests" \
  $PY -m pytest tests/validate/test_global_tracer_content.py \
               tests/validate/test_compare_three_way_nemo.py -q

for arm in mle fwreal; do
  D=$R/nemolev_trp_${arm}_d90
  run_stage "$arm tracer content d30/d60/d90" \
    $PY -u scripts/validate/ocean_fidelity/global_tracer_content.py \
      --snapshot $D/snapshot_day0030.npz $D/snapshot_day0060.npz \
                 $D/snapshot_day0090.npz \
      --mesh-mask "$MESH" \
      --json-out $D/tracer_content.json
done
echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
