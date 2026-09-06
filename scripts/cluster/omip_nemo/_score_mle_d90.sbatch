#!/bin/bash
#SBATCH --job-name=scoremle
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scoremle_%j.out
#
# Score the MLE d90 arm (job 9337800, results/omip_nemo/nemolev_trp_mle_d90).
# One-variable A/B: control = nemolev_trp_fwreal_d90 (already scored at
# compare_fwreal_d90_m03 + arctic_convection_fwreal_d90); this arm adds --mle
# only, same pinned worktree.
#
# PRE-REGISTERED (from _ab_mle_fwreal_d90.sbatch, written before the run):
#   (a) number: genuine-deep Arctic columns (>500 m, not bottom-limited) at
#       day 90 vs NEMO March, and Greenland/Labrador median MLD.
#   (b) CONFIRMS: count falls from 291 toward NEMO's 84 AND the deep-basin
#       medians drop from ~2000 m to O(100 m).
#       REFUTES: count stays > 250, or the columns move instead of shoaling.
#       Count << 84 = over-restratification, also a failure.
#   WATCH: global + Antarctic MLD bands must not degrade (a fix that trades
#   Arctic error for Southern Ocean error is not a fix).
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
[ -r "$GT" ] || { echo "FATAL: NEMO grid_T unreadable"; exit 1; }

SNAP=$R/nemolev_trp_mle_d90/snapshot_day0090.npz
[ -r "$SNAP" ] || { echo "FATAL: $SNAP unreadable"; exit 1; }

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

# Same day (90) and same NEMO month (3) as the control's scoring -- the
# matched-window rule.  Same comparator, same res-deg.
run_stage "MLE arm scorecard vs NEMO month 3" \
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 3 --res-deg 1.0 \
    --grid-label "tripole (fwreal + MLE)" \
    --output-dir $R/compare_mle_d90_m03

run_stage "MLE arm Arctic deep-convection columns" \
  $PY -u scripts/validate/ocean_fidelity/arctic_deep_convection_columns.py \
    --snapshot "$SNAP" --label "fwreal + MLE" \
    --nemo-gridt "$GT" --nemo-month 3 \
    --out-dir $R/arctic_convection_mle_d90

echo
echo "########## control reference numbers (already scored, for side-by-side)"
for f in $R/arctic_convection_fwreal_d90/*.json $R/arctic_convection_fwreal_d90/*.txt; do
  [ -r "$f" ] && { echo "-- $f"; cat "$f"; }
done

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
