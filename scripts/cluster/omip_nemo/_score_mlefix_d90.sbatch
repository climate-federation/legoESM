#!/bin/bash
#SBATCH --job-name=mlefix90s
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mlefix90s_%j.out
#
# PRE-REGISTERED day-90 test of the MLE vertical-branch fix (arm 9362979,
# control = nemolev_trp_gwcorr_d90, one variable: --mle --mle-ce 0.06).
# From _ab_mlefix_d90.sbatch, written before the run:
#   CONFIRMS the lever: genuine-deep Arctic columns fall from the control toward NEMO March 89 (day-30 measured 159 vs control 161 = REFUTED at d30; d90 is the pre-registered horizon) and no
#   cell below -10 psu (pre-fix: -54.7).
#   REFUTES: extremes persist (p99 > 10) -> vertical branch insufficient or
#   wrongly signed in production; kill the MLE default.
# Also scores the arm against NEMO month 1 at day 30 (matched window vs the
# control's own day-30 scorecard).
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
MLE=$R/nemolev_trp_mlefix_d90/snapshot_day0090.npz
CTL=$R/nemolev_trp_gwcorr_d90/snapshot_day0090.npz
for f in "$MLE" "$CTL" "$GT"; do
  [ -r "$f" ] || { echo "FATAL: unreadable $f"; exit 1; }
done

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

run_stage "PRE-REGISTERED: freshening geography + Psi/CFL vs control" \
  $PY -u scripts/validate/ocean_fidelity/mle_psi_diagnostics.py \
    --mle-snapshot "$MLE" --control-snapshot "$CTL" \
    --mle-ce 0.06 --dt-s 150 \
    --mesh-mask $REPO/data/grids/eORCA1.2_mesh_mask.nc \
    --out-dir $R/mlefix_psi_diag_d90

run_stage "MLE-fix arm scorecard vs NEMO month 1 (day 30)" \
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot "$MLE" \
    --nemo-gridt "$GT" --nemo-month 3 --res-deg 1.0 \
    --grid-label "tripole (MLE fixed)" \
    --output-dir $R/compare_mlefix_d90_m03

run_stage "control scorecard vs NEMO month 1 (day 30, matched window)" \
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot "$CTL" \
    --nemo-gridt "$GT" --nemo-month 3 --res-deg 1.0 \
    --grid-label "tripole (control, no MLE)" \
    --output-dir $R/compare_gwcorr_d90_m03

run_stage "Arctic deep-convection columns (MLE fixed, day 30)" \
  $PY -u scripts/validate/ocean_fidelity/arctic_deep_convection_columns.py \
    --snapshot "$MLE" --label "MLE fixed" \
    --nemo-gridt "$GT" --nemo-month 3 \
    --out-dir $R/arctic_convection_mlefix_d90

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
