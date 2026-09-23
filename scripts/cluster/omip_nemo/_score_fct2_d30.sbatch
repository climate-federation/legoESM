#!/bin/bash
#SBATCH --job-name=scfct2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scfct2_%j.out
#
# PRE-REGISTERED day-30 scoring of the advection-scheme A/B (superbee ->
# fct2, arm 9376532).  Thresholds are fixed in _ab_fct2_d30.sbatch and are
# repeated here verbatim so the verdict cannot drift:
#   CONFIRMS: SST rmse < control 0.912 AND MLD rmse does not degrade (37.8).
#   REFUTES : SST rmse rises, or MLD rmse rises by more than ~10 %.
#   INDETERMINATE: both move < 0.02 / < 2 m -> operator-visible but
#                  state-invisible at 30 days (a legitimate answer).
# WATCH (from the run header): superbee is monotone and was chosen partly
# because ppm_fct destabilised the Gulf Stream.  The day-10 check already
# showed fct2's max |u| at 1.17 m/s vs the control's 1.04 -- report the
# western-boundary behaviour, do not sell a clean win if it is noisy there.
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
ARM=$R/nemolev_trp_fct2_d30/snapshot_day0030.npz
[ -r "$ARM" ] || ARM=$R/nemolev_trp_fct2_d30/snapshot_final.npz
CTL=$R/nemolev_trp_gwcorr_d30/snapshot_day0030.npz
[ -r "$CTL" ] || CTL=$R/nemolev_trp_gwcorr_d30/snapshot_final.npz
for f in "$ARM" "$CTL" "$GT"; do
  [ -r "$f" ] || { echo "FATAL: unreadable $f"; exit 1; }
done
echo "[arms] fct2=$ARM"; echo "[arms] ctl =$CTL"

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

run_stage "fct2 arm vs NEMO month 1 (day 30)" \
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot "$ARM" --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --grid-label "tripole (fct2 advection)" \
    --output-dir $R/compare_fct2_d30_m01

run_stage "control (superbee) vs NEMO month 1 (day 30, matched window)" \
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot "$CTL" --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --grid-label "tripole (superbee control)" \
    --output-dir $R/compare_superbee_d30_m01

run_stage "Arctic deep-convection columns (fct2, day 30)" \
  $PY -u scripts/validate/ocean_fidelity/arctic_deep_convection_columns.py \
    --snapshot "$ARM" --label "fct2 advection" \
    --nemo-gridt "$GT" --nemo-month 1 \
    --out-dir $R/arctic_convection_fct2_d30

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
