#!/bin/bash
#SBATCH --job-name=scevd90
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scevd90_%j.out
#
# Day-90 evidence for the EVD hard+adiabatic default at the SHIPPED K_bg
# (1e-5), arm 9355295 -> results/omip_nemo/nemolev_trp_evdkbg_d90.  This is
# the arm codex round-5 named as the missing piece of the ship decision:
# d90 was measured for K_bg=0 (MLD bias/rmse +0.24/45.9 vs control
# +7.58/62.9; Arctic +39.2/124.7) but NOT for the default K_bg.
#
# PRE-REGISTERED (from the memory card, before this scoring):
#   CONFIRMS the default: d90 MLD RMSE lands near the K_bg=0 arm's 45.9 (not
#   back at the control's 62.9) AND Arctic MLD bias falls well below the
#   control's +60.9.
#   REFUTES: MLD RMSE within noise of the control's 62.9 -> the d30 win does
#   not survive to d90 at the shipped background, and the K_bg=0 result was
#   carrying it.
#   GUARDRAIL (co-primary, do not omit): Arctic SST RMSE regressed +15.9 % at
#   d30 (1.05 -> 1.21).  Report the d90 Arctic SST RMSE next to the MLD win.
# Control = nemolev_trpsbc_win_d90 (same protocol, EVD off), NEMO month 3,
# same window on both sides.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
ARM=$R/nemolev_trp_evdkbg_d90/snapshot_day0090.npz
CTL=$R/nemolev_trpsbc_win_d90/snapshot_day0090.npz
for f in "$ARM" "$CTL" "$GT"; do
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

run_stage "EVD K_bg default, d90 vs NEMO month 3" \
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot "$ARM" \
    --nemo-gridt "$GT" --nemo-month 3 --res-deg 1.0 \
    --grid-label "tripole (EVD hard+adiabatic, K_bg default)" \
    --output-dir $R/compare_evdkbg_d90_m03

run_stage "control (EVD off), d90 vs NEMO month 3 (matched window)" \
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot "$CTL" \
    --nemo-gridt "$GT" --nemo-month 3 --res-deg 1.0 \
    --grid-label "tripole (control, EVD off)" \
    --output-dir $R/compare_evdctl_d90_m03

run_stage "Arctic deep-convection columns (EVD arm, d90)" \
  $PY -u scripts/validate/ocean_fidelity/arctic_deep_convection_columns.py \
    --snapshot "$ARM" --label "EVD hard+adiabatic K_bg default" \
    --nemo-gridt "$GT" --nemo-month 3 \
    --out-dir $R/arctic_convection_evdkbg_d90

run_stage "Arctic deep-convection columns (control, d90)" \
  $PY -u scripts/validate/ocean_fidelity/arctic_deep_convection_columns.py \
    --snapshot "$CTL" --label "control EVD off" \
    --nemo-gridt "$GT" --nemo-month 3 \
    --out-dir $R/arctic_convection_evdctl_d90

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
