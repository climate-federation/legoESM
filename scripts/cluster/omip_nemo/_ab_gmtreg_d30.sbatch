#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=gmtreg30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/gmtreg30_%j.out
#
# ONE-VARIABLE A/B: GM coefficient constant 600 -> NEMO's flow-dependent form
# (--gm-treguier = ldf_eiv nn_aei_ijk_t=21, F(baroclinic growth rate)).
#
# WHY THIS RUN EXISTS -- a MEASURED gap, not a hypothesis.  NEMO's own
# emitted aeiu_2d/ahtu_2d (RUN_TRD2 rec 1, 2026-08-12) vs our constant 600
# for BOTH kappa_GM and kappa_Redi, median by region:
#     arctic>60N  aht 180  aei  43      <-- we run 600: 3.3x / 14x too much
#     nh_mid      aht 257  aei 257
#     tropics     aht 752  aei 155
#     sh_mid      aht 539  aei 539
#     antarctic   aht 180  aei  71
# The Arctic overshoot sits exactly where our SSS bias (+1.4) and excess deep
# convection live, and where all three of our grids agree with each other but
# not with NEMO.
#
# WHAT THIS ARM CHANGES: kappa_GM only (the Treguier form replaces it);
# kappa_Redi stays 600.  So it tests the LARGER half (aei, 14x) of the gap.
# aei0 is passed EXPLICITLY as 900 because this pinned worktree predates the
# fix that corrected the default from 1800 (NEMO's laplacian prefactor is
# 1/2*rn_Ue*rn_Le; its aeiu_2d maxes at exactly 900).
# RESIDUAL, STATE IT WITH THE RESULT: kappa_min defaults to 200 m^2/s -- NOT
# NEMO (which has no floor) but required because kappa -> 0 at the equator
# destabilised a 1-degree run.  Arctic NEMO aei is 43, so even this arm still
# runs ~4.6x too much GM there; a lower floor is the follow-up if this helps.
#
# CONTROL: nemolev_trp_gwcorr_d30 (constant kappa_GM = 600), already scored:
# SST rmse 0.9116, SSS 0.8583, MLD 37.8, Arctic SST rmse 1.067, Arctic MLD
# bias +34.7, genuine-deep 161 (NEMO 30).
#
# PRE-REGISTERED (before the run):
#  (a) NUMBERS: day-30 compare_omip_nemo SST/SSS/MLD rmse vs NEMO month 1 and
#      the Arctic deep-column count, same window and res-deg as the control.
#  (b) CONFIRMS: Arctic SSS bias falls by > 0.2 psu OR genuine-deep falls
#      below ~130, with global SST rmse not worse by more than 0.02.
#      REFUTES: Arctic SSS and deep-count both move < 5 %, or global SST rmse
#      degrades > 0.05 (over-weak GM leaking into the mean state).
#      INDETERMINATE: Arctic improves but global SST degrades in the same
#      band -- a trade, to be reported as such and not as a win.
#  (c) WHY NO CHEAPER TEST: the coefficient acts through 30 days of eddy
#      transport; the offline coefficient comparison (already done) gives the
#      gap but not the state response.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_wt_mle_fix
cd $WT || { echo "FATAL: pinned worktree missing"; exit 1; }
echo "=== WT HEAD $(git rev-parse HEAD) ==="
git status --porcelain | head -5
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

CMD_FILE=$WT/scripts/cluster/omip_nemo/_ab_gmtreg_d30.cmd
[ -r "$CMD_FILE" ] || { echo "FATAL: command file missing"; exit 1; }
CMD=$(cat "$CMD_FILE")
grep -q -- "--gm-treguier" <<<"$CMD" || { echo "FATAL: --gm-treguier not in the command"; exit 1; }
grep -q -- "--gm-aei0 900" <<<"$CMD" || { echo "FATAL: aei0 not pinned to NEMO's 900"; exit 1; }

_missing=0
for tok in $CMD; do
  case "$tok" in
    /*|data/*)
      case "$tok" in */nemolev_trp_gmtreg_d30) continue ;; esac
      [ -e "$WT/$tok" ] || [ -e "$tok" ] || { echo "MISSING: $tok"; _missing=1; } ;;
  esac
done
[ $_missing -eq 0 ] || { echo "FATAL: preflight failed"; exit 1; }
echo "[preflight] every path token readable"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader || true

$PY -u $CMD
rc=$?
echo "=== run rc=$rc ==="
exit $rc
