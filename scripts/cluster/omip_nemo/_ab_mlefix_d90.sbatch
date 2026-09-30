#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=mlefix90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mlefix90_%j.out
#
# MLE A/B RELAUNCH on the FIXED operator (vertical bolus branch, commit
# 465823b83).  Control = nemolev_trp_gwcorr_d90 (VSF closure; already run +
# scored).  The fwreal control's command is NOT runnable at the fix SHA: the
# current code rejects --freshwater-closure real_freshwater with
# --sss-restore (the virtual-salt restoring is wrong under volume dilution,
# #1484), a guard added after that control ran -- job 9361926 failed on it.
# The command is the gwcorr manifest command + --mle --mle-ce 0.06 and a new
# --output; the MLE code paths are INACTIVE in the control, so the pair
# remains effectively one-variable even though the code revision differs.
# Day-30 dSSS comparisons therefore use the GWCORR arm as control.
#
# Predecessor arm nemolev_trp_mle_d90 (job 9337800, PRE-fix code) REFUTED its
# own prediction and corrupted the state (-54 psu river-plume extremes, global
# surface SSS -4.5 psu / 90 d) through the missing vertical branch.
#
# PRE-REGISTERED (before this run):
#  (a) NUMBERS: day-30 A/B dSSS map vs the fwreal control
#      (mle_psi_diagnostics.py), day-90 genuine-deep Arctic column count
#      (arctic_deep_convection_columns.py, NEMO March), day-90
#      compare_omip_nemo scorecard.
#  (b) CONFIRMS the fix: day-30 p99|dSSS vs control| < 2 psu (was 26.4) and
#      no cell below -10 psu (was -54); global day-90 SST rmse within 0.1 of
#      the control's 0.826.
#      CONFIRMS the physics goal: genuine-deep count falls 291 -> toward
#      NEMO's 84 without Antarctic/global MLD degradation.
#      REFUTES fix: extremes persist (p99 > 10 psu) -> vertical branch
#      insufficient or wrongly signed in production; kill MLE default.
#      REFUTES lever: count stays > 250 with a healthy state -> MLE is not
#      the Arctic deep-convection lever; keep the fix as correctness, drop
#      the restratification hypothesis.
#  (c) WHY NO CHEAPER TEST: the fix changes the trajectory from step 1; unit
#      tests + offline probes (already run) cannot produce the 90-day
#      coupled response.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_wt_mle_fix
FIXSHA=a2e0a325719ffd48a80a62cea0b158f642bbafe1
cd $WT || { echo "FATAL: worktree missing"; exit 1; }
HEAD=$(git rev-parse HEAD)
echo "=== WT HEAD $HEAD (want $FIXSHA) ==="
[ "$HEAD" = "$FIXSHA" ] || { echo "FATAL: worktree not at pinned SHA"; exit 1; }
git status --porcelain | head -5
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

CMD_FILE=$WT/scripts/cluster/omip_nemo/_ab_mlefix_d90.cmd
[ -r "$CMD_FILE" ] || { echo "FATAL: command file missing"; exit 1; }
CMD=$(cat "$CMD_FILE")

# Preflight: every path-shaped token must be readable BEFORE allocation burn.
_missing=0
for tok in $CMD; do
  case "$tok" in
    /*|data/*)
      # output dir is created by the run; skip it
      case "$tok" in */nemolev_trp_mlefix_d90) continue ;; esac
      [ -e "$WT/$tok" ] || [ -e "$tok" ] || { echo "MISSING: $tok"; _missing=1; } ;;
  esac
done
[ $_missing -eq 0 ] || { echo "FATAL: preflight failed"; exit 1; }
echo "[preflight] every path token readable"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

# Duration comes from the control command itself (--years 0.2466 = 90 days).
$PY -u $CMD
rc=$?
echo "=== run rc=$rc ==="
exit $rc
