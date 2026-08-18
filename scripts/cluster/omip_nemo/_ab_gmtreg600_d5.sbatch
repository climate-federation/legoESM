#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=gmtreg600
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/gmtreg600_%j.out
#
# ONE-VARIABLE A/B: GM coefficient constant 600 -> NEMO's flow-dependent form
# (--gm-treguier = ldf_eiv nn_aei_ijk_t=21, F(baroclinic growth rate)).
#
# DISCRIMINATOR for the day-5 blowup of the aei0=900 Treguier arm (job
# 9379514: [ABORT] non-finite state at step 2880, ice_area -> 0).
#
# THE QUESTION: was the blowup caused by the MAGNITUDE of the flow-dependent
# kappa (cap 900 vs the control's constant 600) or by its FORM (a spatially
# and temporally varying kappa, whatever its size)?
# THIS ARM: identical to the failed one except --gm-aei0 600, i.e. the SAME
# cap as the control's constant kappa.  Only the FORM then differs.
# 5 days only -- the failure appeared by day 5, so that is all that is needed.
#
# PRE-REGISTERED:
#  (a) NUMBER: finite state at the day-5 diagnostic (mean_sst_C not NaN).
#  (b) SURVIVES  -> the blowup was MAGNITUDE/CFL: the fix is a CFL-aware cap,
#      and NEMO's own 900 is reachable only with its limiter (ln_traldf_msc,
#      which we do not implement).
#      BLOWS UP  -> the blowup is STRUCTURAL in our Treguier path (varying
#      kappa x our centred bolus advection), independent of magnitude, and
#      the flow-dependent GM lever is BLOCKED until that is fixed.
#  (c) WHY NO CHEAPER TEST: an offline kappa dump gives the magnitude but not
#      the stability; only integration answers it.  5 days x 1 GPU is the
#      minimum that reproduces the failure.
#
# CONTROL for interpretation: nemolev_trp_gwcorr_d30 (constant kappa 600,
# both adaptive schemes disabled -- verified in its resolved config), stable.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_wt_mle_fix
cd $WT || { echo "FATAL: pinned worktree missing"; exit 1; }
echo "=== WT HEAD $(git rev-parse HEAD) ==="
git status --porcelain | head -5
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

CMD_FILE=$WT/scripts/cluster/omip_nemo/_ab_gmtreg600_d5.cmd
[ -r "$CMD_FILE" ] || { echo "FATAL: command file missing"; exit 1; }
CMD=$(cat "$CMD_FILE")
grep -q -- "--gm-treguier" <<<"$CMD" || { echo "FATAL: --gm-treguier not in the command"; exit 1; }
grep -q -- "--gm-aei0 600" <<<"$CMD" || { echo "FATAL: aei0 not pinned to NEMO's 900"; exit 1; }

_missing=0
for tok in $CMD; do
  case "$tok" in
    /*|data/*)
      case "$tok" in */nemolev_trp_gmtreg600_d5) continue ;; esac
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
