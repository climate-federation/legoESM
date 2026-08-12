#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=fct2d30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fct2d30_%j.out
#
# ONE-VARIABLE A/B: tracer advection superbee -> fct2 (NEMO's own scheme).
#
# WHY THIS RUN EXISTS.  The Stage-C operator test (offline, no GPU) showed
# that when our advection operator is driven by NEMO's OWN effective
# transports, fct2 reproduces NEMO's advection trend BETTER THAN superbee in
# EVERY region and BOTH tracers (T corr 0.592-0.901 vs 0.480-0.868; S
# 0.285-0.979 vs 0.278-0.973).  ORCA1 runs ln_traadv_fct with
# nn_fct_h = nn_fct_v = 2, i.e. exactly fct2; superbee is OUR choice.
# That is an OPERATOR result.  Whether it improves the STATE (SST/SSS/MLD vs
# NEMO) is a different question and needs this run -- the offline test cannot
# answer it because a scheme difference acts through 30 days of feedback.
#
# CONTROL: nemolev_trp_gwcorr_d30 (same command, superbee), already scored.
#
# PRE-REGISTERED (written before the run):
#  (a) THE NUMBERS: day-30 compare_omip_nemo SST rmse / SSS rmse / MLD rmse
#      vs NEMO month 1, on the same res-deg and window as the control.
#  (b) CONFIRMS: SST rmse falls below the control's 0.912 AND MLD rmse does
#      not degrade (control 37.8).  REFUTES: SST rmse rises, or MLD rmse
#      rises by more than ~10 %.  INDETERMINATE: both move < 0.02 / < 2 m --
#      the scheme is operator-visible but state-invisible at 30 days, which
#      is itself a legitimate answer (and would argue for keeping superbee
#      on its other merits).
#  (c) WHY NO CHEAPER TEST: the offline operator test IS the cheaper test and
#      has been run; it cannot produce the 30-day state response.
#
# WATCH: superbee is monotone and was chosen for the Gulf Stream (memory
# omip-finescale-amazon-levers: ppm_fct blew it up).  fct2 is a DIFFERENT
# FCT; if this arm shows grid-scale noise in western boundary currents, that
# is the same failure class and the arm should be reported as such, not as a
# clean win.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_wt_mle_fix
cd $WT || { echo "FATAL: pinned worktree missing"; exit 1; }
echo "=== WT HEAD $(git rev-parse HEAD) ==="
git status --porcelain | head -5
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

CMD_FILE=$WT/scripts/cluster/omip_nemo/_ab_fct2_d30.cmd
[ -r "$CMD_FILE" ] || { echo "FATAL: command file missing"; exit 1; }
CMD=$(cat "$CMD_FILE")
grep -q -- "--tracer-advection fct2" <<<"$CMD" || { echo "FATAL: fct2 not in the command"; exit 1; }

_missing=0
for tok in $CMD; do
  case "$tok" in
    /*|data/*)
      case "$tok" in */nemolev_trp_fct2_d30) continue ;; esac
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
