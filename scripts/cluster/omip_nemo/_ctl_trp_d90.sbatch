#!/bin/bash
#SBATCH --job-name=trpctl90
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpctl90_%j.out
#
# THE 30-DAY PROTOCOL CANNOT ANSWER THE QUESTION WE ARE ASKING.
#
# Thirty days is a local wind response. An equatorial undercurrent equilibrates
# through boundary-wave round trips, which take MONTHS, so every ours-versus-
# NEMO number in the undercurrent thread compares TRANSIENTS and none of them
# excludes an initial-condition phase explanation. That was GLM's closing
# point and it is the reason this run exists. The user chose this option over
# a perturbation ensemble and over parking the thread.
#
# The oracle is already available: the NEMO gateway file covers 20000101 to
# 20000331, eighteen 5-day records, i.e. out to day 90.
#
# RESUMES FROM DAY 15, AND STEP/DAY NUMBERING IS ABSOLUTE.
# CORRECTED AFTER THE RUN COMPLETED. This paragraph originally called the job
# "a fresh 90-day leg", asserted that restart_leg1.npz "IS the day-0 state"
# and that snapshot days are "LEG-RELATIVE", and priced the choice as
# "re-integrating the first thirty days". All four claims are false.
# restart_leg1.npz is at STEP 8640, which at dt = 150 s and 576 steps/day is
# DAY 15, and the run's own log settles the numbering: "43200 steps this leg
# (absolute step 51840)", i.e. 75 days of integration carrying the state from
# day 15 to day 90, with step and day counted ABSOLUTELY rather than from the
# start of the leg. The property the original paragraph was reaching for does
# hold -- snapshot names carry the true absolute day, so the scorer's
# REC = DAY/5 - 1 needs no offset -- but it holds because the numbering is
# absolute, not because the restart was a day-0 state.
#
# BUILT PROGRAMMATICALLY from the control's own run_manifest command line,
# with exactly four values changed and the token count asserted unchanged at
# 122: --years to 90/365, --restart-from to the control's saved restart,
# --restart-save and --output into a new directory. Nothing else differs, so
# this is the same configuration as trp_faithz0_d30 carried out to day 90.
#
# WHAT IT WILL PRODUCE AND WHAT WOULD REFUTE WHAT (pre-registered, because a
# run costing more than one GPU-hour has to name its numbers first):
#   * the 220E undercurrent core speed and 0-400 m net zonal transport at
#     every 5-day mark out to day 90, ours against NEMO's matching record.
#   * If our transport deficit at 220E STOPS growing and settles at a roughly
#     constant fraction of NEMO's, the 30-day divergence was a spin-up
#     transient and the equilibrium difference is that fraction.
#   * If the deficit keeps growing through day 90, it is not a transient and
#     a momentum sink becomes the leading explanation.
#   * If our transport catches NEMO's, the whole undercurrent thread was a
#     transient artefact and closes.
#
# WHAT ACTUALLY HAPPENED, appended so the pre-registration above is recorded
# rather than quietly reinterpreted. The run completed at day 90 (step 51840,
# finite) and the deficit DID settle at a roughly constant fraction -- about
# 0.55 of NEMO's 220E core speed at every mark from day 35 to day 90. I read
# that as the first branch and wrote "spin-up refuted". THAT READING WAS
# ITSELF REFUTED: the undercurrent's transport is a basin-scale mass balance
# that adjusts through western-boundary reflection and Rossby propagation over
# six to twelve months, so ninety days sits INSIDE the adjustment window and a
# flat ratio there does not discriminate between the branches at all. The
# branches above were mis-labelled, not mis-measured -- a constant fraction is
# equally what two runs with different effective mixing look like mid
# transient. See retraction 7fcce5615; structural-versus-transient is OPEN,
# and NEMO's own core is still accelerating at every longitude at day 90.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_ctl_trp_d90.txt)
# ADOPTED 2026-09-23, BY THE USER, AFTER MEASUREMENT -- same change as the
# 30-day pinned card, and the same assertion. This lane now restores surface
# salinity as a real freshwater flux, the way the oracle does, instead of the
# global default's virtual-salt edit. The pre-registered arm (job 9939797,
# scored by 9948227) put ring-1 surface-salinity rmse at 0.5170 against the
# control's 0.6666, a 22 per cent reduction concentrated at the coast. It is
# three flags because the driver refuses water_flux without real_freshwater
# (run_omip_core2.py:8006) -- the virtual-salt closure's tendency is built
# from the NET freshwater, which already contains the restoring term.
#
# THE DAY-90 RESULTS ALREADY RECORDED WERE PRODUCED BEFORE THIS CHANGE. Any
# re-run of this card starts a NEW baseline and its numbers are not
# term-by-term comparable to the twelve-mark undercurrent series.
#
# The .txt is untracked, so the adoption is asserted rather than trusted; the
# pre-adoption line is kept as _ctl_trp_d90.pre_fwadopt.txt.
for _flag in "--sss-restore-channel water_flux" \
             "--sss-restore-normalization live_s" \
             "--freshwater-closure real_freshwater"; do
  case "$CMD" in
    *"$_flag"*) ;;
    *) echo "FATAL: command line lacks the adopted treatment ($_flag)"; exit 1 ;;
  esac
done
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_faithz0_d90/*.npz 2>/dev/null | tail -5
exit $rc
