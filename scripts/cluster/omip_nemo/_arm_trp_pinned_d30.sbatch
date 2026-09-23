#!/bin/bash
#SBATCH --job-name=trppin30
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trppin30_%j.out
#
# THE CLEAN A/B THE EARLIER COMPARISON WAS NOT.
#
# I attributed a six-day SST regression to the TKE surface boundary condition
# by comparing two production lineages that a manifest parity tool called
# equal on declared config. GLM broke it: the two runs were built from
# DIFFERENT CODE -- trp_orca1faithtint_d30 at 26acc418e on branch
# fix/omip-faithful-gap07-gm with a DIRTY tree, trp_faithz0_d30 at 3b5bdfb2a
# on merge/omip-main-sync and clean. A parity tool reads declared keys; it
# cannot see a commit delta. The attribution is retracted and the production
# default was put back untouched.
#
# This is the replacement: ONE binary, ONE variable. The control already
# exists and needs no run -- trp_faithz0_d30 is this exact command line at
# this exact commit with the card's default nemo_z0. This arm is that command
# line with --tke-surface-bc-level interior_pinned appended and nothing else
# changed.
#
# BUILT PROGRAMMATICALLY from the control's own run_manifest.json command
# line: the two restart flags were dropped (this runs as one leg from cold
# start; --years 0.0821918 is 30 days) and the output directory redirected.
# 122 tokens -> 120, as predicted by 122 - 4 + 2, with the count asserted at
# build time along with "no --restart* survived" and "exactly one --output".
#
# PRE-REGISTERED, on nino3 SST bias scored through _score_grids_vs_nemo at
# matched days against trp_faithz0_d30:
#   day 30 arm <= +1.15  -> the boundary condition CAUSES the regression
#                           (the two-lineage gap was +1.107 vs +1.340)
#   day 30 arm >= +1.29  -> the boundary condition is EXONERATED and the
#                           regression lives in the code delta between the
#                           two branches
#   +1.15 to +1.29       -> NO VERDICT, the flag is at most partly responsible
# Day 5 is deliberately NOT the primary: its two-lineage gap was 0.077
# against a 0.060 cross-grid noise floor, which is 1.3x noise and settles
# nothing. The 5-daily snapshots give every intermediate day for free.
#
# RUN FROM _omip_merge, NOT FROM THE MAIN CHECKOUT. The first submission of
# this card did `cd` to main and died in 21 s on "unrecognized arguments" for
# thirteen flags: main's driver is OLDER than the branch the control ran on.
# The control's own manifest says so -- package_path is under _omip_merge --
# and reproducing a run means reproducing its TREE, not just its flags.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_arm_trp_pinned_d30.txt)
# ADOPTED 2026-09-23, BY THE USER, AFTER MEASUREMENT: this lane now restores
# surface salinity the way the oracle does -- as a real freshwater flux --
# instead of the global default's virtual-salt edit. The pre-registered arm
# (job 9939797, scored by 9948227) put ring-1 surface-salinity rmse at 0.5170
# against the control's 0.6666 on a common domain, a 22 per cent reduction
# concentrated entirely at the coast (-0.150 in the edge ring, -0.057 in the
# next, +0.003 in the open ocean). NEMO restores with water; we did not.
#
# IT IS THREE FLAGS AND THE DRIVER MAKES THAT CHOICE: water_flux is refused
# without real_freshwater (run_omip_core2.py:8006), because the virtual-salt
# closure derives its tendency from the NET freshwater, which already contains
# the restoring term, so routing restoring as water under it would apply it
# twice. live_s is forced for the same reason.
#
# EVERY NUMBER FROM THIS CARD IS NOW ON A NEW BASELINE and is not term-by-term
# comparable to runs from before this date.
#
# The command line lives in an untracked .txt, so this asserts the adoption
# rather than trusting it -- a regenerated or restored-from-backup .txt would
# otherwise run the old treatment silently, which is exactly the class of
# defect this campaign keeps paying for. The pre-adoption line is kept beside
# it as _arm_trp_pinned_d30.pre_fwadopt.txt.
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
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_pinned_d30/*.npz 2>/dev/null
echo DONE
# The first version of this card ended on `echo DONE`, so SLURM recorded
# COMPLETED for a run that had died of GPU out-of-memory at day 0. A job's
# exit status is the LAST command's, and a status that cannot fail is not a
# status.
exit $rc
