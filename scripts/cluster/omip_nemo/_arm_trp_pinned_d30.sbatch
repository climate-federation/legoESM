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
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
echo "=== card exit: $? ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_pinned_d30/*.npz 2>/dev/null
echo DONE
