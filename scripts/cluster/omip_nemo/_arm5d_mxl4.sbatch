#!/bin/bash
#SBATCH --job-name=trpmxl4d5
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpmxl4d5_%j.out
#
# ONE VARIABLE: TKE mixing-length choice 3 -> 4, five days, tripole.
#
# WHY FIVE DAYS IS ENOUGH, and this is the whole reason the arm is affordable:
# the equatorial warm bias is already 35-44 percent of its day-30 value at day
# 5 on all three grids, measured from the six matched-day reports. A lever test
# no longer needs a thirty-day run.
#
# WHY NO CONTROL ARM IS SUBMITTED: the control already exists. The harmonised
# tripole's own day-5 snapshot (trp_faithz0_d30/snapshot_day0005.npz) came from
# a fresh leg with no restart, under this identical command line, and its
# viscosity ramp segment (0:1e4:3.0, days 0-8) covers the whole of this arm's
# five days. Submitting a second identical run would buy nothing.
#
# THE COMMAND LINE IS THE HARMONISED TRIPOLE'S OWN, taken from its
# run_manifest.json rather than retyped, with exactly four edits:
#   --years            30/365 -> 5/365
#   --tke-mxl-choice   3 -> 4
#   --output           a new directory
#   --restart-from/--restart-save  removed, so this is a fresh integration from
#                      the same initial condition the control started from
# Building a card from the working card, never from scratch: a from-scratch
# card went unstable inside a day earlier in this campaign and cost a 78-minute
# run plus eight arms.
#
# WHY choice 4: ORCA1's namelist nn_mxl=2 maps to choice 4, and our runs have
# been on 3. This is a faithfulness gap, and the mixed layer is now the leading
# suspect for the equatorial bias -- measured 14.70 m against NEMO's 17.95 m in
# nino3 at day 5, on both the tripole and MPAS.
#
# PRE-REGISTERED, nino3 day-5 SST bias against the control's +0.564 C:
#   <= +0.40   the mixing length is a real lever on the onset; adopt and rerun
#              the thirty-day card on choice 4.
#   +0.40 to +0.70  no verdict; the change is inside the cross-grid spread and
#              cannot be separated from it.
#   >= +0.70   choice 4 makes the onset worse and stays off, reported as a real
#              result about NEMO's own setting on our closure.
#   non-finite before day 5 -> BLOWUP, reported as such.
# Also recorded, not gated: the nino3 mixed-layer depth, which is the quantity
# the hypothesis actually concerns.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_arm_trp_mxl4.txt)
echo "CMD: $PY scripts/run/$CMD"
$PY scripts/run/$CMD
rc=$?; echo "=== rc=$rc ==="
exit $rc
