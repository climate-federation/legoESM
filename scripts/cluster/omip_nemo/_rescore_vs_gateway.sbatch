#!/bin/bash
#SBATCH --job-name=rescore_gw
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/rescore_gw_%j.out
#
# THE REFERENCE HAS BEEN WRONG.  Every SST/SSS/MLD score in this campaign has
# compared a 30-day COLD START against NEMO's FIVE-YEAR JANUARY CLIMATOLOGY
# (RUN_REF monthly grid_T, records 0/12/24/36 averaged).  That is a spun-up
# equilibrium, not the matched transient, so part of every reported "bias" is
# NEMO's own spin-up rather than our error.
#
# NEMO's matched cold start has been on disk the whole time:
#   RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
#   18 five-day records from the SAME January initial state, days 1-90,
#   carrying tos, sos and mldr10_1 -- everything the comparator reads.
# The equatorial instrument already scores against RUN_GATEWAY's grid_U/V/W.
# Only the SST/SSS/MLD scorecard was still pointed at the climatology.
#
# RECORD MAPPING (5-day means from 2000-01-01): rec 3 = days 16-20,
# rec 5 = days 26-30, rec 17 = days 86-90.
#
# RESIDUAL MISMATCH, stated with every number this produces: our snapshots are
# INSTANTANEOUS and NEMO's records are 5-day MEANS.  That is a smaller error
# than a five-year climatology but it is not zero, and it matters most for
# quantities with a short-timescale maximum (mixed-layer depth, velocity
# extrema) rather than for basin SST means.
#
# Costs CPU only: no model steps, existing snapshots re-scored.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu

GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
REF_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc

score () {   # arm  day  record  label
  local arm=$1 day=$2 rec=$3 label=$4
  local snap=$ROOT/results/omip_nemo/nemolev_trp_${arm}/snapshot_day${day}.npz
  echo
  echo "############ $label : $arm day $day vs GATEWAY rec $rec ############"
  [ -f "$snap" ] || { echo "SKIP: no $snap"; return; }
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$snap" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$rec" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label "${label}_gw" \
    --output-dir $ROOT/results/omip_nemo/compare_gw_${label} 2>&1 | tail -38
  echo "### rc=$?"
}

# The three finished cells at the day both the day-20 and day-30 comparisons
# already use, so the new reference is the ONLY thing that changed.
score signedn2_d30 0030 5  ctl_d30
score modfrc_d30   0030 5  modfrc_d30
score ahsmag1_d90  0030 5  ahsmag1_d30
# Day 20, where the corrected-forcing effect was first reported.
score signedn2_d30 0020 3  ctl_d20
score modfrc_d30   0020 3  modfrc_d20
# Day 90, the only cell that has one.
score ahsmag1_d90  0090 17 ahsmag1_d90

echo
echo "############ CONTROL FOR THE CONTROL ############"
echo "The same arm against the OLD climatology reference, so the size of the"
echo "reference change itself is measured rather than assumed."
$PY scripts/validate/compare_omip_nemo.py \
  --legoesm-snapshot $ROOT/results/omip_nemo/nemolev_trp_modfrc_d30/snapshot_day0030.npz \
  --nemo-gridt "$REF_T" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label modfrc_d30_climref \
  --output-dir $ROOT/results/omip_nemo/compare_gw_modfrc_d30_climref 2>&1 | tail -30
echo "### rc=$?"
echo DONE
