#!/bin/bash
#SBATCH --job-name=ahsmodgw
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ahsmodgw_%j.out
#
# THE FOURTH CORNER, SCORED AGAINST THE RIGHT REFERENCE AND ON THE RIGHT
# STATISTIC.  Costs CPU only: the snapshots are already on disk.
#
# The 2x2 factorial is corrected CORE-II forcing x NEMO's own lateral
# viscosity file (plus Smagorinsky 3.0 -> 1.0, which makes that axis a
# composite rather than a clean A_h factorial -- say so with every number).
# Its fourth arm carried its scoring inside the run script, pointed at NEMO's
# FIVE-YEAR JANUARY CLIMATOLOGY.  Our runs are 30-day cold starts, so part of
# every bias that scoring reports is NEMO's own spin-up.  NEMO's matched cold
# start (RUN_GATEWAY, 5-day means from the same January state) is what the
# other three corners were rescored against, and this brings the fourth onto
# it.
#
# It also applies the statistic the campaign has settled on.  The tripole's
# cold tongue is a GROWING drift, so a day-30 value is a rate caught
# mid-flight; the day-10-to-day-30 change per longitude is what separates a
# fix from a coincidence.  Reference numbers for that change at 220-240E:
# NEMO -1.67, FESOM (same forcing, different dycore) -1.68, ours +0.74.
#
# The numbers this produces supersede the fourth arm's own printout, which
# stays in its log against the climatology and must not be quoted.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=$ROOT/results/omip_nemo
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

score () {   # dir  day  rec  label
  local snap=$R/$1/snapshot_day$2.npz
  [ -f "$snap" ] || { echo "SKIP: no $snap"; return; }
  echo
  echo "############ $4 day $2 vs GATEWAY rec $3 ############"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$snap" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$3" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label "$4_d$2" \
    --output-dir $R/compare_gw_$4_d$2 2>&1 | tail -32
  echo "### rc=$?"
}

# The fourth corner at every day it shares with the other three.
score nemolev_trp_ahsmod_d90 0010 1  ahsmod
score nemolev_trp_ahsmod_d90 0020 3  ahsmod
score nemolev_trp_ahsmod_d90 0030 5  ahsmod

echo
echo "############ COLD-TONGUE TENDENCY, day 10 -> day 30, area-weighted ############"
echo "One table, three model arms and the oracle, the SAME box and reduction."
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py \
  --snapshot $R/nemolev_trp_ahsmod_d90/snapshot_day0030.npz --label ahsmod \
  --snapshot $R/nemolev_trp_modfrc_d30/snapshot_day0030.npz --label modfrc \
  --snapshot-early $R/nemolev_trp_ahsmod_d90/snapshot_day0010.npz \
  --snapshot-early $R/nemolev_trp_modfrc_d30/snapshot_day0010.npz \
  --nemo-gridt "$GW_T" --nemo-rec 5 --nemo-rec-early 1 2>&1 | tail -70
echo "### rc=$?"
echo DONE
