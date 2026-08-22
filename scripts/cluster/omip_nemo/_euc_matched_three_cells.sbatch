#!/bin/bash
#SBATCH --job-name=euc3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/euc3_%j.out
#
# THE MISSING CONTROL: the same undercurrent instrument, the same simulated
# day, the same NEMO records, run on all three finished cells of the forcing x
# viscosity factorial.
#
# WHY. The corrected-forcing arm's undercurrent (0.116 m/s at 200E) was being
# compared against a remembered 0.34 from the raw-forcing control -- a number
# from a different instrument invocation on a different day.  Comparing across
# invocations is a confound, not a result, and the whole "the undercurrent got
# WEAKER under stronger trades" reading rests on it.  This job replaces the
# remembered number with a measured one.
#
# Every arm below is scored from its own day-30 snapshot against the same NEMO
# 5-day records 5:6 (simulated days 26-30) and the same NEMO month 1, so the
# three numbers may be compared to each other and to NEMO.
#
#   signedn2  raw winds,       flat A_h + Smagorinsky 3.0   (the control)
#   modfrc    corrected winds, flat A_h + Smagorinsky 3.0
#   ahsmag1   raw winds,       ORCA1 viscosity file + Smag 1.0
#
# COSTS NOTHING BUT CPU: no model steps, only diagnostics on snapshots that
# already exist.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu

GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
GU=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
GV=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_V.nc

for ARM in signedn2_d30 modfrc_d30 ahsmag1_d90; do
  SNAP=$ROOT/results/omip_nemo/nemolev_trp_${ARM}/snapshot_day0030.npz
  echo
  echo "############################################################"
  echo "### ARM $ARM   snapshot $SNAP"
  echo "############################################################"
  [ -f "$SNAP" ] || { echo "FATAL: missing $SNAP -- a cell scored without its snapshot is not a cell"; continue; }
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-month 1 \
    --label ${ARM}_matched_d30 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 \
    --nemo-ufile "$GU" --meridional-lon 240,260 \
    --nemo-vfile "$GV" 2>&1 | tail -150
  echo "### $ARM rc=$?"
done
echo DONE
