#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nino3flx
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nino3flx_%j.out
#
# FORCING OR DYNAMICS, asked of the CORRECTED run.
#
# nino3_flux_vs_nemo.py was written against the old numbers -- its docstring
# still says "+3.03 C at day 30" -- and it has never been run on a snapshot
# that applies NEMO's lateral viscosity correctly.  The bias there is now
# +1.15 degC, and the day-25/day-30 pair shows it GROWING at about 0.06 degC
# per day while every other region agrees with the oracle to a few hundredths.
# A growing bias is a budget imbalance, so the budget is what gets run, and
# the probe's OWN pre-registered bands (in its docstring) are the reading:
#
#   heat within ~10 W/m2 and stress within ~10%  -> forcing is fine, the bias
#                                                   is DYNAMICS
#   our heat more positive by >20 W/m2           -> FORCING; work the bulk
#                                                   formulation first
#   our stress weak by >20%                      -> FORCING-driven weak
#                                                   undercurrent
#
# Run at BOTH days so the answer is not one snapshot.  The record ending on
# day D is D/5-1, and the probe refuses outright if the snapshot day falls
# outside the NEMO window it is scored against -- that refusal is the gate,
# not a nuisance.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

RES=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
SBC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_SBC.nc
# The probe's --mesh DEFAULT is the relative path 'data/grids/...', which only
# resolves from the RUN tree.  This card runs from the merge checkout, where
# that file does not exist, so the mesh is passed absolutely.
MESH=/burg-archive/glab/users/pg2328/_omip_wt/data/grids/eORCA1.2_mesh_mask.nc
for f in "$SBC" "$MESH"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

RC=0
one () {   # one <day> <snapshot>
  local day="$1" snap="$2"
  local rec=$(( day / 5 - 1 ))
  echo; echo "########## day $day, NEMO record $rec (days $((rec*5))-$(((rec+1)*5)))"
  echo "snapshot: $snap"
  [ -e "$snap" ] || { echo "FATAL: missing $snap"; RC=1; return; }
  $PY -u scripts/validate/ocean_fidelity/nino3_flux_vs_nemo.py \
    --snapshot "$snap" --nemo-sbc "$SBC" \
    --nemo-recs "${rec}:$((rec+1))" --day "$day" --mesh "$MESH"
  local rc=$?
  echo "=== day $day rc=$rc"
  [ $rc -eq 0 ] || RC=1
}

one 25 $RES/trp_orca1faithvisc_d30/snapshot_day0025.npz
one 30 $RES/trp_orca1faithvisc_d30/snapshot_final.npz

echo; echo "=== ALL DONE (rc=$RC) ==="
exit $RC
