#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=gwint
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/gwint_%j.out
#
# NEMO'S SIDE OF THE ARCTIC VOLUME INTEGRAL.
#
# Our accumulated net through the four 66N gateways over days 15-30 is
# -1.0050e12 m3 (the Arctic LOSES water), while our Arctic sea level stands
# 2.2557e12 m3 ABOVE NEMO's. Those are not comparable: the first is OURS
# ALONE, the second is a DIFFERENCE. If NEMO's Arctic loses more over the same
# window the two are consistent, and only NEMO's integral can say.
#
# NEMO writes 5-day means, so the integral is a sum over records of
# (transport x 5 days). Records 3, 4 and 5 span days 15-30, matching our CSV's
# accumulation window; records 0-5 span the full 30 days and are printed too,
# because the windows must be stated next to the numbers rather than assumed
# to match.
#
# PRE-REGISTERED: if NEMO's net over days 15-30 is MORE negative than ours by
# roughly 2e12 m3, the sea level and the transports agree and the Arctic
# excess is explained by gateway exchange. If NEMO's is similar or less
# negative, they disagree and something outside the gateways -- a surface
# volume flux, or a difference in what each model calls sea surface height --
# is carrying the excess.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc
for f in "$MESH" "$GW/ORCA1_5d_20000101_20000331_grid_U.nc" "$GW/ORCA1_5d_20000101_20000331_grid_V.nc"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

RC=0
for REC in 0 1 2 3 4 5; do
  echo ""
  echo "=== NEMO record $REC (days $((REC*5))-$((REC*5+5))) ==="
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -u scripts/validate/nemo_transports.py \
    --arctic-gateways --time-idx $REC \
    --grid-u $GW/ORCA1_5d_20000101_20000331_grid_U.nc \
    --grid-v $GW/ORCA1_5d_20000101_20000331_grid_V.nc \
    --mesh $MESH 2>&1 | grep -E "bering|davis|atlantic|siberian|NET|Sv" | head -12
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: record $REC failed"; RC=1; }
done

echo ""
echo "DONE rc=$RC"
echo "Sum records 3,4,5 x 432000 s for the days 15-30 volume, to compare with"
echo "our -1.0050e12 m3 over the same window."
exit $RC
