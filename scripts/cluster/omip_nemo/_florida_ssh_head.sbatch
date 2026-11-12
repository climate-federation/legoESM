#!/bin/bash
#SBATCH --job-name=flhead
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/flhead_%j.out
#
# WHAT DRIVES TWICE THE OBSERVED FLOW THROUGH ONE GRID CELL?
#
# The entire eleven-Sverdrup Florida discrepancy sits in the single cell at
# 280.5E, where we carry 23.7 Sv against NEMO's 11.9, while the offshore cell
# at 283.5E matches NEMO to within a Sverdrup. A channel one cell wide cannot
# resolve a viscous boundary layer, so the Munk-width reasoning does not apply
# to it. What does apply is the sea-level difference the flow runs down.
#
# This bisects rather than ranking guesses, exactly as the same instrument did
# for the Bering:
#   heads differ  -> the FORCING is wrong and the cause is upstream, in the
#                    Gulf of Mexico / subtropical gyre mass distribution;
#   heads agree   -> the RESPONSE is wrong and the cause is local to the
#                    channel: friction, the barotropic solver, partial cells.
#
# TWO HEADS, because a strait flow can be set by either and they are different
# measurements:
#   ALONG   upstream (23-26N, the Straits of Florida) minus downstream
#           (28-31N, downstream of the section) over the strait's own
#           longitudes. This is the head a frictional channel flow runs down.
#   ACROSS  the strait cell's longitudes minus the offshore cell's, in a thin
#           band around the 27N section. A geostrophic jet's transport is set
#           by THIS one, and it is the pairing that matches the per-cell
#           finding: one cell wrong, its neighbour right.
#
# The days are 15 and 20, the interval in which our strait cell goes from
# 15.92 to 24.54 Sv while NEMO does not move. If a head jumps with it, that
# head is the driver; if neither does, the response is local.
#
# Record is DERIVED from the day, never inherited.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
R=/burg-archive/glab/users/pg2328/legoESM
cd $ROOT || exit 1
echo "=== HEAD $(git -C $ROOT log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
MESH=$R/data/grids/eORCA1.2_mesh_mask.nc
RUN=${RUN:-$R/results/omip_nemo/trp_orca1faithtint_d30}

for DAY in 15 20; do
  REC=$(( DAY / 5 - 1 ))
  SNAP=$RUN/snapshot_day00$DAY.npz
  [ -r "$SNAP" ] || { echo "SKIP day $DAY: no $SNAP"; continue; }
  for KIND in along across; do
    if [ "$KIND" = along ]; then
      BANDS="-82,-79" ; LA="23,26" ; LB="28,31" ; NA=upstream ; NB=downstream
    else
      BANDS="-80,-79" ; LA="26.5,27.5" ; LB="26.5,27.5" ; NA=strait ; NB=offshore
    fi
    echo ""
    echo "######## day $DAY, NEMO record $REC, $KIND head ########"
    if [ "$KIND" = across ]; then
      # Two longitude boxes at the SAME latitude, so the across head needs two
      # separate calls and their difference read off the two "ours" numbers.
      for LON in "-80,-79:strait" "-77,-76:offshore"; do
        B=${LON%%:*}; N=${LON##*:}
        echo "-- $N band $B"
        $PY -u scripts/validate/ocean_fidelity/bering_ssh_head.py \
          --snapshot "$SNAP" --mesh "$MESH" --nemo-gridt "$GT" \
          --time-idx "$REC" --lon-bands="$B" \
          --lat-a 26.5,27.5 --lat-b 26.5,27.5 \
          --label-a "$N" --label-b "$N-same"    2>&1
      done
    else
      $PY -u scripts/validate/ocean_fidelity/bering_ssh_head.py \
        --snapshot "$SNAP" --mesh "$MESH" --nemo-gridt "$GT" \
        --time-idx "$REC" --lon-bands="$BANDS" \
        --lat-a "$LA" --lat-b "$LB" --label-a "$NA" --label-b "$NB" \
           2>&1
    fi
  done
done
echo "=== rc=$? ==="
