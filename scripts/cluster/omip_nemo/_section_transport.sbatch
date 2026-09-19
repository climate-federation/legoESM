#!/bin/bash
#SBATCH --job-name=sectrans
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sectrans_%j.out
#
# BAROTROPIC CIRCULATION AT DAY 30: DRAKE PASSAGE AND FRIENDS.
#
# The barotropic mode adjusts in days, so at day 30 it is already near its
# attractor and a large difference is a solver or topography defect rather
# than a transient. That makes this the one remaining day-30 diagnostic with
# a sharp pass/fail, unlike AMOC or the global overturning which are nowhere
# near equilibrium yet and are deliberately not scored here.
#
# SECTIONS, stated rather than defaulted, matching the 26N/30S/140W and
# convection-site precedent:
#   drake     65W, 68S-55S   the ACC. Expect roughly 130-160 Sv EASTWARD;
#                            a value far outside that, or the wrong sign, is
#                            a finding rather than a tuning detail.
#   florida   27N, 80.5W-78.5W  the Florida Current, which runs NORTHWARD
#                              through its strait and so needs a ZONAL
#                              section. Only 2 cells wide on ORCA1, which
#                              is why floridawide exists: a 2-cell window
#                              that clips the current would understate it
#                              silently, exactly as the Antarctic shelf
#                              truncated the thermal-wind integral. If the
#                              two agree the narrow window is capturing the
#                              current; if the wide one is much larger the
#                              narrow number is a clipped artifact.
#   mozambique 17S, 38E-44E    the Mozambique Channel, southward.
#   indonesia 115E, 20S-5S   the Indonesian Throughflow, westward, ~10-20 Sv.
#                            A second section is cheap and guards against a
#                            single lucky agreement being read as a verdict.
#
# The record is DERIVED from the day, never inherited -- an omitted
# --nemo-time-idx once silently scored day 90 against a day-30 run and looked
# like a model regression for half a session.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
R=/burg-archive/glab/users/pg2328/legoESM
cd $ROOT || exit 1
echo "=== HEAD $(git -C $ROOT log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

DAY=$(( 10#${DAY:-30} ))
[ $(( DAY % 5 )) -eq 0 ] && [ "$DAY" -ge 5 ] || { echo "FATAL: DAY=$DAY not a positive multiple of 5"; exit 1; }
REC=$(( DAY / 5 - 1 ))

# SNAP and TAG are overridable so the SAME instrument can score a second run
# at the SAME day: ours-versus-ours at a matched day is the one comparison the
# snapshot-versus-five-day-mean confound cancels out of exactly.  Unset, this
# is the day-30 baseline call it has always been.
SNAP=${SNAP:-$R/results/omip_nemo/trp_orca1faithtint_d30/snapshot_final.npz}
TAG=${TAG:-}
# CELL=LON,LAT dumps that one face level by level on both sides.
# Unset, the card behaves exactly as before.
CELL=${CELL:-}
GU=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
GV=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_V.nc
MESH=$R/data/grids/eORCA1.2_mesh_mask.nc
for f in "$SNAP" "$GU" "$GV" "$MESH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
echo "=== day $DAY, GATEWAY record $REC (0-based) = days $((DAY-5))-$DAY ==="
echo "=== snapshot $SNAP ==="
# Which day is in that file, from the file itself? A leg's snapshot_final.npz
# carries the leg's end state, and pairing it with the wrong NEMO record is
# the exact mistake that once read a day-90 oracle against a day-30 run. The
# step count and the timestep say it outright, so print them and let the
# reader check rather than trusting the filename.
$PY -c "
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
keys = [k for k in z.files if k.lower() in ('_step', 'step', 'n_steps', 'day', 'time', 'dt')]
print('[stamp] ' + ('; '.join(f'{k}={np.asarray(z[k]).ravel()[0]}' for k in keys)
                    if keys else 'no step/day key in this snapshot'))
" "$SNAP"

$PY -u scripts/validate/ocean_fidelity/section_transport_vs_nemo.py \
  --snapshot "$SNAP" --nemo-gridu "$GU" --nemo-gridv "$GV" \
  --mesh-mask "$MESH" --rec "$REC" \
  --section="drake:-65:-68,-55" \
  --section="indonesia:115:-20,-5" \
  --zonal-section="florida:27:-80.5,-78.5" \
  --zonal-section="floridawide:27:-82,-76" \
  --zonal-section="mozambique:-17:38,44" \
  ${CELL:+--cell-profile="$CELL"} \
  --out-json "$R/results/omip_nemo/section_transport/transports_d${DAY}${TAG}.json"
echo "=== probe rc=$? ==="

# Row by row, because a single integral cannot say whether a difference is a
# broad excess or a jet that is too fast in three rows -- different causes,
# different fixes.
$PY -u scripts/plot/plot_section_transport.py \
  --json "$R/results/omip_nemo/section_transport/transports_d${DAY}${TAG}.json" \
  --out "$R/results/omip_nemo/section_transport/transports_d${DAY}${TAG}.png"
echo "=== rc=$? ==="
