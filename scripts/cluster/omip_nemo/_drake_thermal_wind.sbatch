#!/bin/bash
#SBATCH --job-name=drakeTW
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/drakeTW_%j.out
#
# IS OUR STRONG ACC EXPLAINED BY THE DENSITY FIELD, OR NOT?
#
# Our Drake transport is 162.34 Sv against NEMO's 152.27 (ratio 1.066) and
# the jet sits about half a degree north. Separately our Weddell near-surface
# stratification is far too weak. This decides whether those two are the same
# problem or two problems, by measuring the cross-frontal density contrast
# that thermal wind says sets the transport.
#
# The falsifier is PRE-REGISTERED in the probe's docstring with a number and
# a direction: >=1.05 supports a shared cause, <=1.02 says independent,
# between is ambiguous and is reported as ambiguous. The reading also
# requires the excess to sit ABOVE 300 m, where the stratification difference
# lives, and that fraction is printed whichever way the ratio lands.
#
# Record DERIVED from the day, never inherited.
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

SNAP=$R/results/omip_nemo/trp_orca1faithtint_d30/snapshot_final.npz
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
MESH=$R/data/grids/eORCA1.2_mesh_mask.nc
for f in "$SNAP" "$GT" "$MESH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
echo "=== day $DAY, GATEWAY record $REC (0-based) = days $((DAY-5))-$DAY ==="

$PY -u scripts/validate/ocean_fidelity/drake_thermal_wind_vs_nemo.py \
  --snapshot "$SNAP" --nemo-gridt "$GT" --mesh-mask "$MESH" --rec "$REC" \
  --lon=-65 --lat-range=-68,-55 \
  --out-json "$R/results/omip_nemo/section_transport/drake_thermal_wind_d${DAY}.json"
echo "=== edge-box rc=$? ==="

# SENSITIVITY, and it is not optional. The edge default puts the southern box
# on the ANTARCTIC SHELF, so the cross-front difference is undefined below the
# shelf and the double integral silently stops there -- the ACC's deep limb
# never enters the test. This pair places BOTH boxes in deep water. If the
# verdict survives the change it is about the current; if it flips, the first
# answer was about a few hundred metres of shelf water.
$PY -u scripts/validate/ocean_fidelity/drake_thermal_wind_vs_nemo.py \
  --snapshot "$SNAP" --nemo-gridt "$GT" --mesh-mask "$MESH" --rec "$REC" \
  --lon=-65 --lat-range=-68,-55 \
  --south-lat=-62,-60 --north-lat=-57,-55.4 \
  --out-json "$R/results/omip_nemo/section_transport/drake_thermal_wind_deep_d${DAY}.json"
echo "=== rc=$? ==="
