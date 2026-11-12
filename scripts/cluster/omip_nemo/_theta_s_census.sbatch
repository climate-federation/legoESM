#!/bin/bash
#SBATCH --job-name=tscensus
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tscensus_%j.out
#
# VOLUMETRIC theta-S CENSUS, ours vs NEMO, day 30.
#
# Maps and sections both cancel: a global zonal mean cancels opposite-signed
# basin anomalies, a section only sees the water it cuts. A census has no
# geography at all, so it cannot cancel, and at 30 days from rest it is the
# cleanest probe of SPURIOUS DIAPYCNAL MIXING -- a numerically diffusive model
# moves volume out of the distribution's tails toward its centre regardless of
# surface fluxes.
#
# PRE-REGISTERED, sign stated before the number is read: if we mix more than
# NEMO, the extreme temperature/salinity bins must hold LESS volume than
# NEMO's and the central bins MORE, with the tail deficit roughly balancing
# the central surplus. Large but UNSTRUCTURED differences are NOT spurious
# mixing. A null at day 30 is the expected, desirable outcome.
#
# The bin ranges are stated here rather than defaulted, and the probe reports
# what fraction of ocean volume falls OUTSIDE them -- a census whose bins miss
# real water understates exactly the tails the test reads.
set -euo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
# HARDCODED: `ls` lists broken symlinks in the grid directory.
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc

DAY=$(( 10#${DAY:-30} ))
[ $(( DAY % 5 )) -eq 0 ] && [ "$DAY" -ge 5 ] || { echo "FATAL: DAY=$DAY not a positive multiple of 5"; exit 1; }
REC=$(( DAY / 5 - 1 ))
TAG=$(printf 'day%04d' "$DAY")
TRP=${TRP:-trp_orca1faithtint_d30}
SNAP=$R/$TRP/snapshot_$TAG.npz
[ -r "$SNAP" ] || SNAP=$R/$TRP/snapshot_final.npz
for f in "$SNAP" "$GW" "$MESH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

echo "=== day $DAY, GATEWAY record $REC (0-based) = days $((DAY-5))-$DAY ==="
ls -l --time-style=+%Y-%m-%d_%H:%M "$SNAP"

$PY -u scripts/validate/ocean_fidelity/theta_s_census_vs_nemo.py \
  --snapshot "$SNAP" --mesh-mask "$MESH" \
  --nemo-gridt "$GW" --nemo-time-idx "$REC" \
  --label "${LABEL:-tripole}" \
  --t-range=-2.5,32.0 --s-range=30.0,38.0 --n-bins 140 --tail-pct 5.0 \
  --nemo-self-ref-idx $(( REC - 1 )) \
  --out-dir "$R/census_d${DAY}_${TRP}" 2>&1 | grep -vE "Warning|warn" | tail -40

echo "=== figures ==="
ls -l "$R/census_d${DAY}_${TRP}"
