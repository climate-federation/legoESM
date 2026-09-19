#!/bin/bash
#SBATCH --job-name=sections
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sections_%j.out
#
# DEPTH CROSS-SECTIONS, tripole | MPAS | NEMO, at day 30.
#
# Every figure the campaign has produced so far is a surface field. The two
# questions still open -- the equatorial thermocline behind the cold-tongue
# bias, and whether the polar salinity agreement holds below the skin -- are
# both interior questions, and nothing plotted the interior.
#
# THE ORACLE RECORD IS DERIVED FROM THE DAY, NOT DEFAULTED. The GATEWAY grid_T
# file holds 5-day means over 90 days; the record ENDING at snapshot day D is
# D/5-1. Passing no record at all is what scored a day-30 state against the
# day-90 mean in the Treguier scorecard and read as an MLD regression.
set -euo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

DAY=$(( 10#${DAY:-30} ))
[ $(( DAY % 5 )) -eq 0 ] && [ "$DAY" -ge 5 ] || { echo "FATAL: DAY=$DAY not a positive multiple of 5"; exit 1; }
REC=$(( DAY / 5 - 1 ))
TAG=$(printf 'day%04d' "$DAY")

TRP=${TRP:-trp_orca1faithtint_d30}
MPAS=${MPAS:-mpas_gmtreguier_d30}
snap() {   # run dir -> snapshot for DAY, falling back to final when that is
           # the run's target day (the Treguier arm wrote only snapshot_final)
  local f=$R/$1/snapshot_$TAG.npz
  [ -r "$f" ] || f=$R/$1/snapshot_final.npz
  [ -r "$f" ] || { echo "FATAL: $1 has no day-$DAY snapshot" >&2; return 1; }
  echo "$f"
}
ST=$(snap "$TRP") || exit 1
SM=$(snap "$MPAS") || exit 1
[ -r "$GW" ] || { echo "FATAL: missing $GW"; exit 1; }
OUT=$R/sections_d${DAY}_${MPAS}

echo "=== day $DAY, GATEWAY record $REC (0-based) = days $((DAY-5))-$DAY ==="
ls -l --time-style=+%Y-%m-%d_%H:%M "$ST" "$SM"

# BASIN-RESOLVED SECTIONS. A global zonal mean cancels opposite-signed basin
# anomalies, so the Atlantic gets its own windows. Both are stated explicitly
# because the basin's longitude span CHANGES WITH LATITUDE and the repo's
# existing BASINS['atlantic'] (290E onward) would clip the Florida Current and
# the deep western boundary current out of the 26N section -- the RAPID
# latitude, where that boundary current IS the measurement.
#   26N: 279-345E spans Florida to the African margin.
#   30S: 312-17E wraps the prime meridian, South America to Africa.
#
# MERIDIONAL SECTION AT 140W, 20S-20N. The equatorial section already built
# here runs ALONG the equator and shows the thermocline's east-west slope; it
# cannot show how sharply that thermocline is confined to the equator. The
# 140W line is the standard TAO/TRITON view of the cold tongue and the
# Equatorial Undercurrent, and the structure it exposes -- thermocline
# doming and its meridional width -- is readable at day 30 even though the
# cold tongue's AMPLITUDE is not yet a climatology. Window stated rather than
# defaulted: 140W +/- 1.5 deg (three columns at 1 degree), 20S to 20N.
$PY -u scripts/plot/plot_sections_vs_nemo.py \
  --tripole "$ST" --mpas "$SM" --label-mpas "${LABEL_MPAS:-MPAS}" \
  --nemo-gridt "$GW" --nemo-time-idx "$REC" --res-deg 1.0 \
  --lat-section="26:279,345:1.5:atlantic26N" \
  --lat-section="-30:312,17:1.5:atlantic30S" \
  --lon-section="-140:-20,20:1.5:pacific140W" \
  --out-dir "$OUT" 2>&1 | grep -vE "Warning|warn" | tail -60

echo "=== figures ==="
ls -l "$OUT"
