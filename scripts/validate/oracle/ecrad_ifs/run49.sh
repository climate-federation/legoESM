#!/bin/bash
# Run an ecRad oracle executable on one input file with the 49r1 namelist.
# usage: run49.sh <exe> <namelist> <data dir> <input.nc> <output.nc> [key=value ...]
#   key=value pairs override &radiation entries (harness stages, e.g. use_aerosols=false)
set -euo pipefail
EXE=$1; NAM=$2; DATA=$3; IN=$4; OUT=$5; shift 5
W=$(mktemp -d "${TMPDIR:-/tmp}/ecrad49run.XXXX")
sed "s#\"DATA_DIR\"#\"$DATA\"#" "$NAM" > "$W/config.nam"
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  if grep -qE "^$k *=" "$W/config.nam"; then sed -i -E "s#^($k *= *)[^,]*,#\1$v,#" "$W/config.nam"
  else sed -i -E "0,/^&radiation\$/s##\&radiation\n$k = $v,#" "$W/config.nam"; fi
  grep -qE "^$k *= *$v," "$W/config.nam" || { echo "override $kv not applied"; exit 1; }
done
cp "$W/config.nam" "${OUT%.nc}.nam"
( cd "$W" && OMP_NUM_THREADS=1 "$EXE" config.nam "$(readlink -f "$IN")" "$W/out.nc" > "$W/run.log" 2>&1 ) || { tail -20 "$W/run.log"; exit 1; }
mv "$W/out.nc" "$OUT"
for f in "$W"/radiative_properties*.nc; do [ -e "$f" ] && mv "$f" "${OUT%.nc}_$(basename "$f")"; done
cp "$W/run.log" "${OUT%.nc}.log"; rm -rf "$W"
echo "RUN_OK $OUT"
