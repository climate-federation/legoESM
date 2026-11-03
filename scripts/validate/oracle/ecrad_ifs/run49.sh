#!/bin/bash
# Run an ecRad oracle executable on one input file with the 49r1 namelist.
# usage: run49.sh <exe> <namelist> <data dir> <input.nc> <output.nc> [key=value ...]
#   key=value pairs override &radiation entries (harness stages, e.g. use_aerosols=false)
set -euo pipefail
EXE=$(readlink -f "$1"); NAM=$2; DATA=$(readlink -f "$3"); IN=$(readlink -f "$4"); OUT=$(readlink -m "$5"); shift 5
W=$(mktemp -d "${TMPDIR:-/tmp}/ecrad49run.XXXX")
sed "s#\"DATA_DIR\"#\"$DATA\"#" "$NAM" > "$W/config.nam"
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  # array entries are set by slices elsewhere in the file; only scalar keys may be overridden
  grep -qE "^$k *\(" "$W/config.nam" && { echo "override $k: array key, not supported"; exit 1; }
  if grep -qE "^$k *=" "$W/config.nam"; then sed -i -E "s#^($k *= *)[^,]*,#\1$v,#" "$W/config.nam"
  else sed -i -E "0,/^&radiation\$/s##\&radiation\n$k = $v,#" "$W/config.nam"; fi
  grep -qE "^$k *= *$v," "$W/config.nam" || { echo "override $kv not applied"; exit 1; }
done
cp "$W/config.nam" "${OUT%.nc}.nam"
{ sha256sum "$EXE"; echo "OMP_NUM_THREADS=1"; echo "overrides: $*"
  ldd "$EXE" | awk '$3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' | xargs -r sha256sum --; } > "${OUT%.nc}.run"
( cd "$W" && OMP_NUM_THREADS=1 "$EXE" config.nam "$IN" "$W/out.nc" > "$W/run.log" 2>&1 ) || { tail -20 "$W/run.log"; exit 1; }
mv "$W/out.nc" "$OUT"
for f in "$W"/radiative_properties*.nc; do [ -e "$f" ] && mv "$f" "${OUT%.nc}_$(basename "$f")"; done
cp "$W/run.log" "${OUT%.nc}.log"; rm -rf "$W"
# chain of custody: every file the oracle opened, with its sha256
sed -nE 's/^Reading (RRTMG (long|short)wave data|NetCDF) file (.*)$/\3/p' "${OUT%.nc}.log" | sort -u | tr "\n" "\0" | xargs -0 sha256sum -- > "${OUT%.nc}.opened"
grep -q RADRRTM "${OUT%.nc}.opened" || { echo "opened-file log missing RADRRTM"; exit 1; }
echo "RUN_OK $OUT"
