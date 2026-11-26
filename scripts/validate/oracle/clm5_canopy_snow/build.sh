#!/bin/bash
# Build the CLM5.0 canopy-snow oracle (snowveg_flag ON_RAD) from UNMODIFIED
# CTSM release-clm5.0.34 sources: the line ranges below are cut out of the
# shipped files with sed and #include'd verbatim by canopy_snow_driver.F90.
#   CanopyHydrologyMod.F90  294-304  per-step flux reset
#                           309-392  snow interception + below-freezing fall-off
#                           417-443  temperature/wind unloading, ground snow
#                           717-742  FracWet: fwet, fcansno
#   CanopyFluxesMod.F90    1257-1272 wet-canopy evaporation from snocan/liqcan
# usage: build.sh <CTSM checkout> <build dir>
set -euo pipefail
SRC=$1/src/biogeophys
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
sed -n '294,304p'   "$SRC/CanopyHydrologyMod.F90" > "$B/ch_reset.inc"
sed -n '309,392p'   "$SRC/CanopyHydrologyMod.F90" > "$B/ch_intercept.inc"
sed -n '417,443p'   "$SRC/CanopyHydrologyMod.F90" > "$B/ch_unload.inc"
sed -n '717,742p'   "$SRC/CanopyHydrologyMod.F90" > "$B/ch_fracwet.inc"
sed -n '1257,1272p' "$SRC/CanopyFluxesMod.F90"    > "$B/cf_evap.inc"
cd "$B"
gfortran -O0 -ffree-line-length-none -fno-fast-math -cpp -I"$B" \
  -o canopy_snow_driver "$HERE/canopy_snow_driver.F90"
echo BUILD_OK
