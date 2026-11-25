#!/bin/bash
# Build the CAM6 conv_water_4rad oracle behind
# tests/atmosphere/oracle_fixtures/cam6_conv_water.npz.
#
# usage: build.sh <conv_water.F90 from CAM cam_cesm2_1_rel_60> <build dir> <nlev>
# conv_water.F90 (src/physics/cam/conv_water.F90) is compiled UNMODIFIED in
# fp64 with the stand-in modules of stubs.F90; nothing is run here.
set -euo pipefail
SRC=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); mkdir -p "$2"; B=$(cd "$2" && pwd); NLEV=$3
HERE=$(cd "$(dirname "$0")" && pwd)
echo "conv_water.F90 sha256 $(shasum -a 256 "$SRC" | cut -d' ' -f1)"
grep -q 'cu_icwmr = ( sh0_frac \* sh_icwmr(i,k) + dp0_frac\*dp_icwmr(i,k))/max(frac_limit,cu0_frac)' "$SRC"
grep -q 'totg_ice(i,k) = tot0_frac \* tot_icwmr \* wrk1' "$SRC"
FC=${FC:-gfortran}
cp "$SRC" "$B/conv_water.F90"
cp "$HERE/stubs.F90" "$HERE/oracle_driver.F90" "$B/"
cd "$B"
$FC -O0 -ffree-line-length-none -fno-fast-math -DPVER_VAL="$NLEV" -cpp -c stubs.F90
$FC -O0 -ffree-line-length-none -fno-fast-math -cpp -c conv_water.F90
$FC -O0 -ffree-line-length-none -fno-fast-math -c oracle_driver.F90
$FC -o cw_oracle stubs.o conv_water.o oracle_driver.o
echo "built $B/cw_oracle (pver=$NLEV)"
