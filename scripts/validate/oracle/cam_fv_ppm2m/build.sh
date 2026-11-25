#!/bin/sh
# Build the CAM-FV map1_ppm oracle from the UNMODIFIED mapz_module.F90 of
# cam_cesm2_1_rel_60 (src/dynamics/fv/mapz_module.F90).
# usage: build.sh <path to mapz_module.F90> <build dir>
set -eu
SRC=$1
OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cp -f "$SRC" "$OUT/mapz_module.F90"
cd "$OUT"
gfortran -O0 -ffree-line-length-none -cpp \
  "$HERE/stubs.F90" mapz_module.F90 "$HERE/driver.F90" -o map1_ppm_oracle
echo "BUILD_OK $OUT/map1_ppm_oracle"
