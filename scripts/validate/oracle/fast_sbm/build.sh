#!/bin/bash
# Build the WRF v4.8.0 FAST_SBM oracle driver, every real promoted to double
# (the WRF default build mixes real(4)/real(8); the oracle test compares in fp64).
# Usage: build.sh <WRF checkout @ 06d4240ae989cc3e50af412bb472df3d9048783c> <build dir>
set -euo pipefail
WRF=$1; B=$2; H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
bash "$H/extract.sh" "$WRF" "$B"
cp -f "$H/driver.F90" "$H/stubs.F90" "$B/"
cd "$B"
sed -i 's/R4SIZE = 4/R4SIZE = 8/' sbm_collision.F90 sbm_auxiliary.F90 sbm_extract.F90
sed -i 's/real(4)/real(8)/g' sbm_auxiliary.F90
FLAGS="-O0 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -fno-range-check"
for f in stubs.F90 sbm_collision.F90 sbm_auxiliary.F90 sbm_extract.F90 driver.F90; do
  gfortran $FLAGS -c "$f"
done
gfortran $FLAGS -o sbm_driver *.o
printf 'POLYSVP\n290.0\n' | ./sbm_driver
echo "BUILD_OK $B/sbm_driver"
