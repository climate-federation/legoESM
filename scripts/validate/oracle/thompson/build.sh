#!/bin/bash
# Build the WRF v4.8.0 Thompson oracle (phys/module_mp_thompson.F and
# phys/module_mp_radar.F compiled UNMODIFIED) plus the write-only dump copy.
# usage: build.sh <WRF checkout (v4.8.0)> <build dir>
# fp64 everywhere (-fdefault-real-8 -fdefault-double-8), as the tier-2 audit
# ran its oracle; gfortran 8.5.0 for the committed fixture.  The free-form
# .F sources go through cpp with neither DM_PARALLEL nor WRF_CHEM defined.
set -euo pipefail
WRF=$(realpath "$1")
mkdir -p "$2"
B=$(realpath "$2")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B/orig" "$B/dump"
FF="-O2 -cpp -ffree-form -ffree-line-length-none -fdefault-real-8 -fdefault-double-8"
python3 "$HERE/patch_dump.py" "$WRF/phys/module_mp_thompson.F" "$B/dump/module_mp_thompson_dump.F"
for v in orig dump; do
  cd "$B/$v"
  gfortran $FF -c "$HERE/wrf_stubs.F90" "$HERE/thom_dump.F90"
  gfortran $FF -c "$WRF/phys/module_mp_radar.F"
  if [ $v = orig ]; then
    gfortran $FF -c "$WRF/phys/module_mp_thompson.F"
    gfortran $FF -o thompson_driver "$HERE/thompson_driver.F90" *.o
  else
    gfortran $FF -c module_mp_thompson_dump.F
    gfortran $FF -DTHOM_DUMP -o thompson_driver "$HERE/thompson_driver.F90" *.o
  fi
done
echo BUILD_OK
