#!/bin/bash
# Build the gSAM 1.8.8 tracesini oracle used to generate
# tests/atmosphere/oracle_fixtures/ozone_mls.npz.
#
# usage: build.sh <gSAM1.8.8 source dir> <conda env with netcdf-fortran> <build dir>
# tracesini is extracted VERBATIM (rad.f90 lines 1219-1431) and #included; parkind.f90 and
# rrlw_ncpar.f90 are gSAM's own.  Run on a compute node (sbatch).
set -euo pipefail
G=$(cd "$1" && pwd); E=$(cd "$2" && pwd); mkdir -p "$3"; B=$(cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
R=$G/SRC/RAD_RRTM
sha256sum "$R/rad.f90" "$G/RUNDATA/rrtmg_lw.nc"
export FC=${FC:-gfortran}
$FC --version | head -1
cd "$B"
sed -n 1219,1431p "$R/rad.f90" > tracesini.inc
head -1 tracesini.inc; tail -1 tracesini.inc
F="$FC -O2 -fno-fast-math -ffree-line-length-none"
# parkind/rrlw_ncpar keep gSAM's explicit kinds; the stub module gets gSAM's reals in fp64
$F -c "$R/parkind.f90" "$R/rrlw_ncpar.f90"
$F -cpp -fdefault-real-8 -fdefault-double-8 -I"$B" -I"$E/include" -c "$HERE/oracle_driver.F90"
$FC -o "$B/oracle_driver" ./*.o -L"$E/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$E/lib"
echo "BUILD_OK $B/oracle_driver"
