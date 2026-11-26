#!/bin/bash
# Build the Isca v1.0 damping_driver (Rayleigh sponge) oracle used for
# tests/atmosphere/oracle_fixtures/rayleigh_sponge_isca.npz.
#
# usage: build.sh <Isca checkout @ v1.0 5868cfd863b41823a3eeb30754ce06feba7528ec> <build dir>
# src/atmos_param/damping_driver/damping_driver.f90 is compiled UNMODIFIED (copied,
# sha256 printed) against the minimal FMS stubs in stubs_damp.F90.  fp64 via
# -fdefault-real-8 (Isca ships -r8).  Run on a compute node.
set -euo pipefail
I=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "isca $(git -C "$I" rev-parse HEAD)"
FC=${FC:-gfortran}
FFLAGS="-O0 -g -fno-fast-math -fdefault-real-8 -fdefault-double-8 -ffpe-trap=invalid,zero,overflow -ffree-line-length-none -cpp"
$FC --version | head -1
cd "$B"
cp "$I/src/atmos_param/damping_driver/damping_driver.f90" damping_driver.f90
sha256sum damping_driver.f90
cmp damping_driver.f90 "$I/src/atmos_param/damping_driver/damping_driver.f90"
$FC $FFLAGS -c "$HERE/stubs_damp.F90"
$FC $FFLAGS -c damping_driver.f90
$FC $FFLAGS -c "$HERE/drv_damp.F90"
$FC $FFLAGS -o drv_damp stubs_damp.o damping_driver.o drv_damp.o
echo "BUILD_OK $B/drv_damp"
