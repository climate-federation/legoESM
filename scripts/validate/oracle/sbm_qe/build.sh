#!/bin/bash
# Build the Isca v1.0 qe_moist_convection (SIMPLE_BETTS_MILLER) oracle driver.
# usage: build.sh <Isca checkout at tag v1.0 (5868cfd8)> <build dir>
# qe_moist_convection.F90 is compiled with ONE diagnostic-only addition
# (patch_qe.py: a public wrapper exposing the scheme's own CAPE_calculation, no
# numerics changed).  FMS infrastructure is stubbed in stubs_common.F90, whose
# default constants are legoESM's and whose sat_mode 1 is legoESM's Tetens
# curve.  gfortran 8.5.0, fp64 (-fdefault-real-8), -O0, was used for the
# committed fixture (tests/atmosphere/oracle_fixtures/sbm_qe.npz).
set -euo pipefail
ISCA=$1/src/atmos_param/qe_moist_convection/qe_moist_convection.F90
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
python3 "$HERE/patch_qe.py" "$ISCA" "$B/qe_moist_convection_diag.F90"
cd "$B"
FF="-O0 -cpp -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -fno-fast-math -g -fbacktrace -J$B"
gfortran $FF -c "$HERE/stubs_common.F90" -o stubs.o
gfortran $FF -c qe_moist_convection_diag.F90 -o qe.o
gfortran $FF -c "$HERE/drv_sbm.F90" -o drv_sbm.o
gfortran -o drv_sbm.x stubs.o qe.o drv_sbm.o
echo BUILD_OK
