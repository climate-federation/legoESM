#!/bin/bash
# Build the Isca v1.0 hs_forcing oracle used for
# tests/atmosphere/oracle_fixtures/rayleigh_hs_isca.npz.
#
# usage: build.sh <Isca checkout @ v1.0 5868cfd863b41823a3eeb30754ce06feba7528ec> <build dir>
# src/atmos_param/hs_forcing/hs_forcing.F90 is compiled UNMODIFIED (copied, sha256
# printed) against the minimal FMS stubs in stubs_hs.F90.  fp64 via -fdefault-real-8
# (Isca ships -r8).  Run on a compute node.
set -euo pipefail
I=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "isca $(git -C "$I" rev-parse HEAD)"
FC=${FC:-gfortran}
FFLAGS="-O0 -g -fno-fast-math -fdefault-real-8 -fdefault-double-8 -ffpe-trap=invalid,zero,overflow -ffree-line-length-none -cpp"
$FC --version | head -1
cd "$B"
cp "$I/src/atmos_param/hs_forcing/hs_forcing.F90" hs_forcing.F90
sha256sum hs_forcing.F90
cmp hs_forcing.F90 "$I/src/atmos_param/hs_forcing/hs_forcing.F90"
$FC $FFLAGS -c "$HERE/stubs_hs.F90"
$FC $FFLAGS -c hs_forcing.F90
$FC $FFLAGS -c "$HERE/drv_hs.F90"
$FC $FFLAGS -o drv_hs stubs_hs.o hs_forcing.o drv_hs.o
echo "BUILD_OK $B/drv_hs"
