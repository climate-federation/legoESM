#!/bin/bash
# Build the LMDZ r5891 hines_gwd oracle used for
# tests/atmosphere/oracle_fixtures/hines_lmdz.npz.
#
# usage: build.sh <LMDZ libf/phylmd dir (r5891 tarball 20251128)> <build dir>
# hines_gwd.f90 and yoegwd_mod_h.f90 are compiled UNMODIFIED (copied, sha256
# printed) against the dimphy / yomcst_mod_h stand-ins in stubs_hines.F90.
# fp64 via -fdefault-real-8 (LMDZ's gfortran build is -r8).  Run on a compute node.
set -euo pipefail
L=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
FC=${FC:-gfortran}
FFLAGS="-O0 -g -fno-fast-math -fdefault-real-8 -fdefault-double-8 -ffpe-trap=invalid,zero,overflow -ffree-line-length-none -cpp"
$FC --version | head -1
cd "$B"
for f in hines_gwd.f90 yoegwd_mod_h.f90; do
  cp "$L/$f" "$f"; sha256sum "$f"; cmp "$f" "$L/$f"
done
$FC $FFLAGS -c "$HERE/stubs_hines.F90"
$FC $FFLAGS -c yoegwd_mod_h.f90
$FC $FFLAGS -c hines_gwd.f90
$FC $FFLAGS -c "$HERE/drv_hines.F90"
$FC $FFLAGS -o drv_hines stubs_hines.o yoegwd_mod_h.o hines_gwd.o drv_hines.o
echo "BUILD_OK $B/drv_hines"
