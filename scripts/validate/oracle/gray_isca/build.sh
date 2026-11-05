#!/bin/bash
# Build the Isca v1.0 two_stream_gray_rad oracle used for
# tests/atmosphere/oracle_fixtures/gray_isca.npz.
#
# usage: build.sh <Isca checkout @ v1.0 5868cfd863b41823a3eeb30754ce06feba7528ec> <build dir>
# The oracle source is COPIED, never edited in place.  Build-copy edits (diff printed):
#   1. a `public ::` line exposing the module flux arrays (visibility only);
#   2. a memory-only `allocate(lw_dtrans_win)`: v1.0 writes `lw_dtrans_win = 1.`
#      (unread by B_FRIERSON) to an unallocated array; Isca master allocates it.
# fp64 via -fdefault-real-8 (Isca ships -r8).  Run on a compute node.
set -euo pipefail
I=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "isca $(git -C "$I" rev-parse HEAD)"
FC=${FC:-gfortran}
FFLAGS="-O0 -g -fno-fast-math -fdefault-real-8 -fdefault-double-8 -ffpe-trap=invalid,zero,overflow -ffree-line-length-none"
$FC --version | head -1
cd "$B"
cp "$I/src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90" orig.F90
sha256sum orig.F90
sed '/^public :: two_stream_gray_rad_init/a public :: lw_up, lw_down, sw_up, sw_down, tdt_rad, tdt_solar, insolation, olr, net_lw_surf   ! HARNESS: visibility only' orig.F90 \
  | sed '/^allocate (fracsun/a allocate (lw_dtrans_win(ie-is+1, je-js+1, num_levels))   ! HARNESS: memory-only (v1.0 writes it unallocated; master l.323 allocates)' \
  > two_stream_gray_rad.F90
diff orig.F90 two_stream_gray_rad.F90 || true
[ "$(diff orig.F90 two_stream_gray_rad.F90 | grep -c '^>')" = 2 ]
$FC $FFLAGS -c "$HERE/stubs_gray.F90"
$FC $FFLAGS -c two_stream_gray_rad.F90
$FC $FFLAGS -c "$HERE/drv_gray.F90"
$FC $FFLAGS -o drv_gray stubs_gray.o two_stream_gray_rad.o drv_gray.o
echo "BUILD_OK $B/drv_gray"
