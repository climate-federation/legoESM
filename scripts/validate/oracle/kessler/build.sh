#!/bin/bash
# Build the Kessler oracle driver against the UNMODIFIED ESCOMP
# atmospheric_physics kessler.F90 at tag atmos_phys0_31_000.
# usage: build.sh <atmospheric_physics checkout> <build dir>
# The source is taken from the tag itself (git show), so the checkout's HEAD
# does not matter.  gfortran 8.5.0, fp64, was used for the committed fixture
# (tests/atmosphere/oracle_fixtures/kessler.npz).
set -euo pipefail
AP=$1
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
git -C "$AP" show atmos_phys0_31_000:schemes/kessler/kessler.F90 > "$B/kessler.F90"
cd "$B"
FF="-O0 -g -fno-fast-math -ffree-line-length-none -fbacktrace -ffpe-trap=invalid,zero,overflow"
gfortran $FF -c "$HERE/ccpp_kinds_stub.F90" kessler.F90
gfortran $FF -o kessler_driver "$HERE/kessler_driver.F90" ccpp_kinds_stub.o kessler.o
echo BUILD_OK
