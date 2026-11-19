#!/bin/bash
# Build the simple-LW oracle driver against the UNMODIFIED gSAM 1.8.8
# SRC/rad_simple.f90, extracted from the release tarball
# http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz
# (sha256 0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c
# for the copy used for tests/atmosphere/oracle_fixtures/simple_lw_gsam.npz).
# usage: build.sh <gsam1.8.8.tar.gz> <build dir>
# gfortran 8.5.0; -fdefault-real-8 makes gSAM's default-real fp64.
set -euo pipefail
TGZ=$1
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
tar -xzOf "$TGZ" gSAM1.8.8/SRC/rad_simple.f90 > "$B/rad_simple.f90"
cd "$B"
FF="-O0 -g -fno-fast-math -fdefault-real-8 -fdefault-double-8 -ffpe-trap=invalid,zero,overflow -ffree-line-length-none"
gfortran $FF -c "$HERE/stubs_gsam.F90"
gfortran $FF -c rad_simple.f90
gfortran $FF -c "$HERE/drv_simple.F90"
gfortran $FF -o drv_simple stubs_gsam.o rad_simple.o drv_simple.o
echo BUILD_OK
