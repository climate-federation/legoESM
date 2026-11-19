#!/bin/bash
# Build the GFDL atmos_phys 2021.02 moist_conv (Manabe moist convective
# adjustment) oracle driver against the UNMODIFIED moist_conv.F90.
# usage: build.sh <atmos_phys checkout at tag 2021.02 (46817a12)> <build dir>
# FMS infrastructure is stubbed in ../sbm_qe/stubs_common.F90 (legoESM
# constants; sat_mode 1 = legoESM Tetens curve and legoESM q_sat with its exact
# dq_sat/dT for compute_qs).  gfortran 8.5.0, fp64, -O0, was used for the
# committed fixture (tests/atmosphere/oracle_fixtures/dca_moist_conv.npz).
set -euo pipefail
MC=$1/atmos_param/moist_conv/moist_conv.F90
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -cpp -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -fno-fast-math -g -fbacktrace -J$B"
gfortran $FF -c "$HERE/../sbm_qe/stubs_common.F90" -o stubs.o
gfortran $FF -c "$MC" -o mc.o
gfortran $FF -c "$HERE/drv_mca.F90" -o drv_mca.o
gfortran -o drv_mca.x stubs.o mc.o drv_mca.o
echo BUILD_OK
