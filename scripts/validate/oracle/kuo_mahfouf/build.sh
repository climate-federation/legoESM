#!/bin/bash
# Build the Mahfouf MOIST_CONVECTION_KUO oracle driver (Kuo-1965 + Kuo-Anthes).
# usage: build.sh <MOIST_CONVECTION_KUO checkout at 79b78f58> <build dir>
# (called by make_fixture.py)
#
# The oracle is a stand-alone program (src/kuo_schemes.f90) that reads one
# ECMWF profile from data_in/.  driver_vs_oracle.diff turns it into a driver
# with I/O-only edits (columns from a text file, nlev and klev_max at runtime,
# outputs in SI units, Geleyn/symmetric variants dropped); the Kuo-1965 and
# Kuo-Anthes algorithm lines are the oracle's, verbatim.  -DCONST_LEGO selects
# legoESM constants, and thermo_lego.f90 (same function names as the oracle's
# src/thermo_functions3.f90) is legoESM's saturation curve, transcribed from
# packages/core/legoesm/thermo.py.  gfortran 8.5.0, fp64, -O0, was used for the
# committed fixture (tests/atmosphere/oracle_fixtures/kuo_mahfouf.npz).
set -euo pipefail
ORA=$1
B=$2
SHA=79b78f58f793fbf7849bca51d8ec0a92f39ec170
HERE=$(cd "$(dirname "$0")" && pwd)
test "$(git -C "$ORA" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$ORA" status --porcelain -- src)"   # unmodified oracle source
mkdir -p "$B"
cd "$B"
patch -s -o kuo_driver.F90 "$ORA/src/kuo_schemes.f90" "$HERE/driver_vs_oracle.diff"
FF="-O0 -cpp -fdefault-real-8 -fdefault-double-8 -ffree-line-length-0 -fcheck=all -ffpe-trap=zero,overflow,invalid -finit-real=nan -J$B"
gfortran $FF -c "$HERE/thermo_lego.f90" -o thermo_lego.o
gfortran $FF -DCONST_LEGO -o kuo_harm kuo_driver.F90 thermo_lego.o
echo BUILD_OK
