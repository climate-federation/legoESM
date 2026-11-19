#!/bin/bash
# Build the MYJ PBL oracle from the UNMODIFIED WRF v4.8.0 source
# phys/module_bl_myjpbl.F (WRF 06d4240ae989cc3e50af412bb472df3d9048783c).
# The host module module_model_constants (WRF share/) is replaced by
# <constants.F90>, written by make_fixture.py: physical constants from
# legoesm.constants, the MYJ scheme constants (EPSQ2, PQ0, A2-A4) at WRF's values.
# gfortran, fp64 (-fdefault-real-8 -fdefault-double-8), -O0, FPE traps.
# usage: build.sh <WRF checkout> <build dir> <constants.F90>
set -euo pipefail
WRF=$(realpath "$1")
mkdir -p "$2"
B=$(realpath "$2")
CONST=$(realpath "$3")
HERE=$(cd "$(dirname "$0")" && pwd)
FF="-O0 -cpp -ffree-form -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -ffpe-trap=invalid,zero,overflow -fbacktrace"
gfortran --version | head -1
sha256sum "$WRF/phys/module_bl_myjpbl.F" "$CONST"
cd "$B"
cp "$CONST" module_model_constants.F90
cp "$WRF/phys/module_bl_myjpbl.F" module_bl_myjpbl.F
gfortran $FF -c module_model_constants.F90
gfortran $FF -c module_bl_myjpbl.F
gfortran $FF -o myj_driver "$HERE/myj_driver.F90" module_model_constants.o module_bl_myjpbl.o
echo BUILD_OK
