#!/bin/bash
# Build the MYNN-EDMF oracle from UNMODIFIED WRF v4.8.0 sources
# (WRF 06d4240ae989cc3e50af412bb472df3d9048783c, phys/MYNN-EDMF submodule
# 8f8e8b59021249949c04d1686edc5a02ca325627):
#   phys/ccpp_kind_types.F, phys/MYNN-EDMF/WRF/module_bl_mynnedmf_common.F90,
#   phys/MYNN-EDMF/module_bl_mynnedmf.F90.
# The host module module_model_constants (WRF share/) is replaced by
# <constants.F90>, written by make_fixture.py from legoesm.constants: the
# MYNN-EDMF common module takes its physical constants from the host model.
# Two binaries: orig/ (sources as shipped) and dump/ (patch_dump.py: the same
# source plus write-only dumps of intermediate fields).
# gfortran 8.5.0, fp64 (-fdefault-real-8 -fdefault-double-8 -DDOUBLE_PRECISION,
# as the MYNN-EDMF CI Makefile builds it), FPE traps, and the CI Makefile's
# -finit-integer=-999: DMP_mf reads its local k50 (highest level below 50 m)
# unset when no mass level is below 50 m AGL (every L32 column); the CI
# initialisation makes that path deterministic (only level 2 is checked).
# usage: build.sh <WRF checkout> <build dir> <constants.F90>
set -euo pipefail
WRF=$(realpath "$1")
mkdir -p "$2"
B=$(realpath "$2")
CONST=$(realpath "$3")
HERE=$(cd "$(dirname "$0")" && pwd)
M=$WRF/phys/MYNN-EDMF
FF="-O0 -cpp -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -DDOUBLE_PRECISION -ffpe-trap=invalid,zero,overflow -finit-integer=-999 -fbacktrace"
gfortran --version | head -1
sha256sum "$WRF/phys/ccpp_kind_types.F" "$M/WRF/module_bl_mynnedmf_common.F90" "$M/module_bl_mynnedmf.F90" "$CONST"
mkdir -p "$B/orig" "$B/dump"
python3 "$HERE/patch_dump.py" "$M/module_bl_mynnedmf.F90" "$B/dump/module_bl_mynnedmf.F90"
for v in orig dump; do
  cd "$B/$v"
  cp "$WRF/phys/ccpp_kind_types.F" ccpp_kind_types.F90
  cp "$CONST" module_model_constants.F90
  cp "$M/WRF/module_bl_mynnedmf_common.F90" .
  [ $v = orig ] && cp "$M/module_bl_mynnedmf.F90" .
  gfortran $FF -c ccpp_kind_types.F90 module_model_constants.F90 module_bl_mynnedmf_common.F90
  if [ $v = dump ]; then
    gfortran $FF -c "$HERE/mynn_dump_mod.F90"
    gfortran $FF -c module_bl_mynnedmf.F90
    gfortran $FF -DMYNN_DUMP -o mynn_driver "$HERE/mynn_driver.F90" *.o
  else
    gfortran $FF -c module_bl_mynnedmf.F90
    gfortran $FF -o mynn_driver "$HERE/mynn_driver.F90" *.o
  fi
done
echo BUILD_OK
