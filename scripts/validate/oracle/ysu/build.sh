#!/bin/bash
# Build the YSU oracle driver against UNMODIFIED WRF v4.8.0 sources
# (06d4240ae989cc3e50af412bb472df3d9048783c; phys/physics_mmm submodule
# 550b5b494758ad2756394e93cb08b5709b8feb83): phys/ccpp_kind_types.F and
# phys/physics_mmm/bl_ysu.F90.  gfortran 8.5.0, fp64 with -fdefault-real-8
# (WRF RWORDSIZE=8), FPE traps -- the tier-2 audit's primary build; used for
# the committed fixture tests/atmosphere/oracle_fixtures/ysu.npz.
# usage: build.sh <WRF checkout> <build dir>
set -euo pipefail
PHYS=$1/phys
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O2 -fcheck=bounds -ffpe-trap=invalid,zero,overflow -ffree-line-length-none -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
gfortran --version | head -1
cp "$PHYS/ccpp_kind_types.F" ccpp_kind_types.F90   # free-form after cpp, as WRF builds it
cp "$PHYS/physics_mmm/bl_ysu.F90" bl_ysu.F90
gfortran $FF -c ccpp_kind_types.F90
gfortran $FF -c bl_ysu.F90
gfortran $FF -c "$HERE/ysu_driver.F90"
gfortran $FF -o ysu_driver ccpp_kind_types.o bl_ysu.o ysu_driver.o
sha256sum "$PHYS/physics_mmm/bl_ysu.F90" "$PHYS/ccpp_kind_types.F"
echo BUILD_OK
