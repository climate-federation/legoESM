#!/bin/bash
# Build the sfclayrev oracle driver against UNMODIFIED WRF v4.8.0 sources
# (06d4240ae989cc3e50af412bb472df3d9048783c; phys/physics_mmm submodule
# 550b5b494758ad2756394e93cb08b5709b8feb83): phys/ccpp_kind_types.F,
# phys/physics_mmm/sf_sfclayrev.F90 and phys/module_sf_sfclayrev.F (cpp with
# EM_CORE=1, as the ARW build: the optional isftcflx/iz0tlnd/scm_force_flux/
# ustm/ck/cka/cd/cda arguments are passed).  gfortran, fp64 with
# -fdefault-real-8 (WRF RWORDSIZE=8), FPE traps.  Used for the committed
# fixture tests/atmosphere/oracle_fixtures/sfclayrev.npz.
# usage: build.sh <WRF checkout> <build dir>
set -euo pipefail
PHYS=$1/phys
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O2 -fcheck=bounds -ffpe-trap=invalid,zero,overflow -ffree-line-length-none -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
gfortran --version | head -1
cp "$PHYS/ccpp_kind_types.F" ccpp_kind_types.F90
cp "$PHYS/physics_mmm/sf_sfclayrev.F90" sf_sfclayrev.F90
cp "$PHYS/module_sf_sfclayrev.F" module_sf_sfclayrev.F90
gfortran $FF -c ccpp_kind_types.F90
gfortran $FF -c sf_sfclayrev.F90
gfortran $FF -cpp -DEM_CORE=1 -c module_sf_sfclayrev.F90
gfortran $FF -c "$HERE/sfclayrev_driver.F90"
gfortran $FF -o sfclayrev_driver ccpp_kind_types.o sf_sfclayrev.o module_sf_sfclayrev.o sfclayrev_driver.o
sha256sum "$PHYS/physics_mmm/sf_sfclayrev.F90" "$PHYS/module_sf_sfclayrev.F" "$PHYS/ccpp_kind_types.F"
echo BUILD_OK
