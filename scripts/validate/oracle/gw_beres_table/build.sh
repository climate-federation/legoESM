#!/bin/bash
# Build the E3SM v3.0.1 gw oracle for the REAL Beres table: first the unchanged
# text-table build (../gw_spec_e3sm/build.sh), then gw_spec_oracle_nc = the same
# driver.F90 compiled with -DMFCC_FILE plus E3SM's own gw_init_beres netcdf reader
# (gw_drag.F90:513-571 and the maxh/maxuh parameters :102,:104, extracted VERBATIM),
# linked to netcdf-fortran (nf-config must be on PATH, e.g.
# `module load netcdf-c/4.9.3 netcdf-fortran/4.6.2` on Ginsburg).
# Usage: build.sh <E3SM checkout> <build dir> [interpolate_data.F90]
set -euo pipefail
E3SM=${1:?E3SM v3.0.1 checkout}; B=${2:?build dir}
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/../gw_spec_e3sm/build.sh" "$@"
CAMD="$E3SM/components/eam/src/physics/cam"
cd "$B"
FF="-O0 -fno-fast-math -ffpe-trap=invalid,zero,overflow -fcheck=bounds -cpp -ffree-line-length-none -I."  # = ../gw_spec_e3sm/build.sh
grep -qF -- "FF=\"$FF\"" "$HERE/../gw_spec_e3sm/build.sh"
sed -n 513,571p "$CAMD/gw_drag.F90" > gw_init_beres.inc
sed -n '102p;104p' "$CAMD/gw_drag.F90" > gw_beres_dims.inc
grep -q "NF90_GET_VAR(ncid,varid,mfcc,start=\[1,1,ngwv_file-pgwv+1\])" gw_init_beres.inc
grep -q "maxh = 20" gw_beres_dims.inc && grep -q "maxuh = 40" gw_beres_dims.inc
cp -f "$HERE/beres_table.F90" .
NFF=$(nf-config --fflags); NFL=$(nf-config --flibs)
gfortran $FF $NFF -c beres_table.F90
gfortran $FF -DMFCC_FILE -c driver.F90 -o driver_nc.o
gfortran $FF -o gw_spec_oracle_nc stubs.o interpolate_data.o gw_utils.o vdiff_lu_solver.o \
  gw_diffusion.o gw_common.o gw_front.o gw_convect.o gw_oro.o beres_table.o driver_nc.o $NFL
nf-config --version
md5sum gw_init_beres.inc gw_beres_dims.inc
echo BUILD_NC_OK
