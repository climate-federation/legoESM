#!/bin/bash
# Build the standalone E3SM v3.0.1 spectral gravity-wave-drag oracle.  The E3SM
# sources are compiled UNMODIFIED from a checkout of E3SM-Project/E3SM tag
# v3.0.1 (SHA d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84); the two include files
# are extracted VERBATIM from that checkout's gw_drag.F90 (:242-281 the
# alpha0/palph Newtonian table, :901-938 the orographic heating closure, both
# use_gw_energy_fix branches).  stubs.F90 replaces only shr_kind_mod,
# cam_logfile and cam_abortutils.
# Usage: build.sh <E3SM checkout> <build dir> [interpolate_data.F90]
# (the third argument is for sparse checkouts without components/eam/src/control;
# pass that file as fetched raw at the same SHA, md5 a4df0dd42c0ca589040fc5c366f9a507).
set -euo pipefail
E3SM=${1:?E3SM v3.0.1 checkout}; B=${2:?build dir}
INTERP=${3:-$E3SM/components/eam/src/control/interpolate_data.F90}
HERE=$(cd "$(dirname "$0")" && pwd)
CAMD="$E3SM/components/eam/src/physics/cam"
mkdir -p "$B"; cd "$B"
cp -f "$CAMD"/{gw_utils,gw_common,gw_diffusion,vdiff_lu_solver,gw_front,gw_convect,gw_oro}.F90 .
cp -f "$INTERP" interpolate_data.F90
sed -n 242,281p "$CAMD/gw_drag.F90" > alpha_table.inc
sed -n 901,938p "$CAMD/gw_drag.F90" > oro_closure.inc
grep -q "alpha0(nalph)" alpha_table.inc && grep -q "use_gw_energy_fix" oro_closure.inc
cp -f "$HERE/stubs.F90" "$HERE/driver.F90" .
FF="-O0 -fno-fast-math -ffpe-trap=invalid,zero,overflow -fcheck=bounds -cpp -ffree-line-length-none -I."
for f in stubs interpolate_data gw_utils vdiff_lu_solver gw_diffusion gw_common gw_front \
         gw_convect gw_oro driver; do
  gfortran $FF -c $f.F90
done
gfortran $FF -o gw_spec_oracle stubs.o interpolate_data.o gw_utils.o vdiff_lu_solver.o \
  gw_diffusion.o gw_common.o gw_front.o gw_convect.o gw_oro.o driver.o
md5sum gw_utils.F90 gw_common.F90 gw_diffusion.F90 vdiff_lu_solver.F90 gw_front.F90 \
  gw_convect.F90 gw_oro.F90 interpolate_data.F90 alpha_table.inc oro_closure.inc
gfortran --version | head -1
echo BUILD_OK
