#!/bin/bash
# Build the standalone CAM6 orographic gravity-wave-drag oracle.  The CAM
# sources are compiled UNMODIFIED from a checkout of ESCOMP/CAM tag
# cam_cesm2_1_rel_60 (SHA a03b84b7c4e34f965b115686f22a043b85739e56);
# stubs.F90 replaces only kind/log/abort/logfile modules and physconst
# (unused by lininterp).  Usage: build.sh <CAM checkout> <build dir>
set -euo pipefail
CAM=${1:?CAM cam_cesm2_1_rel_60 checkout}; B=${2:?build dir}
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=a03b84b7c4e34f965b115686f22a043b85739e56
test "$(git -C "$CAM" rev-parse HEAD)" = "$SHA" || { echo "CAM not at $SHA" >&2; exit 1; }
test -z "$(git -C "$CAM" status --porcelain)" || { echo "CAM checkout not clean" >&2; exit 1; }
mkdir -p "$B"; cd "$B"
cp -f "$CAM"/src/utils/{coords_1d,linear_1d_operators,interpolate_data}.F90 .
cp -f "$CAM"/src/physics/cam/{gw_utils,vdiff_lu_solver,gw_diffusion,gw_common,gw_oro}.F90 .
cp -f "$HERE/stubs.F90" "$HERE/driver.F90" .
FF="-O0 -fno-fast-math -ffpe-trap=invalid,zero,overflow -fcheck=bounds -cpp -DUSE_CONTIGUOUS= -ffree-line-length-none"
for f in stubs coords_1d gw_utils linear_1d_operators interpolate_data vdiff_lu_solver \
         gw_diffusion gw_common gw_oro driver; do
  gfortran $FF -c $f.F90
done
gfortran $FF -o gw_oro_oracle stubs.o coords_1d.o gw_utils.o linear_1d_operators.o \
  interpolate_data.o vdiff_lu_solver.o gw_diffusion.o gw_common.o gw_oro.o driver.o
md5sum gw_oro.F90 gw_common.F90 gw_utils.F90 gw_diffusion.F90 vdiff_lu_solver.F90 \
  coords_1d.F90 linear_1d_operators.F90 interpolate_data.F90
echo BUILD_OK
