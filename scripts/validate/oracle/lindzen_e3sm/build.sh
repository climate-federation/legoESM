#!/bin/bash
# Build the standalone E3SM v3.0.1 orographic-only gravity-wave-drag oracle
# (the "lindzen" scheme).  The E3SM sources are compiled UNMODIFIED from a
# checkout of E3SM-Project/E3SM tag v3.0.1 (SHA
# d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84); no stubs are needed.  driver.F90
# is the tier-1/tier-2 audit driver (gw_tend orographic_only branch, discrete
# kinetic-energy heating).  Usage: build.sh <E3SM checkout> <build dir>
set -euo pipefail
E3SM=${1:?E3SM v3.0.1 checkout}; B=${2:?build dir}
HERE=$(cd "$(dirname "$0")" && pwd)
CAMD="$E3SM/components/eam/src/physics/cam"
SHA=d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84
[ "$(git -C "$E3SM" rev-parse HEAD)" = "$SHA" ] || { echo "E3SM checkout is not $SHA"; exit 1; }
git -C "$E3SM" diff --quiet HEAD -- components/eam/src/physics/cam || { echo "E3SM sources modified"; exit 1; }
mkdir -p "$B"; cd "$B"
cp -f "$CAMD"/{gw_utils,vdiff_lu_solver,gw_diffusion,gw_common,gw_oro}.F90 .
cp -f "$HERE/driver.F90" .
FF="-O0 -fno-fast-math -ffpe-trap=invalid,zero,overflow -fcheck=bounds -cpp -ffree-line-length-none"
for f in gw_utils vdiff_lu_solver gw_diffusion gw_common gw_oro driver; do
  gfortran $FF -c $f.F90
done
gfortran $FF -o gw_oro_e3sm gw_utils.o vdiff_lu_solver.o gw_diffusion.o gw_common.o gw_oro.o driver.o
md5sum gw_utils.F90 vdiff_lu_solver.F90 gw_diffusion.F90 gw_common.F90 gw_oro.F90
gfortran --version | head -1
echo BUILD_OK
