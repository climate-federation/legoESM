#!/bin/bash
# Build the CAM6 host level-temperature oracle (rrtmg_state.F90, unmodified).
# usage: build_cam_state.sh <CAM @ cam_cesm2_1_rel_60> <cime @ cime5.6.47> <pcols> <pver> <build dir>
# Run on a compute node (sbatch), not a login node.
set -euo pipefail
CAM=$(cd "$1" && pwd); CIME=$(cd "$2" && pwd); NC=$3; NL=$4; mkdir -p "$5"; B=$(cd "$5" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "CAM $(git -C "$CAM" describe --tags) $(git -C "$CAM" rev-parse HEAD)"
echo "cime $(git -C "$CIME" describe --tags) $(git -C "$CIME" rev-parse HEAD)"
FC=${FC:-gfortran}
$FC --version | head -1
FL="-O0 -g -fno-fast-math -ffree-line-length-none -J$B -I$B"
cat > "$B/ppgrid.F90" <<EOT
module ppgrid
  implicit none
  integer, parameter :: pcols = $NC, pver = $NL, pverp = $NL + 1
end module ppgrid
EOT
cd "$B"
for f in "$CIME/src/share/util/shr_kind_mod.F90" "$CIME/src/share/util/shr_const_mod.F90" \
         ppgrid.F90 "$HERE/cam_stubs.F90" "$CAM/src/physics/rrtmg/rrtmg_state.F90" \
         "$HERE/cam_state_driver.F90"; do
  $FC $FL -c "$f" -o "$B/$(basename "${f%.F90}").o"
done
$FC $FL -o "$B/cam_state_driver" shr_kind_mod.o shr_const_mod.o ppgrid.o cam_stubs.o rrtmg_state.o cam_state_driver.o
echo "BUILD_OK $B/cam_state_driver"
