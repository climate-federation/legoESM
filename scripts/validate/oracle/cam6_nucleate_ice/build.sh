#!/bin/bash
# Build the CAM6 cirrus nucleation oracle against UNMODIFIED CAM sources (cam_cesm2_1_rel_60):
# wv_sat_methods.F90 and nucleate_ice.F90, with wv_saturation_stub.F90 (svp_water/svp_ice) and driver.F90.
# usage: build.sh <CAM checkout> <build dir>
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -g -ffree-line-length-none -fno-fast-math -fbacktrace"
gfortran $FF -c "$CAM/wv_sat_methods.F90"
gfortran $FF -c "$HERE/wv_saturation_stub.F90"
gfortran $FF -c "$CAM/nucleate_ice.F90"
gfortran $FF -o nucleate_ice_driver "$HERE/driver.F90" wv_sat_methods.o wv_saturation_stub.o nucleate_ice.o
echo BUILD_OK
