#!/bin/bash
# Build the plain CAM6 MG2 column driver (morrison_mg2_warm_rain/mg2_driver.F90)
# against UNMODIFIED CAM sources, without the probe copy of build.sh there.
# usage: build.sh <CAM checkout (cam_cesm2_1_rel_60)> <build dir>
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -fbacktrace"
gfortran $FF -c "$CAM/wv_sat_methods.F90" "$CAM/micro_mg_utils.F90" "$CAM/micro_mg2_0.F90"
gfortran $FF -o mg2_driver "$HERE/../morrison_mg2_warm_rain/mg2_driver.F90" \
    wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
echo BUILD_OK
