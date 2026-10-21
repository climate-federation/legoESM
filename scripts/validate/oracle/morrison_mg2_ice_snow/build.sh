#!/bin/bash
# Build the CAM6 MG2 two-moment snow oracle drivers against UNMODIFIED CAM
# sources (cam_cesm2_1_rel_60).  The legoESM-qsat stub is the one the warm-rain
# fixture uses (../morrison_mg2_warm_rain/wv_sat_legoesm.F90).
# usage: build.sh <CAM checkout> <build dir>
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -fbacktrace"
gfortran $FF -c "$HERE/../morrison_mg2_warm_rain/wv_sat_legoesm.F90" "$CAM/micro_mg_utils.F90" "$CAM/micro_mg2_0.F90"
gfortran $FF -o mg2_snow_driver "$HERE/mg2_snow_driver.F90" wv_sat_legoesm.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_snow_procs "$HERE/mg2_snow_procs.F90" wv_sat_legoesm.o micro_mg_utils.o micro_mg2_0.o
echo BUILD_OK
