#!/bin/bash
# Build the CAM6 MG2 two-moment snow oracle drivers against UNMODIFIED CAM
# sources (cam_cesm2_1_rel_60), incl. CAM's own wv_sat_methods.F90 (Goff-Gratch).
# usage: build.sh <CAM checkout> <build dir>
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -fbacktrace"
gfortran $FF -c "$CAM/wv_sat_methods.F90" "$CAM/micro_mg_utils.F90" "$CAM/micro_mg2_0.F90"
gfortran $FF -o mg2_snow_driver "$HERE/mg2_snow_driver.F90" wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_snow_procs "$HERE/mg2_snow_procs.F90" wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
echo BUILD_OK
