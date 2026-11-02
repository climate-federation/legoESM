#!/bin/bash
# Build the thin-air (rho < 0.1 kg/m3) MG2 point oracles against UNMODIFIED CAM
# sources (cam_cesm2_1_rel_60): the snow/rain-evaporation point driver of
# ../morrison_mg2_ice_snow, the ice fall-speed point driver and the ice
# sedimentation column driver here.
# usage: build.sh <CAM checkout> <build dir>
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -fbacktrace"
gfortran $FF -c "$CAM/wv_sat_methods.F90" "$CAM/micro_mg_utils.F90" "$CAM/micro_mg2_0.F90"
gfortran $FF -o mg2_snow_procs "$HERE/../morrison_mg2_ice_snow/mg2_snow_procs.F90" \
    wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_ice_fall_procs "$HERE/mg2_ice_fall_procs.F90" \
    wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_ice_sed_cols "$HERE/mg2_ice_sed_cols.F90" \
    wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_rain_frz_pts "$HERE/mg2_rain_frz_pts.F90" \
    wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
echo BUILD_OK
