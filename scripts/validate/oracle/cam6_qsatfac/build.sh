#!/bin/bash
# Build the CAM6 do_subgrid_growth (qsatfac / qsfm) oracle against UNMODIFIED CAM
# sources (cam_cesm2_1_rel_60): wv_sat_methods.F90, micro_mg_utils.F90 and
# micro_mg2_0.F90 compile as shipped; qsatfac_driver.F90 holds the verbatim
# cldfrc2m.F90 and micro_mg2_0.F90 lines (see its header).
# usage: build.sh <CAM checkout> <build dir>
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -fno-fast-math -fbacktrace"
for f in wv_sat_methods micro_mg_utils micro_mg2_0; do gfortran $FF -c "$CAM/$f.F90"; done
gfortran $FF -o qsatfac_driver "$HERE/qsatfac_driver.F90" wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
echo BUILD_OK
