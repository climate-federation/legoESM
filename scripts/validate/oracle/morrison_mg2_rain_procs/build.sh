#!/bin/bash
# Build the MG2 rain process probe: CAM's wv_sat_methods.F90 and
# micro_mg_utils.F90 unmodified, micro_mg2_0.F90 as the probe copy written by
# make_tap.py (assignments to public probe arrays only), and mg2_rain_driver.F90.
# Also builds the plain driver of ../morrison_mg2_warm_rain against the
# UNMODIFIED micro_mg2_0.F90, to check the probe changes no output.
# usage: build.sh <CAM checkout (cam_cesm2_1_rel_60)> <build dir> <python>
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
PY=${3:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B/tap"
cd "$B"
FF="-O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -fbacktrace"
gfortran $FF -c "$CAM/wv_sat_methods.F90" "$CAM/micro_mg_utils.F90" "$CAM/micro_mg2_0.F90"
gfortran $FF -o mg2_driver "$HERE/../morrison_mg2_warm_rain/mg2_driver.F90" \
    wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
"$PY" "$HERE/make_tap.py" "$CAM/micro_mg2_0.F90" tap/micro_mg2_0.F90
test "$(diff "$CAM/micro_mg2_0.F90" tap/micro_mg2_0.F90 | grep -c '^>')" -eq 21
test "$(diff "$CAM/micro_mg2_0.F90" tap/micro_mg2_0.F90 | grep -c '^<')" -eq 0
(cd tap && gfortran $FF -I.. -c "$CAM/micro_mg_utils.F90" micro_mg2_0.F90 \
   && gfortran $FF -I. -I.. -o ../mg2_rain_driver "$HERE/mg2_rain_driver.F90" \
      ../wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o)
echo BUILD_OK
