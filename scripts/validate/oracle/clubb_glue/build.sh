#!/bin/bash
# Build and run the CAM clubb_intr host-glue oracle.
#   build.sh CAM_ROOT WORK_DIR
# CAM_ROOT: ESCOMP/CAM checked out at tag cam_cesm2_1_rel_60
#   (a03b84b7c4e34f965b115686f22a043b85739e56).  WORK_DIR must already hold
#   oracle_in_15000.txt / oracle_in_100.txt from build_inputs.py.
# Every physics line of the driver is a VERBATIM line range of the oracle,
# extracted here into include files; main.F90 only sequences them and
# stubs.F90 supplies legoESM's constants and minimal CAM types.
set -euo pipefail
CAM_ROOT=${1:?usage: build.sh CAM_ROOT WORK_DIR}
WORK=${2:?usage: build.sh CAM_ROOT WORK_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=a03b84b7c4e34f965b115686f22a043b85739e56
test "$(git -C "$CAM_ROOT" rev-parse HEAD)" = "$SHA" || { echo "CAM_ROOT not at $SHA"; exit 1; }
O=$CAM_ROOT/src/physics/cam
CI=$O/clubb_intr.F90; TR=$O/tropopause.F90; CF=$O/cldfrc2m.F90
B=$WORK/fortran; mkdir -p "$B"; cd "$B"
x(){ sed -n "$2p" "$1" > "$3"; echo "$3 <- $(basename "$1"):$2"; }
x $CI 1534,1538 inc_exner.inc
x $CI 1546,1550 inc_init.inc
x $CI 1601,1606 inc_thv.inc
x $CI 1626,1674 inc_grid.inc
x $CI 1677,1680 inc_wm.inc
x $CI 1774,1784 inc_sfc.inc
x $CI 2009,2014 inc_toplev_reset.inc
x $CI 2042,2161 inc_post.inc
x $CI 2478,2486 inc_alst.inc
x $CI 2492,2514 inc_deepcu.inc
x $CI 2535,2559 inc_aist.inc
x $CI 2572,2588 inc_ast.inc
x $TR 720,812 inc_twmo.inc
x $TR 818,878 inc_tropopause_twmo.inc
x $TR 367,417 inc_tropopause_analytic.inc
x $TR 1192,1200 inc_polar.inc
x $CF 901,1154 inc_aist_vector.inc
x $CF 46,47 inc_qist.inc
cp -f $O/wv_sat_methods.F90 .
cp -f "$HERE/main.F90" "$HERE/stubs.F90" .
gfortran --version | head -1
FFL="-O0 -g -ffree-line-length-none -fcheck=bounds -ffpe-trap=invalid,zero,overflow"
gfortran $FFL -c wv_sat_methods.F90
gfortran $FFL -c stubs.F90
gfortran $FFL -c main.F90
gfortran $FFL -o oracle_glue wv_sat_methods.o stubs.o main.o
./oracle_glue 15000.0 "$WORK/oracle_in_15000.txt" > "$WORK/oracle_out_15000.txt"
./oracle_glue 100.0 "$WORK/oracle_in_100.txt" > "$WORK/oracle_out_100.txt"
