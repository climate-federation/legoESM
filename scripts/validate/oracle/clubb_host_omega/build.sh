#!/bin/bash
# Build the CAM6 host-omega oracle behind
# tests/atmosphere/oracle_fixtures/clubb_host_omega_cam.npz.
#
# usage: build.sh <clubb_intr.F90> <grid_class.F90> <interpolation.F90> <build dir> <nlev>
#   clubb_intr.F90: ESCOMP/CAM cam_cesm2_1_rel_60 src/physics/cam/clubb_intr.F90
#     (sha256 f4b7c3d213e694abb35c8a2e7d020a039ee538b668a5d9d798e32902314354b4).
#   grid_class.F90, interpolation.F90: CLUBB src/CLUBB_core at
#     larson-group clubb_release 324635f52 (2016-09-01), the ghost-level
#     single-column grid of the r8099 era that CAM6 vendors.
# Every physics line is a VERBATIM line range of those files, extracted into
# include files; main.F90 only sequences them and stubs.F90 holds legoESM's
# constants and a minimal physics_state.  Nothing is run here.
set -euo pipefail
CI=$1; GC=$2; IP=$3; mkdir -p "$4"; B=$(cd "$4" && pwd); NLEV=$5
HERE=$(cd "$(dirname "$0")" && pwd)
echo "clubb_intr.F90 sha256 $(shasum -a 256 "$CI" | cut -d' ' -f1)"
x(){ sed -n "$2p" "$1" > "$B/$3"; echo "$3 <- $(basename "$1"):$2"; }
x "$CI" 1626,1674 inc_grid.inc
x "$CI" 1677,1680 inc_wm.inc
x "$CI" 1778,1778 inc_wm_zm.inc
x "$GC" 662,665 inc_zt_grid_type3.inc
x "$GC" 1662,1680 inc_weights_zt2zm.inc
x "$GC" 1286,1303 inc_linear_interpolated_azm.inc
x "$IP" 109,109 inc_linear_interp_factor.inc
grep -q 'wm_zt(k+1) = -1._r8\*state1%omega(i,pver-k+1)/(rho_in(k+1)\*gravit)' "$B/inc_wm.inc"
grep -q 'rho_ds_zt(k+1)       = (1._r8/gravit)\*(state1%pdel(i,pver-k+1)/dz_g(pver-k+1))' "$B/inc_grid.inc"
grep -q 'wm_zm           = zt2zm_api(wm_zt)' "$B/inc_wm_zm.inc"
grep -q 'gr%zt(1) = gr%zm(1) - 0.5_core_rknd \* ( gr%zm(2) - gr%zm(1) )' "$B/inc_zt_grid_type3.inc"
grep -q 'linear_interp_factor( gr%weights_zt2zm(1, k), azt(k+1), azt(k) )' "$B/inc_linear_interpolated_azm.inc"
FC=${FC:-gfortran}
cp -f "$HERE/stubs.F90" "$HERE/main.F90" "$B/"
cd "$B"
FFL="-O0 -ffree-line-length-none -fno-fast-math -fcheck=bounds"
$FC $FFL -DPVER_VAL="$NLEV" -cpp -c stubs.F90
$FC $FFL -c main.F90
$FC -o wm_oracle stubs.o main.o
echo "built $B/wm_oracle (pver=$NLEV)"
