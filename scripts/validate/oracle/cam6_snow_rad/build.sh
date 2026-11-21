#!/bin/bash
# Build the CAM6 snow-for-radiation oracle behind
# tests/atmosphere/oracle_fixtures/cam6_snow_rad.npz.
#
# usage: build.sh <dir holding clubb_intr.F90, micro_mg_cam.F90, micro_mg_utils.F90
#                  from CAM cam_cesm2_1_rel_60> <build dir> <gravit>
# micro_mg_utils.F90 is compiled UNMODIFIED; the other statements are copied
# verbatim (line ranges below).  Nothing is run here.
set -euo pipefail
C=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd); G=$3
HERE=$(cd "$(dirname "$0")" && pwd)
for f in clubb_intr.F90 micro_mg_cam.F90 micro_mg_utils.F90; do
  echo "$f sha256 $(shasum -a 256 "$C/$f" | cut -d' ' -f1)"
done
sed -n '2512p'      "$C/clubb_intr.F90"   > "$B/ci_concld.inc"
sed -n '2309,2321p' "$C/micro_mg_cam.F90" > "$B/mgcam_snow.inc"
sed -n '2576,2583p' "$C/micro_mg_cam.F90" > "$B/mgcam_des.inc"
sed -n '2639p'      "$C/micro_mg_cam.F90" > "$B/mgcam_desum.inc"
grep -q 'concld(i,k) = min(cloud_frac(i,k)-alst(i,k)+deepcu(i,k),0.80_r8)' "$B/ci_concld.inc"
head -1 "$B/mgcam_snow.inc" | grep -q 'cldfsnow(i,k) = cld(i,k)'
tail -1 "$B/mgcam_snow.inc" | grep -q 'icswp(i,k) = qsout(i,k) / max( mincld, cldfsnow(i,k) ) \* state_loc%pdel(i,k) / gravit'
head -1 "$B/mgcam_des.inc" | grep -q 'where (qs_grid(:ngrdcol,top_lev:) >= 1.e-7_r8)'
tail -1 "$B/mgcam_des.inc" | grep -q '3._r8 \* rhosn/rhows'
grep -q 'des_grid(i,k) = des_grid(i,k) \* 1.e6_r8' "$B/mgcam_desum.inc"
echo "      end where" >> "$B/mgcam_des.inc"
FC=${FC:-gfortran}
cd "$B"
$FC -O0 -ffree-line-length-none -fno-fast-math -DHAVE_GAMMA_INTRINSICS -c "$C/micro_mg_utils.F90"
$FC -O0 -ffree-line-length-none -fno-fast-math -cpp -DGRAVIT_VAL="$G" -I"$B" -c "$HERE/oracle_driver.F90"
$FC -o "$B/snow_oracle" micro_mg_utils.o oracle_driver.o
echo "built $B/snow_oracle"
