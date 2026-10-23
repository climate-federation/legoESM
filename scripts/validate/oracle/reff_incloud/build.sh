#!/bin/bash
# Build the CAM6 MG2 effective-radius oracle used to generate
# tests/atmosphere/oracle_fixtures/reff_incloud.npz.
#
# usage: build.sh <CAM checkout @ cam_cesm2_1_rel_60> <WRF module_mp_morr_two_moment.F (v4.6.1)> <build dir>
# micro_mg_utils.F90 is compiled UNMODIFIED; the MG2 statements are copied verbatim
# (line ranges below) from micro_mg2_0.F90; the WRF M2005 PSD lines get only the
# renames needed to share one Fortran scope with MG2 (Fortran is case-insensitive:
# PGAM/LAMC/EFFC/DUM/QSMALL clash with MG2 locals) and legoESM's moist density for
# DUM (the constants-only variant certified in the tier-1 cloud-optics audit).
# Run on a compute node (sbatch), not a login node.
set -euo pipefail
C=$(cd "$1" && pwd); W=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); mkdir -p "$3"; B=$(cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "CAM $(git -C "$C" describe --tags) $(git -C "$C" rev-parse HEAD)"
echo "WRF file sha256 $(sha256sum "$W" | cut -d' ' -f1)"
M="$C/src/physics/cam"
sed -n '877,879p'   "$M/micro_mg2_0.F90" > "$B/mg2_cld.inc"
sed -n '2739,2768p' "$M/micro_mg2_0.F90" > "$B/mg2_dum.inc"
sed -n '2772,2808p' "$M/micro_mg2_0.F90" > "$B/mg2_ice.inc"
sed -n '2811,2865p' "$M/micro_mg2_0.F90" > "$B/mg2_liq.inc"
grep -q 'lcldm = max(liqcldf,mincld)' "$B/mg2_cld.inc"
grep -q 'dumc(i,k) = max(qc(i,k)+qctend(i,k)\*deltat,0._r8)/lcldm(i,k)' "$B/mg2_dum.inc"
grep -q 'dumnc(i,k)=ncnst/rho(i,k)' "$B/mg2_dum.inc"
grep -q 'effi(i,k) = 1.5_r8/lami(i,k)\*1.e6_r8' "$B/mg2_ice.inc"
grep -q 'effc(i,k) = (pgam(i,k)+3._r8)/lamc(i,k)/2._r8\*1.e6_r8' "$B/mg2_liq.inc"
REN='s/\bPGAM\b/PGAM3/g; s/\bLAMC\b/LAMC3/g; s/\bEFFC\b/EFFC3/g; s/\bQSMALL\b/QSMALLW/g; s/\bDUM\b/DUM3/g'
sed -n '3918,3949p' "$W" | sed -e 's#DUM = PRES(K)/(287.15\*T3D(K))#DUM3 = RHOM(K)#' -e "$REN" > "$B/wrf_liq.inc"
sed -n '4024,4029p' "$W" | sed -e "$REN" > "$B/wrf_effc.inc"
grep -q 'DUM3 = RHOM(K)' "$B/wrf_liq.inc"
grep -q 'PGAM3(K)=0.0005714\*(NC3D(K)/1.E6\*DUM3)+0.2714' "$B/wrf_liq.inc"
grep -q 'EFFC3(K) = GAMMA(PGAM3(K)+4.)/' "$B/wrf_effc.inc"
FC=${FC:-gfortran}
$FC --version | head -1
cd "$B"
$FC -O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -c "$M/micro_mg_utils.F90"
$FC -O0 -g -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -cpp -I"$B" \
  -c "$HERE/oracle_driver.F90"
$FC -o "$B/oracle_driver" micro_mg_utils.o oracle_driver.o
echo "BUILD_OK $B/oracle_driver"
