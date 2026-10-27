#!/bin/bash
# Build the CESM2.1 (cime5.6.47) oracle used to generate
# tests/atmosphere/oracle_fixtures/rad_boundary.npz.
#
# usage: build.sh <cime checkout @ cime5.6.47> <build dir>
# shr_kind_mod / shr_const_mod / shr_orb_mod are compiled UNMODIFIED from the checkout;
# the ocean-albedo lines are copied verbatim from seq_flux_mct.F90 L881-894.
# Run on a compute node (sbatch), not a login node.
set -euo pipefail
C=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "cime $(git -C "$C" describe --tags) $(git -C "$C" rev-parse HEAD)"
U="$C/src/share/util"
sed -n '881,894p' "$C/src/drivers/mct/main/seq_flux_mct.F90" > "$B/ocnalb.inc"
grep -q 'anidr = (.026_r8/(cosz\*\*1.7_r8 + 0.065_r8))' "$B/ocnalb.inc"
FC=${FC:-gfortran}
FL="-O0 -ffree-line-length-none -fno-range-check"
$FC --version | head -1
cd "$B"
$FC $FL -c "$U/shr_kind_mod.F90" "$HERE/stubs.F90" "$U/shr_const_mod.F90" "$U/shr_orb_mod.F90"
$FC $FL -I"$B" -c "$HERE/oracle_driver.F90"
$FC -o "$B/oracle_driver" shr_kind_mod.o stubs.o shr_const_mod.o shr_orb_mod.o oracle_driver.o
echo "BUILD_OK $B/oracle_driver"
