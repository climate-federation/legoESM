#!/bin/bash
# Build the standalone AeroBulk oracle driver (TURB_COARE3P0 + TURB_NCAR).
# Usage: build.sh <aerobulk checkout> <build dir>
# The checkout must be brodeau/aerobulk tag 1.0.0,
# SHA ca40928872eb9c00f96a51cf3a6eed0bb4a70580.
#
# Sources are compiled UNMODIFIED except mod_const.f90, where two constants are
# set to legoESM's values exactly as the tier-2 oracle audit ran them
# (oracle_audit/surface_layer_most_coare_ly09 "build_lego"): grav 9.8 -> 9.80616
# (constants.g) and R_vap 461.495 -> 461.51 (constants.R_v).  They stay
# default-real (single-precision) literals, as every mod_const constant in
# AeroBulk is; the JAX port reproduces that rounding.
set -euo pipefail
AB=${1:?aerobulk 1.0.0 checkout}; B=${2:?build dir}
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$AB" rev-parse HEAD)
[ "$SHA" = ca40928872eb9c00f96a51cf3a6eed0bb4a70580 ] || { echo "wrong aerobulk SHA $SHA" >&2; exit 1; }
git -C "$AB" diff --quiet HEAD -- src || { echo "aerobulk src/ has local changes" >&2; exit 1; }
mkdir -p "$B"; cd "$B"
for f in mod_const mod_phymbl mod_skin_coare mod_common_coare mod_blk_coare3p0 mod_blk_ncar; do
  cp -f "$AB/src/$f.f90" .
done
cp -f "$HERE/driver.f90" .
sed -i 's/grav  = 9.8      !/grav  = 9.80616  !/' mod_const.f90
sed -i 's/R_vap = 461.495    !/R_vap = 461.51     !/' mod_const.f90
grep -q "grav  = 9.80616  !" mod_const.f90 && grep -q "R_vap = 461.51     !" mod_const.f90
FF="-O0 -fno-fast-math -ffree-line-length-none"
for f in mod_const mod_phymbl mod_skin_coare mod_common_coare mod_blk_coare3p0 mod_blk_ncar driver; do
  gfortran $FF -c $f.f90
done
gfortran -o drv driver.o mod_blk_ncar.o mod_blk_coare3p0.o mod_common_coare.o mod_skin_coare.o mod_phymbl.o mod_const.o
sha1sum mod_*.f90 driver.f90
