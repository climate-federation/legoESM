#!/bin/bash
# Build the CAM CLUBB-MF oracle driver against UNMODIFIED physics sources:
#   ESCOMP/atmospheric_physics atmos_phys0_34_002 (cec8c93ed22e6dd60bd676459d4fc0468826fe7d)
#     schemes/clubb/clubb_mf.F90 (the tag CAM cam_development 417212dff0 pins),
#     to_be_ccppized/{wv_saturation,wv_sat_methods}.F90
#   ESCOMP/CESM_share share1.1.22 (f256faa25edb103f09f3c372d021ea6e1a633516) RandNum/ (KISS RNG)
# The only source edit: clubb_mf's private `poisson` is made public so the
# driver can replay CAM's draws (diffed below; must be the one added line).
# stubs.F90 supplies CAM/CCPP infrastructure only.  gfortran, fp64, FPE traps.
# Optional 4th argument `bsort`: also switch on CAM's buoyancy-sorting
# detrainment, a local `logical :: bsort = .false.` that CAM never sets
# (one more edited line, diffed below).
# usage: build.sh <atmos_phys> <CESM_share> <build dir> [bsort]
set -euo pipefail
AP=$1; SH=$2/RandNum/src; B=$3; MODE=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
test "$(git -C "$AP" rev-parse HEAD)" = cec8c93ed22e6dd60bd676459d4fc0468826fe7d
test "$(git -C "$2" rev-parse HEAD)" = f256faa25edb103f09f3c372d021ea6e1a633516
# consumed sources must be the committed ones (no working-tree edits)
test -z "$(git -C "$AP" status --porcelain -- schemes/clubb/clubb_mf.F90 to_be_ccppized/)"
test -z "$(git -C "$2" status --porcelain -- RandNum)"
mkdir -p "$B"; cd "$B"; rm -f ./*.o ./*.mod
sed 's/^\(  public :: clubb_mf_init, &\)$/\1\n            poisson, \&/' "$AP/schemes/clubb/clubb_mf.F90" > clubb_mf.F90
NEDIT=1
if [ "$MODE" = bsort ]; then
  sed -i 's/^\(     logical  *:: bsort = \)\.false\.$/\1.true./' clubb_mf.F90
  NEDIT=2
elif [ -n "$MODE" ]; then
  echo "unknown mode $MODE" >&2; exit 2
fi
test "$(diff "$AP/schemes/clubb/clubb_mf.F90" clubb_mf.F90 | grep -c '^>')" = "$NEDIT"
FF="-O0 -g -fbacktrace -cpp -ffree-line-length-none -fcheck=bounds -ffpe-trap=invalid,zero,overflow"
gfortran --version | head -1
gfortran $FF -c "$HERE/stubs.F90"
gcc -O0 -c "$SH/kissvec/kissvec.c"
gcc -O0 -DDSFMT_MEXP=19937 -I"$2/RandNum/include" -c "$SH/dsfmt_f03/dSFMT.c" "$SH/dsfmt_f03/dSFMT_utils.c"
gfortran $FF -c "$SH/kissvec/kissvec_mod.F90" "$SH/mt19937/mersennetwister_mod.F90"
gfortran $FF -DDSFMT_MEXP=19937 -c "$SH/dsfmt_f03/dSFMT_interface.F90"
gfortran $FF -c "$SH/shr_RandNum_mod.F90"
gfortran $FF -c "$AP/to_be_ccppized/wv_sat_methods.F90"
gfortran $FF -c "$AP/to_be_ccppized/wv_saturation.F90"
gfortran $FF -c clubb_mf.F90
gfortran $FF -c "$HERE/driver.F90"
gfortran $FF -o clubb_mf_driver ./*.o
sha256sum clubb_mf.F90 clubb_mf_driver "$AP/schemes/clubb/clubb_mf.F90" "$AP/to_be_ccppized/wv_saturation.F90" "$AP/to_be_ccppized/wv_sat_methods.F90" \
  "$SH/shr_RandNum_mod.F90" "$SH/kissvec/kissvec.c"
echo BUILD_OK
