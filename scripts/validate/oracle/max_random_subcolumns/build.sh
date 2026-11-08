#!/bin/bash
# Build the CAM maximum-random subcolumn oracle used to generate
# tests/atmosphere/oracle_fixtures/max_random_subcolumns.npz.
#
# usage: build.sh <CAM checkout @ cam_cesm2_1_rel_60> <cime checkout @ cime5.6.47> <build dir>
# The only change to CAM source is ONE inserted line (the uniform injection) before the case(2)
# recursion of mcica_subcol_gen_sw.f90; the patch is printed.  Run on a compute node (sbatch).
set -euo pipefail
CAMR=$(cd "$1" && pwd); CIMER=$(cd "$2" && pwd); mkdir -p "$3"; B=$(cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
CAM=$CAMR/src/physics/rrtmg/aer_src
CIME=$CIMER/src/share
echo "CAM $(git -C "$CAMR" describe --tags 2>/dev/null || true) $(git -C "$CAMR" rev-parse HEAD)"
echo "cime $(git -C "$CIMER" describe --tags 2>/dev/null || true) $(git -C "$CIMER" rev-parse HEAD 2>/dev/null || true)"
export FC=${FC:-gfortran}
$FC --version | head -1
cd "$B"
cp "$CAM/mcica_subcol_gen_sw.f90" "$CAM/parrrsw.f90" "$CAM/rrsw_wvn.f90" .
python3 - <<'PY'
import re
f = "mcica_subcol_gen_sw.f90"
s = open(f).read()
i = s.index("case(2)")
m = re.compile(r"\n(\s*)do ilev = 2, ?nlay").search(s, i)
s = s[:m.start()] + "\n      if (use_inject) CDF(:,:,:) = cdf_inject_sw(:nsubcol,:ncol,:nlay)" + s[m.start():]
j = s.index("\n", s.index("ShrF95MtRandGen, ShrDsfmtRandGen"))
s = s[:j] + "\n   use mcica_inject_data, only: use_inject, cdf_inject_sw" + s[j:]
open(f, "w").write(s)
PY
diff "$CAM/mcica_subcol_gen_sw.f90" mcica_subcol_gen_sw.f90 || true
RN=$CIME/RandNum/src
gcc -O2 -fPIC -DDSFMT_MEXP=19937 -I"$CIME/RandNum/include" -c "$RN/dsfmt_f03/dSFMT.c" \
  "$RN/dsfmt_f03/dSFMT_utils.c" "$RN/kissvec/kissvec.c"
F="$FC -O2 -g -fno-fast-math -ffree-line-length-none -cpp -I$CIME/RandNum/include"
$F -c "$CIME/util/shr_kind_mod.F90"
$F -c "$RN/mt19937/mersennetwister_mod.F90" "$RN/dsfmt_f03/dSFMT_interface.F90" "$RN/kissvec/kissvec_mod.F90"
$F -c "$RN/shr_RandNum_mod.F90"
$F -c "$HERE/stubs.F90"
$F -c parrrsw.f90 rrsw_wvn.f90
$F -c mcica_subcol_gen_sw.f90
$F -c "$HERE/oracle_driver.F90"
$FC -o "$B/oracle_driver" ./*.o
echo "BUILD_OK $B/oracle_driver"
