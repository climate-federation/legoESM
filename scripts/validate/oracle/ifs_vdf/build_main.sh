#!/bin/bash
# Build the OpenIFS 49r1 VDFMAIN oracle (driver_main.F90): build.sh's modules and
# kernels plus vdfmain.F90 and vdffblend.F90 compiled unchanged, the surface
# routines SURFEXCDRIVER/SURFPP stubbed from their own interface headers
# (gen_surf_stubs.py), and the 180-argument call generated from vdfmain.F90
# (gen_vdfmain_driver.py).
# usage: build_main.sh <OpenIFS checkout at 8f6f722> <build dir> <python>
set -euo pipefail
OIFS=$1; B=$2; PY=$3; HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$OIFS" rev-parse HEAD)
[ "$SHA" = 8f6f722f732b92fed2c39af763d8fcd1ab15f573 ] || { echo "OpenIFS SHA $SHA is not 8f6f722"; exit 1; }
S=$OIFS/ifs-source/arpifs
MODS="$S/module/yomcst.F90 $S/module/yoethf.F90 $S/module/yoevdf.F90 $S/module/yoevdfs.F90 \
      $S/module/yoegwd.F90 $S/module/yoephli.F90 $S/module/parphy.F90 $S/module/yomjfh.F90"
RTNS="$S/phys_ec/suvdf.F90 $S/phys_ec/suvdfs.F90 $S/phys_ec/satur.F90 $S/phys_ec/cuadjtq.F90 \
      $S/phys_ec/cuinin.F90 $S/phys_ec/cubasen.F90 $S/phys_ec/vdfeis.F90 $S/phys_ec/vdfdpbl.F90 \
      $S/phys_ec/vdfhghtn.F90 $S/phys_ec/vdfexcu.F90 $S/phys_ec/vdftofdc.F90 \
      $S/phys_ec/vdfdifm.F90 $S/phys_ec/vdfdifh.F90 $S/phys_ec/vdfincr.F90 $S/phys_ec/vdfdifc.F90 \
      $S/phys_ec/vdffblend.F90 $S/phys_ec/vdfmain.F90"
mkdir -p "$B/inc" "$B/gen"; cd "$B"
for f in $(grep -ho '#include "[a-z0-9_]*\.intfb\.h"' $RTNS "$HERE/driver_main.F90" "$HERE/oracle_setup.F90" "$HERE/stubs_main.F90" | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  : > inc/$f
done
cp "$HERE/../bechtold_ifs/cuadjtq.intfb.h" inc/
cp "$HERE/surfseb_stub.h" inc/surfseb.h
cat > inc/surf_inq.h <<'EOH'
INTERFACE
SUBROUTINE SURF_INQ(YDSURF, PREPUST)
USE PARKIND1, ONLY : JPRB
USE YOEPHY, ONLY : TSURF_STUB
TYPE(TSURF_STUB), INTENT(IN) :: YDSURF
REAL(KIND=JPRB), OPTIONAL, INTENT(OUT) :: PREPUST
END SUBROUTINE SURF_INQ
END INTERFACE
EOH
for h in surfexcdriver surfpp; do
  git -C "$OIFS" show "$SHA:ifs-source/surf/interface/$h.h" > gen/$h.orig.h
  "$PY" "$HERE/gen_surf_stubs.py" gen/$h.orig.h inc/$h.h gen/${h}_stub.F90
done
"$PY" "$HERE/gen_vdfmain_driver.py" "$S/phys_ec/vdfmain.F90" inc
for f in $(grep -ho '#include "[a-z0-9_]*\.nam\.h"' $RTNS | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  git -C "$OIFS" show "$SHA:ifs-source/arpifs/namelist/$f" > inc/$f
done
FF="-O1 -ffp-contract=off -fno-fast-math -cpp -ffree-line-length-none -fbacktrace -g"
INC="-I$B/inc -I$S/function -I$S/include"
gfortran $FF $INC -c "$HERE/stubs.F90"
for f in $MODS "$HERE/stubs_main.F90" "$HERE/oracle_setup.F90" gen/surfexcdriver_stub.F90 gen/surfpp_stub.F90 \
         $RTNS "$HERE/driver_main.F90"; do
  gfortran $FF $INC -c "$f"
done
gfortran $FF -o oracle_main.x *.o
gfortran --version | head -1
echo BUILD_OK
