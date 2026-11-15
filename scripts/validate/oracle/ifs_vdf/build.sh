#!/bin/bash
# Build the OpenIFS 49r1 VDF column-kernel oracle: vdfdpbl.F90, vdfhghtn.F90
# (with its callees satur, cuinin, cubasen, cuadjtq, vdfeis), vdfexcu.F90, vdftofdc.F90,
# suvdf.F90, suvdfs.F90 and the YOMCST/YOETHF/YOEVDF/YOEVDFS/YOEGWD/YOEPHLI/
# PARPHY modules, all compiled unchanged, with stubs.F90 and driver.F90.
# usage: build.sh <OpenIFS checkout at 8f6f722> <build dir>
set -euo pipefail
OIFS=$1; B=$2; HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$OIFS" rev-parse HEAD)
[ "$SHA" = 8f6f722f732b92fed2c39af763d8fcd1ab15f573 ] || { echo "OpenIFS SHA $SHA is not 8f6f722"; exit 1; }
S=$OIFS/ifs-source/arpifs
MODS="$S/module/yomcst.F90 $S/module/yoethf.F90 $S/module/yoevdf.F90 $S/module/yoevdfs.F90 \
      $S/module/yoegwd.F90 $S/module/yoephli.F90 $S/module/parphy.F90"
RTNS="$S/phys_ec/suvdf.F90 $S/phys_ec/suvdfs.F90 $S/phys_ec/satur.F90 $S/phys_ec/cuadjtq.F90 \
      $S/phys_ec/cuinin.F90 $S/phys_ec/cubasen.F90 $S/phys_ec/vdfeis.F90 $S/phys_ec/vdfdpbl.F90 \
      $S/phys_ec/vdfhghtn.F90 $S/phys_ec/vdfexcu.F90 $S/phys_ec/vdftofdc.F90"
mkdir -p "$B/inc"; cd "$B"
# empty interface blocks (implicit interfaces) except the two routines with
# OPTIONAL arguments: CUADJTQ (as bechtold_ifs) and SURF_INQ
for f in $(grep -ho '#include "[a-z0-9_]*\.intfb\.h"' $RTNS "$HERE/driver.F90" | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  : > inc/$f
done
cp "$HERE/../bechtold_ifs/cuadjtq.intfb.h" inc/
cat > inc/surf_inq.h <<'EOF'
INTERFACE
SUBROUTINE SURF_INQ(YDSURF, PREPUST)
USE PARKIND1, ONLY : JPRB
USE YOEPHY, ONLY : TSURF_STUB
TYPE(TSURF_STUB), INTENT(IN) :: YDSURF
REAL(KIND=JPRB), OPTIONAL, INTENT(OUT) :: PREPUST
END SUBROUTINE SURF_INQ
END INTERFACE
EOF
for f in $(grep -ho '#include "[a-z0-9_]*\.nam\.h"' $RTNS | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  git -C "$OIFS" show "$SHA:ifs-source/arpifs/namelist/$f" > inc/$f
done
FF="-O1 -ffp-contract=off -fno-fast-math -cpp -ffree-line-length-none -fbacktrace -g"
INC="-I$B/inc -I$S/function -I$S/include"
gfortran $FF $INC -c "$HERE/stubs.F90"
for f in $MODS $RTNS "$HERE/driver.F90"; do
  gfortran $FF $INC -c "$f"
done
gfortran $FF -o oracle.x *.o
gfortran --version | head -1
echo BUILD_OK
