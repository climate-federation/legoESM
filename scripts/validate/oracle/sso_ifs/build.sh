#!/bin/bash
# Build the OpenIFS 49r1 subgrid-orography drag oracle: gwdrag.F90, gwsetup.F90,
# gwprofil.F90, sugwd.F90, suvdf.F90, vdftofdc.F90 and the YOMCST/YOEGWD/
# YOEPHLI/YOEVDF/PARPHY modules, compiled unchanged, with stubs.F90 and driver.F90.
# usage: build.sh <OpenIFS checkout at 8f6f722> <build dir>
set -euo pipefail
OIFS=$1; B=$2; HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$OIFS" rev-parse HEAD)
[ "$SHA" = 8f6f722f732b92fed2c39af763d8fcd1ab15f573 ] || { echo "OpenIFS SHA $SHA is not 8f6f722"; exit 1; }
S=$OIFS/ifs-source/arpifs
SRC="$S/module/yomcst.F90 $S/module/yoegwd.F90 $S/module/yoephli.F90 $S/module/yoevdf.F90 \
     $S/module/parphy.F90 $S/phys_ec/sugwd.F90 $S/phys_ec/gwsetup.F90 $S/phys_ec/gwprofil.F90 \
     $S/phys_ec/gwdrag.F90 $S/phys_ec/suvdf.F90 $S/phys_ec/vdftofdc.F90"
mkdir -p "$B/inc"; cd "$B"
# empty interface blocks (implicit interfaces; none of these routines has OPTIONAL arguments)
for f in $(grep -ho '#include "[a-z0-9_]*\.intfb\.h"' $SRC "$HERE/driver.F90" | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  : > inc/$f
done
for f in $(grep -ho '#include "[a-z0-9_]*\.nam\.h"' $SRC | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  git -C "$OIFS" show "$SHA:ifs-source/arpifs/namelist/$f" > inc/$f
done
FF="-O1 -ffp-contract=off -fno-fast-math -cpp -ffree-line-length-none -fbacktrace -g"
INC="-I$B/inc -I$S/function -I$S/include"
gfortran $FF $INC -c "$HERE/stubs.F90"
for f in $SRC "$HERE/driver.F90"; do
  gfortran $FF $INC -c "$f"
done
gfortran $FF -o oracle.x *.o
gfortran --version | head -1
echo BUILD_OK
