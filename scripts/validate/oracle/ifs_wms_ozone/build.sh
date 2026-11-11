#!/bin/bash
# Build the OpenIFS 49r1 oracle for the WMS non-orographic GWD (sugwwms.F90,
# gwdrag_wms.F90), Cariolle ozone chemistry (o3chem.F90) and methane
# oxidation (sumethox.F90, methox.F90), with the real YOMCST/YOEGWWMS/YOEGWD/
# YOEMETH/YOMOZO modules, all compiled unchanged, plus stubs.F90 and driver.F90.
# usage: build.sh <OpenIFS checkout at 8f6f722> <build dir>
set -euo pipefail
OIFS=$1; B=$2; HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$OIFS" rev-parse HEAD)
[ "$SHA" = 8f6f722f732b92fed2c39af763d8fcd1ab15f573 ] || { echo "OpenIFS SHA $SHA is not 8f6f722"; exit 1; }
mkdir -p "$B/inc" "$B/src"; cd "$B"
SRCS="module/yomcst.F90 module/yoegwwms.F90 module/yoegwd.F90 module/yoemeth.F90 module/yomozo.F90
      phys_ec/sugwwms.F90 phys_ec/gwdrag_wms.F90 phys_ec/o3chem.F90 phys_ec/sumethox.F90 phys_ec/methox.F90"
# read every source from the pinned commit (the working checkout is sparse)
for f in $SRCS; do git -C "$OIFS" show "$SHA:ifs-source/arpifs/$f" > "src/$(basename "$f")"; done
# empty interface blocks for every *.intfb.h (implicit interfaces; no OPTIONAL arguments)
for f in $(grep -ho '#include "[a-z0-9_]*\.intfb\.h"' src/*.F90 | sed 's/.*"\(.*\)"/\1/' | sort -u); do : > inc/$f; done
for f in $(grep -ho '#include "[a-z0-9_]*\.nam\.h"' src/*.F90 | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  git -C "$OIFS" show "$SHA:ifs-source/arpifs/namelist/$f" > inc/$f
done
for f in $(grep -ho '#include "[a-z0-9_]*\.func\.h"' src/*.F90 | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  git -C "$OIFS" show "$SHA:ifs-source/arpifs/function/$f" > inc/$f
done
FF="-O1 -ffp-contract=off -fno-fast-math -cpp -ffree-line-length-none -fbacktrace -g"
INC="-I$B/inc"
gfortran $FF $INC -c "$HERE/stubs.F90"
for f in yomcst yoegwwms yoegwd yoemeth yomozo sugwwms gwdrag_wms o3chem sumethox methox; do
  gfortran $FF $INC -c "src/$f.F90"
done
gfortran $FF $INC -c "$HERE/driver.F90"
gfortran $FF -o oracle.x *.o
gfortran --version | head -1
echo BUILD_OK
