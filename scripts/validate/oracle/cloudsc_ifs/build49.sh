#!/bin/bash
# Build the OpenIFS 49r1 CLOUDSC oracle (cloudsc.F90, cloud_supersatcheck.F90,
# sucldp.F90 and the YOMCST/YOETHF/YOECLDP/YOEPHLI modules, compiled unchanged)
# with stubs49.F90 and driver49.F90.
# usage: build49.sh <OpenIFS checkout at 8f6f722> <build dir>
set -euo pipefail
OIFS=$1; B=$2; HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$OIFS" rev-parse HEAD)
[ "$SHA" = 8f6f722f732b92fed2c39af763d8fcd1ab15f573 ] || { echo "OpenIFS SHA $SHA is not 8f6f722"; exit 1; }
S=$OIFS/ifs-source/arpifs
mkdir -p "$B/inc"; cd "$B"
# empty interface blocks for every *.intfb.h the compiled sources include
# (implicit interfaces; none of these routines has OPTIONAL arguments)
for f in $(grep -ho '#include "[a-z0-9_]*\.intfb\.h"' $S/module/yomcst.F90 $S/module/yoecldp.F90 \
             $S/phys_ec/cloud_supersatcheck.F90 $S/phys_ec/sucldp.F90 $S/phys_ec/cloudsc.F90 | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  : > inc/$f
done
# namelist-group includes are outside the sparse checkout: read them from the pinned commit
for f in $(grep -ho '#include "[a-z0-9_]*\.nam\.h"' $S/module/yomcst.F90 $S/module/yoecldp.F90 \
             $S/phys_ec/sucldp.F90 $S/phys_ec/cloudsc.F90 | sed 's/.*"\(.*\)"/\1/' | sort -u); do
  git -C "$OIFS" show "$SHA:ifs-source/arpifs/namelist/$f" > inc/$f
done
FF="-O1 -ffp-contract=off -fno-fast-math -cpp -ffree-line-length-none -fbacktrace -g"
INC="-I$B/inc -I$S/function -I$S/include"
gfortran $FF $INC -c "$HERE/stubs49.F90"
# utility function outside the sparse checkout, read from the pinned commit (compiled after YOMCST)
git -C "$OIFS" show "$SHA:ifs-source/arpifs/utility/fcgeneralized_gamma.F90" > fcgeneralized_gamma.F90

for f in $S/module/yomcst.F90 $S/module/yoethf.F90 $S/module/yoecldp.F90 $S/module/yoephli.F90 fcgeneralized_gamma.F90 \
         $S/phys_ec/cloud_supersatcheck.F90 $S/phys_ec/sucldp.F90 $S/phys_ec/cloudsc.F90 "$HERE/driver49.F90"; do
  gfortran $FF $INC -c "$f"
done
gfortran $FF -o oracle49.x *.o
gfortran --version | head -1
echo BUILD_OK
