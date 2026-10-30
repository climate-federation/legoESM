#!/bin/bash
# Build the OpenIFS cumulus oracle (CUMASTRN and callees) as a standalone
# one-column driver.
# usage: build.sh <OpenIFS checkout at 8f6f722f73> <build dir>
# The SAME executable runs both fixture arms: the driver sets the constants
# at run time (CONST_MODE 1 = constants from the namelist, 2 = IFS native).
# fcttre.func.h / cuadjtq.func.h are the stock OpenIFS statement functions
# (mixed phase).  cumastrn.F90 is compiled from a copy made by
# instrument_cumastrn.py, which only inserts write-only stage dumps
# (asserts no source line removed).  gfortran 8.5.0 was used for the
# committed fixture (tests/atmosphere/oracle_fixtures/bechtold_ifs.npz).
set -euo pipefail
OIFS=$1
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$OIFS" rev-parse HEAD)
[ "$SHA" = 8f6f722f732b92fed2c39af763d8fcd1ab15f573 ] || { echo "OpenIFS SHA $SHA is not 8f6f722"; exit 1; }
S=$OIFS/ifs-source/arpifs/phys_ec
mkdir -p "$B/inc"
cd "$B"
# interface blocks: only CUADJTQ needs one (OPTIONAL argument); the rest are
# empty so the #include lines resolve to implicit interfaces
for f in abor1 cuadjtqs cuascn cubasen cubasmcn cubidiag cuctracer cuddrafn cudlfsn \
         cudtdqn cududv cuentr cuflxn cuinin; do : > inc/$f.intfb.h; done
cp "$HERE/cuadjtq.intfb.h" inc/
python3 "$HERE/instrument_cumastrn.py" "$S/cumastrn.F90" "$B/cumastrn_instr.F90"
FF="-O1 -ffp-contract=off -fno-fast-math -cpp -ffree-line-length-none -fbacktrace -g"
INC="-I$B/inc -I$OIFS/ifs-source/arpifs/function"
gfortran $FF $INC -c "$HERE/stubs.F90" "$HERE/dumpmod.F90"
for f in cuadjtq cuinin cubasen cubasmcn cuentr cuascn cudlfsn cuddrafn cuflxn cudtdqn cubidiag cududv; do
  gfortran $FF $INC -c "$S/$f.F90"
done
gfortran $FF $INC -c cumastrn_instr.F90
gfortran $FF $INC -c "$HERE/driver.F90"
gfortran $FF -o oracle.x stubs.o dumpmod.o cuadjtq.o cuinin.o cubasen.o cubasmcn.o cuentr.o \
  cuascn.o cudlfsn.o cuddrafn.o cuflxn.o cudtdqn.o cubidiag.o cududv.o cumastrn_instr.o driver.o
gfortran --version | head -1
echo BUILD_OK
