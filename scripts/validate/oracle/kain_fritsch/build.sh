#!/bin/bash
# Build the WRF KF-Eta oracle (module_cu_kfeta.F, WRF v4.8.0 06d4240ae9) as a
# standalone one-column driver, fp64 (-fdefault-real-8), gfortran 8.5.0 for the
# committed fixture (tests/atmosphere/oracle_fixtures/kain_fritsch.npz).
# usage: build.sh <WRF checkout at 06d4240ae9> <build dir>
set -euo pipefail
WRF=$1
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$WRF" rev-parse HEAD)
[ "$SHA" = 06d4240ae989cc3e50af412bb472df3d9048783c ] || { echo "WRF SHA $SHA is not v4.8.0 06d4240ae9"; exit 1; }
mkdir -p "$B"
cd "$B"
python3 "$HERE/instrument_kfeta.py" "$WRF/phys/module_cu_kfeta.F" module_cu_kfeta_dbg.F
FF="-O2 -cpp -ffree-line-length-none -fno-fast-math -fdefault-real-8 -fdefault-double-8"
gfortran $FF -c "$HERE/stubs.F90"
gfortran $FF -ffree-form -c module_cu_kfeta_dbg.F
gfortran $FF -c "$HERE/driver.F90"
gfortran $FF -c "$HERE/lut_dump.F90"
gfortran -o kf_driver.x stubs.o module_cu_kfeta_dbg.o driver.o
gfortran -o lut_dump.x stubs.o module_cu_kfeta_dbg.o lut_dump.o
gfortran --version | head -1
echo BUILD_OK
