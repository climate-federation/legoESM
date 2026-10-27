#!/bin/bash
# Build the WRF v4.8.0 Tiedtke oracle driver (fp64, legoESM constants and
# saturation curve: audit variant "V1").
# usage: build.sh <WRF checkout at v4.8.0> <build dir>
# The module is compiled from an INSTRUMENTED copy made by patch_oracle.py
# (adds `use tdbg` + dump assignments only; asserts 0 lines removed).
# -DLEGOCONST: stubs.F90 module_model_constants = legoesm.constants values,
#   water only (xls = xlv, xlf = 0).  -DLEGOSAT: Tetens coefficients =
#   legoesm.thermo.saturation_vapor_pressure on the liquid and ice branch.
# gfortran 8.5.0 was used for the committed fixture
# (tests/atmosphere/oracle_fixtures/tiedtke_wrf.npz).
set -euo pipefail
WRF=$1
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$WRF" rev-parse HEAD)
[ "$SHA" = 06d4240ae989cc3e50af412bb472df3d9048783c ] || { echo "WRF SHA $SHA is not v4.8.0"; exit 1; }
MD5=$(md5sum "$WRF/phys/module_cu_tiedtke.F" | cut -d' ' -f1)
[ "$MD5" = 1d9127ddd05b9e314ae95b847043e155 ] || { echo "module md5 $MD5 mismatch"; exit 1; }
mkdir -p "$B"
cd "$B"
python3 "$HERE/patch_oracle.py" "$WRF/phys/module_cu_tiedtke.F" module_cu_tiedtke_dbg.F
FF="-O2 -cpp -ffree-line-length-none -fno-fast-math -fdefault-real-8 -fdefault-double-8 -DLEGOCONST -DLEGOSAT"
gfortran $FF -c "$HERE/stubs.F90"
gfortran $FF -ffree-form -c module_cu_tiedtke_dbg.F
gfortran $FF -c "$HERE/driver.F90"
gfortran -o tdriver.x stubs.o module_cu_tiedtke_dbg.o driver.o
gfortran $FF -fcheck=bounds -ffree-form -o tdriver_chk.x "$HERE/stubs.F90" module_cu_tiedtke_dbg.F "$HERE/driver.F90"
gfortran --version | head -1
echo BUILD_OK
