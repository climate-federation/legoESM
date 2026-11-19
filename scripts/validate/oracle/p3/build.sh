#!/bin/bash
# Build the WRF v4.8.0 P3 oracle driver used by make_fixture.py.
#
# Usage: PYTHONPATH=<legoESM packages> bash build.sh <WRF checkout> <build_dir>
#
# Oracle: https://github.com/wrf-model/WRF tag v4.8.0
# (SHA 06d4240ae989cc3e50af412bb472df3d9048783c), phys/module_mp_p3.F (P3 v4.5.2)
# and run/p3_lookupTable_1.dat-v5.4_2momI (the table p3_init requests for
# 2-moment ice).  patch_p3.py writes the legoESM-fed scratch copy (legoESM
# constants + saturation, process-rate dumps); it is compiled with gfortran in
# fp64 (-fdefault-real-8 -fdefault-double-8), as the tier-1/2 oracle fixtures are.
# Compute: run inside a SLURM job, not on a login node.
set -euo pipefail
WRF=$(readlink -f "$1"); B=$(readlink -m "$2")
HERE=$(cd "$(dirname "$0")" && pwd)
test "$(git -C "$WRF" rev-parse HEAD)" = "06d4240ae989cc3e50af412bb472df3d9048783c"
echo "716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6  $WRF/phys/module_mp_p3.F" | sha256sum -c -
echo "be1ab6fb03481e376e47c6c79d808af5d8ab069f2b242931e9c54801bad4ae84  $WRF/run/p3_lookupTable_1.dat-v5.4_2momI" | sha256sum -c -
mkdir -p "$B/lookup"
cp -f "$WRF/run/p3_lookupTable_1.dat-v5.4_2momI" "$B/lookup/"
python3 "$HERE/patch_p3.py" "$WRF/phys/module_mp_p3.F" "$B"
gfortran --version | head -1
FF="-O0 -g -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -fno-range-check -fbacktrace -cpp"
cd "$B"
gfortran $FF -c "$HERE/stubs_lego.F90"
gfortran $FF -ffree-form -c module_mp_p3_lego.F
gfortran $FF -c "$HERE/p3_driver.F90"
gfortran $FF -o p3_driver p3_driver.o module_mp_p3_lego.o stubs_lego.o
gfortran $FF -c "$HERE/qsat_check.F90"
gfortran $FF -o qsat_check qsat_check.o stubs_lego.o
test -x p3_driver && test -x qsat_check && echo BUILD_OK
