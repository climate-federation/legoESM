#!/bin/bash
# Build the gSAM 1.8.8 M2005 oracle driver (fp64) used by the warm-rain /
# saturation-adjustment oracle fixtures.
# usage: build.sh <gSAM1.8.8 source root> <build dir>
#
# Oracle: gSAM 1.8.8 (no git tag/SHA); pin = tarball
#   http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz
#   sha256 0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c
# patch_gsam.py makes exact-string scratch copies of SRC/MICRO_M2005/
# {micro_params.f90, module_mp_graupel.f90, microphysics.f90:satadj_liquid}:
# legoESM saturation curve and constants fed in, process-rate dumps (ODIAG),
# run-time satadj tolerance; the unified diffs are written next to the copies.
# gfortran 8.5.0 was used for the committed fixtures.
set -euo pipefail
SRC=$1/SRC/MICRO_M2005
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
python3 "$HERE/patch_gsam.py" "$SRC" "$B"
cd "$B"
FF="-O0 -g -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -fno-range-check -fbacktrace"
gfortran $FF -c micro_params_lego.f90
gfortran $FF -c "$HERE/gsam_stubs.F90"
gfortran $FF -c module_mp_graupel_lego.f90
gfortran $FF -cpp -I"$B" -c "$HERE/satadj_mod.F90"
gfortran $FF -c "$SRC/wrf_debug.f90"
gfortran $FF -c "$HERE/gsam_driver.F90"
gfortran $FF -o gsam_driver gsam_driver.o satadj_mod.o module_mp_graupel_lego.o \
    gsam_stubs.o micro_params_lego.o wrf_debug.o
echo BUILD_OK
