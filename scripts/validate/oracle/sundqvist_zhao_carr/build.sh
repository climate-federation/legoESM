#!/bin/bash
# Build the Zhao-Carr precpd oracle driver against UNMODIFIED ccpp-physics sources.
# usage: build.sh <ccpp-physics checkout (v7.0.1, da75531b)> <build dir>
# zhaocarr_precpd.f and hooks/machine.F compile as shipped; stub_funcphys.F90
# replaces funcphys::fpvs (the GFS lookup table) by legoESM's liquid Tetens curve,
# exactly as the 2026-10-19 tier-2 audit did.  -fdefault-real-8 makes the
# oracle's default-real literals (0.025, 0.01, 5.0e-8, ...) double precision:
# the audit's "r8" faithfulness build.  gfortran 8.5.0 was used for the
# committed fixture (tests/atmosphere/oracle_fixtures/sundqvist_zhao_carr.npz).
set -euo pipefail
CCPP=$1/physics
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FL="-O0 -g -fno-fast-math -ffpe-trap=invalid,zero,overflow -fdefault-real-8 -fdefault-double-8"
gfortran $FL -cpp -ffixed-form -c "$CCPP/hooks/machine.F"
gfortran $FL -c "$HERE/stub_funcphys.F90"
gfortran $FL -c "$CCPP/MP/Zhao_Carr/zhaocarr_precpd.f"
gfortran $FL -c "$HERE/drv_zc.F90"
gfortran -o drv_zc machine.o stub_funcphys.o zhaocarr_precpd.o drv_zc.o
echo BUILD_OK
