#!/bin/bash
# Build the Zhao-Carr gscond oracle driver against UNMODIFIED ccpp-physics sources.
# usage: build.sh <ccpp-physics checkout (v7.0.1, da75531b)> <build dir>
# zhaocarr_gscond.f and hooks/machine.F compile as shipped; funcphys::fpvs is the
# legoESM Tetens stub shared with the precpd oracle (../sundqvist_zhao_carr).
# The shipped file does not compile with any gfortran (8.5, 11.2, 13.0): line
# 107 continues the "optional ::" declaration with a leading comma
# ("&,  tp(:,:), ..."), "Invalid character in name".  The build compiles a COPY
# with that one comma removed (declaration only; no executable line changes)
# and checks both checksums.
set -euo pipefail
CCPP=$1/physics
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FL="-O0 -g -fno-fast-math -ffpe-trap=invalid,zero,overflow -fdefault-real-8 -fdefault-double-8"
gfortran $FL -cpp -ffixed-form -c "$CCPP/hooks/machine.F"
gfortran $FL -c "$HERE/../sundqvist_zhao_carr/stub_funcphys.F90"
SRC="$CCPP/MP/Zhao_Carr/zhaocarr_gscond.f"
echo "d183b9cc74fde7c99f58b9e2dd8c0f1c504ae8a90ec0d3bc002d4295fa6b5f42  $SRC" | sha256sum -c -
sed '107s/^     &,                     tp(:,:)/     \&                      tp(:,:)/' "$SRC" > zhaocarr_gscond.f
[ "$(diff "$SRC" zhaocarr_gscond.f | grep -c '^[<>]')" = 2 ] || { echo "patch did not apply"; exit 1; }
gfortran $FL -c zhaocarr_gscond.f
gfortran $FL -c "$HERE/drv_gscond.F90"
gfortran -o drv_gscond machine.o stub_funcphys.o zhaocarr_gscond.o drv_gscond.o
echo BUILD_OK
