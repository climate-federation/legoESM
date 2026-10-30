#!/bin/bash
# Build the two Louis oracle drivers against UNMODIFIED sources:
#   OpenIFS github.com/ecmwf-ifs/openifs @ 8f6f722f732b92fed2c39af763d8fcd1ab15f573
#     ifs-source/arpifs/phys_ec/vdfexcu.F90 (whole subroutine), module/parphy.F90,
#     module/yoevdfs.F90, function/fcvdfs.func.h
#   ICON icon-2026.04-public 8597da45ef4b86323f3fb844caedc4ae5e1ffc01
#     src/atm_phy_schemes/mo_turb_vdiff_diag.f90 lines 518-530 (the ECHAM/ICON
#     "Louis 79" unstable block), extracted verbatim by sed into icon_frag.inc
# stubs_ifs.F90 + inc/surf_inq.h supply only the IFS infrastructure modules
# vdfexcu imports.  gfortran 8.5.0, fp64 (explicit kinds), FPE traps; used for
# the committed fixture tests/atmosphere/oracle_fixtures/louis.npz.
# usage: build.sh <openifs checkout> <ICON checkout> <build dir>
set -euo pipefail
OIFS=$1/ifs-source/arpifs
ICON=$2/src/atm_phy_schemes
B=$3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B/ifs" "$B/icon"
FF="-O2 -g -fbacktrace -cpp -DUSE_CONTIGUOUS= -ffree-line-length-none -fcheck=bounds -ffpe-trap=invalid,zero,overflow"
gfortran --version | head -1
( cd "$B/ifs"
  gfortran $FF -c "$HERE/stubs_ifs.F90"
  gfortran $FF -c "$OIFS/module/parphy.F90"
  gfortran $FF -c "$OIFS/module/yoevdfs.F90"
  gfortran $FF -I"$HERE/inc" -I"$OIFS/function" -c "$OIFS/phys_ec/vdfexcu.F90"
  gfortran $FF -c "$HERE/ifs_driver.F90"
  gfortran $FF -o "$B/ifs_exe" ./*.o )
sed -n '518,530p' "$ICON/mo_turb_vdiff_diag.f90" > "$B/icon/icon_frag.inc"
( cd "$B/icon"; gfortran $FF -I. -o "$B/icon_exe" "$HERE/icon_driver.F90" )
sha256sum "$OIFS/phys_ec/vdfexcu.F90" "$OIFS/module/parphy.F90" "$OIFS/module/yoevdfs.F90" \
  "$OIFS/function/fcvdfs.func.h" "$ICON/mo_turb_vdiff_diag.f90" "$ICON/mo_turb_vdiff_params.f90" \
  "$B/icon/icon_frag.inc"
echo BUILD_OK
