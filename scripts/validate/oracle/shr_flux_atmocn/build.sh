#!/bin/bash
# Build the standalone CESM2.1 shr_flux_atmOcn oracle (UNMODIFIED cime5.6.47
# shr_flux_mod.F90 + real shr_kind_mod / shr_const_mod; stubs.F90 replaces only
# shr_log_mod, shr_sys_mod and water_isotopes, none of which touch the flux
# numerics).  Usage: build.sh <cime5.6.47 checkout> <build dir>
# The cime checkout must be tag cime5.6.47, SHA d8e9b894d7eb96fe96f8f8a6943b421147f94606.
set -euo pipefail
CIME=${1:?cime5.6.47 checkout}; B=${2:?build dir}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"; cd "$B"
U=$CIME/src/share/util
cp -f "$U/shr_kind_mod.F90" "$U/shr_const_mod.F90" "$U/shr_flux_mod.F90" .
cp -f "$HERE/stubs.F90" "$HERE/driver.F90" .
FF="-O0 -ffree-line-length-none"
gfortran $FF -c shr_kind_mod.F90 shr_const_mod.F90
gfortran $FF -c stubs.F90
gfortran $FF -cpp -c shr_flux_mod.F90
gfortran $FF -c driver.F90
gfortran -o drv driver.o shr_flux_mod.o stubs.o shr_const_mod.o shr_kind_mod.o
sha1sum shr_flux_mod.F90
