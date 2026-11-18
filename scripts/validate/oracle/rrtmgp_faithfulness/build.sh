#!/bin/bash
# Build the latest RTE+RRTMGP (out of tree, unpatched) and the audit driver.
# usage: build.sh <rte-rrtmgp checkout> <build dir>      (run through sbatch)
set -euo pipefail
R=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
module load gcc/13.4.0-gcc-13.4.0 cmake/4.2.4-gcc-13.4.0 netcdf-fortran/4.6.2-openmpi-5.0.10-gcc-13.4.0
echo "rte-rrtmgp $(git -C "$R" describe --tags --always) $(git -C "$R" rev-parse HEAD)"
FF="-O2 -g -fno-fast-math -ffree-line-length-none -fPIC"
gfortran --version | head -1
cmake -S "$R" -B "$B/cm" -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_BUILD_TYPE=None \
  -DCMAKE_Fortran_FLAGS="$FF" -DRTE_KERNEL_MODE=default -DRTE_ENABLE_SP=OFF > "$B/cmake.log" 2>&1 \
  || { tail -30 "$B/cmake.log"; exit 1; }
cmake --build "$B/cm" -j8 > "$B/make.log" 2>&1 || { tail -40 "$B/make.log"; exit 1; }
NFI=$(nf-config --includedir); NFL=$(nf-config --flibs)
NCD=$(dirname "$(ldd "$(nf-config --prefix)/lib/libnetcdff.so" | awk '/libnetcdf\.so/{print $3}')")
MODS=$(find "$B/cm" -name '*.mod' -printf '%h\n' | sort -u | sed 's/^/-I/' | tr '\n' ' ')
LIBS=$(find "$B/cm" -name '*.a' | grep -E 'rrtmgp|rte' | sort -r | tr '\n' ' ')
cd "$B"
for f in "$R/examples/shared-utils/mo_testing_utils.F90" "$R/examples/shared-utils/mo_simple_netcdf.F90" \
         "$R/rrtmgp/data-loading-examples/mo_optics_utils_rrtmgp.F90" "$HERE/oracle_driver.F90"; do
  gfortran $FF $MODS -I"$B" -I"$NFI" -J"$B" -c "$f" -o "$B/$(basename "${f%.F90}").o"
done
gfortran $FF -o "$B/oracle_driver" mo_testing_utils.o mo_simple_netcdf.o mo_optics_utils_rrtmgp.o \
  oracle_driver.o -Wl,--start-group $LIBS -Wl,--end-group $NFL -L"$NCD" -Wl,-rpath,"$(nf-config --prefix)/lib" -Wl,-rpath,"$NCD"
echo "BUILD_OK $B/oracle_driver"
