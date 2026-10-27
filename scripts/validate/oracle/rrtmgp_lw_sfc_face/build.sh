#!/bin/bash
# Build the RTE+RRTMGP v1.8 oracle driver used to generate
# tests/atmosphere/oracle_fixtures/rrtmgp_lw_sfc_face.npz.
#
# usage: build.sh <rte-rrtmgp checkout @ v1.8> <conda env with netcdf-fortran> <build dir>
# Upstream is built out of tree and UNPATCHED: the fixture only reads the gas-optics
# Planck sources (compute_Planck_source), not the two-stream solver.
# Run on a compute node (sbatch), not a login node.
set -euo pipefail
R=$(cd "$1" && pwd); E=$(cd "$2" && pwd); mkdir -p "$3"; B=$(cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "rte-rrtmgp $(git -C "$R" describe --tags) $(git -C "$R" rev-parse HEAD)"
cp "$R/build/Makefile" "$B/"
cd "$B"
sed -i "s#\.\./#$R/#g" Makefile
export FC=${FC:-gfortran}
export FCFLAGS="-O2 -g -fno-fast-math -ffree-line-length-none -fPIC"
$FC --version | head -1
make -j4 librte.a librrtmgp.a > make.log 2>&1 || { tail -40 make.log; exit 1; }
for f in mo_simple_netcdf mo_load_coefficients; do
  $FC $FCFLAGS -I"$B" -I"$E/include" -J"$B" -c "$R/examples/$f.F90" -o "$B/$f.o"
done
$FC $FCFLAGS -I"$B" -I"$E/include" -c "$HERE/oracle_driver.F90" -o "$B/oracle_driver.o"
$FC $FCFLAGS -o "$B/oracle_driver" "$B/oracle_driver.o" "$B/mo_load_coefficients.o" \
  "$B/mo_simple_netcdf.o" -L"$B" -lrrtmgp -lrte -L"$E/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$E/lib"
echo "BUILD_OK $B/oracle_driver"
