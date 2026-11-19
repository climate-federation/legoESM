#!/bin/bash
# Build the AMR-Wind amd_muvel oracle used to generate
# tests/atmosphere/oracle_fixtures/amd.npz.
#
# usage: build.sh <amr-wind checkout @ v4.2.0> <build dir>
# amd_oracle.cpp carries src/turbulence/LES/AMD.H::amd_muvel VERBATIM (checked below
# against the checkout) plus minimal amrex type stubs, so the kernel compiles without
# AMReX. Run on a compute node (sbatch), not a login node.
set -euo pipefail
R=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git -C "$R" rev-parse HEAD)
echo "amr-wind $(git -C "$R" describe --tags) $SHA"
[ "$SHA" == "0ee09c22b9afeca06231e4910276c0ba05fb52dd" ] || { echo "WRONG AMR-Wind SHA"; exit 1; }
diff <(sed -n '/amrex::Real amd_muvel(/,/^}/p' "$HERE/amd_oracle.cpp") \
     <(sed -n '/amrex::Real amd_muvel(/,/^}/p' "$R/src/turbulence/LES/AMD.H") > /dev/null \
  || { echo "amd_muvel NOT VERBATIM"; exit 1; }
g++ --version | head -1
g++ -O0 -std=c++17 -o "$B/amd_oracle" "$HERE/amd_oracle.cpp"
echo "BUILD_OK $B/amd_oracle"
