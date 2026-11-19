#!/bin/bash
# Build the microhh rte-rrtmgp-cpp device-function oracle used to generate
# tests/atmosphere/oracle_fixtures/mc3d_kernels.npz.
#
# usage: build.sh <rte-rrtmgp-cpp checkout @ v0.3> <build dir>
# The headers are used unmodified; create_knull_grid is cut verbatim from
# src_cuda_rt/Raytracer.cu (lines 49-84 at v0.3).  Double precision (no RTE_USE_SP).
set -euo pipefail
R=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "rte-rrtmgp-cpp $(git -C "$R" describe --tags) $(git -C "$R" rev-parse HEAD)"
sed -n 49,84p "$R/src_cuda_rt/Raytracer.cu" > "$B/knull_kernel.inc"
head -3 "$B/knull_kernel.inc"; tail -1 "$B/knull_kernel.inc"
g++ --version | head -1
g++ -O2 -std=c++17 -fno-fast-math -D__device__= -D__host__= -D__global__= \
  -I"$HERE/shim" -I"$R/include_rt" -I"$R/include" -I"$B" \
  -o "$B/oracle_driver" "$HERE/oracle_driver.cpp"
echo "BUILD_OK $B/oracle_driver"
