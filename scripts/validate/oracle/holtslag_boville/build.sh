#!/bin/bash
# Build the Holtslag-Boville oracle driver against UNMODIFIED sources:
#   E3SM v3.0.1 (d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84)
#     components/eam/src/physics/cam/{pbl_utils,hb_diff}.F90
#   CAM cam_cesm2_1_rel_60 (a03b84b7c4e34f965b115686f22a043b85739e56)
#     src/utils/{coords_1d,linear_1d_operators}.F90,
#     src/physics/cam/{vdiff_lu_solver,diffusion_solver}.F90
# stubs.F90 supplies only the CAM infrastructure modules those import.
# gfortran 8.5.0, fp64 (r8 kinds), FPE traps; used for the committed fixture
# tests/atmosphere/oracle_fixtures/holtslag_boville.npz.
# usage: build.sh <E3SM checkout> <CAM checkout> <build dir>
set -euo pipefail
E3=$1/components/eam/src/physics/cam
CAM=$2/src
B=$3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O2 -g -fbacktrace -cpp -DUSE_CONTIGUOUS= -ffree-line-length-none -fcheck=bounds -ffpe-trap=invalid,zero,overflow"
gfortran --version | head -1
gfortran $FF -c "$HERE/stubs.F90"
gfortran $FF -c "$E3/pbl_utils.F90"
gfortran $FF -c "$E3/hb_diff.F90"
gfortran $FF -c "$CAM/utils/coords_1d.F90"
gfortran $FF -c "$CAM/utils/linear_1d_operators.F90"
gfortran $FF -c "$CAM/physics/cam/vdiff_lu_solver.F90"
gfortran $FF -c "$CAM/physics/cam/diffusion_solver.F90"
gfortran $FF -c "$HERE/hb_driver.F90"
gfortran $FF -o hb_driver ./*.o
sha256sum "$E3/hb_diff.F90" "$E3/pbl_utils.F90" "$CAM/physics/cam/diffusion_solver.F90" \
  "$CAM/physics/cam/vdiff_lu_solver.F90" "$CAM/utils/coords_1d.F90" "$CAM/utils/linear_1d_operators.F90"
echo BUILD_OK
