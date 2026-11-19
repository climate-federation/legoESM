#!/bin/bash
# Build the CAM6 post-CLUBB PBL-height oracle (clubb_intr.F90:2594-2617)
# against UNMODIFIED CAM cam_cesm2_1_rel_60
# (a03b84b7c4e34f965b115686f22a043b85739e56) src/physics/cam/{pbl_utils,hb_diff}.F90.
# The CAM infrastructure stubs are the Holtslag-Boville harness's
# (../holtslag_boville/stubs.F90; pver = 32 = the input columns).
# usage: build.sh <CAM checkout> <build dir>
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O2 -g -fbacktrace -cpp -ffree-line-length-none -fcheck=bounds -ffpe-trap=invalid,zero,overflow"
gfortran --version | head -1
gfortran $FF -c "$HERE/../holtslag_boville/stubs.F90"
gfortran $FF -c "$CAM/pbl_utils.F90"
gfortran $FF -c "$CAM/hb_diff.F90"
gfortran $FF -c "$HERE/pblintd_driver.F90"
gfortran $FF -o pblintd_driver stubs.o pbl_utils.o hb_diff.o pblintd_driver.o
sha256sum "$CAM/pbl_utils.F90" "$CAM/hb_diff.F90" "$CAM/clubb_intr.F90"
echo BUILD_OK
