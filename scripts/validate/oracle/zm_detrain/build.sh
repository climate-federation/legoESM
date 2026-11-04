#!/bin/bash
# Build the CAM6 detrainment oracle: the clubb_intr.F90:2345-2386 loop of
# UNMODIFIED CAM cam_cesm2_1_rel_60 (a03b84b7c4e34f965b115686f22a043b85739e56),
# extracted line for line into block.inc and #included by driver.F90.
# usage: build.sh <CAM checkout> <build dir>
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
test "$(git -C "$1" rev-parse HEAD)" = a03b84b7c4e34f965b115686f22a043b85739e56
sed -n '2345,2386p' "$CAM/clubb_intr.F90" > block.inc
head -1 block.inc | grep -q 'do k=1,pver'
tail -1 block.inc | grep -q 'enddo'
gfortran --version | head -1
gfortran -O2 -cpp -ffree-line-length-none -fcheck=bounds -I. -o detrain_driver "$HERE/driver.F90"
sha256sum "$CAM/clubb_intr.F90" block.inc
echo BUILD_OK
