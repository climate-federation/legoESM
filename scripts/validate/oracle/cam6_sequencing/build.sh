#!/bin/bash
# Build the CAM6 physics-sequencing oracle used for
# tests/atmosphere/oracle_fixtures/cam6_sequencing.npz.
#
# usage: build.sh <CAM checkout @ cam_cesm2_1_rel_60 a03b84b7> <build dir>
# oracle_seq.F90 holds verbatim extracts of CAM's physics_update / ptend_scale /
# qneg3 arithmetic and the physpkg tphysbc/tphysac control flow (line references in
# its header), driving analytic stub schemes identical to stubs.py.  The CAM SHA is
# printed so the extracts can be re-checked.  fp64.  Run on a compute node.
set -euo pipefail
C=$(cd "$1" && pwd); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "cam $(git -C "$C" rev-parse HEAD)"
FC=${FC:-gfortran}
$FC --version | head -1
cd "$B"
$FC -O0 -g -fno-fast-math -fdefault-real-8 -ffree-line-length-none -fcheck=all -o oracle_seq "$HERE/oracle_seq.F90"
echo "BUILD_OK $B/oracle_seq"
