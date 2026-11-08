#!/bin/bash
# Build the CONVECT v4.3c (Emanuel convect43c.f) oracle driver.
# usage: build.sh <convect43c.f> <build dir>
#
# The oracle is Kerry Emanuel's convect43c.f, CONVECT version 4.3c (20 May
# 2002), sha256 below -- the copy the 2026-10-19 tier-2 audit compiled.  Its
# upstream still serves it (texmex.mit.edu/pub/emanuel/CONVECT4/convect43c.f,
# re-downloaded 2026-10-31 with this sha256 for the #2266 FU/FV regeneration);
# climt (CliMT/climt 3ebe9427, climt/_lib/emanuel/convect43c.f90) carries the
# same algorithm lines.  patch_oracle.py (from the audit, unchanged) adds only
# COMMON-block switches and debug copies: legoESM constants (CPD CPV CL RV RD
# LV0 G) in place of CONVECT's stock values, as CONVECT asks of its host; an
# IDDOFF switch that takes the oracle's own EP(INB)<1e-4 skip of the
# downdraught (used for the downdraught-off fixture); COEFFR/COEFFS and ICOLD
# switches, set to the oracle's values by make_fixture.py.  driver.f90 reads
# columns from text and writes outputs.  gfortran 8.5.0, fp64, -O2
# -fno-fast-math (the audit's flags) was used for the committed fixture.
#
# Provenance (#2158 ask 6): the oracle is the sha256-pinned audited copy
# above, which is NOT in git.  Regenerating the fixture needs a copy with
# exactly that sha256; a different file (including climt's convect43c.f90,
# a reading copy of the same algorithm, not this oracle) is refused by the
# check below.  CI never rebuilds the oracle: the tests load the committed
# fixture tests/atmosphere/oracle_fixtures/emanuel_convect43c.npz and assert
# the sha256 recorded in it.
set -euo pipefail
SRC=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
mkdir -p "$2"
B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
echo "3eb50b9cf65817966ab89c56fbbdaf09fa546056d7f58b2ceb8564a4c254cc23  $SRC" | sha256sum -c -
cd "$B"
cp "$HERE/emdbg.inc" .
python3 "$HERE/patch_oracle.py" "$SRC" convect43c_patched.f
FF="-O2 -fno-fast-math -ffixed-line-length-132 -ffree-line-length-none -fdefault-real-8 -fdefault-double-8"
gfortran $FF -c convect43c_patched.f -o conv8.o
gfortran $FF -c "$HERE/driver.f90" -o drv8.o
gfortran -o em_driver_r8.x conv8.o drv8.o
echo "FLAGS: $FF"
echo BUILD_OK
