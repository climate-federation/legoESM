#!/bin/bash
# Build the Vreman reference-Fortran oracle used to generate
# tests/atmosphere/oracle_fixtures/vreman.npz.
#
# usage: build.sh <Vreman_Subgridmodel_Fortran.txt> <build dir>
# The oracle is A.W. Vreman's own code, http://www.vremanresearch.nl/Vreman_Subgridmodel_Fortran.txt
# (unversioned; the copy used has sha256 below, downloaded 2026-10-19). The file is a
# template (pseudo-loop "do n= ....", undeclared arrays), so vreman_oracle.f carries its
# body verbatim with harness-only edits marked AUDIT. This script checks that every
# arithmetic line of vreman_oracle.f's kernel is a verbatim line of the downloaded file.
# Run on a compute node (sbatch), not a login node.
set -euo pipefail
SRC=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); mkdir -p "$2"; B=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=8a7efd8c64f63602746917e81c978ff71f442863cfdbcf4c2f378c0004598feb
echo "$SHA  $SRC" | sha256sum -c -
# verbatim check: the gradient-mapping block and the whole kernel block (b11= .. end if)
# of vreman_oracle.f must equal the downloaded file line for line.
for rng in '/d1v1=du1dx1/,/d3v3=du3dx3/p' '/b11=/,/end if/p'; do
  diff <(sed -n "$rng" "$HERE/vreman_oracle.f" | sed 's/[[:space:]]*$//') \
       <(sed -n "$rng" "$SRC" | sed 's/[[:space:]]*$//') > /dev/null \
    || { echo "NOT VERBATIM: $rng"; exit 1; }
done
gfortran --version | head -1
gfortran -O0 -fno-fast-math -ffixed-line-length-none -o "$B/vreman_oracle" "$HERE/vreman_oracle.f"
echo "BUILD_OK $B/vreman_oracle"
