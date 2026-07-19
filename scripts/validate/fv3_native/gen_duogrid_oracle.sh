#!/usr/bin/env bash
# Regenerate the phase-3 duogrid halo oracle fixture.
#
# Compiles the VERBATIM duo-grid variant global_grid.F90 (mirror:
# luanfs/FV3_container @ 7d06431e, see duogrid_ref/SOURCE_COMMIT.txt)
# against the minimal shims, runs the driver (serial, < 5 s), and packs the
# k2e remap tables (all six staggers) + extended/kinked supergrid coords for
# C12 and C24 into
#   tests/grids/fixtures/fv3_duogrid_oracle.npz
#
# Reference tree default: /burg-archive/glab/users/pg2328/Code/FV3/duogrid_ref
# (override with DUOGRID_REF=...).
#
# Build flags: -fdefault-real-8 -fdefault-double-8 (r8-default production
# pair; real-8 alone promotes d0 literals to quad and breaks mixed-kind
# array constructors) and -DNO_QUAD_PRECISION (the module's double branch;
# the quad default trips a strict-gfortran kind bug in the same
# constructors — documented substitution, affects grid factors at ~1e-15).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
REF="${DUOGRID_REF:-/burg-archive/glab/users/pg2328/Code/FV3/duogrid_ref}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

FF="-O2 -fdefault-real-8 -fdefault-double-8 -cpp -DNO_QUAD_PRECISION"
gfortran $FF -J "$WORKDIR" -c "$HERE/duogrid_shim.F90" -o "$WORKDIR/shim.o"
gfortran $FF -J "$WORKDIR" -c "$REF/global_grid.F90" -o "$WORKDIR/gg.o"
gfortran $FF -J "$WORKDIR" -o "$WORKDIR/duogrid_oracle" \
    "$HERE/duogrid_oracle_driver.F90" "$WORKDIR/shim.o" "$WORKDIR/gg.o"
(cd "$WORKDIR" && ./duogrid_oracle)

mkdir -p "$REPO_ROOT/tests/grids/fixtures"
python - "$WORKDIR" "$REPO_ROOT/tests/grids/fixtures/fv3_duogrid_oracle.npz" < "$HERE/_pack_duogrid_oracle.py"
