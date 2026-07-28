#!/usr/bin/env bash
# NORD-2 AUTHORITATIVE fixture (2026-07-27 root cause): the live
# model runs k2e_nord=2; this variant runs the auth generator AT ITS
# OWN DEFAULTS (driver_n2 FATALs if the tree default is not 2) at
# C12/C24/C48.  The forced-4 fixture (gen_duogrid_oracle_auth.sh)
# remains as the named HISTORICAL regression fixture.
# Regenerate the phase-3 duogrid halo oracle fixture from the
# AUTHORITATIVE modular global_grid tree (Zenodo 8327578 symmetryclean,
# tools/global_grid/*.F90 — Xi Chen's purpose-built generator with the
# dedicated global_grid_gen_k2e.F90) instead of the luanfs mirror
# monolith.  Same shim + same driver + same pack format; output
#   tests/grids/fixtures/fv3_duogrid_oracle_n2.npz
# The re-verify test compares it table-by-table against the
# mirror-pinned fv3_duogrid_oracle.npz (the phase-3 provenance caveat).
#
# Build flags identical to gen_duogrid_oracle.sh (see its header for
# the -DNO_QUAD_PRECISION rationale).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
AUTH="${DUOGRID_AUTH:-/burg-archive/glab/users/pg2328/Code/FV3/duogrid_symmetryclean/atmos_cubed_sphere-symmetryclean/tools/global_grid}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

FF="-O2 -fdefault-real-8 -fdefault-double-8 -cpp -DNO_QUAD_PRECISION"
OBJS=""
gfortran $FF -J "$WORKDIR" -c "$HERE/duogrid_shim.F90" -o "$WORKDIR/shim.o"
OBJS="$WORKDIR/shim.o"
# dependency order (module use graph: lib -> data -> util/alloc ->
# gen_* -> base -> global_grid)
for f in lib_grid.F90 global_grid_data.F90 global_grid_util.F90 \
         global_grid_alloc.F90 global_grid_gen_lonlat.F90 \
         global_grid_gen_coords.F90 global_grid_gen_cell.F90 \
         global_grid_gen_vec.F90 global_grid_gen_k2e.F90 \
         global_grid_base.F90 global_grid.F90; do
  gfortran $FF -J "$WORKDIR" -c "$AUTH/$f" -o "$WORKDIR/${f%.F90}.o"
  OBJS="$OBJS $WORKDIR/${f%.F90}.o"
done
gfortran $FF -J "$WORKDIR" -o "$WORKDIR/duogrid_oracle" \
    "$HERE/duogrid_oracle_driver_n2.F90" $OBJS
(cd "$WORKDIR" && ./duogrid_oracle)

mkdir -p "$REPO_ROOT/tests/grids/fixtures"
python - "$WORKDIR" \
    "$REPO_ROOT/tests/grids/fixtures/fv3_duogrid_oracle_n2.npz" \
    < "$HERE/_pack_duogrid_oracle.py"
echo "AUTH fixture written."
