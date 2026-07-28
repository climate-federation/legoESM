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
python3 - "$WORKDIR" "$REPO_ROOT/tests/grids/fixtures/fv3_duogrid_oracle_n2.npz" << 'PACKEOF'
import sys

import numpy as np

wd, out = sys.argv[1], sys.argv[2]
data = {}
for res in (12, 24, 48):
    fams = {}
    with open(f"{wd}/duogrid_k2e_n2_c{res}.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            fams.setdefault(p[0], []).append(
                (int(p[1]), int(p[2]), int(p[3]), int(p[4]),
                 [float(x) for x in p[5:]]))
    for fam, rows in fams.items():
        t1 = [r for r in rows if r[0] == 1]
        data[f"c{res}_{fam}_ij"] = np.array([(r[1], r[2]) for r in t1],
                                            dtype=np.int64)
        data[f"c{res}_{fam}_loc"] = np.array([r[3] for r in t1],
                                             dtype=np.int64)
        data[f"c{res}_{fam}_coef"] = np.array([r[4] for r in t1])
    data[f"c{res}_k2e_nord"] = np.array(
        len(next(iter(fams.values()))[0][4]))
np.savez_compressed(out, **data)
print("packed", out, len(data), "arrays")
PACKEOF
echo "AUTH fixture written."
