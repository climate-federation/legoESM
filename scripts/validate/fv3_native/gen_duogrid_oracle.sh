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
python - "$WORKDIR" "$REPO_ROOT/tests/grids/fixtures/fv3_duogrid_oracle.npz" <<'EOF'
import sys
import numpy as np

workdir, out_path = sys.argv[1], sys.argv[2]
arrays = {}
for res in (12, 24):
    # k2e tables: rec stag n i j loc coef(1:nord). Tile-symmetric (verified
    # at pack time); store tile 1 only, keyed by stagger.
    per_stag = {}
    nord = None
    for line in open(f"{workdir}/duogrid_k2e_c{res}.txt"):
        if line.startswith("#"):
            if "k2e_nord =" in line:
                nord = int(line.split()[-1])
            continue
        p = line.split()
        stag, tile, i, j, loc = p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])
        coef = np.array([float(x) for x in p[5:]])
        per_stag.setdefault(stag, {}).setdefault(tile, {})[(i, j)] = (loc, coef)
    assert nord is not None
    for stag, tiles in per_stag.items():
        t1 = tiles[1]
        for t, recs in tiles.items():
            for key, (loc, coef) in recs.items():
                l1, c1 = t1[key]
                assert l1 == loc and np.abs(c1 - coef).max() < 1e-13, (
                    f"tile asymmetry {stag} {t} {key}")
        keys = np.array(sorted(t1), dtype=np.int64)
        locs = np.array([t1[tuple(k)][0] for k in keys], dtype=np.int64)
        coefs = np.stack([t1[tuple(k)][1] for k in keys])
        arrays[f"k2e_{stag}_ij_c{res}"] = keys
        arrays[f"k2e_{stag}_loc_c{res}"] = locs
        arrays[f"k2e_{stag}_coef_c{res}"] = coefs
    arrays[f"k2e_nord_c{res}"] = np.array(nord)

    # ext/kik supergrid coords (tile 1; verified tile-symmetric)
    data = np.loadtxt(f"{workdir}/duogrid_coords_c{res}.txt")
    t1 = data[data[:, 0] == 1]
    ii = t1[:, 1].astype(int)
    jj = t1[:, 2].astype(int)
    lo = ii.min()
    size = ii.max() - lo + 1
    for col, name in ((3, "ext_x"), (4, "ext_y"), (5, "kik_x"), (6, "kik_y")):
        a = np.full((size, size), np.nan)
        a[ii - lo, jj - lo] = t1[:, col]
        arrays[f"duo_{name}_c{res}"] = a
    arrays[f"duo_index_lo_c{res}"] = np.array(lo)
np.savez_compressed(out_path, **arrays)
print(f"wrote {out_path}")
EOF
