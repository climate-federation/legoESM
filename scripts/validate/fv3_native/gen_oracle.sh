#!/usr/bin/env bash
# Regenerate the FV3 gnomonic_ed source-derived oracle fixture.
#
# Compiles the verbatim FV3 grid-generation extraction
# (fv3_gnomonic_oracle.f90), runs it (< 1 s, single core), and packs the
# C1/C8/C36 corner coordinates into the tracked test fixture
#   tests/grids/fixtures/fv3_gnomonic_ed_oracle.npz
#
# Usage (from the repo root):
#   bash scripts/validate/fv3_native/gen_oracle.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

gfortran -O2 -J "$WORKDIR" -o "$WORKDIR/fv3_gnomonic_oracle" "$HERE/fv3_gnomonic_oracle.f90"
(cd "$WORKDIR" && ./fv3_gnomonic_oracle)

mkdir -p "$REPO_ROOT/tests/grids/fixtures"
python - "$WORKDIR" "$REPO_ROOT/tests/grids/fixtures/fv3_gnomonic_ed_oracle.npz" <<'EOF'
import sys
import numpy as np

workdir, out_path = sys.argv[1], sys.argv[2]
arrays = {}
for im in (1, 8, 36):
    # face-1 panel corners: (im+1, im+1)
    data = np.loadtxt(f"{workdir}/fv3_gnomonic_ed_c{im}.txt")
    n = im + 1
    lon = np.zeros((n, n))
    lat = np.zeros((n, n))
    for i, j, lo, la in data.reshape(-1, 4):
        lon[int(i) - 1, int(j) - 1] = lo
        lat[int(i) - 1, int(j) - 1] = la
    arrays[f"lon_c{im}"] = lon
    arrays[f"lat_c{im}"] = lat

    # mirror_grid 6-face corners: (6, im+1, im+1)
    data = np.loadtxt(f"{workdir}/fv3_gnomonic_ed_faces_c{im}.txt")
    lon6 = np.zeros((6, n, n))
    lat6 = np.zeros((6, n, n))
    for f, i, j, lo, la in data.reshape(-1, 5):
        lon6[int(f) - 1, int(i) - 1, int(j) - 1] = lo
        lat6[int(f) - 1, int(i) - 1, int(j) - 1] = la
    arrays[f"lon6_c{im}"] = lon6
    arrays[f"lat6_c{im}"] = lat6

    # cell_center2 A-grid centres: (6, im, im)
    data = np.loadtxt(f"{workdir}/fv3_gnomonic_ed_agrid_c{im}.txt")
    clon = np.zeros((6, im, im))
    clat = np.zeros((6, im, im))
    for f, i, j, lo, la in data.reshape(-1, 5):
        clon[int(f) - 1, int(i) - 1, int(j) - 1] = lo
        clat[int(f) - 1, int(i) - 1, int(j) - 1] = la
    arrays[f"agrid_lon_c{im}"] = clon
    arrays[f"agrid_lat_c{im}"] = clat
np.savez_compressed(out_path, **arrays)
print(f"wrote {out_path}")
EOF
