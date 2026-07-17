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

# -fdefault-real-8 is REQUIRED: great_circle_dist returns default `real`
# upstream (FV3 production builds compile with r8 default reals).
gfortran -O2 -fdefault-real-8 -J "$WORKDIR" -o "$WORKDIR/fv3_gnomonic_oracle" "$HERE/fv3_gnomonic_oracle.f90"
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

    # init_grid/grid_area metrics (unit sphere): fieldid f i j value
    shapes = {
        1: ("area",   (6, im, im)),
        2: ("dx",     (6, im, im + 1)),
        3: ("dy",     (6, im + 1, im)),
        4: ("dxa",    (6, im, im)),
        5: ("dya",    (6, im, im)),
        6: ("dxc",    (6, im + 1, im)),
        7: ("dyc",    (6, im, im + 1)),
        8: ("area_c", (6, im + 1, im + 1)),
    }
    fields = {fid: np.zeros(shp) for fid, (_, shp) in shapes.items()}
    data = np.loadtxt(f"{workdir}/fv3_metrics_c{im}.txt")
    for fid, f, i, j, v in data.reshape(-1, 5):
        fields[int(fid)][int(f) - 1, int(i) - 1, int(j) - 1] = v
    for fid, (name, _) in shapes.items():
        arrays[f"{name}_unit_c{im}"] = fields[fid]

    # grid_utils_init angle fields: fieldid f i j value (dimensionless)
    ang_shapes = {}
    for ip in range(1, 10):
        ang_shapes[10 + ip] = (f"cos_sg{ip}", (6, im, im))
    ang_shapes.update({
        20: ("cosa_u", (6, im + 1, im)),
        21: ("sina_u", (6, im + 1, im)),
        22: ("rsin_u", (6, im + 1, im)),
        23: ("cosa_v", (6, im, im + 1)),
        24: ("sina_v", (6, im, im + 1)),
        25: ("rsin_v", (6, im, im + 1)),
        26: ("cosa_s", (6, im, im)),
        27: ("rsin2", (6, im, im)),
        28: ("cosa_b", (6, im + 1, im + 1)),
        29: ("sina_b", (6, im + 1, im + 1)),
        30: ("rsina_b", (6, im + 1, im + 1)),
    })
    ang = {fid: np.zeros(shp) for fid, (_, shp) in ang_shapes.items()}
    data = np.loadtxt(f"{workdir}/fv3_angles_c{im}.txt")
    for fid, f, i, j, v in data.reshape(-1, 5):
        ang[int(fid)][int(f) - 1, int(i) - 1, int(j) - 1] = v
    for fid, (name, _) in ang_shapes.items():
        arrays[f"{name}_c{im}"] = ang[fid]
np.savez_compressed(out_path, **arrays)
print(f"wrote {out_path}")
EOF
