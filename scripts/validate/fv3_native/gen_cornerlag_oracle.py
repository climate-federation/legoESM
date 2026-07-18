"""Generate the corner-Lagrange oracle fixture.

Certifies `_CornerLagrange` (fv3_native_ext_vector) — weights AND the
nine-slot fill sequence at all four staggerings — against the verbatim
Fortran `compute_lagrange_coeff` + `fill_corner_region_2d`
(fv3_cornerlag_extract.F90).  This directly tests the signed-arc-
coordinate Lagrange-weight equivalence claim (gnomonic lines are great
circles) against upstream's great-circle-distance-ratio products.

Usage: gen_cornerlag_oracle.py --n 12 --ng 3 --build-dir DIR [--tile 1]
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

_STAGS = ((0, 0), (1, 1), (0, 1), (1, 0))
_FLD = {(0, 0): "F00", (1, 1): "F11", (0, 1): "F01", (1, 0): "F10"}


def _stag_lonlat(n, ng, istag, jstag, tile):
    """EXT lon/lat at the (istag,jstag) supergrid parity, ref layout."""
    from legoesm.grids.fv3_native_halos import _ED_CARTS, _ed_line

    line = _ed_line(n, 2 * (ng + 2))
    iv = np.array([line[2 * i - istag]
                   for i in range(1 - ng, n + istag + ng + 1)])
    jv = np.array([line[2 * j - jstag]
                   for j in range(1 - ng, n + jstag + ng + 1)])
    xg, yg = np.meshgrid(iv, jv, indexing="ij")
    cx, cy, cz = _ED_CARTS[tile - 1](xg, yg)
    r = np.sqrt(cx * cx + cy * cy + cz * cz)
    lon = np.mod(np.arctan2(cy, cx), 2.0 * np.pi)
    lat = np.arcsin(cz / r)
    return lon, lat


def _smooth(lon, lat):
    return np.sin(lat) * 40.0 + 12.0 * np.cos(lon) * np.cos(lat)


def serialize_inputs(n, ng, a_lon_w, a_lat_w, fields) -> str:
    lines = [f"{n} {ng}"]
    # a_pt one ring wider: fort indices 1-(ng+1) .. n+(ng+1);
    # numpy index i maps to fort i + (1 - (ng + 1))
    lo_w = 1 - (ng + 1)
    m = a_lon_w.shape[0]
    for i in range(m):
        for j in range(m):
            lines.append(f"APT {i + lo_w} {j + lo_w} "
                         f"{a_lon_w[i, j]:.17e} {a_lat_w[i, j]:.17e}")
    for (istag, jstag), f in fields.items():
        tagf = _FLD[(istag, jstag)]
        ni, nj = f.shape
        for i in range(ni):
            for j in range(nj):
                lines.append(f"{tagf} {i + 1 - ng} {j + 1 - ng} "
                             f"{f[i, j]:.17e} 0.0")
    lines.append("COEF 0 0 0 0")
    for (istag, jstag) in _STAGS:
        lines.append(f"RUN {istag} {jstag} 0 0")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--tile", type=int, default=1)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--out", default=str(
        REPO / "tests/grids/fixtures/fv3_cornerlag_oracle.npz"))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO / "packages/core"))

    n, ng, tile = args.n, args.ng, args.tile
    # a_pt on the ONE-RING-WIDER A lattice (upstream dg%bd analog)
    a_lon_w, a_lat_w = _stag_lonlat(n, ng + 1, 0, 0, tile)

    fields = {}
    for (istag, jstag) in _STAGS:
        lon, lat = _stag_lonlat(n, ng, istag, jstag, tile)
        fields[(istag, jstag)] = _smooth(lon, lat)

    text = serialize_inputs(n, ng, a_lon_w, a_lat_w, fields)
    sha = hashlib.sha256(text.encode()).hexdigest()

    bd = Path(args.build_dir)
    bd.mkdir(parents=True, exist_ok=True)
    src = REPO / "scripts/validate/fv3_native"
    extract_sha = hashlib.sha256(
        (src / "fv3_cornerlag_extract.F90").read_bytes()).hexdigest()
    subprocess.run(
        ["gfortran", "-O0", "-g", "-fdefault-real-8", "-fdefault-double-8",
         "-o", str(bd / "cornerlag_driver"),
         str(src / "fv3_cornerlag_shim.F90"),
         str(src / "fv3_cornerlag_extract.F90"),
         str(src / "fv3_cornerlag_driver.F90")],
        check=True, cwd=bd)
    run = subprocess.run([str(bd / "cornerlag_driver")], input=text,
                         capture_output=True, text=True, check=True)

    outs: dict = {}
    for line in run.stdout.splitlines():
        p = line.split()
        if p[0] != "OUT":
            continue
        key = f"out_{p[1]}{p[2]}"
        outs.setdefault(key, []).append(
            [int(p[3]), int(p[4]), float(p[5])])
    np.savez_compressed(
        args.out, n=n, ng=ng, tile=tile,
        input_sha256=sha, extract_sha256=extract_sha,
        **{k: np.array(v) for k, v in outs.items()},
        **{f"fld_{istag}{jstag}": fields[(istag, jstag)]
           for (istag, jstag) in _STAGS},
        lineage=("verbatim compute_lagrange_coeff + fill_corner_region_2d "
                 "(Zenodo 8327578 symmetryclean fv_duogrid.F90:1719-1903,"
                 "2159-2272,2369-2449 + great_circle_dist); a_pt on the "
                 "ng+1 lattice per the duogrid_alloc bounds split; "
                 "-fdefault-real-8"))
    print("saved", args.out, "input sha", sha[:16],
          "extract sha", extract_sha[:16])


if __name__ == "__main__":
    main()
