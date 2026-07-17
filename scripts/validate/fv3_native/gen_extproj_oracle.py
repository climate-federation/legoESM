"""Generate the ext-projection oracle fixture (independent basis cert).

Exports the ext A/B lattices (ext_parity_lonlat_ref — the dg%a_pt/b_pt
analogs) + analytic geographic winds, runs the verbatim Fortran
a2stag_metrics + cubed_a2d/a2c_halo driver, and packs the dumped bases
and projections into a fixture the pytest gate compares against
`_compute_ext_vectors_native` + `_a2d_project`/`_a2c_project`
(closing the shared-oracle P1: the Fortran side derives its OWN bases).

Usage: gen_extproj_oracle.py --n 12 --ng 4 --build-dir DIR [--tile 1]
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def serialize_inputs(n, ng, a_lon, a_lat, b_lon, b_lat, ug, vg) -> str:
    lines = [f"{n} {ng}"]
    m = n + 2 * ng
    lo = 1 - ng
    for i in range(m):
        for j in range(m):
            lines.append(f"APT {i + lo} {j + lo} "
                         f"{a_lon[i, j]:.17e} {a_lat[i, j]:.17e}")
    for i in range(m + 1):
        for j in range(m + 1):
            lines.append(f"BPT {i + lo} {j + lo} "
                         f"{b_lon[i, j]:.17e} {b_lat[i, j]:.17e}")
    for i in range(m):
        for j in range(m):
            lines.append(f"UG {i + lo} {j + lo} {ug[i, j]:.17e} 0.0")
            lines.append(f"VG {i + lo} {j + lo} {vg[i, j]:.17e} 0.0")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ng", type=int, default=4)
    ap.add_argument("--tile", type=int, default=1)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--out", default=str(
        REPO / "tests/grids/fixtures/fv3_extproj_oracle.npz"))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO / "packages/core"))
    from legoesm.grids.fv3_native_ext_vector import ext_parity_lonlat_ref

    n, ng = args.n, args.ng
    a_lon6, a_lat6 = ext_parity_lonlat_ref(n, ng, "A")
    b_lon6, b_lat6 = ext_parity_lonlat_ref(n, ng, "B")
    t = args.tile - 1
    a_lon, a_lat = a_lon6[t], a_lat6[t]
    b_lon, b_lat = b_lon6[t], b_lat6[t]
    u0 = 38.61068276698372
    ug = u0 * np.cos(a_lat)
    # a nonzero, asymmetric v discriminates every basis slot
    vg = 7.5 * np.sin(a_lon) * np.cos(a_lat)

    text = serialize_inputs(n, ng, a_lon, a_lat, b_lon, b_lat, ug, vg)
    sha = hashlib.sha256(text.encode()).hexdigest()

    bd = Path(args.build_dir)
    bd.mkdir(parents=True, exist_ok=True)
    src = REPO / "scripts/validate/fv3_native"
    extract_sha = hashlib.sha256(
        (src / "fv3_extproj_extract.F90").read_bytes()).hexdigest()
    subprocess.run(
        ["gfortran", "-O0", "-g", "-fdefault-real-8", "-fdefault-double-8",
         "-o", str(bd / "extproj_driver"),
         str(src / "fv3_extproj_shim.F90"),
         str(src / "fv3_extproj_extract.F90"),
         str(src / "fv3_extproj_driver.F90")],
        check=True, cwd=bd)
    run = subprocess.run([str(bd / "extproj_driver")], input=text,
                         capture_output=True, text=True, check=True)

    recs: dict = {}
    for line in run.stdout.splitlines():
        p = line.split()
        recs.setdefault(p[0], []).append([float(x) for x in p[1:]])
    out = {k: np.array(v) for k, v in recs.items()}
    np.savez_compressed(
        args.out, n=n, ng=ng, tile=args.tile,
        input_sha256=sha, extract_sha256=extract_sha,
        ug=ug, vg=vg,
        **{k.lower(): v for k, v in out.items()},
        lineage=("verbatim a2stag_metrics + cubed_a2d/a2c_halo "
                 "(Zenodo 8327578 symmetryclean fv_duogrid.F90:2590-2763,"
                 "2846-2992); shim fill_corner_region = identity "
                 "(fully-defined analytic inputs; corner-fill operator "
                 "certified separately); -fdefault-real-8"))
    print("saved", args.out, "input sha", sha[:16],
          "extract sha", extract_sha[:16])


if __name__ == "__main__":
    main()
