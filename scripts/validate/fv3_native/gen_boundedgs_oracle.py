"""Generate the BOUNDED-conventions gridstruct oracle fixture.

Feeds the extended own-face ED B-lattice (the duo gen_k2e lattice;
--halo-mode) to the verbatim bounded-branch metric+angle init
(bounded_domain=.true. — PROVEN the duo lane by the Zenodo C48 logs:
duo da_min_c=23543093086.1030 differs from the plain run's
23335991574.8811, so duo grid init takes the bounded arms and consumes
real wedge-halo geometry) and packs every dumped field.

Usage: gen_boundedgs_oracle.py --n 12 --ng 3 [--tile 1]
       [--halo-mode extended|kinked]
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--tile", type=int, default=1)
    ap.add_argument("--halo-mode", default="extended",
                    choices=("extended", "kinked"),
                    help="B-lattice halo convention fed to the bounded "
                         "init.  'extended' = own-face ED gnomonic "
                         "continuation (ext_parity_lonlat_ref; the duo "
                         "gen_k2e lattice — real values everywhere incl "
                         "corner wedges).  PROVEN convention: the C48 "
                         "Zenodo duo run prints da_min_c distinct from "
                         "the plain run (23543093086.1030 vs "
                         "23335991574.8811), so duo's bounded arms "
                         "consumed real wedge agrid — only the extended "
                         "lattice supplies that.  'kinked' = mpp-state "
                         "side strips + sentinel wedges (diagnostic "
                         "only; poisons vertex area_c).")
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--out", default=str(
        REPO / "tests/grids/fixtures/fv3_boundedgs_oracle.npz"))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO / "packages/core"))

    n, ng = args.n, args.ng
    if args.halo_mode == "extended":
        from legoesm.grids.fv3_native_ext_vector import ext_parity_lonlat_ref
        lon6, lat6 = ext_parity_lonlat_ref(n, ng, "B")
        lon, lat = lon6[args.tile - 1], lat6[args.tile - 1]
    else:
        from legoesm.grids.fv3_native_gridstruct import (
            build_kinked_corner_lonlat,
        )
        lon, lat = build_kinked_corner_lonlat(n, ng, tile=args.tile)
    lines = [f"{n} {ng}"]
    m = n + 2 * ng + 1
    lo = 1 - ng
    for i in range(m):
        for j in range(m):
            lines.append(f"GRD {i + lo} {j + lo} "
                         f"{lon[i, j]:.17e} {lat[i, j]:.17e}")
    text = "\n".join(lines) + "\n"
    sha = hashlib.sha256(text.encode()).hexdigest()

    bd = Path(args.build_dir)
    bd.mkdir(parents=True, exist_ok=True)
    src = REPO / "scripts/validate/fv3_native"
    ff = ["-O0", "-g", "-fdefault-real-8", "-fdefault-double-8",
          "-cpp", "-DDYCORE_SOLO", "-ffree-line-length-none"]
    subprocess.run(
        ["gfortran", *ff, "-o", str(bd / "boundedgs_driver"),
         str(src / "fv3_boundedgs_shim.F90"),
         str(src / "fv3_boundedgs_extract.F90"),
         str(src / "fv3_boundedgs_driver.F90")],
        check=True, cwd=bd)
    run = subprocess.run([str(bd / "boundedgs_driver")], input=text,
                         capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stdout[-2000:])
        print(run.stderr[-2000:])
        raise SystemExit(f"driver rc={run.returncode}")

    known = {"DX", "DY", "DXA", "DYA", "DXC", "DYC", "AREA", "ARC",
             "SINA", "COSA", "RSNA", "RSN2", "RSNU", "RSNV", "CSAU",
             "CSAV", "CSAS", "SNAU", "SNAV", "DVGU", "DVGV", "DL6U",
             "DL6V", "SSG", "CSG", "AGX", "AGY", "DAMN",
             "RDX", "RDY", "RDXC", "RDYC", "RARA", "RDXA", "RDYA",
             "RARC"}
    recs: dict = {}
    for line in run.stdout.splitlines():
        p = line.split()
        if not p or p[0] not in known:
            continue                    # upstream is_master diagnostics
        if p[0] in ("SSG", "CSG"):
            recs.setdefault(f"{p[0].lower()}_{p[3]}", []).append(
                [int(p[1]), int(p[2]), float(p[4])])
        elif p[0] == "DAMN":
            recs["da_min"] = [[0, 0, float(p[1])]]
            recs["da_min_c"] = [[0, 0, float(p[2])]]
        else:
            recs.setdefault(p[0].lower(), []).append(
                [int(p[1]), int(p[2]), float(p[3])])
    for line in run.stdout.splitlines():
        if "da_max" in line or "Orthogonal" in line or "Aspect" in line:
            print("[oracle]", line.strip())
    np.savez_compressed(
        args.out, n=n, ng=ng, tile=args.tile, halo_mode=args.halo_mode,
        input_sha256=sha,
        extract_sha256=hashlib.sha256(
            (src / "fv3_boundedgs_extract.F90").read_bytes()).hexdigest(),
        **{k: np.array(v) for k, v in recs.items()},
        lineage=("BOUNDED-conventions gridstruct: verbatim grid_utils_init "
                 "(symmetryclean, bounded_domain=T) + fv_grid_tools metric "
                 "loops (plain clone, documented substitution) + bounded "
                 "grid_area arms; B-lattice input per halo_mode (extended "
                 "= own-face ED gnomonic continuation, the duo gen_k2e "
                 "lattice); sorted_inta = plain-order documented "
                 "substitution (LIVE under bounded, ULP-class; C48 log "
                 "match 1.5e-12 rel); C48 cross-check targets from the "
                 "Zenodo duo run fms.out: da_max/da_min=2.26548304435260, "
                 "da_max_c=53362939642.3731, da_min_c=23543093086.1030"))
    print("saved", args.out, "input sha", sha[:16])


if __name__ == "__main__":
    main()
