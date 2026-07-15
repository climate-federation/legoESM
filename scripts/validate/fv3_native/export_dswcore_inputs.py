#!/usr/bin/env python
"""Export the phase-4b d_sw one-step oracle inputs (dswcore_input.txt).

Builds the single-tile gridstruct (all d_sw-consumed fields) and the
analytic SW state, runs the CERTIFIED python c_sw once to produce the
time-centred uc/vc (and ua/va), and dumps everything as Fortran-index
text records for ``fv3_dswcore_oracle_driver``.  Feeding the c_sw outputs
as explicit inputs keeps both oracle sides byte-identical — re-deriving
them in Fortran could flip PPM sign branches on last-bit differences.

Also writes ``dswcore_input.npz`` for the python-side reconciliation test.

Usage: export_dswcore_inputs.py [--res 12] [--ng 3] [--dt 225.0]
                                [--outdir .]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "..", "packages", "core"))

from legoesm.grids.fv3_native_gridstruct import (  # noqa: E402
    FV3_OMEGA,
    FV3_RADIUS_M,
    analytic_swcore_state,
    build_fv3_native_gridstruct,
)

_GS_2D = (
    ("RAREA", "rarea"), ("AREA", "area"),
    ("AREA_C", "area_c"), ("RAREA_C", "rarea_c"),
    ("DXA", "dxa"), ("DYA", "dya"), ("RDXA", "rdxa"), ("RDYA", "rdya"),
    ("COSA_S", "cosa_s"), ("RSIN2", "rsin2"),
    ("DX", "dx"), ("DY", "dy"), ("RDX", "rdx"), ("RDY", "rdy"),
    ("DXC", "dxc"), ("DYC", "dyc"), ("RDXC", "rdxc"), ("RDYC", "rdyc"),
    ("COSA_U", "cosa_u"), ("SINA_U", "sina_u"), ("RSIN_U", "rsin_u"),
    ("COSA_V", "cosa_v"), ("SINA_V", "sina_v"), ("RSIN_V", "rsin_v"),
    ("COSA", "cosa"), ("SINA", "sina"), ("RSINA", "rsina"),
    ("FC", "fC"), ("F0", "f0"),
    ("DIVG_U", "divg_u"), ("DIVG_V", "divg_v"),
    ("DEL6_U", "del6_u"), ("DEL6_V", "del6_v"),
    ("GRID_LON", "grid_lon"), ("GRID_LAT", "grid_lat"),
    ("AGRID_LON", "agrid_lon"), ("AGRID_LAT", "agrid_lat"),
)
_EDGES = (("EDGE_S", "edge_s"), ("EDGE_N", "edge_n"),
          ("EDGE_W", "edge_w"), ("EDGE_E", "edge_e"))
_STATE = (("DELP", "delp"), ("PT", "pt"), ("W", "w"),
          ("U", "u"), ("V", "v"),
          ("UC", "uc"), ("VC", "vc"), ("UA", "ua"), ("VA", "va"),
          ("DIVGD_IN", "divg_d_in"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=12)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--dt", type=float, default=225.0)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    from legoesm.core.fv3_native_sw_core import Bounds, c_sw
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_bgrid_scalar_halos,
        exchange_cgrid_vector_halos,
    )

    bd = Bounds.single_tile(args.res, args.ng)

    # run the certified python c_sw on ALL SIX faces: dyn_core exchanges
    # uc/vc (CGRID_NE vector) AND divgd (CORNER scalar) between c_sw and
    # d_sw, so the deep halos carry the NEIGHBOUR faces' c_sw results.
    uc6, vc6, divg6 = [], [], []
    gs = st = None
    for t in range(1, 7):
        gs_t = build_fv3_native_gridstruct(args.res, args.ng, tile=t,
                                           radius=FV3_RADIUS_M,
                                           omega=FV3_OMEGA)
        st_t = analytic_swcore_state(gs_t)
        csw_t = c_sw(delp=st_t["delp"], pt=st_t["pt"],
                     w=np.zeros_like(st_t["delp"]),
                     u=st_t["u"], v=st_t["v"], gs=gs_t, bd=bd,
                     npx=args.res + 1, npy=args.res + 1,
                     dt2=0.5 * args.dt, nord=1,
                     hydrostatic=True, dord4=True, grid_type=0)
        for key in ("uc", "vc", "ua", "va"):
            a = np.asarray(csw_t[key], dtype=np.float64).copy()
            a[~np.isfinite(a)] = 0.0
            csw_t[key] = a
        uc6.append(csw_t["uc"])
        vc6.append(csw_t["vc"])
        divg6.append(np.asarray(csw_t["divg_d"], dtype=np.float64).copy())
        if t == 1:
            gs, st, csw = gs_t, st_t, csw_t

    exchange_cgrid_vector_halos(uc6, vc6, 1, args.res, args.ng)
    # divg_d halo: the CORNER scalar exchange fills tile-1's B-node halo
    # from the neighbour faces (dyn_core.F90:451/577).  d_sw (nord=1) reads
    # divg_d(0,npx+1, 1:npx) and (1:npx, 0,npy+1) in the divergence-damping
    # n-loop, so a zeroed halo is materially wrong (codex p4b r2).
    exchange_bgrid_scalar_halos(divg6, 1, args.res, args.ng)

    state = dict(st)
    state["w"] = np.zeros_like(st["delp"])
    state["uc"] = uc6[0]
    state["vc"] = vc6[0]
    state["ua"] = csw["ua"]
    state["va"] = csw["va"]
    # real c_sw divergence (compute + exchanged halo); the remaining
    # corner-diagonal region (unread by d_sw at nord=1) stays NaN->0.
    divg_in = divg6[0].copy()
    divg_in[~np.isfinite(divg_in)] = 0.0
    state["divg_d_in"] = divg_in

    lo = 1 - args.ng
    txt = os.path.join(args.outdir, "dswcore_input.txt")
    npz = os.path.join(args.outdir, "dswcore_input.npz")
    with open(txt, "w") as f:
        f.write(f"# res {args.res}\n# ng {args.ng}\n")
        f.write(f"# dt {args.dt!r}\n")

        def dump(name: str, a: np.ndarray):
            ni, nj = a.shape[:2]
            for i in range(ni):
                for j in range(nj):
                    if a.ndim == 2:
                        f.write(f"{name} {i + lo} {j + lo} "
                                f"{a[i, j]:.17e}\n")
                    else:
                        for k in range(a.shape[2]):
                            f.write(f"{name} {i + lo} {j + lo} {k + 1} "
                                    f"{a[i, j, k]:.17e}\n")

        for name, key in _GS_2D:
            dump(name, gs[key])
        dump("SIN_SG", gs["sin_sg"])
        dump("COS_SG", gs["cos_sg"])
        for name, key in _EDGES:
            arr = gs[key]
            for i in range(arr.shape[0]):
                f.write(f"{name} {i + 1} {arr[i]:.17e}\n")
        f.write(f"DA_MIN {gs['da_min']:.17e}\n")
        f.write(f"DA_MIN_C {gs['da_min_c']:.17e}\n")
        for name, key in _STATE:
            dump(name, state[key])

    save = {k: gs[k] for _, k in _GS_2D}
    save.update({k: gs[k] for _, k in _EDGES})
    save.update({k: state[k] for _, k in _STATE})
    save.update({"sin_sg": gs["sin_sg"], "cos_sg": gs["cos_sg"],
                 "da_min": gs["da_min"], "da_min_c": gs["da_min_c"],
                 "res": args.res, "ng": args.ng, "dt": args.dt})
    np.savez_compressed(npz, **save)
    print(f"wrote {txt} and {npz}")


if __name__ == "__main__":
    main()
