#!/usr/bin/env python
"""Export the phase-4 c_sw one-step oracle inputs (swcore_input.txt).

Builds the single-tile FV3 gridstruct (certified builders + kinked
mpp-equivalent halos, ``legoesm.grids.fv3_native_gridstruct``) and the
analytic SW state, then dumps EVERY slot of every consumed array as
Fortran-indexed text records for ``fv3_swcore_oracle_driver``.  Also
writes ``swcore_input.npz`` so the python-side reconciliation test reads
byte-identical inputs.

Usage: export_swcore_inputs.py [--res 12] [--ng 3] [--dt2 112.5]
                               [--nord 1] [--outdir .]
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

# (record name, dict key, fortran lo bounds (ilo, jlo)) per 2-D field;
# ilo/jlo expressed via ng at runtime: "c" -> 1-ng (cell axis),
# "b" -> 1-ng (node axis has the same lower bound; only lengths differ)
_FIELDS_2D = (
    ("RAREA", "rarea"), ("DXA", "dxa"), ("DYA", "dya"),
    ("COSA_S", "cosa_s"), ("RSIN2", "rsin2"),
    ("DX", "dx"), ("DY", "dy"), ("DXC", "dxc"), ("DYC", "dyc"),
    ("RDXC", "rdxc"), ("RDYC", "rdyc"),
    ("COSA_U", "cosa_u"), ("SINA_U", "sina_u"), ("RSIN_U", "rsin_u"),
    ("COSA_V", "cosa_v"), ("SINA_V", "sina_v"), ("RSIN_V", "rsin_v"),
    ("RAREA_C", "rarea_c"), ("FC", "fC"),
)
_STATE_2D = (("DELP", "delp"), ("PT", "pt"), ("U", "u"), ("V", "v"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=12)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--dt2", type=float, default=112.5)
    ap.add_argument("--nord", type=int, default=1)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    gs = build_fv3_native_gridstruct(args.res, args.ng,
                                     radius=FV3_RADIUS_M, omega=FV3_OMEGA)
    st = analytic_swcore_state(gs)
    lo = 1 - args.ng   # every axis' Fortran lower bound

    txt = os.path.join(args.outdir, "swcore_input.txt")
    npz = os.path.join(args.outdir, "swcore_input.npz")
    with open(txt, "w") as f:
        f.write(f"# res {args.res}\n# ng {args.ng}\n")
        f.write(f"# nord {args.nord}\n# dt2 {args.dt2!r}\n")

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

        for name, key in _FIELDS_2D:
            dump(name, gs[key])
        dump("SIN_SG", gs["sin_sg"])
        dump("COS_SG", gs["cos_sg"])
        for name, key in _STATE_2D:
            dump(name, st[key])

    save = {k: gs[k] for _, k in _FIELDS_2D}
    save.update({k: st[k] for _, k in _STATE_2D})
    save.update({"sin_sg": gs["sin_sg"], "cos_sg": gs["cos_sg"],
                 "res": args.res, "ng": args.ng,
                 "dt2": args.dt2, "nord": args.nord})
    np.savez_compressed(npz, **save)
    print(f"wrote {txt} and {npz}")


if __name__ == "__main__":
    main()
