#!/usr/bin/env python
"""Dump OUR ext_vector D-grid pipeline stage arrays on the case-8 IC.

Companion to the instrumented Zenodo model (fv_duogrid.F90 ext_vector
stage dumps S0-S6, first D-grid call = dyn_core entry on the IC): runs
the same first D-vector exchange from the same case-8 two-burst IC on
the bounded-conventions six-face context and writes every stage array
plus the static operands (a11..a22, dx, dy, sin_sg5, agrid) to one npz.

compare_extvec_stages.py diffs the two sides stage by stage; the first
divergent stage localizes the live-path defect (codex fix-advice
rank-2).

Usage: extvec_stage_dump.py --n 48 --out ours_extvec.npz
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pathlib import Path
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[3]))   # repo root: tests.test_cases
    from tests.test_cases.colliding_modons import _MODON_H0, _modon_winds_geo

    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.grids.fv3_native_ext_vector import (
        center_a_matrix,
        ext_vector_dgrid_sixface,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_RADIUS_M,
        analytic_swcore_state,
    )

    ctx = build_six_face_duo_context(args.n, 3, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)

    # FV3 constants_mod GRAV (case-8 delp = 5000*Grav)
    # const-ok: oracle pins upstream's gravity, not legoESM's
    fv3_grav = 9.80665

    def wind_fn(ll):
        u_e, v_n = _modon_winds_geo(ll[..., 0], ll[..., 1], FV3_RADIUS_M)
        return np.asarray(u_e), np.asarray(v_n)

    def scalars_fn(ll):
        delp = np.full(ll.shape[:-1], fv3_grav * _MODON_H0)
        return delp, np.ones_like(delp)

    u6, v6 = [], []
    for gs in ctx["gs6"]:
        st = analytic_swcore_state(gs, wind_fn=wind_fn,
                                   scalars_fn=scalars_fn)
        u6.append(np.array(st["u"], copy=True))
        v6.append(np.array(st["v"], copy=True))

    out = {}

    def hook(stage, t, name, arr):
        out[f"{stage}_t{t + 1}_{name}"] = np.array(arr, copy=True)

    ectx = ctx["ectx"]
    ectx["stage_dump"] = hook
    ext_vector_dgrid_sixface(u6, v6, ectx)

    # static operands (S0), full data domain, per tile
    for t, gs in enumerate(ctx["gs6"]):
        a11, a12, a21, a22 = center_a_matrix(gs)
        for nm, a in (("a11", a11), ("a12", a12), ("a21", a21),
                      ("a22", a22)):
            out[f"S0_t{t + 1}_{nm}"] = a
        out[f"S0_t{t + 1}_dx"] = np.asarray(gs["dx"])
        out[f"S0_t{t + 1}_dy"] = np.asarray(gs["dy"])
        out[f"S0_t{t + 1}_sinsg5"] = np.asarray(gs["sin_sg"][..., 4])
        out[f"S0_t{t + 1}_aglon"] = np.asarray(gs["agrid_lon"])
        out[f"S0_t{t + 1}_aglat"] = np.asarray(gs["agrid_lat"])

    root = here.parents[3]
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status",
                            "--porcelain"], capture_output=True,
                           text=True).stdout.strip()
    out["git_sha"] = np.array(sha + ("-dirty" if dirty else ""))
    out["n"] = np.array(args.n)
    out["ng"] = np.array(ctx["ng"])
    out["protocol"] = np.array(
        "bounded-conventions ctx, use_ext_bundle, case-8 two-burst IC "
        "(tests.test_cases.colliding_modons, FV3 GRAV delp), first "
        "D-grid ext_vector call; numpy index [p,q] = Fortran "
        "(p+1-ng_lattice, q+1-ng_lattice); model lattice ng=3, "
        "geographic p1 lattice ng=4")
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out} ({len(out)} arrays)", flush=True)


if __name__ == "__main__":
    main()
