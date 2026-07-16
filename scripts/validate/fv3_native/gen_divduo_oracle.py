#!/usr/bin/env python
"""Gen divergence_corner_duo oracle inputs, or pack the fixture.

Two modes (no regeneration on pack — the fixture inputs, the hash, and the
Fortran output all come from ONE generation, so "bit-exact on identical
inputs" is actually enforced; codex p4c r2 P1):

  gen_divduo_oracle.py <W>          build inputs + run the python port;
                                    write divduo_input.txt (the Fortran
                                    driver reads it) + staging.npz.
  gen_divduo_oracle.py <W> --pack   read the SAME staging.npz + the
                                    first-run divduo_input.txt (hash it)
                                    + the driver's divduo_output.txt, and
                                    write the committed fixture npz.
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "..", "packages", "core"))

W = sys.argv[1]
PACK = len(sys.argv) > 2 and sys.argv[2] == "--pack"
res, ng = 12, 3
lo = 1 - ng
m_b = res + 2 * ng + 1
FIX = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                   "tests", "grids", "fixtures", "divduo_oracle_c12.npz")

if not PACK:
    from legoesm.core.fv3_native_duo_sw_core import divergence_corner_duo
    from legoesm.core.fv3_native_sw_core import Bounds, c_sw
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        analytic_swcore_state,
        build_fv3_native_gridstruct,
    )

    gs = build_fv3_native_gridstruct(res, ng, radius=FV3_RADIUS_M,
                                     omega=FV3_OMEGA)
    st = analytic_swcore_state(gs)
    bd = Bounds.single_tile(res, ng)
    csw = c_sw(delp=st["delp"], pt=st["pt"], w=np.zeros_like(st["delp"]),
               u=st["u"], v=st["v"], gs=gs, bd=bd, npx=res + 1, npy=res + 1,
               dt2=112.5, nord=1, hydrostatic=True, dord4=True, grid_type=0)
    ua = np.nan_to_num(np.asarray(csw["ua"], float), nan=0.0)
    va = np.nan_to_num(np.asarray(csw["va"], float), nan=0.0)
    u = np.nan_to_num(np.asarray(st["u"], float), nan=0.0)
    v = np.nan_to_num(np.asarray(st["v"], float), nan=0.0)

    def dump(f, name, a):
        ni, nj = a.shape[:2]
        for i in range(ni):
            for j in range(nj):
                if a.ndim == 2:
                    f.write(f"{name} {i + lo} {j + lo} {a[i, j]:.17e}\n")
                else:
                    for k in range(a.shape[2]):
                        f.write(f"{name} {i + lo} {j + lo} {k + 1} "
                                f"{a[i, j, k]:.17e}\n")

    with open(f"{W}/divduo_input.txt", "w") as f:
        f.write(f"# res {res}\n# ng {ng}\n")
        dump(f, "RAREA_C", gs["rarea_c"])
        dump(f, "DXC", gs["dxc"])
        dump(f, "DYC", gs["dyc"])
        dump(f, "SIN_SG", gs["sin_sg"])
        dump(f, "COS_SG", gs["cos_sg"])
        dump(f, "U", u)
        dump(f, "V", v)
        dump(f, "UA", ua)
        dump(f, "VA", va)

    # stage the exact input arrays the fixture will carry (so pack does
    # NOT regenerate them)
    np.savez_compressed(
        f"{W}/staging.npz",
        u=u, v=v, ua=ua, va=va, rarea_c=gs["rarea_c"],
        dxc=gs["dxc"], dyc=gs["dyc"], sin_sg=gs["sin_sg"],
        cos_sg=gs["cos_sg"])
    # smoke: the python port runs (its output is re-derived from the
    # STORED inputs at test time, so nothing else needs saving here)
    divergence_corner_duo(u, v, ua, va, gs, bd, res + 1, res + 1,
                          grid_type=0)
    print("gen: inputs + staging written")

else:
    stag = np.load(f"{W}/staging.npz")
    fd = np.full((m_b, m_b), np.nan)
    for line in open(f"{W}/divduo_output.txt"):
        pp = line.split()
        fd[int(pp[1]) - lo, int(pp[2]) - lo] = float(pp[3])
    inp_hash = hashlib.sha256(
        open(f"{W}/divduo_input.txt", "rb").read()).hexdigest()
    np.savez_compressed(
        FIX, divg_d=fd,
        u=stag["u"], v=stag["v"], ua=stag["ua"], va=stag["va"],
        rarea_c=stag["rarea_c"], dxc=stag["dxc"], dyc=stag["dyc"],
        sin_sg=stag["sin_sg"], cos_sg=stag["cos_sg"],
        res=res, ng=ng, input_sha256=inp_hash,
        input_lineage="plain-c_sw ua/va (ROUTINE-TRANSLATION gate; the "
        "full DUO pipeline needs dg-initialized d2a2c_vect ua/va)")
    print("fixture packed; input_sha256", inp_hash)
