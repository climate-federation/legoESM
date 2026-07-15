#!/usr/bin/env python
"""Gen divergence_corner_duo oracle inputs + run the python port."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","..","..","packages","core"))
from legoesm.core.fv3_native_duo_sw_core import divergence_corner_duo
from legoesm.core.fv3_native_sw_core import Bounds, c_sw
from legoesm.grids.fv3_native_gridstruct import (
    FV3_OMEGA,
    FV3_RADIUS_M,
    analytic_swcore_state,
    build_fv3_native_gridstruct,
)

W = sys.argv[1]; res, ng = 12, 3; lo = 1 - ng
gs = build_fv3_native_gridstruct(res, ng, radius=FV3_RADIUS_M, omega=FV3_OMEGA)
st = analytic_swcore_state(gs)
bd = Bounds.single_tile(res, ng)
csw = c_sw(delp=st["delp"], pt=st["pt"], w=np.zeros_like(st["delp"]),
           u=st["u"], v=st["v"], gs=gs, bd=bd, npx=res+1, npy=res+1,
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
                f.write(f"{name} {i+lo} {j+lo} {a[i,j]:.17e}\n")
            else:
                for k in range(a.shape[2]):
                    f.write(f"{name} {i+lo} {j+lo} {k+1} {a[i,j,k]:.17e}\n")

with open(f"{W}/divduo_input.txt","w") as f:
    f.write(f"# res {res}\n# ng {ng}\n")
    dump(f,"RAREA_C",gs["rarea_c"]); dump(f,"DXC",gs["dxc"]); dump(f,"DYC",gs["dyc"])
    dump(f,"SIN_SG",gs["sin_sg"]); dump(f,"COS_SG",gs["cos_sg"])
    dump(f,"U",u); dump(f,"V",v); dump(f,"UA",ua); dump(f,"VA",va)

# python port
py = divergence_corner_duo(u, v, ua, va, gs, bd, res+1, res+1, grid_type=0)
np.save(f"{W}/py_divduo.npy", py)

# after the Fortran driver runs, build the committed fixture from ITS
# output (packed here so the fixture is reproducible + carries inputs +
# provenance) — invoked as a second pass with --pack once the driver ran.
if len(sys.argv) > 2 and sys.argv[2] == "--pack":
    import hashlib
    m_b = res + 2 * ng + 1
    fd = np.full((m_b, m_b), np.nan)
    for line in open(f"{W}/divduo_output.txt"):
        pp = line.split()
        fd[int(pp[1]) - lo, int(pp[2]) - lo] = float(pp[3])
    # active mask: the Fortran compute loops write i,j in isd+1..ied,
    # jsd+1..jed = (res+2ng)^2 - the outer isd/jsd ring stays at the 1e25
    # init (72 sentinel slots on the 19x19 B array); certify only written.
    inp_hash = hashlib.sha256(
        open(f"{W}/divduo_input.txt", "rb").read()).hexdigest()[:16]
    np.savez_compressed(
        "%s/../../../tests/grids/fixtures/divduo_oracle_c12.npz"
        % os.path.dirname(__file__),
        divg_d=fd, u=u, v=v, ua=ua, va=va,
        rarea_c=gs["rarea_c"], dxc=gs["dxc"], dyc=gs["dyc"],
        sin_sg=gs["sin_sg"], cos_sg=gs["cos_sg"],
        res=res, ng=ng, input_sha256=inp_hash,
        input_lineage="plain-c_sw ua/va (routine-arithmetic gate; the "
        "full DUO pipeline needs dg-initialized d2a2c_vect ua/va)")
    print("fixture packed; input_sha256", inp_hash)
else:
    print("gen + python port done")
