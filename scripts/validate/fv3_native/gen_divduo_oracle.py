#!/usr/bin/env python
"""Gen divergence_corner_duo oracle inputs + run the python port."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","..","..","packages","core"))
from legoesm.grids.fv3_native_gridstruct import (
    FV3_OMEGA, FV3_RADIUS_M, analytic_swcore_state, build_fv3_native_gridstruct)
from legoesm.core.fv3_native_sw_core import Bounds, c_sw
from legoesm.core.fv3_native_duo_sw_core import divergence_corner_duo

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
print("gen + python port done")
