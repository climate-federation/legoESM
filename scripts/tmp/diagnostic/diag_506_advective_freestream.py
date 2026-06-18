"""506b prototype: advective-form free-stream correction for PROGNOSTIC flow.

The flux-form update h += div(F(h))/area violates free-stream on the cube
(div(F(1)) != 0, the GCL violation).  Subtracting h*div(F(1)) converts it to
the ADVECTIVE form dh/dt = -(div(hv) - h*div(v)), which is free-stream
preserving by construction (a constant has zero gradient) and needs NO
streamfunction -> applicable to the evolving Case-6 flow.

Validate on the cube cosine bell with the ORIGINAL d2a2c winds (the path that
fragmented): does the correction fix free-stream (h=1) and the bell?
"""
from __future__ import annotations
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, jax, jax.numpy as jnp
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_sw_core import d2a2c_vect
from legoesm.core.fv_tp_2d import compute_transport_quantities, fv_tp_2d
from legoesm.core.conservation import conservation_accumulator
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, cosine_bell_error_norms, cosine_bell_exact)

N=48; DT=1800.0; DAYS=12.0; NSUB=6; B=math.pi/4
grid=create_cubed_sphere(N); cd=create_cubed_sphere_cdgrid(grid); area=grid.area
_acc=conservation_accumulator()
st=cosine_bell_cubesphere(grid,cd,beta=B)
ua,va,uc,vc,ut,vt=d2a2c_vect(st.u_d,st.v_d,cd)
crx,cry,xfx,yfx,ra_x,ra_y=compute_transport_quantities(ut,vt,DT/NSUB,cd)
# spurious area-flux divergence (= div(F(h=1))); advective correction subtracts h*eps
eps=(xfx[:,:-1,:]-xfx[:,1:,:]+yfx[:,:,:-1]-yfx[:,:,1:])

def step(h, advective, mass):
    fx,fy=fv_tp_2d(h,crx,cry,xfx,yfx,ra_x,ra_y,cd,hord=10,apply_fortran_xppm_boundary=True)
    a=area.astype(_acc); h64=h.astype(_acc)
    div=(fx.astype(_acc)[:,:-1,:]-fx.astype(_acc)[:,1:,:]+fy.astype(_acc)[:,:,:-1]-fy.astype(_acc)[:,:,1:])
    if advective:
        div=div - h64*eps.astype(_acc)   # flux-form -> advective form
    hn=(h64+div/a).astype(h.dtype)
    if mass is not None:
        hp=jnp.maximum(hn,0.0); mp=jnp.sum(hp.astype(_acc)*a); hn=hp*(mass/jnp.maximum(mp,1.0)).astype(hp.dtype)
    return hn

def run(advective):
    h=st.h; mass=float(jnp.sum(h*area))
    @jax.jit
    def outer(h_):
        def b(hh,_): return step(hh,advective,mass),None
        o,_=jax.lax.scan(b,h_,None,length=NSUB); return o
    for _ in range(int(round(DAYS*86400/DT))): h=outer(h)
    he=cosine_bell_exact(grid.lon,grid.lat,grid.radius,DAYS*86400.0,B)
    return cosine_bell_error_norms(h,he,area)

def freestream(advective):
    h=jnp.ones((6,N,N))
    @jax.jit
    def outer(h_):
        def b(hh,_): return step(hh,advective,None),None
        o,_=jax.lax.scan(b,h_,None,length=NSUB); return o
    for _ in range(int(round(2*86400/DT))): h=outer(h)
    return float(jnp.max(jnp.abs(h-1.0)))

print(f"free-stream h=1 (2d):  flux-form={freestream(False):.3e}   advective={freestream(True):.3e}")
nf=run(False); na=run(True)
print(f"cosine bell pi/4 L2:   flux-form={nf['l2']:.3f}   advective={na['l2']:.3f}")
