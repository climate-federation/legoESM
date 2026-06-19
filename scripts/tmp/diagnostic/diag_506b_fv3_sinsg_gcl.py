"""506b decisive test: does the FV3-faithful edge sin_sg reduce the free-stream
/ GCL violation?  iter-678 only measured W2 alpha=0 L2 (which it worsened);
nobody measured the GCL effect.  If FV3 sin_sg shrinks the area-flux divergence
of a non-divergent wind, the metric IS the lever (and W2 regression is a
re-tuning problem); if not, the rewrite won't help free-stream.

Build cos_sg(1..4)=W,S,E,N exactly per fv_grid_utils.F90:342-350:
  p3 = cell center; p1_<edge> = mid_pt3_cart of the two edge corners;
  cos_sg = cos_angle(...), sin_sg = sin(spherical_angle(...)).
Swap into cd.sin_sg[...,0:4], recompute the cosine-bell d2a2c area-flux
divergence, compare corner vs interior magnitude to the current metric.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, jax.numpy as jnp
from legoesm.grids.cubed_sphere import (
    create_cubed_sphere, latlon2xyz, mid_pt3_cart, spherical_angle)
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_sw_core import d2a2c_vect
from legoesm.core.fv_tp_2d import compute_transport_quantities
from tests.test_cases.cosine_bell import cosine_bell_cubesphere

N = 48; DT = 300.0; B = math.pi / 4
grid = create_cubed_sphere(N); cd = create_cubed_sphere_cdgrid(grid); area = np.asarray(grid.area)


def _xyz(lon, lat):
    x, y, z = latlon2xyz(lon, lat)
    return jnp.stack([x, y, z], axis=-1)


def fv3_edge_sin_sg():
    c = _xyz(cd.lon_corner, cd.lat_corner)          # (6,n+1,n+1,3)
    ctr = _xyz(grid.lon, grid.lat)                  # (6,n,n,3)
    c00 = c[:, :-1, :-1]; c10 = c[:, 1:, :-1]; c01 = c[:, :-1, 1:]; c11 = c[:, 1:, 1:]
    pW = mid_pt3_cart(c00, c01); pS = mid_pt3_cart(c00, c10)
    pE = mid_pt3_cart(c10, c11); pN = mid_pt3_cart(c01, c11)
    sW = jnp.sin(spherical_angle(pW, ctr, c01))     # cos_sg(1) args
    sS = jnp.sin(spherical_angle(pS, c10, ctr))     # cos_sg(2)
    sE = jnp.sin(spherical_angle(pE, ctr, c10))     # cos_sg(3)
    sN = jnp.sin(spherical_angle(pN, c01, ctr))     # cos_sg(4)
    return np.stack([np.asarray(sW), np.asarray(sS), np.asarray(sE), np.asarray(sN)], axis=-1)


def gcl_map(cd_):
    st = cosine_bell_cubesphere(grid, cd_, beta=B)
    ua, va, uc, vc, ut, vt = d2a2c_vect(st.u_d, st.v_d, cd_)
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(ut, vt, DT, cd_)
    fdiv = np.abs(np.asarray((xfx[:, :-1, :] - xfx[:, 1:, :]
                              + yfx[:, :, :-1] - yfx[:, :, 1:]) / area) / DT)
    cor = np.zeros((6, N, N), bool)
    for ci in (slice(0, 2), slice(-2, None)):
        for cj in (slice(0, 2), slice(-2, None)):
            cor[:, ci, cj] = True
    interior = fdiv[:, N//4:3*N//4, N//4:3*N//4]
    return fdiv.max(), fdiv[cor].mean(), interior.mean()


print("metric        | gcl_max   corner_mean  interior_mean  corner/int")
mx, cm, im = gcl_map(cd)
print(f"current       | {mx:.2e}  {cm:.2e}    {im:.2e}     {cm/im:.1f}x")

fv3 = fv3_edge_sin_sg()
sg = np.asarray(cd.sin_sg).copy()
# sanity: how different is FV3 edge sin_sg from current?
print(f"  FV3 vs current edge sin_sg: max|diff|={np.abs(fv3-sg[...,0:4]).max():.3e} "
      f"mean|diff|={np.abs(fv3-sg[...,0:4]).mean():.3e}")
sg[..., 0:4] = fv3
cd_fv3 = cd._replace(sin_sg=jnp.asarray(sg))
mx2, cm2, im2 = gcl_map(cd_fv3)
print(f"FV3 edge sin_sg| {mx2:.2e}  {cm2:.2e}    {im2:.2e}     {cm2/im2:.1f}x")
print(f"\ncorner GCL change: {cm:.2e} -> {cm2:.2e}  ({100*(cm2-cm)/cm:+.0f}%)")
