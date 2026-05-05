"""Iter-988: probe Lin-Rood inter-sweep pad_halo for q_i / q_j at the seam.

After Y-sweep `_yppm` produces fy2 and q_i, the X-sweep needs
pad_halo(q_i) to compute fx1.  This probe checks whether the
inter-sweep halo amplifies the cube-edge seam at face=1 i=0 j=21.

Finding: the Y-sweep correction (q_i - zeta_abs) is small (~0.4%)
and the halo structure is preserved.  No anomaly at the inter-sweep.

The FB chain v_ll_Linf=55.6 m/s gap is architectural per the
`FV3FBShallowWaterModel` docstring ("known unstable") rather than
a single missing operator.  Production `FV3EdgeShallowWaterModel`
(Arakawa-Lamb + RK3) achieves 0.132 m/s.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import (
    _c_sw,
    _d_sw1_recompute_ut_vt,
    _p_grad_c,
)
from legoesm.core.fv_tp_2d import (
    _yppm,
    compute_transport_quantities,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


def main():
    N = 36
    grid = create_cubed_sphere(N, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    h = sw.h.data
    h_s = sw.h_s.data
    dt = 300.0
    g = 9.80616

    h_star, uc_new, vc_new, ua, va = _c_sw(h, u_d, v_d, h_s, cdgrid, dt, g)
    dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt * 0.5, g)
    uc_new = uc_new + dp_x
    vc_new = vc_new + dp_y

    ut, vt = _d_sw1_recompute_ut_vt(
        uc_new, vc_new, cdgrid, dt, u_d_old=u_d, v_d_old=v_d)

    dx_u = cdgrid.dx_edge_y
    dy_v = cdgrid.dy_edge_x
    vt_circ = u_d * dx_u
    ut_circ = v_d * dy_v
    rarea = 1.0 / cdgrid.base.area
    zeta = rarea * (vt_circ[:, :, :-1] - vt_circ[:, :, 1:]
                     + ut_circ[:, 1:, :] - ut_circ[:, :-1, :])
    zeta_abs = zeta + cdgrid.base.f

    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    dg = grid.duogrid
    q_full = pad_halo(zeta_abs, halo=2, duogrid=dg)
    fy2 = _yppm(q_full[:, 2:-2, :], cry, N, None, None, None, None,
                 use_duogrid=True, apply_fortran_xppm_boundary=False,
                 bounded_domain=False)
    fyy = yfx * fy2
    area = cdgrid.base.area
    q_i = (zeta_abs * area + fyy[:, :, :-1] - fyy[:, :, 1:]) / ra_y
    q_i_pad = pad_halo(q_i, halo=2, duogrid=dg)

    print("=== Iter-988 Lin-Rood inter-sweep at face=1 west j=21 ===")
    print(f"zeta_abs[1, 0, 21] = {float(zeta_abs[1, 0, 21]):.4e}")
    print(f"q_i[1, 0, 21]      = {float(q_i[1, 0, 21]):.4e}")
    print(f"delta = q_i - zeta = {float(q_i[1, 0, 21] - zeta_abs[1, 0, 21]):.4e}")

    print(f"\nq_i_pad halo at face=1 west j_pad=23:")
    for i_pad in [0, 1, 2, 3]:
        print(f"  i_pad={i_pad} → {float(q_i_pad[1, i_pad, 23]):.4e}")


if __name__ == "__main__":
    main()
