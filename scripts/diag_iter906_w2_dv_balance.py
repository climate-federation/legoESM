"""Iter-906 diagnostic: per-component decomposition of dv_d/dt at the
W2 cube-vertex hot spots identified in iter-904.

Background.  iter-904's t=0 W2 C36 tendency decomposition showed
the top-20 |dv_d/dt| cells cluster at lat ±33.9° on faces 0 and 2
(the cube-vertex band).  iter-905's path 3 also failed.  iter-906
takes a different angle — instead of trying yet another time-
integration swap, it decomposes dv_d/dt at these hot spots into its
constituent terms, exposing WHICH component drives the geostrophic-
cancellation residual.

Production fv3_sw_tendencies computes (per `operators_cdgrid.py:
2009-2010`):

  du_cc =  zeta_abs * v_cc - dB_dx_cc
  dv_cc = -zeta_abs * u_cc - dB_dy_cc

For the W2 IC, the analytical solution has `v_cc ≡ 0` and exact
geostrophic balance `f * u = -g * dh/dy`, which means
`-zeta_abs*u_cc - dB_dy_cc = 0` IF the discretization is exact.

iter-906 measures the magnitudes of `+zeta_abs*u_cc` and
`-dB_dy_cc` separately at each W2 hot-spot D-grid v-edge.  Their
sum is dv_cc; if both terms are large but cancel poorly, the
cancellation failure is the bias source.  If one term dominates,
that term's discretization is the issue.

Output:
  - For each top-10 D-grid v-edge by |dv_d/dt|:
      * Coriolis-vorticity term: zeta_abs * u_cc (at projecting cell)
      * Pressure-gradient term:  dB_dy_cc (at projecting cell)
      * Their sum:               dv_cc (= residual; ideally 0)
      * Ratio: |sum| / |max(term1, term2)| (= cancellation quality)

If ratio is small (~1e-4), the terms cancel ~99.99% — bias is
limited by floating-point precision and the 0.132 m/s W2 v_ll_Linf
is structural.

If ratio is large (~1e-1 to 1), one or both terms have meaningful
discretization errors at the cube vertex.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)

# Replicate fv3_sw_tendencies' internal computation up to the
# pressure-gradient and Coriolis-vorticity terms.
from legoesm.core.operators_cdgrid import (
    fv3_d2cc, fv3_cc2c, _arakawa_lamb_gradient,
    _interp_corner_to_center, dgrid_vorticity)
from legoesm.grids.halo import pad_halo_vector


def main():
    n = 36
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    g = 9.80616
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    h = sw.h.data
    h_s = sw.h_s.data
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    # === Reproduce fv3_sw_tendencies logic ===
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    KE = 0.5 * (u_cc ** 2 + v_cc ** 2)
    B = KE + g * (h + h_s)
    dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)

    base = cdgrid.base
    dg = base.duogrid
    offsets = None if dg is not None else base.halo_interp_offsets
    u_cc_pad, v_cc_pad = pad_halo_vector(
        u_cc, v_cc,
        base.cos_angle, base.sin_angle,
        base.cos_angle_padded, base.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    u_corner = 0.25 * (u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
                        + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:])
    v_corner = 0.25 * (v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
                        + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:])
    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
    zeta_abs = zeta + base.f

    dB_dx_cc = _interp_corner_to_center(dB_dx)
    dB_dy_cc = _interp_corner_to_center(dB_dy_perp)

    coriolis_term = -np.asarray(zeta_abs * u_cc)        # = -zeta_abs * u_cc
    pressure_term = -np.asarray(dB_dy_cc)               # = -dB_dy_cc
    dv_cc_residual = coriolis_term + pressure_term      # = dv_cc

    abs_dv = np.abs(dv_cc_residual)
    flat = abs_dv.ravel()
    n_face = n * n
    print("=" * 72)
    print("iter-906: W2 dv tendency decomposition at cell centres")
    print(f"(C36 t=0; |dv_cc| stats and top-10 hot spots)")
    print("=" * 72)
    print(f"max  |dv_cc|         = {abs_dv.max():.3e}  m/s^2")
    print(f"mean |dv_cc|         = {abs_dv.mean():.3e}  m/s^2")
    print(f"max  |coriolis term| = {np.max(np.abs(coriolis_term)):.3e}  m/s^2")
    print(f"max  |pressure term| = {np.max(np.abs(pressure_term)):.3e}  m/s^2")
    print()
    print(f"{'face':>4}  {'i':>3}  {'j':>3}  {'lat':>8}  {'lon':>8}  "
          f"{'-zeta*u':>13}  {'-dB/dy':>13}  {'dv_cc':>13}  {'|cancel|':>10}")
    print("-" * 100)
    top10 = np.argsort(flat)[-10:][::-1]
    for idx in top10:
        face = idx // n_face
        rem = idx % n_face
        i = rem // n
        j = rem % n
        lat_deg = float(np.degrees(base.lat[face, i, j]))
        lon_deg = float(np.degrees(base.lon[face, i, j]))
        cor_v = float(coriolis_term[face, i, j])
        pre_v = float(pressure_term[face, i, j])
        sum_v = float(dv_cc_residual[face, i, j])
        max_term = max(abs(cor_v), abs(pre_v))
        cancel = abs(sum_v) / max_term if max_term > 0 else float('nan')
        print(f"{face:>4}  {i:>3}  {j:>3}  "
              f"{lat_deg:>+8.2f}  {lon_deg:>+8.2f}  "
              f"{cor_v:>+13.4e}  {pre_v:>+13.4e}  {sum_v:>+13.4e}  "
              f"{cancel:>10.3e}")
    print()

    # Aggregate cancellation quality at hot spots vs everywhere.
    max_abs_term = np.maximum(np.abs(coriolis_term),
                                np.abs(pressure_term))
    cancel_field = np.where(max_abs_term > 0,
                              abs_dv / np.maximum(max_abs_term, 1e-30),
                              0.0)
    print(f"Cancellation quality (|dv_cc| / max(|coriolis|, |pressure|)):")
    print(f"  hot-spot top-10 mean:  {cancel_field.ravel()[top10].mean():.3e}")
    print(f"  global mean (all cells): {cancel_field.mean():.3e}")
    print(f"  global max  (worst cell): {cancel_field.max():.3e}")
    print()

    # Compare hot-spot cancellation to global to see if cube vertices
    # have WORSE cancellation than average.
    hot_cancel = cancel_field.ravel()[top10].mean()
    global_cancel = cancel_field.mean()
    if hot_cancel > 5.0 * global_cancel:
        print(f"VERDICT: cube-vertex hot spots have CANCELLATION {hot_cancel/global_cancel:.1f}x")
        print(f"         WORSE than global average — geostrophic")
        print(f"         cancellation breakdown is localized to cube vertices.")
        print(f"         Likely target: improve zeta_abs or dB_dy_cc")
        print(f"         discretization at the cube-vertex region.")
    else:
        print(f"VERDICT: cube-vertex cancellation is comparable to global")
        print(f"         (ratio {hot_cancel/global_cancel:.1f}x).  Cube vertices")
        print(f"         are not anomalously bad — the W2 bias is amplification")
        print(f"         of a uniformly-distributed cancellation error during")
        print(f"         time integration.")


if __name__ == "__main__":
    main()
