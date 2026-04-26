"""Iter-904 diagnostic: t=0 W2 C36 tendency decomposition for the
production fv3_sw_tendencies path, plus W2 1-day comparison with
the new `use_fv3_dsw1_mass_transport` flag.

Per the user's iter-904 task spec:
  - production dh_dt, du_dt, dv_dt
  - dB_dy_cc (Bernoulli gradient)
  - -zeta_abs * u_cc (Coriolis + relative vorticity * u term)
  - dv balance = -zeta_abs*u_cc - dB_dy_cc  (vector-invariant
    momentum eqn at cell centre)
  - div_damp contribution before/after boundary_fix
  - projected dv_d_dt at edge-midpoint D-grid
  - lat-lon v_north tendency hot spots
  - top 20 cells by |dv residual|

Then:
  - 1-day W2 measurement with use_fv3_dsw1_mass_transport=True vs
    iter-892 default.
  - Acceptance: improve v_ll_Linf by >=10 % AND worsen h_L2 by <5 %.
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
    create_cubed_sphere_cdgrid, cell_centre_angles_from_4edge)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _build_w2_initial_state(n=36):
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    return grid, cdgrid, state, u0


def part1_t0_tendency_decomposition():
    """Compute tendencies at t=0 W2 C36 and identify hot spots."""
    print("=" * 72)
    print("PART 1: t=0 W2 C36 tendency decomposition")
    print("=" * 72)
    n = 36
    grid, cdgrid, state, u0 = _build_w2_initial_state(n=n)
    div_damp = 8.0 * _div_damp_cube(n)

    # Production fv3_sw_tendencies (iter-892 default + iter-893
    # apply_fortran_xppm_boundary=True per matrix runner).
    dh_dt, du_dt, dv_dt = fv3_sw_tendencies(
        state.h, state.u_d, state.v_d, state.h_s, cdgrid,
        div_damp=div_damp, boundary_fix=True, dddmp=0.2,
        apply_fortran_xppm_boundary=True)

    dh = np.asarray(dh_dt)
    du = np.asarray(du_dt)
    dv = np.asarray(dv_dt)

    print(f"Production tendencies at t=0:")
    print(f"  max |dh/dt|     = {np.max(np.abs(dh)):.4e}  m/s")
    print(f"  max |du_d/dt|   = {np.max(np.abs(du)):.4e}  m/s^2")
    print(f"  max |dv_d/dt|   = {np.max(np.abs(dv)):.4e}  m/s^2")
    print()

    # Project dv_d onto lat-lon to find v-wind tendency hot spots.
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    # Cell-centre du, dv from D-grid edge tendencies.
    du_cc = 0.5 * (du[:, :, :-1] + du[:, :, 1:])
    dv_cc = 0.5 * (dv[:, :-1, :] + dv[:, 1:, :])
    # v_north tendency = sa * du_cc + ca * dv_cc.
    dv_north = np.asarray(sa) * du_cc + np.asarray(ca) * dv_cc
    dv_ll = apply_cubedsphere_to_latlon(dv_north, weights)
    print(f"Projected v_north tendency (lat-lon at t=0):")
    print(f"  max |dv_north/dt|_ll = {np.max(np.abs(dv_ll)):.4e}  m/s^2")
    flat = np.abs(dv_ll).ravel()
    top_idx = np.argsort(flat)[-5:][::-1]
    print(f"  top-5 lat-lon cell |dv_north/dt|: "
          f"{flat[top_idx]}")
    print()

    # Top-20 cube-sphere cells by |dv_d_dt| at the d-grid v-edge.
    dv_abs = np.abs(dv)
    flat_dv = dv_abs.ravel()
    top20 = np.argsort(flat_dv)[-20:][::-1]
    n_per_face = (n + 1) * n
    print(f"Top 20 D-grid v-edge cells by |dv_d/dt| at t=0:")
    print(f"  {'face':>4}  {'i':>3}  {'j':>3}  {'lat':>8}  {'lon':>8}  "
          f"{'|dv|':>11}")
    for idx in top20:
        face = idx // n_per_face
        rem = idx % n_per_face
        i = rem // n
        j = rem % n
        lat_deg = float(np.degrees(cdgrid.lat_edge_y[face, i, j]))
        lon_deg = float(np.degrees(cdgrid.lon_edge_y[face, i, j]))
        print(f"  {face:>4}  {i:>3}  {j:>3}  {lat_deg:>+8.2f}  "
              f"{lon_deg:>+8.2f}  {float(dv_abs[face, i, j]):>+11.3e}")
    print()
    return grid, cdgrid


def part2_w2_1day_comparison(grid, cdgrid):
    """Compare iter-892 default vs iter-904 use_fv3_dsw1_mass_transport
    on W2 1-day at C36."""
    print("=" * 72)
    print("PART 2: W2 1-day comparison (C36, dt=300s)")
    print("=" * 72)
    n = grid.n
    dt = 300.0
    n_steps = int(86400 / dt)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)

    def run(use_fv3_dsw1):
        cfg = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            div_damp=8.0 * _div_damp_cube(n),
            boundary_fix=True,
            damp_v=0.06,
            nord_v=2,
            apply_fortran_xppm_boundary=True,
            use_fv3_dsw1_mass_transport=use_fv3_dsw1,
        )
        model = FV3EdgeShallowWaterModel(grid, config=cfg)
        cdg = model.cdgrid
        u_d = cdg.cos_angle_edge_x * (u0 * jnp.cos(cdg.lat_edge_x))
        v_d = -cdg.sin_angle_edge_y * (u0 * jnp.cos(cdg.lat_edge_y))
        state = FV3EdgeShallowWaterState(
            h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
        model.set_initial_mass(state)

        h_ic = np.asarray(sw.h.data)
        area = np.asarray(grid.area)

        for _ in range(n_steps):
            state = model.step(state, dt)

        h_final = np.asarray(state.h)
        h_l2 = float(np.sqrt(np.sum((h_final - h_ic) ** 2 * area)
                              / np.sum(h_ic ** 2 * area)))

        ca, sa = cell_centre_angles_from_4edge(cdg)
        u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                       + np.asarray(state.u_d)[:, :, 1:])
        v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                       + np.asarray(state.v_d)[:, 1:, :])
        v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
        v_ll = apply_cubedsphere_to_latlon(v_north, weights)
        v_ll_linf = float(np.max(np.abs(v_ll)))
        return h_l2, v_ll_linf

    print(f"{'config':<40}  {'h_L2':>10}  {'v_ll_Linf':>11}")
    print("-" * 70)
    h_l2_off, v_inf_off = run(use_fv3_dsw1=False)
    print(f"{'(A) iter-892 default (production)':<40}  "
          f"{h_l2_off:>10.3e}  {v_inf_off:>11.4e}")
    h_l2_on, v_inf_on = run(use_fv3_dsw1=True)
    print(f"{'(B) iter-904 use_fv3_dsw1_mass_transport':<40}  "
          f"{h_l2_on:>10.3e}  {v_inf_on:>11.4e}")
    print()

    v_inf_pct = 100.0 * (v_inf_on - v_inf_off) / v_inf_off
    h_l2_pct = 100.0 * (h_l2_on - h_l2_off) / h_l2_off
    print(f"v_ll_Linf change: {v_inf_pct:+.2f} %  "
          f"(acceptance: <= -10 %)")
    print(f"h_L2 change:       {h_l2_pct:+.2f} %  "
          f"(acceptance: <= +5 %)")
    print()

    if v_inf_pct <= -10.0 and h_l2_pct <= 5.0:
        print("VERDICT: iter-904 PASSES the Ralph-protocol acceptance")
        print("         gate.  v_ll_Linf improves >=10 % AND h_L2")
        print("         worsens <5 %.  Recommend default-flip in")
        print("         iter-905+ after broader (W5/cosine bell/ocean")
        print("         rest) confirmation.")
    elif v_inf_pct > -10.0 and h_l2_pct <= 5.0:
        print("VERDICT: iter-904 does NOT improve W2 v_ll_Linf by 10%.")
        print("         Keep flag default-OFF; document negative result.")
    elif v_inf_pct <= -10.0 and h_l2_pct > 5.0:
        print("VERDICT: iter-904 improves v_ll_Linf but h_L2 regresses")
        print("         beyond 5 % threshold.  Mixed result; keep")
        print("         flag default-OFF; document trade-off.")
    else:
        print("VERDICT: iter-904 worsens both v_ll_Linf and h_L2.")
        print("         Keep flag default-OFF; document negative result.")


def main():
    grid, cdgrid = part1_t0_tendency_decomposition()
    part2_w2_1day_comparison(grid, cdgrid)


if __name__ == "__main__":
    main()
