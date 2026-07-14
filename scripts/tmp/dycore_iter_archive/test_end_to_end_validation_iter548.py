"""FV3_3D iter 548: end-to-end validation — min_edge factory +
scan_step + 30 steps + comprehensive metrics.

Combines all the validated pieces into one user-realistic run:
- ``make_legoesm_nh_min_edge_config`` (iter-466 factory)
- ``make_clipped_scan_step`` (iter-544 helper)
- 30 SBR steps (longest measured run that stays stable @ C8)
- Reports: mass drift, edge_std, interior_std, max field
  magnitudes per prognostic.

This is the user-facing "happy-path production run" test.

Tests
-----

1. ``test_end_to_end_30_steps_sbr`` — single comprehensive
   validation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_legoesm_nh_min_edge_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import (
    create_cubed_sphere,
    rotate_winds_geo_to_grid,
)
from legoesm.grids.halo import make_clipped_scan_step
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _edge_and_interior_std(field_data):
    n_face, n_x, n_y, n_lev = field_data.shape
    edge_mask = np.zeros((n_x, n_y), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask_b = np.broadcast_to(
        edge_mask[None, :, :, None], field_data.shape,
    )
    interior_mask = ~edge_mask_b
    arr = np.asarray(field_data)
    return (
        float(arr[edge_mask_b].std()),
        float(arr[interior_mask].std()),
    )


def _build_sbr_state(n=8, U_0=20.0):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lat = grid.lat
    u_east = U_0 * jnp.cos(lat)
    v_north = jnp.zeros_like(lat)
    u_grid, v_grid = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle,
    )
    u_p = jnp.broadcast_to(
        u_grid[..., None], (6, n, n, nlev),
    ).astype(jnp.float64)
    v_p = jnp.broadcast_to(
        v_grid[..., None], (6, n, n, nlev),
    ).astype(jnp.float64)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=u_p, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_p, name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1), dtype=jnp.float64),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n), dtype=jnp.float64),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0), dtype=jnp.float64),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_end_to_end_30_steps_sbr(capsys):
    n = 8
    n_steps = 30
    grid, hc, tm, state = _build_sbr_state(n)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    scan_step = make_clipped_scan_step(
        m, state, dt=10.0, n_steps=n_steps, slack=0.5,
    )
    final = scan_step(state)

    # Mass conservation (rho_prime is identically zero at start; should
    # stay near zero with no net source)
    rho_arr = np.asarray(final.rho_prime.data)
    area = np.asarray(grid.area)
    dz = np.asarray(hc.dz)
    final_mass = float(np.sum(rho_arr * area[..., None] * dz[None, None, None, :]))

    # Edge / interior diagnostics
    e_theta, i_theta = _edge_and_interior_std(final.theta_prime.data)
    e_u, i_u = _edge_and_interior_std(final.u.data)
    e_v, i_v = _edge_and_interior_std(final.v.data)

    with capsys.disabled():
        print(
            f"\n[iter-548 end-to-end @ C8, {n_steps} steps SBR, "
            f"min_edge + scan_step]"
        )
        print(f"  total rho' mass (should be near 0): {final_mass:.3e}")
        print(f"  θ′ edge_std / int_std: "
              f"{e_theta:.3e} / {i_theta:.3e}  ratio {e_theta/max(i_theta, 1e-30):.2f}×")
        print(f"  u  edge_std / int_std: "
              f"{e_u:.3e} / {i_u:.3e}  ratio {e_u/max(i_u, 1e-30):.2f}×")
        print(f"  v  edge_std / int_std: "
              f"{e_v:.3e} / {i_v:.3e}  ratio {e_v/max(i_v, 1e-30):.2f}×")
    # All finite
    for fld in ("u", "v", "theta_prime", "rho_prime", "w"):
        d = getattr(final, fld).data
        assert jnp.all(jnp.isfinite(d)), f"{fld} non-finite at 30 steps"
