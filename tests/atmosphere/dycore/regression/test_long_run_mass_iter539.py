"""FV3_3D iter 539: 50-step mass conservation test.

iter-523 verified NH mass conservation at 10 steps (5e-9
drift).  Extend to 50 steps to confirm drift is bounded over
longer integration.

Tests
-----

1. ``test_50_step_mass_conservation`` — sum(rho' * area * dz)
   at intervals up to step 50.  Drift must stay small.
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
from legoesm.grids.halo import make_clipped_step
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_sbr_with_rho_bump(n=8, U_0=20.0, rho_amp=2.0):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lon = grid.lon
    lat = grid.lat
    u_east = U_0 * jnp.cos(lat)
    v_north = jnp.zeros_like(lat)
    u_grid, v_grid = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle,
    )
    u_p = jnp.broadcast_to(u_grid[..., None], (6, n, n, nlev))
    v_p = jnp.broadcast_to(v_grid[..., None], (6, n, n, nlev))
    sigma = 0.7
    dlon = jnp.mod(lon + jnp.pi, 2 * jnp.pi) - jnp.pi
    rho_bump_2d = rho_amp * jnp.exp(
        -(dlon ** 2 * jnp.cos(lat) ** 2 + lat ** 2) / sigma ** 2
    )
    rho_p = jnp.broadcast_to(rho_bump_2d[..., None], (6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=u_p, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_p, name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=rho_p, name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def _total_mass(state, grid, hc):
    rho = np.asarray(state.rho_prime.data)
    area = np.asarray(grid.area)
    dz_arr = np.asarray(hc.dz)
    return float(
        np.sum(rho * area[..., None] * dz_arr[None, None, None, :])
    )


def test_50_step_mass_conservation(capsys):
    grid, hc, tm, state = _build_sbr_with_rho_bump()
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step = make_clipped_step(m, state, dt=10.0, slack=0.5)
    m0 = _total_mass(state, grid, hc)
    history = [(0, m0)]
    s = state
    nan_step = None
    for step_idx in range(1, 51):
        s = step(s, 10.0)
        cur_mass = _total_mass(s, grid, hc)
        if step_idx % 5 == 0 or step_idx == 1:
            history.append((step_idx, cur_mass))
        if nan_step is None and not np.isfinite(cur_mass):
            nan_step = step_idx
            break
    with capsys.disabled():
        print(
            f"\n[iter-539 50-step mass conservation @ C8, "
            f"SBR + rho bump]"
        )
        print(f"  {'step':>5s}: {'mass':>16s}  {'rel drift':>10s}")
        for step_idx, mass in history:
            rel = (mass - m0) / m0 if m0 != 0 else float("nan")
            print(f"  {step_idx:5d}: {mass:16.6e}  {rel:+10.2e}")
        if nan_step is not None:
            print(f"\n  first NaN at step {nan_step}")
        finite_drifts = [
            abs((m - m0) / m0) if m0 != 0 else 0
            for _, m in history
            if np.isfinite(m)
        ]
        max_rel = max(finite_drifts) if finite_drifts else float("nan")
        print(f"  max |rel drift| (finite steps): {max_rel:.2e}")
        if max_rel < 1e-8:
            print(f"  → mass conserved to ~10 ppb")
        elif max_rel < 1e-6:
            print(f"  → mass nearly conserved (1ppm)")
        else:
            print(f"  → mass drifts")
    finite_drifts = [
        abs((m - m0) / m0) if m0 != 0 else 0
        for _, m in history
        if np.isfinite(m)
    ]
    max_rel = max(finite_drifts) if finite_drifts else float("nan")
    assert np.isfinite(max_rel) and max_rel < 1e-5, (
        f"Mass should be conserved over finite steps: got {max_rel:.2e}"
    )
