"""FV3_3D iter 523: mass conservation during SBR multi-step run.

iter-521 showed SBR IC produces small (~0.06 K) theta_prime
noise at edges after 10 steps.  Open question: is mass
(rho_prime) conserved, or does it drift?

Mass conservation is a first-class FV3 design goal.  This
iter measures sum(rho_prime * dV) over the cube — a
properly-discretized FV3 dycore should keep this constant to
machine precision.

Tests
-----

1. ``test_rho_prime_total_mass_drift_sbr`` — compute total
   mass = sum(rho * area * dz) at step 0, 5, 10.  Report
   absolute and relative drift.
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
from legoesm.grids.halo import monotone_halo_clip_context
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_sbr_state(n, U_0=20.0, theta_amp=2.0):
    """SBR winds + nonzero theta_prime so total integrand is meaningful."""
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
    u_p = jnp.broadcast_to(u_grid[..., None], (6, n, n, nlev))
    v_p = jnp.broadcast_to(v_grid[..., None], (6, n, n, nlev))
    # rho_prime: smooth gaussian bump near equator
    lon = grid.lon
    sigma = 0.7
    dlon = jnp.mod(lon + np.pi, 2 * np.pi) - np.pi
    rho_bump_2d = theta_amp * jnp.exp(
        -(dlon ** 2 * jnp.cos(lat) ** 2 + lat ** 2) / sigma ** 2
    )
    rho_p = jnp.broadcast_to(rho_bump_2d[..., None], (6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=u_p, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_p, name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=rho_p, name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def _total_mass(state, grid, hc):
    """Sum rho_prime * area * dz over the full cube."""
    rho = np.asarray(state.rho_prime.data)            # (6, n, n, nlev)
    area = np.asarray(grid.area)                       # (6, n, n)
    dz_arr = np.asarray(hc.dz)                         # (nlev,)
    # mass = sum_f sum_ij sum_k rho[f, i, j, k] * area[f, i, j] * dz[k]
    return float(
        np.sum(rho * area[..., None] * dz_arr[None, None, None, :])
    )


def test_rho_prime_total_mass_drift_sbr(capsys):
    n = 8
    grid, hc, tm, state = _build_sbr_state(n)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    mass_history = []
    mass_history.append((0, _total_mass(state, grid, hc)))
    with monotone_halo_clip_context(slack=0.5):
        m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
        s = state
        for step in range(1, 11):
            s = m.step(s, dt=10.0)
            if step in (5, 10):
                mass_history.append((step, _total_mass(s, grid, hc)))
    m0 = mass_history[0][1]
    with capsys.disabled():
        print(
            f"\n[iter-523 SBR mass conservation @ C8, 10 steps]"
        )
        print(f"  {'step':>5s}: {'mass':>16s}  {'rel drift':>10s}")
        for step, mass in mass_history:
            rel = (mass - m0) / m0 if m0 != 0 else float("nan")
            print(f"  {step:5d}: {mass:16.6e}  {rel:+10.2e}")
        max_rel = max(
            abs((m - m0) / m0) if m0 != 0 else 0
            for _, m in mass_history
        )
        print(f"\n  max |rel drift|: {max_rel:.2e}")
        if max_rel < 1e-10:
            print(f"  → mass conserved to machine precision")
        elif max_rel < 1e-6:
            print(f"  → mass nearly conserved (1ppm)")
        else:
            print(f"  → mass drifts (>1ppm)")
    # Total should be bounded — drift < 1% is hard pass; conservation
    # to machine precision is the ideal.
    max_rel = max(
        abs((m - m0) / m0) if m0 != 0 else 0
        for _, m in mass_history
    )
    assert max_rel < 1e-2, (
        f"mass should not drift by more than 1%: got {max_rel:.2e}"
    )
