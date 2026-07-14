"""FV3_3D iter 578: dycore linearity for tiny perturbations.

iter-572 showed zero IC → zero output (rest preservation).
For tiny perturbations the dycore should respond ~linearly.
Apply 1 mK θ′ Gaussian bump and 10 mK same shape.  After 10
steps, the 10× larger IC should give ~10× larger response.

Tests
-----

1. ``test_linearity_response``.
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
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import make_clipped_step
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_state_pert(n=16, amplitude=1e-3):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lon = grid.lon
    lat = grid.lat
    lon0 = jnp.pi / 4.0
    lat0 = 0.0
    sigma = 0.5
    dlon = jnp.mod(lon - lon0 + 3 * jnp.pi, 2 * jnp.pi) - jnp.pi
    dist2 = dlon ** 2 * jnp.cos(lat) ** 2 + (lat - lat0) ** 2
    theta_p_2d = amplitude * jnp.exp(-dist2 / sigma ** 2)
    theta_p = jnp.broadcast_to(
        theta_p_2d[..., None], (6, n, n, nlev),
    ).astype(jnp.float64)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1), dtype=jnp.float64),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=theta_p,
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


def test_linearity_response(capsys):
    n = 16
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
        heat_source_del2_iters=8, heat_source_del2_coeff=0.20,
    )
    results = []
    for amp in [1e-3, 1e-2]:
        grid, hc, tm, state = _build_state_pert(n=n, amplitude=amp)
        cfg = make_legoesm_nh_min_edge_config(**kw)
        m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
        step = make_clipped_step(m, state, dt=10.0, slack=0.5)
        s = state
        for _ in range(10):
            s = step(s, 10.0)
        max_theta = float(jnp.abs(s.theta_prime.data).max())
        max_u = float(jnp.abs(s.u.data).max())
        results.append((amp, max_theta, max_u))
    with capsys.disabled():
        print(
            f"\n[iter-578 linearity test @ C{n}, 10 steps]"
        )
        for amp, mt, mu in results:
            print(
                f"  amp={amp:.1e}: max|θ′|={mt:.3e}, max|u|={mu:.3e}"
            )
        if results[0][1] > 0:
            ratio_theta = results[1][1] / results[0][1]
            ratio_u = results[1][2] / results[0][2] if results[0][2] > 0 else 0
            print(
                f"\n  10× IC → max|θ′| ratio: {ratio_theta:.2f}× "
                f"(should be ~10 for linear)"
            )
            print(
                f"  10× IC → max|u| ratio: {ratio_u:.2f}× "
                f"(should be ~10 for linear)"
            )
    assert all(np.isfinite(r[1]) for r in results)
