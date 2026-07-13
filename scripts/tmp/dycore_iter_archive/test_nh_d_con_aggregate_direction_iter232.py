"""FV3_3D iter 232: aggregate direction test for the full NH
d_con stack (mirror of PE iter-231).

iter-230 verified each NH d_con site has linear scaling.
iter-232 verifies that the AGGREGATE behaves correctly under
integration: 10 NH steps with all 5 NH d_con knobs
(damp_v_d_con, damp_w_d_con, corner_div_damp_d_con,
div_damp_d_con, ah_d_con) ON simultaneously and damping
mechanisms removing KE → ``mean(θ_p)_ON > mean(θ_p)_OFF``
(energy conservation: KE→heat).

Tests
-----

1. ``test_nh_full_d_con_stack_heats_column_vs_off`` — strong
   wind + w perturbation IC, 10 NH steps with all damping ON.
   ``mean(θ_p)_d_con_ON > mean(θ_p)_d_con_OFF``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
)


def test_nh_full_d_con_stack_heats_column_vs_off():
    """Full NH damping stack ON, with all 5 d_con knobs ON, must
    heat the column relative to the d_con-OFF baseline."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=232)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev + 1))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )

    common = dict(
        # Full NH damping stack ON (mirrors iter-184 umbrella).
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        damp_w=0.030, nord_w=1,
        A_h=1e6, smagorinsky_cs=0.20,
    )
    cfg_off = CDGridCompressibleEulerConfig(
        **common,
        damp_v_d_con=0.0,
        damp_w_d_con=0.0,
        corner_div_damp_d_con=0.0,
        div_damp_d_con=0.0,
        ah_d_con=0.0,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        **common,
        damp_v_d_con=1.0,
        damp_w_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
    )

    def run(cfg, n_steps=10, dt=10.0):
        m = CDGridCompressibleEulerModel(
            grid, height_coord, terrain_metric, cfg,
        )
        s = state
        for _ in range(n_steps):
            s = m.step(s, dt)
        return s

    s_off = run(cfg_off)
    s_on = run(cfg_on)

    mean_theta_off = float(jnp.mean(s_off.theta_prime.data))
    mean_theta_on = float(jnp.mean(s_on.theta_prime.data))

    delta = mean_theta_on - mean_theta_off
    assert delta > 0.0, (
        f"All-d_con-ON NH column must be WARMER than all-OFF: "
        f"mean(θ_p)_ON={mean_theta_on:.6f}, "
        f"mean(θ_p)_OFF={mean_theta_off:.6f}, Δ={delta:.6e}.  "
        f"Either an iter-203/207/209/222/224/226 d_con site has "
        f"a sign error or energy conservation is broken in the "
        f"NH aggregate."
    )

    # Sanity: bounded.
    assert delta < 5.0, (
        f"Δmean(θ_p)={delta:.4e} K is unphysically large for "
        f"10 NH steps with C8 IC."
    )
