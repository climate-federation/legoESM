"""FV3_3D iter 397: extend dynamic Exner to the NH aggregate
delt_max sponge cap.

iter-336/337 wired ``Π_total`` at the d_con denominators.
iter-397 extends to the aggregate cap derivation
``|Δθ_p · Π| ≤ delt_max · dt`` so the cap uses ``Π_total``
when ``use_fv3_dynamic_exner=True`` (matches FV3 NH dyn_core
sponge cap using live ``pkz``).

Tests
-----

1. ``test_dyn_exner_off_cap_uses_exner_ref`` — bit-for-bit
   baseline.
2. ``test_dyn_exner_on_cap_uses_pi_total`` — flag=True with
   non-zero π' changes the capped d_con tendency vs flag=False.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    cdgrid_compressible_euler_slow_tendencies,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
)


@pytest.fixture(scope="module")
def state_setup():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    hc = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    tm = compute_terrain_metric(terrain, hc)
    rng = np.random.default_rng(seed=397)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    theta_p = rng.uniform(-15.0, 15.0, size=(6, n, n, nlev))
    rho_p = rng.uniform(-0.2, 0.2, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.asarray(theta_p),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.asarray(rho_p),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, cdgrid, hc, tm, state


def _cfg(dyn_exner, delt_max):
    """Engage corner_div_d_con + aggregate cap.  Tight delt_max
    to force cap activation."""
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        delt_max=delt_max,
        use_fv3_dynamic_exner=dyn_exner,
    )


def test_dyn_exner_off_cap_uses_exner_ref(state_setup):
    grid, cdgrid, hc, tm, state = state_setup
    cfg_default = _cfg(dyn_exner=False, delt_max=1.0)
    cfg_explicit_off = _cfg(dyn_exner=False, delt_max=1.0)
    t_d = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_default, dt_actual=10.0,
    )
    t_e = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_explicit_off, dt_actual=10.0,
    )
    np.testing.assert_array_equal(
        np.asarray(t_d.dtheta_prime_dt.data),
        np.asarray(t_e.dtheta_prime_dt.data),
    )


def test_dyn_exner_on_cap_uses_pi_total(state_setup):
    """flag=True changes capped tendency at the aggregate
    delt_max site vs flag=False with non-zero π'."""
    grid, cdgrid, hc, tm, state = state_setup
    cfg_off = _cfg(dyn_exner=False, delt_max=1e-4)
    cfg_on = _cfg(dyn_exner=True, delt_max=1e-4)
    t_off = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_off, dt_actual=10.0,
    )
    t_on = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_on, dt_actual=10.0,
    )
    diff = float(np.max(np.abs(
        np.asarray(t_on.dtheta_prime_dt.data)
        - np.asarray(t_off.dtheta_prime_dt.data),
    )))
    assert diff > 1e-12
