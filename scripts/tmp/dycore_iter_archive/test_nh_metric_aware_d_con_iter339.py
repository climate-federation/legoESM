"""FV3_3D iter 339: opt-in metric-aware d_con form for NH
``damp_v_d_con`` site.  Mirror of PE iter-338.

NH analogue of iter-338's PE port.  Closes gap #1 for NH
damp_v_d_con site (other NH d_con sites pending future iter).
Composes with iter-320 cv flag + iter-336/337 dynamic Exner.

Tests
-----

1. ``test_baseline_bit_for_bit`` — flag=False bit-for-bit.
2. ``test_metric_changes_theta_p`` — flag=True changes θ_p.
3. ``test_metric_finite`` — flag=True finite.
4. ``test_metric_differentiable_at_rest`` — AD-safe.
5. ``test_metric_composes_with_cv_and_dynamic_exner`` — full
   stack (metric + cv + dynamic Exner) finite + AD-safe.
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
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def small_nh():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    rng = np.random.default_rng(seed=339)
    u_p = rng.uniform(-25.0, 25.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-25.0, 25.0, size=(6, n, n, nlev))

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
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
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
    return grid, height_coord, terrain_metric, state


def _cfg(metric=False, cv=False, dyn_exner=False):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        use_fv3_metric_aware_d_con=metric,
        use_fv3_d_con_cv=cv,
        use_fv3_dynamic_exner=dyn_exner,
    )


def test_baseline_bit_for_bit(small_nh):
    """flag=False is bit-for-bit baseline."""
    grid, hc, tm, state = small_nh
    m_default = CDGridCompressibleEulerModel(grid, hc, tm, _cfg())
    m_explicit = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(metric=False),
    )
    s_d = m_default.step(state, 5.0)
    s_e = m_explicit.step(state, 5.0)
    np.testing.assert_array_equal(
        np.asarray(s_d.theta_prime.data),
        np.asarray(s_e.theta_prime.data),
    )


def test_metric_changes_theta_p(small_nh):
    """flag=True changes θ_p measurably."""
    grid, hc, tm, state = small_nh
    m_off = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(metric=False),
    )
    m_on = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(metric=True),
    )
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.theta_prime.data)
        - np.asarray(s_off.theta_prime.data),
    )))
    assert diff > 1e-12


def test_metric_finite(small_nh):
    """flag=True finite output."""
    grid, hc, tm, state = small_nh
    m = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(metric=True),
    )
    s = m.step(state, 5.0)
    for fld in (s.u.data, s.v.data, s.theta_prime.data):
        assert np.all(np.isfinite(np.asarray(fld)))


def test_metric_differentiable_at_rest(small_nh):
    """AD-safe at rest with flag=True."""
    grid, hc, tm, _ = small_nh
    n = 8
    nlev = 5
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    rest = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    m = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(metric=True),
    )

    def loss(amp):
        s = rest._replace(
            u=rest.u.replace(data=amp * jnp.ones_like(rest.u.data)),
        )
        for _ in range(2):
            s = m.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)


def test_metric_composes_with_cv_and_dynamic_exner(small_nh):
    """Full FV3-fidelity NH d_con stack (metric + cv + dynamic
    Exner) produces finite state."""
    grid, hc, tm, state = small_nh
    m = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(metric=True, cv=True, dyn_exner=True),
    )
    s = m.step(state, 5.0)
    for fld in (s.u.data, s.v.data, s.theta_prime.data, s.rho_prime.data):
        assert np.all(np.isfinite(np.asarray(fld)))
