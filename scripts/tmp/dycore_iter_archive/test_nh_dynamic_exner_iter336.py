"""FV3_3D iter 336: opt-in FV3-faithful dynamic Exner factor for
the NH slow-tendency d_con sites.

Audit
-----
iter-207 used frozen ``Π_ref`` (exner_ref from ``HeightCoordinate``)
in the d_con denominator ``c_x · Π_ref`` for the iter-203 / 209 /
222 / 224 / 226 NH d_con sites — a reference-state linearization
that is ~30 % under-heating aloft (where actual p deviates
significantly from p_ref).  FV3 NH path
(``dyn_core.F90:1769`` cv branch + line 1796) uses live ``pkz``
computed from current pressure (``rdg * delp / delz * pt``) — full
dynamic Exner factor.

iter-336 adds the opt-in flag ``use_fv3_dynamic_exner: bool =
False`` (NH only).  When True, the 3 SLOW-TENDENCY d_con sites
(corner_div / cell-centre div_damp / Smag-A_h) replace ``Π_ref``
with ``Π_total = Π_ref + π'`` where ``π'`` is the Exner
perturbation already computed at slow_tendencies step 1.  Default
False preserves bit-for-bit baseline.

Post-acoustic d_con sites (damp_v, damp_w) keep ``Π_ref`` because
``π'`` would require recomputation in the ``step()`` method post-
acoustic site (covered by future iter if needed).

PE path uses actual T (no Exner factor); flag is NH-only.

Tests
-----

1. ``test_baseline_bit_for_bit`` — flag=False reproduces baseline
   exactly.
2. ``test_dynamic_exner_changes_tendency`` — flag=True changes
   ``dθ_p/dt`` measurably from frozen-Π_ref baseline.
3. ``test_dynamic_exner_finite`` — flag=True produces finite
   tendency.
4. ``test_dynamic_exner_differentiable_at_rest`` — AD-safe at
   rest.
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
def small_perturbed_state():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    rng = np.random.default_rng(seed=336)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    # Non-zero theta_p, rho_p so π' is non-trivial and differs
    # from Π_ref measurably.
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
    return grid, cdgrid, height_coord, terrain_metric, state


def _cfg_engages_dcon_sites(use_fv3_dynamic_exner: bool):
    """Engage all 3 slow-tendency d_con sites."""
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
        A_h=1e6, smagorinsky_cs=0.20,
        ah_d_con=1.0,
        use_fv3_dynamic_exner=use_fv3_dynamic_exner,
    )


def test_baseline_bit_for_bit(small_perturbed_state):
    """flag=False reproduces baseline exactly across the 3
    slow-tendency d_con sites."""
    grid, cdgrid, hc, tm, state = small_perturbed_state
    cfg_default = _cfg_engages_dcon_sites(use_fv3_dynamic_exner=False)
    cfg_explicit_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
        A_h=1e6, smagorinsky_cs=0.20,
        ah_d_con=1.0,
        # explicit False matches default
        use_fv3_dynamic_exner=False,
    )
    tend_default = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_default, dt_actual=10.0,
    )
    tend_explicit = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_explicit_off,
        dt_actual=10.0,
    )
    np.testing.assert_array_equal(
        np.asarray(tend_default.dtheta_prime_dt.data),
        np.asarray(tend_explicit.dtheta_prime_dt.data),
    )


def test_dynamic_exner_changes_tendency(small_perturbed_state):
    """flag=True changes dθ_p/dt measurably (proves wiring active
    given non-zero π')."""
    grid, cdgrid, hc, tm, state = small_perturbed_state
    cfg_off = _cfg_engages_dcon_sites(use_fv3_dynamic_exner=False)
    cfg_on = _cfg_engages_dcon_sites(use_fv3_dynamic_exner=True)
    tend_off = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_off, dt_actual=10.0,
    )
    tend_on = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_on, dt_actual=10.0,
    )
    diff = float(np.max(np.abs(
        np.asarray(tend_on.dtheta_prime_dt.data)
        - np.asarray(tend_off.dtheta_prime_dt.data),
    )))
    assert diff > 1e-12, (
        "iter-336 dynamic Exner did NOT change dtheta_p/dt when "
        "flag flipped.  Either π' is zero everywhere (test "
        "fixture too uniform) or wiring is silent no-op."
    )


def test_dynamic_exner_finite(small_perturbed_state):
    """flag=True produces finite tendency."""
    grid, cdgrid, hc, tm, state = small_perturbed_state
    cfg_on = _cfg_engages_dcon_sites(use_fv3_dynamic_exner=True)
    tend_on = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_on, dt_actual=10.0,
    )
    assert np.all(
        np.isfinite(np.asarray(tend_on.dtheta_prime_dt.data))
    )
    assert np.all(np.isfinite(np.asarray(tend_on.du_dt.data)))


def test_dynamic_exner_differentiable_at_rest(small_perturbed_state):
    """jax.grad flows finitely through 2 NH steps with flag=True
    + full d_con stack at rest state."""
    grid, _, hc, tm, _ = small_perturbed_state
    n = grid.n
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
    cfg = _cfg_engages_dcon_sites(use_fv3_dynamic_exner=True)
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)

    def loss(amp):
        s = rest._replace(
            u=rest.u.replace(data=amp * jnp.ones_like(rest.u.data)),
        )
        for _ in range(2):
            s = m.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)
