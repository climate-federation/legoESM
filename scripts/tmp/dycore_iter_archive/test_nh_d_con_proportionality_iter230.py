"""FV3_3D iter 230: proportionality tests for the iter-222
(corner-div), iter-224 (cell-centre div_damp), and iter-226
(Smagorinsky-A_h) NH d_con sites.

Unlike the PE path (iter-228 / iter-229 bit-for-bit), the NH
``NonHydrostaticTendencies`` does not expose D-grid wind
tendencies in the public return.  We therefore validate the
NH d_con wiring via a LINEARITY test: the dθ_p/dt contribution
must scale linearly with the d_con knob.

Tests
-----

1. ``test_nh_corner_div_damp_d_con_scales_linearly`` — iter-222.
2. ``test_nh_div_damp_d_con_scales_linearly`` — iter-224.
3. ``test_nh_ah_d_con_scales_linearly`` — iter-226.

Each test runs the slow-tendency function 3 times with d_con in
{0.5, 1.0, 2.0} and verifies:

    Δ(dθ_p_dt | d=2.0 vs d=0.5) == 3.0 * Δ(dθ_p_dt | d=1.0 vs d=0.5)
    Δ(dθ_p_dt | d=1.0 vs d=0.5) == 1.0 * Δ(dθ_p_dt | d=1.0 vs d=0.5)

i.e., d_con increment behaves like a scalar coefficient on a
fixed (state-dependent) heating tendency.  Catches non-linear
mistakes (e.g., accidental d_con² coupling) and verifies the
gate is actually proportional, not a clamp/cap.
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
def small_nh_state_with_winds():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=230)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
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
    return grid, cdgrid, height_coord, terrain_metric, state


def _check_linearity(make_cfg, state, grid, height_coord, terrain_metric, cdgrid):
    """Verify dθ_p/dt scales linearly with d_con."""
    cfg_05 = make_cfg(0.5)
    cfg_10 = make_cfg(1.0)
    cfg_20 = make_cfg(2.0)
    dt = 10.0

    tend_05 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric, cdgrid,
        cfg_05, dt_actual=dt,
    )
    tend_10 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric, cdgrid,
        cfg_10, dt_actual=dt,
    )
    tend_20 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric, cdgrid,
        cfg_20, dt_actual=dt,
    )

    delta_10 = tend_10.dtheta_prime_dt.data - tend_05.dtheta_prime_dt.data
    delta_20 = tend_20.dtheta_prime_dt.data - tend_05.dtheta_prime_dt.data

    # delta_10 corresponds to a 0.5 increment in d_con; delta_20
    # to a 1.5 increment.  So delta_20 / delta_10 should be 3.0.
    np.testing.assert_allclose(
        delta_20, 3.0 * delta_10,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "NH d_con contribution to dθ_p/dt is NOT linear in "
            "d_con.  Expected (Δ@2.0 - Δ@0.5) == 3.0 * (Δ@1.0 - "
            "Δ@0.5)."
        ),
    )

    # Also verify that delta_10 is NON-zero (the mechanism must
    # actually be active so the test isn't vacuous).
    assert float(jnp.max(jnp.abs(delta_10))) > 1e-10, (
        "d_con=1.0 vs d_con=0.5 gives no measurable change in "
        "dθ_p/dt; either the mechanism is gated off or the wind "
        "state is too quiet to test."
    )


def test_nh_corner_div_damp_d_con_scales_linearly(small_nh_state_with_winds):
    """iter-222 NH corner-div d_con scales linearly with the
    d_con knob."""
    grid, cdgrid, hc, tm, state = small_nh_state_with_winds

    def make_cfg(d):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            corner_div_damp_d2_bg=0.0005,
            corner_div_damp_dddmp=0.20,
            corner_div_damp_d_con=d,
        )

    _check_linearity(make_cfg, state, grid, hc, tm, cdgrid)


def test_nh_div_damp_d_con_scales_linearly(small_nh_state_with_winds):
    """iter-224 NH cell-centre div_damp d_con scales linearly."""
    grid, cdgrid, hc, tm, state = small_nh_state_with_winds

    def make_cfg(d):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            div_damp_coeff=1e6, div_damp_dddmp=0.20,
            div_damp_d_con=d,
        )

    _check_linearity(make_cfg, state, grid, hc, tm, cdgrid)


def test_nh_ah_d_con_scales_linearly(small_nh_state_with_winds):
    """iter-226 NH Smagorinsky-A_h d_con scales linearly."""
    grid, cdgrid, hc, tm, state = small_nh_state_with_winds

    def make_cfg(d):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            A_h=1e6, smagorinsky_cs=0.20,
            ah_d_con=d,
        )

    _check_linearity(make_cfg, state, grid, hc, tm, cdgrid)
