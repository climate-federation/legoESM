"""FV3_3D iter 271: NH stability across different
``n_acoustic_substeps`` values.

The NH compressible-Euler integrator uses an acoustic substep
within each large step.  ``n_acoustic_substeps`` controls how
many substeps run per outer step.  Default in tests is 4
(matches FV3 standard ``n_split=4``).  This iter verifies
stability across the range {2, 4, 8} with the full d_con
stack engaged.

Tests
-----

1. ``test_nh_n_acoustic_substeps_2_stable`` — 2 substeps.
2. ``test_nh_n_acoustic_substeps_4_stable`` — 4 substeps.
3. ``test_nh_n_acoustic_substeps_8_stable`` — 8 substeps.
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


def _state(seed):
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

    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.2, 0.2, size=(6, n, n, nlev + 1))

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
    return grid, height_coord, terrain_metric, state


def _run_with_n_substeps(n_substeps):
    grid, hc, tm, state = _state(seed=271 + n_substeps)
    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=n_substeps,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        damp_w_d_con=1.0,
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        delt_max=1.0,
    )
    model = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    s = state
    for _ in range(10):
        s = model.step(s, 10.0)
    return s


def test_nh_n_acoustic_substeps_2_stable():
    """2 acoustic substeps."""
    s = _run_with_n_substeps(2)
    assert jnp.all(jnp.isfinite(s.u.data))
    assert jnp.all(jnp.isfinite(s.theta_prime.data))


def test_nh_n_acoustic_substeps_4_stable():
    """4 acoustic substeps (umbrella default)."""
    s = _run_with_n_substeps(4)
    assert jnp.all(jnp.isfinite(s.u.data))
    assert jnp.all(jnp.isfinite(s.theta_prime.data))


def test_nh_n_acoustic_substeps_8_stable():
    """8 acoustic substeps (more conservative)."""
    s = _run_with_n_substeps(8)
    assert jnp.all(jnp.isfinite(s.u.data))
    assert jnp.all(jnp.isfinite(s.theta_prime.data))
