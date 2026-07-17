"""FV3_3D iter 173: async-halo overlap for the NH cell-centre
divergence damping gradient call.

Mirror of the PE wiring at ``primitive_eq_cdgrid.py:594-598``.
Under MPI backend, ``use_async_halo=True`` dispatches the
``arakawa_lamb_gradient(div_v)`` call inside the iter-171
div-damp block to the async-overlapping variant.

Tests
-----
1. Single-device fall-through bit-for-bit equivalence:
   ``use_async_halo=True`` and ``use_async_halo=False`` produce
   identical state under the local (non-MPI) backend.
2. ``jax.grad`` flows through 5 steps with the field set.
3. AST regression: the dispatch + helper import are present.
"""
from __future__ import annotations

import ast

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

from tests.legoesm_paths import legoesm_source_path


@pytest.fixture(scope="module")
def small_nh_state():
    """Same fixture as the iter-168/169/170/171/172 NH tests."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, grid.n, grid.n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def test_nh_async_halo_single_device_equivalence(small_nh_state):
    """On single-device backend, ``use_async_halo`` is a no-op:
    True and False produce bit-for-bit identical output."""
    grid, height_coord, terrain_metric, state = small_nh_state

    # Use div_damp_coeff > 0 so the async-halo branch is reachable.
    cfg_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=1e7, div_damp_dddmp=0.0,
        use_async_halo=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=1e7, div_damp_dddmp=0.0,
        use_async_halo=True,    # no effect on single-device
    )

    model_off = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_off,
    )
    model_on = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_on,
    )

    s_off = state
    s_on = state
    for _ in range(5):
        s_off = model_off.step(s_off, 10.0)
        s_on = model_on.step(s_on, 10.0)

    np.testing.assert_array_equal(
        np.asarray(s_off.u.data), np.asarray(s_on.u.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_off.v.data), np.asarray(s_on.v.data),
    )


def test_nh_async_halo_differentiable(small_nh_state):
    """``jax.grad`` flows through 5 steps with the flag set (single
    device).  Catches AD breakage in the dispatch path."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=1e7, div_damp_dddmp=0.20,
        use_async_halo=True,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(theta_p_data):
        s = state._replace(
            theta_prime=state.theta_prime.replace(data=theta_p_data),
        )
        for _ in range(5):
            s = model.step(s, 10.0)
        return jnp.mean(s.u.data ** 2 + s.v.data ** 2)

    grad = jax.grad(loss_fn)(state.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad))


def test_nh_async_halo_ast_regression():
    """AST regression: the iter-173 wiring (config field +
    dispatch + helper import) must be present in the source.
    Catches a refactor that drops the async-halo dispatch
    silently."""
    src_path = legoesm_source_path(
        "atmosphere/dynamics/gcm/compressible_euler_cdgrid.py"
    )
    src_text = src_path.read_text()

    expected_substrings = [
        "use_async_halo: bool = False",              # config field
        "config.use_async_halo and _ghb_div() ==",   # dispatch gate
        "overlapped_arakawa_lamb_gradient",          # helper name
    ]
    missing = [s for s in expected_substrings if s not in src_text]
    assert not missing, (
        f"iter-173 AST regression: NH async-halo wiring missing "
        f"{missing}.  This was added by iter 173; its absence "
        f"silently disables the MPI overlap optimization without "
        f"breaking unit tests."
    )
