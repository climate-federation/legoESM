"""Physics hook of the spectral NH dycore (``step(..., physics_fn=...)``).

Pins the contract used by the matrix DCMIP-2025 TC3 spectral row: the
callback returns SPECTRAL tendencies added at every RK stage.  A callback
returning zeros must reproduce the plain dynamics step; the Kessler callback
(shared grid-space core with the MPAS arm) must change theta' and the water
tracers.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.spectral_nh import (  # noqa: E402
    SpectralCompressibleEulerModel,
    SpectralNHConfig,
    dcmip25_tc3_init_spectral,
)
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from scripts.matrix.run_atmosphere_test_matrix import (  # noqa: E402
    _make_kessler_spectral_nh_physics_fn,
)

_DT = 3.0


@pytest.fixture(scope="module")
def tc3():
    state, hcoord, tmetric = dcmip25_tc3_init_spectral(
        create_gaussian_grid(21), n_levels=40)
    cfg = SpectralNHConfig(
        n_acoustic_substeps=25, semi_implicit_acoustic=True,
        sponge_width=8000.0, sponge_coeff=0.15, hyperdiff_coeff=0.0,
        small_earth_factor=60.0, rotating=False)
    model = SpectralCompressibleEulerModel(
        create_gaussian_grid(21, radius=constants.R_earth / 60.0),
        hcoord, tmetric, cfg)
    return model, state, hcoord


def _leaves(s):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(s)]


def test_zero_physics_fn_equals_plain_step(tc3):
    model, state, _ = tc3

    def zero_fn(s, grid, hcoord, tmetric):
        return jax.tree_util.tree_map(jnp.zeros_like, s)

    plain = _leaves(model.step(state, _DT))
    with_zero = _leaves(model.step(state, _DT, physics_fn=zero_fn))
    for a, b in zip(plain, with_zero):
        np.testing.assert_array_equal(a, b)


def test_kessler_physics_fn_changes_step(tc3):
    model, state, hcoord = tc3
    phys = _make_kessler_spectral_nh_physics_fn(hcoord, _DT)

    tend = phys(state, model.grid, model.height_coord, model.terrain_metric)
    assert float(jnp.max(jnp.abs(tend.theta_prime_hat.data))) > 0.0
    assert float(jnp.max(jnp.abs(tend.tracers_hat.data))) > 0.0
    for f in ("vor_hat", "div_hat", "w_hat", "rho_prime_hat", "phis_hat"):
        assert float(jnp.max(jnp.abs(getattr(tend, f).data))) == 0.0

    plain = model.step(state, _DT)
    moist = model.step(state, _DT, physics_fn=phys)
    assert np.all(np.isfinite(np.asarray(moist.tracers_hat.data)))
    d_theta = np.max(np.abs(np.asarray(
        moist.theta_prime_hat.data - plain.theta_prime_hat.data)))
    d_q = np.max(np.abs(np.asarray(
        moist.tracers_hat.data - plain.tracers_hat.data)))
    assert d_theta > 0.0 and d_q > 0.0


def test_physics_fn_evaluated_at_every_rk_stage(tc3):
    """The callback runs once per outer SSP-RK3 stage (3 trace-time calls
    in one step), and a ``(tendencies, diagnostics)`` tuple is unwrapped."""
    model, state, _ = tc3
    calls = []

    def counting_zero_fn(s, grid, hcoord, tmetric):
        calls.append(1)
        return jax.tree_util.tree_map(jnp.zeros_like, s), {}

    out = model.step(state, _DT, physics_fn=counting_zero_fn)
    assert len(calls) == 3
    plain = _leaves(model.step(state, _DT))
    for a, b in zip(plain, _leaves(out)):
        np.testing.assert_array_equal(a, b)
