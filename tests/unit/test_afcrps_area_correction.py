"""#1413: area_weighted_afcrps must carry the n_lat/Σw correction.

Its three siblings apply it and say so; the bare `jnp.mean(crps * w)` this
replaces was low by exactly `n_lat / Σw` — a resolution-DEPENDENT factor (5x at
T5, ~32x at T42) on the live S2S training objective.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ml.loss import area_weighted_afcrps, area_weighted_mse

jax.config.update("jax_enable_x64", True)


def _cos_weights(n_lat):
    lat = jnp.linspace(-np.pi / 2, np.pi / 2, n_lat)
    return jnp.cos(lat)


@pytest.mark.parametrize("n_lat", [10, 16, 64])
def test_constant_crps_field_returns_that_constant(n_lat):
    """A spatially constant error must score as itself at ANY resolution.

    That is the whole point of the correction: without it the value scales
    with n_lat/Σw.
    """
    n_lon, n_ch, n_mem = 8, 2, 4
    w = _cos_weights(n_lat)
    rng = np.random.default_rng(0)
    target = jnp.asarray(rng.normal(size=(n_lat, n_lon, n_ch)))
    # A deterministic "ensemble" of identical members => CRPS = |x - y|.
    offset = 0.025
    ensemble = jnp.broadcast_to(target + offset,
                                (n_mem, n_lat, n_lon, n_ch))
    got = float(area_weighted_afcrps(ensemble, target, w))
    assert got == pytest.approx(offset, rel=1e-6), (
        f"n_lat={n_lat}: got {got}, expected {offset}. A value scaling with "
        f"n_lat/sum(w) = {n_lat / float(jnp.sum(w)):.3f} is the #1413 defect.")


def test_scale_matches_the_mse_sibling_on_the_same_field():
    """CRPS of a constant offset vs sqrt of the MSE of the same offset."""
    n_lat, n_lon, n_ch = 32, 8, 1
    w = _cos_weights(n_lat)
    target = jnp.zeros((n_lat, n_lon, n_ch))
    offset = 0.5
    ensemble = jnp.broadcast_to(jnp.full_like(target, offset),
                                (3, n_lat, n_lon, n_ch))
    crps = float(area_weighted_afcrps(ensemble, target, w))
    mse = float(area_weighted_mse(jnp.full_like(target, offset), target, w))
    assert crps == pytest.approx(offset, rel=1e-6)
    assert mse ** 0.5 == pytest.approx(offset, rel=1e-6)


def test_gradient_is_finite_and_nonzero():
    n_lat, n_lon, n_ch = 8, 4, 1
    w = _cos_weights(n_lat)
    rng = np.random.default_rng(1)
    target = jnp.asarray(rng.normal(size=(n_lat, n_lon, n_ch)))
    ens = jnp.asarray(rng.normal(size=(4, n_lat, n_lon, n_ch)))
    g = jax.grad(lambda e: area_weighted_afcrps(e, target, w))(ens)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.max(jnp.abs(g))) > 0.0
