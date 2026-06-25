"""PR C: ML correctness fixes from the codebase review.

- ``weighted_mae`` now applies the area-weighting resolution correction.
- ``compute_normalization_stats`` aligns latitude weights to the LAT axis.
- ``TrainablePhysicsParams.from_defaults`` seeds softplus params overflow-safe.
- ``_resolve_variables`` raises (not silently drops) on an unknown ERA5 var.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest


def test_weighted_mae_is_resolution_independent():
    from legoesm.ml.loss import weighted_mae

    # A CONSTANT absolute error must give MAE == that constant regardless of
    # n_lat (the area-weighting correction divides by Σw, not the array size).
    for n_lat in (8, 16, 32):
        n_lon, n_ch = 5, 3
        pred = jnp.zeros((2, n_lat, n_lon, n_ch))
        target = jnp.full((2, n_lat, n_lon, n_ch), 0.7)
        # Gauss-Legendre-like weights summing to 2 (non-uniform).
        mu = jnp.linspace(-0.99, 0.99, n_lat)
        weights = (1.0 - mu ** 2)
        weights = weights / jnp.sum(weights) * 2.0
        mae = float(weighted_mae(pred, target, weights))
        assert mae == pytest.approx(0.7, abs=1e-6), (
            f"n_lat={n_lat}: weighted_mae={mae} not resolution-independent "
            "(missing the n_lat/Σw correction?)"
        )


def test_normalization_weights_align_to_latitude_axis():
    from legoesm.ml.normalization import compute_normalization_stats

    # data layout (n_samples, n_lat, n_lon, n_channels) with n_channels != n_lat
    # so a weight misaligned onto the channel axis would BROADCAST-ERROR.
    n_s, n_lat, n_lon, n_ch = 2, 4, 5, 3
    lat_vals = jnp.arange(n_lat, dtype=jnp.float64)  # value varies by latitude
    data = jnp.broadcast_to(
        lat_vals[None, :, None, None], (n_s, n_lat, n_lon, n_ch),
    )
    weights = jnp.array([1.0, 3.0, 3.0, 1.0])  # latitude weights (n_lat,)

    stats = compute_normalization_stats(data, weights=weights)
    # Hand-computed latitude-weighted mean of [0,1,2,3] with w=[1,3,3,1]:
    expected = float(jnp.sum(lat_vals * weights) / jnp.sum(weights))  # = 1.5
    assert stats.mean.shape == (n_ch,)
    np.testing.assert_allclose(np.asarray(stats.mean), expected, rtol=1e-6)


def test_softplus_seed_is_overflow_safe_for_large_default():
    from legoesm.training.trainable_params import (
        TrainablePhysicsParams, ParamConstraint,
    )

    # Midpoint default = 50500 -> exp(default) overflows to inf under the naive
    # inverse softplus, seeding the raw leaf with inf.
    c = ParamConstraint("big_softplus", 1.0e3, 1.0e5, "softplus")
    params = TrainablePhysicsParams.from_defaults([c])
    raw = params.raw_values["big_softplus"]
    assert jnp.isfinite(raw), f"softplus seed is non-finite: {raw}"
    # Round-trip: softplus(seed) recovers the default (= midpoint 50500).
    recovered = float(params.as_dict()["big_softplus"])
    assert recovered == pytest.approx(50500.0, rel=1e-4)


def test_resolve_variables_raises_on_unknown():
    xr = pytest.importorskip("xarray")
    from legoesm.ml.data.era5_loader import _resolve_variables

    ds = xr.Dataset({
        "temperature": (("x",), np.zeros(3)),
        "u_component_of_wind": (("x",), np.zeros(3)),
    })
    # Known variable resolves.
    assert _resolve_variables(ds, ("temperature",)) == ["temperature"]
    # An unresolvable variable raises (not a silent drop -> short channel list).
    with pytest.raises(ValueError, match="(?i)not found"):
        _resolve_variables(ds, ("temperature", "definitely_not_a_var"))
