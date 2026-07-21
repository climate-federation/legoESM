"""Unit tests for the AIMIP radiation-flux training loss.

Covers :func:`radiation_flux_loss` (TOA + surface flux supervision), the
lifted shared :func:`_lat_weighted_mean`, and its integration into
``carry_mse``.  Runnable directly (``python test_radiation_flux_loss.py``)
as a cheap compute-node self-check, or under pytest.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.training.losses import (
    LossConfig,
    _lat_weighted_mean,
    carry_mse,
    radiation_flux_loss,
)

N_LAT, N_LON, NLEV = 8, 16, 4


class _FluxCarry(NamedTuple):
    """Minimal carry stub exposing only the held radiation flux fields."""
    held_sw_up_toa: jax.Array
    held_lw_up_toa: jax.Array
    held_sw_net_sfc: jax.Array
    held_lw_net_sfc: jax.Array


class _StateCarry(NamedTuple):
    """Full-enough stub for ``carry_mse`` (state + flux fields)."""
    T: jax.Array
    u: jax.Array
    v: jax.Array
    q_v: jax.Array
    p_s: jax.Array
    held_sw_up_toa: jax.Array
    held_lw_up_toa: jax.Array
    held_sw_net_sfc: jax.Array
    held_lw_net_sfc: jax.Array


def _flux_carry(key, base=250.0):
    fields = [base + jax.random.normal(k, (N_LAT, N_LON)) * 10.0
              for k in jax.random.split(key, 4)]
    return _FluxCarry(*fields)


def _flux_cfg(**over):
    base = dict(w_T=0.0, w_u=0.0, w_v=0.0, w_q=0.0, w_ps=0.0,
                w_flux_rsut=1.0, w_flux_olr=1.0,
                w_flux_sfc_sw=1.0, w_flux_sfc_lw=1.0, flux_scale=20.0)
    base.update(over)
    return LossConfig(**base)


def test_identical_fluxes_zero_loss():
    c = _flux_carry(jax.random.PRNGKey(0))
    loss = radiation_flux_loss(c, c, lat_weights=None, config=_flux_cfg())
    assert float(loss) == 0.0


def test_offset_positive_and_scaled():
    """Constant +c OLR offset, only w_flux_olr → loss = c²/scale²."""
    c0 = _flux_carry(jax.random.PRNGKey(1))
    offset = 8.0
    pred = c0._replace(held_lw_up_toa=c0.held_lw_up_toa + offset)
    cfg = _flux_cfg(w_flux_rsut=0.0, w_flux_olr=1.0,
                    w_flux_sfc_sw=0.0, w_flux_sfc_lw=0.0)
    loss = radiation_flux_loss(pred, c0, lat_weights=None, config=cfg)
    expected = offset ** 2 / cfg.flux_scale ** 2
    assert np.isclose(float(loss), expected, rtol=1e-5)


def test_bias_term_catches_mean_only():
    """A zero-mean perturbation gives ~0 bias term but >0 MSE term."""
    c0 = _flux_carry(jax.random.PRNGKey(2))
    pert = jax.random.normal(jax.random.PRNGKey(3), (N_LAT, N_LON))
    pert = pert - jnp.mean(pert)  # zero global mean
    pred = c0._replace(held_lw_up_toa=c0.held_lw_up_toa + pert)
    bias_cfg = _flux_cfg(w_flux_rsut=0.0, w_flux_olr=0.0,
                         w_flux_sfc_sw=0.0, w_flux_sfc_lw=0.0,
                         w_bias_flux_olr=1.0)
    mse_cfg = _flux_cfg(w_flux_rsut=0.0, w_flux_olr=1.0,
                        w_flux_sfc_sw=0.0, w_flux_sfc_lw=0.0)
    bias_loss = radiation_flux_loss(pred, c0, lat_weights=None, config=bias_cfg)
    mse_loss = radiation_flux_loss(pred, c0, lat_weights=None, config=mse_cfg)
    assert float(bias_loss) < 1e-10        # mean offset is ~0
    assert float(mse_loss) > 1e-3          # but per-cell error is real


def test_zero_weights_exactly_zero():
    c0 = _flux_carry(jax.random.PRNGKey(4))
    pred = c0._replace(held_lw_up_toa=c0.held_lw_up_toa + 50.0)
    loss = radiation_flux_loss(pred, c0, lat_weights=None, config=LossConfig())
    assert float(loss) == 0.0


def test_differentiable():
    c0 = _flux_carry(jax.random.PRNGKey(5))
    tgt = _flux_carry(jax.random.PRNGKey(6))
    cfg = _flux_cfg(w_bias_flux_olr=10.0)

    def loss_of(olr):
        pred = c0._replace(held_lw_up_toa=olr)
        return radiation_flux_loss(pred, tgt, lat_weights=None, config=cfg)

    g = jax.grad(loss_of)(c0.held_lw_up_toa)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.sum(jnp.abs(g))) > 0.0


def test_lat_weighted_mean_matches_uniform_and_weighted():
    field = jnp.arange(N_LAT * N_LON, dtype=jnp.float64).reshape(N_LAT, N_LON)
    # No weights → plain mean.
    assert np.isclose(float(_lat_weighted_mean(field, None)), float(jnp.mean(field)))
    # Uniform lat_weights → same as plain mean (resolution-independent norm).
    w = jnp.ones((N_LAT,))
    assert np.isclose(float(_lat_weighted_mean(field, w)), float(jnp.mean(field)))
    # cos(lat)-like weights → differs and equals manual area-weighted mean.
    latw = jnp.linspace(0.2, 1.0, N_LAT)
    manual = float(jnp.sum(jnp.mean(field, axis=1) * latw) / jnp.sum(latw))
    assert np.isclose(float(_lat_weighted_mean(field, latw)), manual, rtol=1e-6)


def test_ambiguous_lat_axis_uses_first():
    """Square field with two axes == n_lat: weight the FIRST (lat) axis.

    (Behaviour changed from raising to axis-0 convention so that adding
    LatLonGrid.weights doesn't crash valid n_lev==n_lat configs.)  With
    uniform weights the result must still equal the plain mean.
    """
    field = jnp.arange(N_LAT * N_LAT, dtype=jnp.float64).reshape(N_LAT, N_LAT)
    w = jnp.ones((N_LAT,))
    out = float(_lat_weighted_mean(field, w))
    assert np.isclose(out, float(jnp.mean(field)))   # uniform w -> plain mean
    # Non-uniform weights weight axis 0 (rows):
    latw = jnp.linspace(0.2, 1.0, N_LAT)
    manual = float(jnp.sum(jnp.mean(field, axis=1) * latw) / jnp.sum(latw))
    assert np.isclose(float(_lat_weighted_mean(field, latw)), manual, rtol=1e-6)


def test_carry_mse_includes_flux_when_weighted():
    """carry_mse with state matched but fluxes offset → loss from flux term."""
    k = jax.random.split(jax.random.PRNGKey(7), 9)
    T = jax.random.normal(k[0], (N_LAT, N_LON, NLEV)) * 5 + 250
    u = jax.random.normal(k[1], (N_LAT, N_LON, NLEV))
    v = jax.random.normal(k[2], (N_LAT, N_LON, NLEV))
    q = jnp.abs(jax.random.normal(k[3], (N_LAT, N_LON, NLEV))) * 1e-3
    ps = jax.random.normal(k[4], (N_LAT, N_LON)) * 100 + 1e5
    fl = [jax.random.normal(k[5 + i], (N_LAT, N_LON)) * 10 + 200 for i in range(4)]
    tgt = _StateCarry(T, u, v, q, ps, *fl)
    pred = tgt._replace(held_lw_up_toa=tgt.held_lw_up_toa + 5.0)  # only OLR off
    sigma = jnp.linspace(0.1, 0.95, NLEV)

    no_flux = LossConfig()
    with_flux = LossConfig(w_flux_olr=1.0, flux_scale=20.0)
    l0 = carry_mse(pred, tgt, sigma, lat_weights=None, config=no_flux)
    l1 = carry_mse(pred, tgt, sigma, lat_weights=None, config=with_flux)
    # State is identical → no_flux loss is 0; flux term lifts it.
    assert float(l0) < 1e-12
    assert float(l1) > 1e-3


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("radiation_flux_loss: all self-checks passed")


if __name__ == "__main__":
    _run_all()
