"""Direct tests for the SFNO lat-lon <-> Gaussian coupling step (#797 sfno mode).

The WB scale trainer's sfno mode runs the dycore on a LAT-LON grid while the
SFNO's spherical-harmonic transforms need Gaussian quadrature latitudes —
``make_sfno_step_unified_latlon`` remaps prognostics lat-lon -> Gaussian
(precomputed IDW weights), runs SFNOPhysics there, and remaps the predicted
tendencies back. The shipped factory referenced this function before it
existed (the old symbols-resolve test SKIPPED instead of failing).

Needs JAX_ENABLE_X64=1 (Gaussian grid / SH transforms).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.regridding import compute_latlon_to_voronoi_weights
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.training.sfno_dycore_coupling import (
    SFNOPhysics,
    make_sfno_step_unified_latlon,
)

NLEV = 2
N_LAT_LL, N_LON_LL = 8, 16


def _weights(src_lat, src_lon, tgt_lat, tgt_lon, tgt_shape):
    lon2d, lat2d = np.meshgrid(np.asarray(tgt_lon), np.asarray(tgt_lat))
    w = compute_latlon_to_voronoi_weights(
        np.asarray(src_lat), np.asarray(src_lon),
        lat2d.ravel(), lon2d.ravel())
    return w._replace(target_shape=tgt_shape)


@pytest.fixture(scope="module")
def setup():
    gauss = create_gaussian_grid(10)
    n_ch = 4 * NLEV + 2
    sfno = SFNO(
        SFNOConfig(in_channels=n_ch, out_channels=n_ch,
                   embed_dim=8, n_blocks=1, residual_prediction=False),
        gauss, key=jax.random.PRNGKey(0))
    sfno_ph = SFNOPhysics(sfno=sfno, grid=gauss, nlev=NLEV)

    ll_lat = jnp.linspace(-np.pi / 2 * 0.95, np.pi / 2 * 0.95, N_LAT_LL)
    ll_lon = jnp.linspace(0.0, 2 * np.pi, N_LON_LL, endpoint=False)
    w_ll2g = _weights(ll_lat, ll_lon, gauss.lat, gauss.lon,
                      (int(gauss.n_lat), int(gauss.n_lon)))
    w_g2ll = _weights(gauss.lat, gauss.lon, ll_lat, ll_lon,
                      (N_LAT_LL, N_LON_LL))
    return sfno_ph, w_ll2g, w_g2ll, ll_lat, ll_lon


def _call_step(step, T, p_s, q_v):
    """Invoke with the legacy step_unified tail layout."""
    zeros2 = jnp.zeros_like(p_s)
    u = jnp.zeros_like(T)
    v = jnp.zeros_like(T)
    held = [zeros2] * 5
    return step(
        True, T, p_s, q_v, jnp.zeros_like(T), jnp.zeros_like(T),
        u, v, zeros2 + 290.0, zeros2,            # sst, sic
        jnp.zeros_like(p_s), jnp.zeros_like(p_s),  # lat, lon (2D)
        jnp.asarray(0.0), jnp.asarray(0.0),      # day_of_year, seconds_of_day
        jnp.asarray(300.0),                      # dt
        jnp.ones_like(p_s), jnp.asarray(1361.0),  # solar_weights, s_0
        None, None,                              # o3_vmr, aerosol_od
        jnp.zeros_like(T), *held,                # held_dT_rad + 5 held fluxes
    )


def test_latlon_step_returns_latlon_shaped_finite_tendencies(setup):
    sfno_ph, w_ll2g, w_g2ll, _, _ = setup
    step = make_sfno_step_unified_latlon(sfno_ph, w_ll2g, w_g2ll)
    T = jnp.full((N_LAT_LL, N_LON_LL, NLEV), 280.0)
    p_s = jnp.full((N_LAT_LL, N_LON_LL), 1.0e5)
    q_v = jnp.full((N_LAT_LL, N_LON_LL, NLEV), 5e-3)
    out, held_new = _call_step(step, T, p_s, q_v)
    assert out.dT_dt.shape == T.shape
    assert out.dq_v_dt.shape == T.shape
    assert out.precip.shape == p_s.shape
    assert bool(jnp.all(jnp.isfinite(out.dT_dt)))
    assert bool(jnp.all(jnp.isfinite(out.dq_v_dt)))
    assert len(held_new) == 6


def test_zero_decoder_gives_exactly_zero_tendencies(setup):
    """Epoch-0 stability contract, sfno flavor: with a ZERO-initialized
    decoder (what the scale_build factory ships), the untrained SFNO must
    emit exactly-zero tendencies so the first rollout is the pure dycore."""
    sfno_ph, w_ll2g, w_g2ll, _, _ = setup
    zeroed = eqx.tree_at(
        lambda m: (m.sfno.decoder.weight, m.sfno.decoder.bias), sfno_ph,
        (jnp.zeros_like(sfno_ph.sfno.decoder.weight),
         jnp.zeros_like(sfno_ph.sfno.decoder.bias)))
    step = make_sfno_step_unified_latlon(zeroed, w_ll2g, w_g2ll)
    T = jnp.full((N_LAT_LL, N_LON_LL, NLEV), 280.0)
    p_s = jnp.full((N_LAT_LL, N_LON_LL), 1.0e5)
    q_v = jnp.full((N_LAT_LL, N_LON_LL, NLEV), 5e-3)
    out, _ = _call_step(step, T, p_s, q_v)
    assert bool(jnp.all(out.dT_dt == 0.0))
    assert bool(jnp.all(out.dq_v_dt == 0.0))


def test_remap_roundtrip_preserves_smooth_field():
    """The IDW weights pair must approximately preserve a smooth field on a
    lat-lon -> Gaussian -> lat-lon round trip (interior points)."""
    from legoesm.grids.regridding import regrid_scalar

    gauss = create_gaussian_grid(10)
    ll_lat = jnp.linspace(-np.pi / 2 * 0.95, np.pi / 2 * 0.95, N_LAT_LL)
    ll_lon = jnp.linspace(0.0, 2 * np.pi, N_LON_LL, endpoint=False)
    w_ll2g = _weights(ll_lat, ll_lon, gauss.lat, gauss.lon,
                      (int(gauss.n_lat), int(gauss.n_lon)))
    w_g2ll = _weights(gauss.lat, gauss.lon, ll_lat, ll_lon,
                      (N_LAT_LL, N_LON_LL))
    f = jnp.sin(ll_lat)[:, None] * jnp.cos(ll_lon)[None, :]
    back = regrid_scalar(regrid_scalar(f, w_ll2g), w_g2ll)
    assert back.shape == f.shape
    # coarse IDW round trip: loose tolerance, interior rows only
    err = jnp.abs(back - f)[2:-2, :]
    assert float(err.max()) < 0.25
