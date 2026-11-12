"""GSE alleviation: weight normalisation, conservation and gating."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.wave.config import WaveGSEConfig
from legoesm.wave.grid import prepare_env, wave_cell_area
from legoesm.wave.physics.gse import GSESpatialAveraging


def _fields(config, depth):
    env = prepare_env(config, depth)
    spectra = jnp.ones_like(jnp.broadcast_to(env.group_velocity, (1, 6, 5, 4, 4))) * 0.02
    spectra = spectra * (1.0 + 0.1 * jnp.arange(4).reshape(1, 1, 1, 4, 1))
    return env, spectra


def test_weights_sum_to_one(tiny_config, tiny_depth):
    env, spectra = _fields(tiny_config, tiny_depth)
    gse = GSESpatialAveraging(
        WaveGSEConfig(enabled=True, alpha_s=1.0, alpha_n=1.0, periodic_x=True))
    cg = jnp.broadcast_to(env.group_velocity, spectra.shape)
    weights = gse.compute_average_weights(cg, env, dt=1800.0)
    total = jnp.sum(weights, axis=(-2, -1))
    assert np.allclose(np.asarray(total), 1.0, atol=1e-6)


def test_preserves_column_total(tiny_config, tiny_depth):
    env, spectra = _fields(tiny_config, tiny_depth)
    mask = jnp.ones(spectra.shape[:3], dtype=bool)
    gse = GSESpatialAveraging(WaveGSEConfig(
        enabled=True, alpha_s=1.0, alpha_n=1.0, periodic_x=True,
        preserve_spectral_totals=True))
    out = gse(spectra, env, mask, dt=1800.0)
    area = wave_cell_area(env)
    assert np.allclose(
        np.asarray(jnp.sum(out * area)), np.asarray(jnp.sum(spectra * area)),
        rtol=1e-5)


def test_disabled_or_zero_dt_is_identity(tiny_config, tiny_depth):
    env, spectra = _fields(tiny_config, tiny_depth)
    mask = jnp.ones(spectra.shape[:3], dtype=bool)
    assert np.array_equal(
        np.asarray(GSESpatialAveraging(
            WaveGSEConfig(enabled=False))(spectra, env, mask, 1800.0)),
        np.asarray(spectra))
    assert np.array_equal(
        np.asarray(GSESpatialAveraging(
            WaveGSEConfig(enabled=True))(spectra, env, mask, 0.0)),
        np.asarray(spectra))


def test_eager_and_jit_agree(tiny_config, tiny_depth):
    env, spectra = _fields(tiny_config, tiny_depth)
    mask = jnp.ones(spectra.shape[:3], dtype=bool)
    gse = GSESpatialAveraging(
        WaveGSEConfig(enabled=True, alpha_s=1.0, alpha_n=1.0, periodic_x=True))
    eager = gse(spectra, env, mask, 1800.0)
    jitted = jax.jit(lambda s, e, m: gse(s, e, m, 1800.0))(
        spectra, env, mask)
    assert np.allclose(np.asarray(eager), np.asarray(jitted), atol=1e-6)


def test_displacement_guard_rejects_a_too_large_footprint(tiny_config, tiny_depth):
    env = prepare_env(tiny_config, tiny_depth)
    gse = GSESpatialAveraging(WaveGSEConfig(enabled=True, substeps=1))
    with pytest.raises(ValueError, match="requires at least"):
        gse.validate_displacement(env.group_velocity, env, 1e7)


def test_enabled_gse_requires_a_positive_cadence(tiny_config):
    from legoesm.wave.config import WaveGSEConfig, WaveTimeConfig

    config = tiny_config._replace(
        time=WaveTimeConfig(dt_gse_s=0),
        gse=WaveGSEConfig(enabled=True),
    )
    try:
        config.validate()
    except ValueError as exc:
        assert "dt_gse_s" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validate() accepted GSE with dt_gse_s=0")


def test_latitude_packet_conserves_each_area_weighted_spectral_bin(tiny_config, tiny_depth):
    config = tiny_config._replace(grid=tiny_config.grid._replace(lat_deg=(60.,30.,0.,-30.,-60.)))
    env = prepare_env(config,tiny_depth)
    spectra = jnp.zeros((1,6,5,4,4),dtype=jnp.float32).at[:,2,0,:,:].set(.02)
    mask = jnp.ones(spectra.shape[:3],dtype=bool).at[:,2,1].set(False)
    gse = GSESpatialAveraging(WaveGSEConfig(
        enabled=True, alpha_s=1., alpha_n=1., periodic_x=True,
        renormalize_land_blocked=False, preserve_spectral_totals=True,
        substeps=2))
    gse.validate_displacement(env.group_velocity,env,1800.)
    area = wave_cell_area(env)
    before = np.asarray(jnp.sum(spectra*area,axis=(1,2)))
    result = gse(spectra,env,mask,1800.)
    after = np.asarray(jnp.sum(result*area,axis=(1,2)))
    assert result.dtype == spectra.dtype
    np.testing.assert_allclose(after,before,rtol=2e-6)
    assert bool(jnp.all(result >= 0))
    weights = gse.compute_average_weights(
        jnp.broadcast_to(jnp.asarray(env.group_velocity, dtype=spectra.dtype), spectra.shape),
        env, 1800. / gse.config.substeps)
    weights = gse.prepare_valid_scatter_weights(weights,mask)
    legacy_density_scatter = spectra
    for _ in range(gse.config.substeps):
        legacy_density_scatter = gse.scatter_by_weights(legacy_density_scatter,weights,mask)
    legacy_total = np.asarray(jnp.sum(legacy_density_scatter*area,axis=(1,2)))
    assert np.max(np.abs(legacy_total/before-1)) > 2e-6
