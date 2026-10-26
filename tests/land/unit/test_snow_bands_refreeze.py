"""Rain-on-cold-snow refreeze must be capped at the pack cold content, not the
rain supply.  Uncapped, heavy rain on thin cold snow released ~kW/m2 of
spurious latent (L_f) surface heating (snow_bands gap-6)."""
from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.land.snow_bands import (
    ElevationSnowBandConfig,
    band_elevation_anomalies,
    step_snow_bands,
)


def test_refreeze_capped_by_cold_content():
    cfg = ElevationSnowBandConfig(
        band_dz=band_elevation_anomalies(jnp.asarray([0.0]))  # flat -> uniform bands
    )
    n_bands = cfg.band_dz.shape[-1]
    swe = jnp.full((1, n_bands), 2.0)                     # thin snow [kg/m2]
    T_sfc = jnp.asarray([constants.T_freeze - 10.0])      # cold pack (10 K cold content)
    zeros = jnp.zeros((1, n_bands))
    dt = 3600.0
    rain = jnp.full((1, n_bands), 1e-2)                   # 36 kg/m2 over dt -- far excess
    out = step_snow_bands(
        swe, zeros, zeros, T_sfc, zeros, dt,
        Q_net=jnp.zeros(1), cfg=cfg, precip_rain_bands=rain, snow_age_activation_K=0.0
    )
    cap = 2.0 * constants.c_pi * 10.0 / constants.L_f     # ~0.126 kg/m2
    assert float(out.refreeze[0]) <= cap + 1e-9           # capped at cold content
    assert float(out.refreeze[0]) < float(cfg.refreeze_frac * 1e-2 * dt)  # cap fired
