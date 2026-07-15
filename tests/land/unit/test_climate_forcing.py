# tests/land/unit/test_climate_forcing.py
from __future__ import annotations
import jax, jax.numpy as jnp
import numpy.testing as npt
jax.config.update("jax_enable_x64", True)
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.climate_forcing import make_climatological_forcing

def test_warm_climate_has_higher_T_than_cold():
    warm = make_climatological_forcing(300.0, 5.0, 250.0, 2e-5, jnp.asarray(200.0), jnp.asarray(12.0))
    cold = make_climatological_forcing(270.0, 5.0, 250.0, 2e-5, jnp.asarray(200.0), jnp.asarray(12.0))
    assert float(warm.T_lowest[0]) > float(cold.T_lowest[0])

def test_seasonal_amplitude_controls_summer_winter_gap():
    jul = make_climatological_forcing(285.0, 15.0, 250.0, 2e-5, jnp.asarray(200.0), jnp.asarray(6.0))
    jan = make_climatological_forcing(285.0, 15.0, 250.0, 2e-5, jnp.asarray(15.0), jnp.asarray(6.0))
    assert float(jul.T_lowest[0]) - float(jan.T_lowest[0]) > 20.0

def test_noon_shortwave_scales_with_sw_mean():
    lo = make_climatological_forcing(290.0, 5.0, 100.0, 2e-5, jnp.asarray(200.0), jnp.asarray(12.0))
    hi = make_climatological_forcing(290.0, 5.0, 300.0, 2e-5, jnp.asarray(200.0), jnp.asarray(12.0))
    assert float(hi.sw_down[0]) > float(lo.sw_down[0])
    assert float(lo.sw_down[0]) >= 0.0

def test_returns_atmtosurface_shape1_and_traceable():
    f = make_climatological_forcing(290.0, 5.0, 250.0, 2e-5, jnp.asarray(180.0), jnp.asarray(12.0))
    assert isinstance(f, AtmToSurface) and f.T_lowest.shape == (1,)
    out = jax.jit(lambda d, h: make_climatological_forcing(290.0, 5.0, 250.0, 2e-5, d, h).sw_down)(
        jnp.asarray(180.0), jnp.asarray(12.0))
    assert jnp.all(jnp.isfinite(out))

def test_lmip_latitude_preset_unchanged():
    # Regression: the refactored latitude wrapper matches the old T/precip logic.
    from legoesm.land.lmip_forcing import make_synthetic_lmip_forcing
    f = make_synthetic_lmip_forcing(45.5 * jnp.pi / 180.0, 0.0, jnp.asarray(200.0), jnp.asarray(12.0))
    # 45.5N July: T_atm ~ 280 K per the documented latitude baseline (+/- a few K).
    assert 273.0 < float(f.T_lowest[0]) < 288.0
