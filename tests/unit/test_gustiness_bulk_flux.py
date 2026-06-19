"""Single-column (SCM-style) tests for the convective-gustiness bulk-flux floor.

The resolved grid-mean surface wind misses sub-grid boundary-layer gustiness,
which dominates the air-sea latent/sensible flux in light-wind convective
regions.  ``apply_gustiness`` adds the Wing (2018) / Beljaars (1995) floor
``|U|_eff = sqrt(u^2 + v^2 + u_gust^2)`` so coarse coupled runs evaporate
realistically (the dry-column / weak-hydrological-cycle fix).  The same helper
is shared by the SCM (rce_surface_flux), the slab/two-layer ocean, and the
coupler ocean bulk flux.  The floor lives inside the single sqrt so the
gradient is finite even at calm wind (AD-safe for the differentiable coupler).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.bulk_flux import apply_gustiness, simple_bulk_fluxes


def test_apply_gustiness_math():
    # gustiness=0 recovers the bare wind speed sqrt(u^2+v^2).
    u = jnp.array([2.0, 3.0]); v = jnp.array([0.0, 4.0])
    assert np.allclose(np.asarray(apply_gustiness(u, v, 0.0)),
                       np.asarray(jnp.sqrt(u ** 2 + v ** 2)))
    # |U|_eff = sqrt(u^2 + v^2 + u_gust^2).
    assert float(apply_gustiness(jnp.array(2.0), jnp.array(0.0), 5.0)) == \
        pytest.approx(float(np.sqrt(4.0 + 25.0)))
    # floored at the gustiness value at calm wind.
    assert float(apply_gustiness(jnp.array(0.0), jnp.array(0.0), 5.0)) == \
        pytest.approx(5.0)


def test_gustiness_raises_evaporation_at_low_wind():
    """At a LIGHT resolved wind (2 m/s) over a warm sub-saturated ocean, the
    gustiness floor (Wing 5 m/s) must boost the latent flux — the bulk flux is
    linear in wind speed, so a starved light-wind evaporation is lifted toward
    a realistic magnitude (the dry-column fix)."""
    u = jnp.array(2.0); v = jnp.array(0.0)
    T_air = jnp.array(297.0); q_air = jnp.array(0.014)
    sst = jnp.array(300.0); q_sfc = jnp.array(0.0223)   # ~q_sat(300K, 1000 hPa)
    rho = jnp.array(1.15); Ch = 1.5e-3; Cd = 1.5e-3
    wind_raw = apply_gustiness(u, v, 0.0)               # 2.0 m/s
    wind_gust = apply_gustiness(u, v, 5.0)              # sqrt(29) = 5.39 m/s
    *_, lhflx_raw = simple_bulk_fluxes(
        u, v, T_air, q_air, sst, q_sfc, rho, wind_raw, Cd, Ch)
    *_, lhflx_gust = simple_bulk_fluxes(
        u, v, T_air, q_air, sst, q_sfc, rho, wind_gust, Cd, Ch)
    assert abs(float(lhflx_gust)) > abs(float(lhflx_raw))
    # latent flux is linear in wind speed.
    assert abs(float(lhflx_gust)) / abs(float(lhflx_raw)) == pytest.approx(
        float(wind_gust) / float(wind_raw), rel=1e-6)
    # >2x lift at this light wind (2 -> 5.39 m/s).
    assert abs(float(lhflx_gust)) > 2.0 * abs(float(lhflx_raw))


def test_apply_gustiness_ad_safe_at_calm_wind():
    """Gradient finite at EXACT calm wind u=v=0 — the floor is inside the single
    sqrt (gustiness>0 keeps the argument strictly positive).  A two-stage
    sqrt(sqrt(u^2+v^2)^2 + g^2) would NaN here (codex regression guard)."""
    f = lambda u, v: apply_gustiness(u, v, 5.0)
    gu, gv = jax.grad(f, argnums=(0, 1))(jnp.array(0.0), jnp.array(0.0))
    assert bool(jnp.isfinite(gu)) and bool(jnp.isfinite(gv))
    gu2, gv2 = jax.grad(f, argnums=(0, 1))(jnp.array(3.0), jnp.array(1.0))
    assert bool(jnp.isfinite(gu2)) and bool(jnp.isfinite(gv2))


def test_simple_ocean_config_uses_gustiness():
    """The slab/two-layer ocean exposes a gustiness field (Wing default 5.0)."""
    from legoesm.ocean.simple_ocean import SimpleOceanConfig
    assert SimpleOceanConfig().gustiness == pytest.approx(5.0)
