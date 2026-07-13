"""Phase 2b Stage 3 — multi-layer snow column coupled to the canopy + soil.

Covers the ``snow_scheme="multilayer"`` path in ``step_multilayer_land`` (two-leaf
canopy): it must (1) accumulate snow and INSULATE the soil (the boreal cold-bias
fix — the soil surface stays warmer than the snow-blind ``"single"`` budget under
cold air), (2) conserve WATER over a run (Δstorage = ∫(precip − ET − runoff)), and
(3) fail loudly on the unsupported scheme combinations it is scoped away from.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land,
)
from legoesm.land.snow_bands import ElevationSnowBandConfig
from legoesm.land.snow_column import total_water
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig

NCOL = 3
DT = 1800.0


def _cfg(scheme):
    return MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=20), snow_scheme=scheme)


def _forcing(T_air, snowfall, rain):
    return AtmToSurface(
        T_lowest=jnp.full(NCOL, T_air), q_lowest=jnp.full(NCOL, 0.001),
        u_lowest=jnp.full(NCOL, 4.0), v_lowest=jnp.full(NCOL, 1.0),
        p_lowest=jnp.full(NCOL, 98000.0), p_surface=jnp.full(NCOL, 101325.0),
        rho_lowest=jnp.full(NCOL, 1.3), sw_down=jnp.full(NCOL, 80.0),
        lw_down=jnp.full(NCOL, 230.0), cos_zenith=jnp.full(NCOL, 0.3),
        precip_total=jnp.full(NCOL, snowfall + rain),
        precip_snow=jnp.full(NCOL, snowfall),
        co2_ppmv=jnp.full(NCOL, 420.0), has_radiation=True, has_precipitation=True)


def _run(scheme, snowfall, rain, T_air, T_init, nsteps):
    cfg = _cfg(scheme)
    state = init_multilayer_land_state(
        NCOL, cfg, T_init=T_init, theta_init=0.3,
        TgC_init=T_init - constants.T_freeze)
    lat = jnp.full(NCOL, jnp.deg2rad(60.0))
    step = jax.jit(lambda s, f: step_multilayer_land(
        s, f, cfg, U_min=1.0, dt=DT, lat=lat, doy=15.0))
    f = _forcing(T_air, snowfall, rain)
    resp = None
    for _ in range(nsteps):
        state, resp, _ = step(state, f)
    return cfg, state, resp, f


def _column_water(cfg, state):
    """Total land water storage [kg/m^2]: snow + soil moisture + surface pond."""
    grid = make_soil_grid(cfg.soil_grid)
    swe = (total_water(state.snow_column) if state.snow_column is not None
           else state.snow_depth)
    soil = jnp.sum(state.theta_soil * grid.dz[None, :], axis=-1) * constants.rho_water
    pond = (state.surface_water * constants.rho_water
            if state.surface_water is not None else 0.0)
    return swe + soil + pond


def test_multilayer_snow_accumulates_and_insulates():
    """A cold column (T_air=258 K) with steady snowfall: the multilayer column
    accumulates a real pack and keeps the SOIL SURFACE warmer than the snow-blind
    single budget (the boreal cold-bias mechanism)."""
    _, st_ml, resp_ml, _ = _run("multilayer", 1e-3, 0.0, 258.0, 272.0, 120)
    _, st_s, _, _ = _run("single", 1e-3, 0.0, 258.0, 272.0, 120)

    assert jnp.all(jnp.isfinite(st_ml.T_soil))
    assert jnp.all(jnp.isfinite(resp_ml.T_sfc))
    # A real pack built up (well above the thermal-activation threshold).
    assert float(total_water(st_ml.snow_column)[0]) > 100.0
    # Insulation: the multilayer soil surface is meaningfully warmer than single,
    # which overcools toward the 258 K air.
    assert float(st_ml.T_soil[0, 0]) > float(st_s.T_soil[0, 0]) + 3.0
    # And it has NOT overcooled to near air temperature the way single does.
    assert float(st_ml.T_soil[0, 0]) > 264.0


def test_multilayer_snow_water_conservation():
    """Δ(snow + soil + pond) == ∫(precip − ET − runoff) dt over the run, to
    machine tolerance (the column drainage + Richards both conserve mass)."""
    cfg = _cfg("multilayer")
    state = init_multilayer_land_state(
        NCOL, cfg, T_init=270.0, theta_init=0.3, TgC_init=-3.0)
    lat = jnp.full(NCOL, jnp.deg2rad(60.0))
    step = jax.jit(lambda s, f: step_multilayer_land(
        s, f, cfg, U_min=1.0, dt=DT, lat=lat, doy=15.0))
    # Mixed snow + rain so accumulation, drainage, infiltration and ET all fire.
    f = _forcing(T_air=272.0, snowfall=5e-4, rain=2e-4)

    w0 = _column_water(cfg, state)
    net_in = jnp.zeros(NCOL)
    for _ in range(60):
        state, resp, _ = step(state, f)
        # precip in; ET (surface_mass_flux, +up=drying) + runoff (freshwater_flux)
        # out — all kg/m^2/s.
        net_in = net_in + (f.precip_total - resp.surface_mass_flux
                           - resp.freshwater_flux) * DT
    w1 = _column_water(cfg, state)
    resid = np.asarray(w1 - w0 - net_in)
    # Storage change must match the integrated boundary fluxes (no spurious source
    # or sink at the snow<->soil seam).
    assert np.max(np.abs(resid)) < 1e-6, f"water budget residual {resid}"


def test_multilayer_requires_two_leaf_canopy():
    """Scoped to the two-leaf canopy: multilayer + SimpleSEB raises (not a silent
    fall-through to the single-node budget)."""
    cfg = MultiLayerLandConfig(surface_scheme=SimpleSEBConfig(),
                               snow_scheme="multilayer")
    state = init_multilayer_land_state(NCOL, cfg, T_init=270.0)
    f = _forcing(270.0, 1e-4, 0.0)
    with pytest.raises(ValueError, match="TwoLeafCanopyConfig"):
        step_multilayer_land(state, f, cfg, U_min=1.0, dt=DT,
                             lat=jnp.zeros(NCOL))


def test_multilayer_and_bands_mutually_exclusive():
    """Multilayer column and elevation bands are two snow representations — enabling
    both must raise."""
    cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(),
        snow_scheme="multilayer",
        elev_bands=ElevationSnowBandConfig(band_dz=jnp.zeros((NCOL, 5))))
    state = init_multilayer_land_state(NCOL, cfg, T_init=270.0)
    f = _forcing(270.0, 1e-4, 0.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        step_multilayer_land(state, f, cfg, U_min=1.0, dt=DT,
                             lat=jnp.zeros(NCOL))
