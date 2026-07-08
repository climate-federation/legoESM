"""Per-cell soil/vegetation albedo is preserved through the canopy path.

Regression for the LMIP two-leaf-canopy albedo bug: ``CanopyLandParams`` stores
its surface albedo in ``ALB_VIS``/``ALB_NIR`` (there is no ``albedo_veg`` field),
so ``multilayer_land`` used to read the snow-free base via
``_get(lp, "albedo_veg", config.albedo_land)`` and silently collapse the canopy
base to the scalar ``config.albedo_land``.  With ``snow_albedo_feedback=True``
(the LMIP 10-yr production setting) that made the reported/coupled albedo a
uniform base + snow bands, discarding all soil-colour + vegetation structure
even though the canopy energy balance itself used the real per-cell albedo.

The fix feeds the canopy's own RT-diagnosed ``surface_out.albedo`` (soil-colour
ALB_VIS/NIR + vegetation) as the snow-free base for the snow feedback.  These
tests assert the reported albedo (a) varies per column with the soil albedo and
(b) is NOT collapsed to the scalar ``config.albedo_land``.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy import CanopyLandParams
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land,
)
from legoesm.land.surface_scheme import TwoLeafCanopyConfig


def _daytime_forcing(ncol: int) -> AtmToSurface:
    return AtmToSurface(
        T_lowest=jnp.full(ncol, 295.0),
        q_lowest=jnp.full(ncol, 0.010),
        u_lowest=jnp.full(ncol, 3.0),
        v_lowest=jnp.full(ncol, 0.5),
        p_lowest=jnp.full(ncol, 98000.0),
        p_surface=jnp.full(ncol, 101325.0),
        rho_lowest=jnp.full(ncol, 1.18),
        sw_down=jnp.full(ncol, 700.0),
        lw_down=jnp.full(ncol, 350.0),
        cos_zenith=jnp.full(ncol, 0.8),
        precip_total=jnp.zeros(ncol),
        precip_snow=jnp.zeros(ncol),
        co2_ppmv=jnp.full(ncol, 420.0),
        has_radiation=True,
        has_precipitation=False,
    )


def _bare_params_two_soils() -> CanopyLandParams:
    """Two BARE columns (LAI=0, FNonVeg=1) with distinct soil albedos:
    col 0 = dark soil, col 1 = bright desert.  Bare so the canopy RT albedo is
    essentially the soil albedo, isolating the soil-colour signal.  Built
    directly from the public ``CanopyLandParams`` (no private helper import)."""
    f = lambda v: jnp.full(2, v)
    return CanopyLandParams(
        LAI=f(0.0), hc=f(0.1), fC4=f(0.0), FNonVeg=f(1.0),
        CI=f(1.0), kn=f(0.3),
        Vcmax25_C3_leaf=f(0.0), Vcmax25_C4_leaf=f(0.0),
        m_C3=f(9.0), m_C4=f(4.0), b0_C3=f(0.01), b0_C4=f(0.04),
        alf=f(0.3), TgC=f(20.0),
        ALB_VIS=jnp.array([0.05, 0.30]),
        ALB_NIR=jnp.array([0.10, 0.40]),
        emissivity=f(0.96), rz0m=f(0.1), rd=f(0.0),
    )


def test_canopy_reported_albedo_tracks_soil_color():
    """Reported albedo varies per column with the soil albedo and is not the
    scalar ``config.albedo_land`` (the pre-fix collapse value)."""
    ncol = 2
    cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=20),
        snow_albedo_feedback=True,   # the LMIP production setting
    )
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0)
    forcing = _daytime_forcing(ncol)
    lp = _bare_params_two_soils()

    _, response, _ = step_multilayer_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.array([0.3, 0.3]), doy=180.0, land_params=lp)

    alb = response.albedo
    assert jnp.all(jnp.isfinite(alb))
    assert jnp.all((alb >= 0.0) & (alb <= 1.0))
    # Snow-free columns: the bright-desert cell must be brighter than the dark
    # cell (per-cell soil-colour structure preserved), by a clear margin.
    assert float(alb[1] - alb[0]) > 0.10, (alb[0], alb[1])
    # Neither column collapsed onto the scalar fallback config.albedo_land (0.2):
    # at least one differs from it substantially (pre-fix, BOTH were ~0.2).
    assert abs(float(alb[0]) - cfg.albedo_land) > 0.05
    assert abs(float(alb[1]) - cfg.albedo_land) > 0.05


def test_snow_blends_on_top_of_percell_base():
    """Snow feedback still blends on top of the per-cell soil base: a snow-laden
    column is brighter than the same soil snow-free."""
    ncol = 2
    cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=20),
        snow_albedo_feedback=True,
    )
    lp = _bare_params_two_soils()
    forcing = _daytime_forcing(ncol)
    lat = jnp.array([0.3, 0.3])

    # Snow-free reference.
    state0 = init_multilayer_land_state(ncol, cfg, T_init=290.0)
    _, resp0, _ = step_multilayer_land(
        state0, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=lat, doy=180.0, land_params=lp)

    # Deep fresh snow on both columns.
    state1 = init_multilayer_land_state(ncol, cfg, T_init=270.0)
    state1 = state1._replace(
        snow_depth=jnp.full(ncol, 200.0),
        snow_age=jnp.zeros(ncol),
    )
    _, resp1, _ = step_multilayer_land(
        state1, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=lat, doy=180.0, land_params=lp)

    # Snow brightens both columns above their snow-free soil albedo.
    assert jnp.all(resp1.albedo > resp0.albedo)
