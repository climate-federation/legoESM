"""Two-leaf canopy aerodynamic surface temperature ``Tc`` (T_canopy_air).

The two-leaf canopy presents the atmosphere with TWO distinct surface
temperatures:

* the **aerodynamic** temperature for sensible-heat coupling — the canopy
  air-space temperature ``Tc`` (a conductance-weighted blend of the
  above-canopy air, sunlit/shaded leaves, and ground; the exchange node
  ``H_tot = rho*cp*(Tc - Ta)/Ra``), exposed as ``SurfaceFluxOutput.T_canopy_air``
  and reported by the coupler as the tile ``T_sfc``; and
* the **radiometric** temperature for longwave — the LW-derived equivalent
  ``T_surface`` (``eps_col*sigma*T_surface^4 = LW_emit``), reported as ``T_rad``.

These tests pin that ``T_canopy_air`` is populated, is a genuine convex blend of
its sources (so bounded by their min/max), and is distinct from the radiometric
``T_surface`` for a sunlit vegetated cell.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.surface_scheme.two_leaf_canopy import (
    compute_two_leaf_canopy_fluxes,
)


def _forcing(ncol: int, *, sw_down: float, cos_zenith: float, Ta: float) -> AtmToSurface:
    return AtmToSurface(
        T_lowest=jnp.full(ncol, Ta),
        q_lowest=jnp.full(ncol, 0.010),
        u_lowest=jnp.full(ncol, 3.0),
        v_lowest=jnp.full(ncol, 0.0),
        p_lowest=jnp.full(ncol, 98000.0),
        p_surface=jnp.full(ncol, 101325.0),
        rho_lowest=jnp.full(ncol, 1.18),
        sw_down=jnp.full(ncol, sw_down),
        lw_down=jnp.full(ncol, 360.0),
        cos_zenith=jnp.full(ncol, cos_zenith),
        precip_total=jnp.zeros(ncol),
        precip_snow=jnp.zeros(ncol),
        co2_ppmv=jnp.full(ncol, 420.0),
        has_radiation=True,
        has_precipitation=False,
    )


def _run(ncol: int, *, sw_down: float, cos_zenith: float, Ta: float, T_soil: float):
    """Drive the canopy with a fixed-soil-temperature thermal callback."""
    T_soil_top = jnp.full(ncol, T_soil)
    forcing = _forcing(ncol, sw_down=sw_down, cos_zenith=cos_zenith, Ta=Ta)
    # Trivial soil thermal closure: hold the soil skin temperature fixed so the
    # test exercises only the canopy closure (no soil-thermal feedback needed).
    def _soil_thermal_fn(G, dt_):
        return T_soil_top
    return compute_two_leaf_canopy_fluxes(
        T_soil_top=T_soil_top,
        forcing=forcing,
        canopy_config=TwoLeafCanopyConfig(max_iters=30),
        land_config=MultiLayerLandConfig(),
        canopy_params=None,
        w_frac_rz=jnp.full(ncol, 0.7),
        wind_speed=jnp.full(ncol, 3.0),
        wind_dir_x=jnp.ones(ncol),
        wind_dir_y=jnp.zeros(ncol),
        soil_thermal_fn=_soil_thermal_fn,
        dt=1800.0,
    )


def test_t_canopy_air_populated_and_finite():
    out = _run(2, sw_down=700.0, cos_zenith=0.8, Ta=295.0, T_soil=292.0)
    assert out.T_canopy_air is not None
    assert out.T_canopy_air.shape == (2,)
    assert jnp.all(jnp.isfinite(out.T_canopy_air))
    # Physically reasonable surface temperature.
    assert jnp.all(out.T_canopy_air > 250.0)
    assert jnp.all(out.T_canopy_air < 340.0)


def test_t_canopy_air_is_convex_blend_of_sources():
    # Tc = conductance-weighted blend of (Ta, Tf_Sun, Tf_Sh, Ts) -> bounded by
    # the min/max of those four sources (all conductances >= 0).
    out = _run(1, sw_down=700.0, cos_zenith=0.8, Ta=295.0, T_soil=292.0)
    Ta = 295.0
    sources = jnp.stack([
        jnp.full((1,), Ta),
        out.Tf_Sun,
        out.Tf_Sh,
        out.Ts_solve,
    ])
    lo = jnp.min(sources, axis=0)
    hi = jnp.max(sources, axis=0)
    tol = 1e-6
    assert jnp.all(out.T_canopy_air >= lo - tol)
    assert jnp.all(out.T_canopy_air <= hi + tol)


def test_t_canopy_air_distinct_from_radiometric_when_sunlit():
    # For a sunlit vegetated cell the aerodynamic Tc (near the air/leaf blend)
    # and the LW-derived radiometric T_surface are genuinely different temps.
    out = _run(1, sw_down=800.0, cos_zenith=0.9, Ta=295.0, T_soil=290.0)
    assert float(jnp.abs(out.T_canopy_air[0] - out.T_surface[0])) > 0.05


def test_t_canopy_air_between_air_and_leaf_in_daytime():
    # Strong daytime SW heats the leaves above the air; the aerodynamic node Tc
    # sits between the air temperature and the (warmer) sunlit leaf.
    out = _run(1, sw_down=850.0, cos_zenith=0.9, Ta=295.0, T_soil=293.0)
    Tc = float(out.T_canopy_air[0])
    Tf_sun = float(out.Tf_Sun[0])
    assert Tf_sun > 295.0          # leaves warmer than air at midday
    assert 295.0 - 1.0 <= Tc <= Tf_sun + 1.0
