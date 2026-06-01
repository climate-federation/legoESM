"""F11 two-way ice <-> ocean coupling: the sea-ice tile's ice->ocean exchange,
mapped by ``ice_ocean_forcing_from_ice_response`` to the ocean's surface forcing
structs, makes a PROGNOSTIC (salinity + momentum) ocean RESPOND — salinity
freshens under ice melt and currents are driven by the ice stress.  The
slab-ocean driver drops this exchange intentionally; this verifies the
prognostic two-way capability and the ICE-ONLY forcing contract."""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import init_dynamic_ice_state
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.coupler.config import TileConfig
from legoesm.coupler.tile_fractions import compute_tile_fractions
from legoesm.coupler.ocean_forcing import ice_ocean_forcing_from_ice_response
from legoesm import constants


def _zero_tile(shape):
    z = jnp.zeros(shape)
    return TileResponse(
        T_surface=z, albedo=z, emissivity=z, z0=z, q_surface=z, shflx=z,
        lhflx=z, tau_x=z, tau_y=z, lw_up=z, u_ocean_sfc=z, v_ocean_sfc=z,
        co2_flux=z, freshwater_flux=z, ocean_heat_extraction=z,
        ocean_stress_x=z, ocean_stress_y=z, surface_mass_flux=z, salt_flux=z)


def _warm_forcing(shape):
    f = lambda v: jnp.full(shape, v)
    return AtmToSurface(
        sw_down=f(400.0), lw_down=f(340.0), precip_total=jnp.zeros(shape),
        precip_snow=jnp.zeros(shape), T_lowest=f(282.0), q_lowest=f(6e-3),
        u_lowest=f(6.0), v_lowest=f(2.0), p_lowest=f(9.5e4), p_surface=f(1e5),
        rho_lowest=f(1.2), cos_zenith=f(0.5), co2_ppmv=f(400.0),
        has_radiation=jnp.ones(shape), has_precipitation=jnp.ones(shape))


def _ocean(nlat, nlon, n_levels=8):
    grid = create_latlon_grid(n_lat=nlat, n_lon=nlon)
    zc = create_ocean_z_star(n_levels=n_levels, H_max=4000.0,
                             dz_surface=10.0, dz_deep=500.0)
    ocean = rest_state_latlon_cgrid_ocean(grid, zc, S_uniform=35.0)
    model = LatLonCGridOceanModel(grid, zc, config=LatLonCGridOceanConfig())
    return grid, ocean, model


def test_helper_is_ice_only_with_correct_weights():
    """ice_fw / q_net / tau depend ONLY on the ice TileResponse and the tile
    fractions, with blend_tiles' weights (f_water for FW/heat, f_ice for
    stress).  Other tiles' freshwater can never enter ice_fw (the helper takes
    the ice response alone), so a driver adding atmospheric P-E-R does not
    double count."""
    nlat, nlon = 12, 24
    shape = (nlat, nlon)
    om = jnp.ones(shape)  # all ocean
    ice = _zero_tile(shape)._replace(
        freshwater_flux=jnp.full(shape, 3e-4),
        ocean_heat_extraction=jnp.full(shape, 20.0),
        ocean_stress_x=jnp.full(shape, 0.05),
        ocean_stress_y=jnp.full(shape, -0.02))
    fracs = compute_tile_fractions(
        TileConfig(f_land=jnp.zeros(shape), f_lake=jnp.zeros(shape)),
        jnp.full(shape, 0.7))
    fw, sf = ice_ocean_forcing_from_ice_response(ice, fracs)
    f_water = fracs.f_ocean + fracs.f_ice
    assert jnp.allclose(fw.ice_fw, f_water * ice.freshwater_flux)
    assert jnp.allclose(fw.precip, 0.0) and jnp.allclose(fw.evap, 0.0)
    assert jnp.allclose(fw.runoff, 0.0)  # ice-only: no atmospheric P-E-R
    assert jnp.allclose(sf.q_net, -(f_water * ice.ocean_heat_extraction))
    assert jnp.allclose(sf.tau_x, -(fracs.f_ice * ice.ocean_stress_x))
    assert jnp.allclose(sf.tau_y, -(fracs.f_ice * ice.ocean_stress_y))
    # sw_down stays None (driver's solar concern); freshwater MIRRORS ice_fw so
    # the KPP buoyancy path sees the ice freshwater (not double-counted: it is a
    # different consumer than the FreshwaterForcing salinity tendency).
    assert sf.sw_down is None
    assert jnp.allclose(sf.freshwater, fw.ice_fw)


def test_ice_melt_freshens_ocean_and_drives_currents():
    nlat, nlon = 36, 72
    shape = (nlat, nlon)
    grid, ocean, model = _ocean(nlat, nlon)
    om = ocean.land_mask.data
    st = init_dynamic_ice_state(shape)
    st = st._replace(
        h_ice=st.h_ice.replace(data=jnp.where(om > 0.5, 0.6, 0.0)),
        concentration=st.concentration.replace(
            data=jnp.where(om > 0.5, 0.8, 0.0)),
        T_ice=st.T_ice.replace(data=jnp.full(shape, 271.0)))
    ice_cfg = SeaIceConfig(n_categories=1, dynamics="free_drift", transport="none")
    warm_sst = jnp.full(shape, constants.T_freeze_ocean + 3.0)
    z = jnp.zeros(shape)
    new_ice, ice_resp = step_sea_ice(
        st, _warm_forcing(shape), warm_sst, z, z, ice_cfg,
        U_min=1.0, dt=1800.0, grid=grid)

    fracs = compute_tile_fractions(
        TileConfig(f_land=(1.0 - om), f_lake=jnp.zeros(shape)),
        new_ice.concentration.data)
    fw, sf = ice_ocean_forcing_from_ice_response(ice_resp, fracs)
    assert float(jnp.max(fw.ice_fw)) > 0.0  # melting somewhere

    S0 = ocean.S.data[..., 0]
    o = ocean
    for _ in range(6):
        o = model.step(o, 1800.0, freshwater=fw, surface_forcing=sf)
    dS = o.S.data[..., 0] - S0
    melt = (fw.ice_fw > 1e-12) & (om > 0.5)
    assert float(jnp.min(jnp.where(melt, dS, 0.0))) < -1e-6, (
        "ice melt freshwater did not freshen the prognostic ocean")
    assert float(jnp.max(jnp.abs(o.u.data))) > 1e-5, (
        "ice stress did not drive prognostic ocean currents")
    assert jnp.all(jnp.isfinite(o.S.data)) and jnp.all(jnp.isfinite(o.u.data))


def test_freeze_extracts_freshwater_raises_salinity():
    nlat, nlon = 24, 48
    shape = (nlat, nlon)
    grid, ocean, model = _ocean(nlat, nlon, n_levels=6)
    om = ocean.land_mask.data
    # Synthetic FREEZE: ice forms -> negative freshwater (water removed).
    ice = _zero_tile(shape)._replace(
        freshwater_flux=jnp.where(om > 0.5, -1e-4, 0.0))
    fracs = compute_tile_fractions(
        TileConfig(f_land=(1.0 - om), f_lake=jnp.zeros(shape)),
        jnp.where(om > 0.5, 0.6, 0.0))
    fw, sf = ice_ocean_forcing_from_ice_response(ice, fracs)
    S0 = ocean.S.data[..., 0]
    o = ocean
    for _ in range(6):
        o = model.step(o, 1800.0, freshwater=fw, surface_forcing=sf)
    dS = jnp.where(om > 0.5, o.S.data[..., 0] - S0, 0.0)
    assert float(jnp.max(dS)) > 1e-6, "freeze did not raise ocean salinity"


def test_ice_stress_drives_currents_in_correct_direction():
    """A purely EASTWARD on-ocean ice stress must drive a net EASTWARD current.
    Guards the tau sign (helper negates ocean_stress; ocean consumer negates
    external tau -> net applied = +on-ocean stress).  A single missing negation
    reverses the current."""
    nlat, nlon = 24, 48
    shape = (nlat, nlon)
    grid, ocean, model = _ocean(nlat, nlon, n_levels=6)
    om = ocean.land_mask.data
    ice = _zero_tile(shape)._replace(
        ocean_stress_x=jnp.where(om > 0.5, 0.1, 0.0),
        ocean_stress_y=jnp.zeros(shape))
    fracs = compute_tile_fractions(
        TileConfig(f_land=(1.0 - om), f_lake=jnp.zeros(shape)),
        jnp.where(om > 0.5, 0.9, 0.0))
    fw, sf = ice_ocean_forcing_from_ice_response(ice, fracs)
    o = ocean
    for _ in range(6):
        o = model.step(o, 1800.0, freshwater=fw, surface_forcing=sf)
    u_mean = float(jnp.mean(o.u.data[..., 0]))
    assert u_mean > 1e-5, (
        f"eastward ice stress did not drive eastward current (u_mean={u_mean:.3e}); "
        "tau sign convention is wrong")


def test_kpp_sees_ice_freshwater_buoyancy():
    """With KPP boundary-layer mixing enabled, the ice freshwater must reach the
    KPP salt-buoyancy path (surface_forcing.freshwater) — not just the salinity
    tendency.  The helper now populates surface_forcing.freshwater, so the
    KPP-on ocean steps finite and the salinity responds (Codex)."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    nlat, nlon = 24, 48
    shape = (nlat, nlon)
    grid = create_latlon_grid(n_lat=nlat, n_lon=nlon)
    zc = create_ocean_z_star(n_levels=6, H_max=4000.0, dz_surface=10.0, dz_deep=500.0)
    base = OceanPhysicsConfig()
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"),
        lateral_mixing=type(base.lateral_mixing)(scheme="none"),
        surface_forcing=type(base.surface_forcing)(scheme="none"),
        shortwave_penetration=None,
    )
    ocean = rest_state_latlon_cgrid_ocean(grid, zc, S_uniform=35.0)
    model = LatLonCGridOceanModel(
        grid, zc, config=LatLonCGridOceanConfig(physics=physics))
    om = ocean.land_mask.data
    ice = _zero_tile(shape)._replace(
        freshwater_flux=jnp.where(om > 0.5, 2e-4, 0.0))   # melt
    fracs = compute_tile_fractions(
        TileConfig(f_land=(1.0 - om), f_lake=jnp.zeros(shape)),
        jnp.where(om > 0.5, 0.7, 0.0))
    fw, sf = ice_ocean_forcing_from_ice_response(ice, fracs)
    # The KPP buoyancy channel is fed (the bug was this being None/zero).
    assert sf.freshwater is not None
    assert float(jnp.max(jnp.abs(sf.freshwater))) > 0.0
    S0 = ocean.S.data[..., 0]
    o = ocean
    for _ in range(6):
        o = model.step(o, 1800.0, freshwater=fw, surface_forcing=sf)
    assert jnp.all(jnp.isfinite(o.S.data)) and jnp.all(jnp.isfinite(o.u.data))
    dS = jnp.where(om > 0.5, o.S.data[..., 0] - S0, 0.0)
    assert float(jnp.min(dS)) < -1e-6, "KPP-on: ice melt did not freshen ocean"
