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
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.coupler.config import TileConfig
from legoesm.coupler.tile_fractions import compute_tile_fractions
from legoesm.coupler.ocean_forcing import ice_ocean_forcing_from_ice_response
from legoesm import constants


def _zero_tile(shape):
    z = jnp.zeros(shape)
    return TileResponse(
        T_sfc=z, albedo=z, emissivity=z, z0=z, q_surface=z, shflx=z,
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


def test_brine_real_salt_flux_raises_ocean_salinity():
    """The EXPLICIT sea-ice salt_flux (real brine-rejection salt mass, +into
    ocean) must be delivered as a real top-layer salt source and RAISE salinity
    — independent of (and in addition to) the freshwater virtual-salt path
    (#F11 explicit brine salt-mass ocean source)."""
    nlat, nlon = 24, 48
    shape = (nlat, nlon)
    grid, ocean, model = _ocean(nlat, nlon, n_levels=6)
    om = ocean.land_mask.data
    # Synthetic brine: salt INTO ocean (>0), ZERO freshwater (isolate salt path).
    ice = _zero_tile(shape)._replace(
        salt_flux=jnp.where(om > 0.5, 1e-4, 0.0),
        freshwater_flux=jnp.zeros(shape))
    fracs = compute_tile_fractions(
        TileConfig(f_land=(1.0 - om), f_lake=jnp.zeros(shape)),
        jnp.where(om > 0.5, 0.6, 0.0))
    fw, sf = ice_ocean_forcing_from_ice_response(ice, fracs)
    f_water = fracs.f_ocean + fracs.f_ice
    assert jnp.allclose(sf.salt_flux, f_water * ice.salt_flux)
    assert jnp.allclose(fw.ice_fw, 0.0)  # FW channel inactive here
    S0 = ocean.S.data[..., 0]
    o = ocean
    for _ in range(6):
        o = model.step(o, 1800.0, freshwater=fw, surface_forcing=sf)
    dS = jnp.where(om > 0.5, o.S.data[..., 0] - S0, 0.0)
    assert float(jnp.max(dS)) > 1e-6, "brine salt-in did not raise ocean salinity"


def test_cubed_sphere_external_scheme_two_way():
    """F11 two-way on the CUBED-SPHERE OceanModel via the 'external' surface-
    forcing scheme: the SAME helper output drives FW freshening, real brine-salt
    salinification, and ice-stress currents on the cubed sphere (#F11 phase 2)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.state import OceanConfig, OceanSurfaceForcing
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.init import rest_state_ocean
    n = 16
    shape = (6, n, n)
    grid = create_cubed_sphere(n)
    zc = create_ocean_z_star(n_levels=6, H_max=4000.0, dz_surface=10.0, dz_deep=500.0)
    ocean = rest_state_ocean(grid, zc, S_uniform=35.0, H_max=4000.0)
    base = OceanPhysicsConfig()
    phys = OceanPhysicsConfig(
        surface_forcing=type(base.surface_forcing)(scheme="external"),
        vertical_mixing=type(base.vertical_mixing)(scheme="none"),
        lateral_mixing=type(base.lateral_mixing)(scheme="none"),
        shortwave_penetration=None)
    model = OceanModel(grid, zc, config=OceanConfig(physics=phys))
    om = (ocean.H_bathy.data > 1.0).astype(ocean.S.data.dtype)

    # Synthetic ice exchange: melt (FW>0), brine (salt>0), eastward stress.
    ice = _zero_tile(shape)._replace(
        freshwater_flux=jnp.where(om > 0.5, 2e-4, 0.0),
        salt_flux=jnp.where(om > 0.5, 1e-4, 0.0),
        ocean_stress_x=jnp.where(om > 0.5, 0.1, 0.0),
        ocean_stress_y=jnp.zeros(shape))
    fracs = compute_tile_fractions(
        TileConfig(f_land=(1.0 - om), f_lake=jnp.zeros(shape)),
        jnp.where(om > 0.5, 0.8, 0.0))
    fw, sf = ice_ocean_forcing_from_ice_response(ice, fracs)

    # Cubed-sphere OceanModel.step takes only surface_forcing (external scheme).
    S0 = ocean.S.data[..., 0]
    o = ocean
    for _ in range(5):
        o = model.step(o, 1800.0, surface_forcing=sf)
    assert jnp.all(jnp.isfinite(o.S.data)) and jnp.all(jnp.isfinite(o.u.data))
    # Net of FW dilution (-) + brine salt (+): with these magnitudes salt wins,
    # so salinity rises somewhere; the channels are both active + finite.
    dS = jnp.where(om > 0.5, o.S.data[..., 0] - S0, 0.0)
    assert float(jnp.max(jnp.abs(dS))) > 1e-6, "external scheme: no salinity response"
    # Ice stress drove currents.
    assert float(jnp.max(jnp.abs(o.u.data))) > 1e-5, "external scheme: no currents"


def test_cubed_sphere_external_fw_freshens_salt_salinifies():
    """On the cubed sphere, FW-only freshens and salt-only salinifies (isolated
    channels through the 'external' scheme), confirming both signs."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.state import OceanConfig, OceanSurfaceForcing
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.init import rest_state_ocean
    n = 16
    grid = create_cubed_sphere(n)
    zc = create_ocean_z_star(n_levels=6, H_max=4000.0, dz_surface=10.0, dz_deep=500.0)
    ocean = rest_state_ocean(grid, zc, S_uniform=35.0, H_max=4000.0)
    base = OceanPhysicsConfig()
    model = OceanModel(grid, zc, config=OceanConfig(physics=OceanPhysicsConfig(
        surface_forcing=type(base.surface_forcing)(scheme="external"),
        vertical_mixing=type(base.vertical_mixing)(scheme="none"),
        lateral_mixing=type(base.lateral_mixing)(scheme="none"),
        shortwave_penetration=None)))
    om = (ocean.H_bathy.data > 1.0).astype(ocean.S.data.dtype)
    S0 = ocean.S.data[..., 0]

    o = ocean
    for _ in range(5):
        o = model.step(o, 1800.0, surface_forcing=OceanSurfaceForcing(
            freshwater=jnp.where(om > 0.5, 2e-4, 0.0)))
    dS_fw = jnp.where(om > 0.5, o.S.data[..., 0] - S0, jnp.nan)
    assert float(jnp.nanmean(dS_fw)) < -1e-6, "FW-only did not freshen"

    o = ocean
    for _ in range(5):
        o = model.step(o, 1800.0, surface_forcing=OceanSurfaceForcing(
            salt_flux=jnp.where(om > 0.5, 1e-4, 0.0)))
    dS_salt = jnp.where(om > 0.5, o.S.data[..., 0] - S0, jnp.nan)
    assert float(jnp.nanmean(dS_salt)) > 1e-6, "salt-only did not salinify"


def test_external_scheme_uses_atmosphere_tau_convention():
    """The 'external' scheme applies ocean reaction = -tau (ATMOSPHERE
    convention), the OPPOSITE of 'prescribed' (+tau on-ocean).  A raw POSITIVE
    tau_x (NOT routed through the F11 helper's pre-negation) must therefore drive
    a WESTWARD current.  Pins the documented sign contract directly."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.state import OceanConfig, OceanSurfaceForcing
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.init import rest_state_ocean
    n = 16
    grid = create_cubed_sphere(n)
    zc = create_ocean_z_star(n_levels=6, H_max=4000.0, dz_surface=10.0, dz_deep=500.0)
    ocean = rest_state_ocean(grid, zc, S_uniform=35.0, H_max=4000.0)
    base = OceanPhysicsConfig()
    model = OceanModel(grid, zc, config=OceanConfig(physics=OceanPhysicsConfig(
        surface_forcing=type(base.surface_forcing)(scheme="external"),
        vertical_mixing=type(base.vertical_mixing)(scheme="none"),
        lateral_mixing=type(base.lateral_mixing)(scheme="none"),
        shortwave_penetration=None)))
    om = (ocean.H_bathy.data > 1.0).astype(ocean.u.data.dtype)
    o = ocean
    # ONE step isolates the direct stress tendency before Coriolis rotates it.
    o = model.step(o, 1800.0, surface_forcing=OceanSurfaceForcing(
        tau_x=jnp.where(om > 0.5, 0.1, 0.0)))
    u_top = jnp.where(om > 0.5, o.u.data[..., 0], jnp.nan)
    assert float(jnp.nanmean(u_top)) < -1e-6, (
        "external scheme: positive tau_x must drive WESTWARD current "
        "(ocean reaction = -tau)")


def test_latlon_rejects_external_scheme_to_avoid_double_apply():
    """The lat-lon C-grid applies OceanSurfaceForcing DIRECTLY in its dynamics,
    so routing the SAME forcing through the 'external' physics scheme would
    double-apply momentum/heat/salt.  The model must reject that config at
    construction (fail loud) rather than silently double-count."""
    import pytest
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    grid = create_latlon_grid(n_lat=12, n_lon=24)
    zc = create_ocean_z_star(n_levels=6, H_max=4000.0, dz_surface=10.0, dz_deep=500.0)
    ocean = rest_state_latlon_cgrid_ocean(grid, zc, S_uniform=35.0)
    base = OceanPhysicsConfig()
    bad_phys = OceanPhysicsConfig(
        surface_forcing=type(base.surface_forcing)(scheme="external"))
    with pytest.raises(ValueError, match="external.*not supported on the"):
        LatLonCGridOceanModel(
            grid, zc, config=LatLonCGridOceanConfig(physics=bad_phys))


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


def test_kpp_sees_real_salt_buoyancy():
    """With KPP enabled, the REAL brine salt flux reaches the KPP salt-buoyancy
    path (surface_forcing.salt_flux) and raises salinity — the KPP-on analogue
    of the freshwater-buoyancy test for the explicit salt channel (#F11)."""
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
        salt_flux=jnp.where(om > 0.5, 1e-4, 0.0),     # brine into ocean
        freshwater_flux=jnp.zeros(shape))
    fracs = compute_tile_fractions(
        TileConfig(f_land=(1.0 - om), f_lake=jnp.zeros(shape)),
        jnp.where(om > 0.5, 0.7, 0.0))
    fw, sf = ice_ocean_forcing_from_ice_response(ice, fracs)
    assert float(jnp.max(jnp.abs(sf.salt_flux))) > 0.0
    S0 = ocean.S.data[..., 0]
    o = ocean
    for _ in range(6):
        o = model.step(o, 1800.0, freshwater=fw, surface_forcing=sf)
    assert jnp.all(jnp.isfinite(o.S.data)) and jnp.all(jnp.isfinite(o.u.data))
    dS = jnp.where(om > 0.5, o.S.data[..., 0] - S0, 0.0)
    assert float(jnp.max(dS)) > 1e-6, "KPP-on: brine salt did not raise salinity"
