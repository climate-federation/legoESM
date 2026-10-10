"""Coupled CMIP lane exchange closure (review 2026-10-10, F17/F18/F35/F36/F38).

* F17: land runoff on dry ocean cells is routed to the nearest wet cell, so
  what the land exports equals what the ocean applies (shared grid: roundoff).
* F18: across grids the ice->ocean channels (basal heat, salt, melt water,
  stress) pre-weighted by the atmosphere water fraction keep their integral on
  the wet ocean instead of losing the dry-cell share.
* F38: the flux handback is the dt-weighted mean of the sub-step responses
  (integral received == integral sent) and carries the surface stress, which
  the compiled lane packs into the atmosphere's prescribed-stress channel.
* F35: under diurnal_cycle the coupler forces the surfaces with the
  segment-MEAN net radiation, not one instantaneous sun position.
* F36: in a coupled run a land-mask file sets the land fraction of the
  coupler's land tile without activating the atmosphere's own slab land.

Builds grids and runs tiny JAX models: run on a compute node (sbatch).
"""
from __future__ import annotations

import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.driver.coupled_esm_driver import CoupledESMDriver  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tile(shape):
    z = jnp.zeros(shape)
    return types.SimpleNamespace(
        albedo=jnp.full(shape, 0.06), lw_up=jnp.full(shape, 400.0),
        shflx=jnp.full(shape, 10.0), lhflx=jnp.full(shape, 50.0),
        surface_mass_flux=jnp.full(shape, 50.0 / constants.L_v),
        tau_x=z, tau_y=z)


def _prev(shape, **chan):
    z = jnp.zeros(shape)
    names = ("river_runoff_flux", "ice_lake_freshwater_flux",
             "ocean_heat_extraction", "salt_flux",
             "ocean_stress_x", "ocean_stress_y")
    return types.SimpleNamespace(
        **{n: jnp.asarray(chan.get(n, z)) for n in names})


def _assemble(stub, prev, ocean_shape):
    """Real _assemble_ocean_forcing on a light stub (ocean-grid forcing)."""
    stub._last_sfc_response = prev
    sst = jnp.full(ocean_shape, 290.0)
    cur = jnp.zeros(ocean_shape)
    stub._coupler_cfg = None
    stub._ocean_surface_KuvC = lambda: (sst, cur, cur)
    atm_forcing = types.SimpleNamespace(
        sw_down=jnp.full(ocean_shape, 300.0),
        lw_down=jnp.full(ocean_shape, 350.0),
        precip_total=jnp.full(ocean_shape, 2.0e-5))
    with mock.patch("legoesm.coupler.coupler.ocean_tile_response",
                    return_value=_tile(ocean_shape)):
        return CoupledESMDriver._assemble_ocean_forcing(stub, atm_forcing)


def _integral(field, area, wet=None):
    f = np.asarray(field, np.float64)
    if wet is not None:
        f = f * np.asarray(wet, np.float64)
    return float(np.sum(f * np.asarray(area, np.float64)))


# ---------------------------------------------------------------------------
# F17: shared grid, all runoff on dry cells
# ---------------------------------------------------------------------------
def _shared_grid_stub(route=True):
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(8)
    shape = tuple(g.grid_shape_2d)
    wet = np.ones(shape)
    wet[2:5, 3:9] = 0.0          # a continent
    wet[0, :] = 0.0              # an Antarctic row, far from some wet cells
    stub = types.SimpleNamespace(
        _ocean_land_mask=jnp.asarray(wet), _ocean_grid=g,
        _ocean_area_w=g.grid_area, _atm=types.SimpleNamespace(grid=g))
    if route:
        CoupledESMDriver._build_dry_to_wet_map(stub)
    else:
        stub._dry_to_wet_map = None
    return stub, g, wet, shape


def test_f17_land_runoff_exported_equals_applied_shared_grid():
    stub, g, wet, shape = _shared_grid_stub()
    rng = np.random.default_rng(0)
    # land runoff = f_land * R with f_land = 1 - wet (the from_ocean partition)
    river = (1.0 - wet) * rng.uniform(1e-6, 1e-5, shape)
    sf, fw = _assemble(stub, _prev(shape, river_runoff_flux=river), shape)
    exported = _integral(river, g.grid_area)
    applied = _integral(fw.runoff, g.grid_area, wet)   # the ocean's * mask
    assert exported > 0.0
    np.testing.assert_allclose(applied, exported, rtol=1e-12)
    assert np.all(np.asarray(fw.runoff)[wet == 0.0] == 0.0)
    # the driver's own tripwire now reads zero (to roundoff)
    np.testing.assert_allclose(
        float(stub._last_runoff_applied_integral_ranklocal),
        float(stub._last_runoff_export_integral_ranklocal), rtol=1e-12)


def test_f17_control_without_routing_loses_all_land_runoff():
    """Non-vacuity: the pre-fix path (no map) applies none of it."""
    stub, g, wet, shape = _shared_grid_stub(route=False)
    river = (1.0 - wet) * 5e-6
    _, fw = _assemble(stub, _prev(shape, river_runoff_flux=river), shape)
    assert _integral(fw.runoff, g.grid_area, wet) == 0.0


def test_f17_map_refuses_fractional_wet_mask():
    stub, *_ = _shared_grid_stub(route=False)
    stub._ocean_land_mask = stub._ocean_land_mask * 0.5
    with pytest.raises(ValueError, match="binary"):
        CoupledESMDriver._build_dry_to_wet_map(stub)


def test_f17_no_map_for_all_wet_or_no_dynamic_ocean():
    stub = types.SimpleNamespace(_ocean_land_mask=None)
    CoupledESMDriver._build_dry_to_wet_map(stub)
    assert stub._dry_to_wet_map is None
    stub, *_ = _shared_grid_stub(route=False)
    stub._ocean_land_mask = jnp.ones_like(stub._ocean_land_mask)
    CoupledESMDriver._build_dry_to_wet_map(stub)
    assert stub._dry_to_wet_map is None


# ---------------------------------------------------------------------------
# F18: lat-lon atm -> tripole ocean, pre-weighted ice channels
# ---------------------------------------------------------------------------
def _cross_grid_case():
    from legoesm.coupler.grid_remap import make_grid_remapper, remap_field
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.tripole import create_synthetic_tripole
    atm = create_latlon_grid(10)
    trip = create_synthetic_tripole(n_lat=16)
    oshape = (int(trip.n_lat), int(trip.n_lon))
    wet = np.ones(oshape)
    j0 = oshape[0] // 2
    wet[j0:j0 + 3, :] = 0.0      # coastline crossing atm cells
    rem = make_grid_remapper(atm, trip)
    f_water_atm = np.clip(np.asarray(
        remap_field(jnp.asarray(wet), rem.o2a)), 0.0, 1.0)
    return atm, trip, oshape, wet, rem, f_water_atm


@pytest.mark.parametrize("route", [True, False])
def test_f18_cross_grid_ice_channels_conserved_on_wet_ocean(route):
    from legoesm.coupler.grid_remap import remap_field
    atm, trip, oshape, wet, rem, f_w = _cross_grid_case()
    ashape = tuple(atm.grid_shape_2d)
    rng = np.random.default_rng(1)
    chans = {   # per-water-area ice exchange x atm f_water (tile_fractions)
        "ice_lake_freshwater_flux": f_w * rng.uniform(1e-6, 2e-6, ashape),
        "ocean_heat_extraction": f_w * rng.uniform(5.0, 20.0, ashape),
        "salt_flux": f_w * rng.uniform(1e-8, 3e-8, ashape),
        "ocean_stress_x": f_w * rng.uniform(0.01, 0.1, ashape),
        "ocean_stress_y": f_w * rng.uniform(-0.1, -0.01, ashape),
    }
    stub = types.SimpleNamespace(
        _ocean_land_mask=jnp.asarray(wet), _ocean_grid=trip,
        _grid_remapper=rem, _ocean_area_w=trip.grid_area,
        _atm=types.SimpleNamespace(grid=atm))
    if route:
        CoupledESMDriver._build_dry_to_wet_map(stub)
    else:
        stub._dry_to_wet_map = None
    sf0, _ = _assemble(stub, _prev(ashape), oshape)
    sf, fw = _assemble(stub, _prev(ashape, **chans), oshape)
    applied = {
        "ice_lake_freshwater_flux": fw.ice_fw,
        "ocean_heat_extraction": sf0.q_net - sf.q_net,   # q_net -= ohe
        "salt_flux": sf.salt_flux,
        "ocean_stress_x": sf0.tau_x - sf.tau_x,          # tau -= ice stress
        "ocean_stress_y": sf0.tau_y - sf.tau_y,
    }
    area_o = trip.grid_area
    for name, a_field in chans.items():
        r_field = remap_field(jnp.asarray(a_field), rem.a2o)
        remapped = _integral(r_field, area_o)
        got = _integral(applied[name], area_o, wet)
        if route:
            # exact w.r.t. what the conservative remap delivered
            np.testing.assert_allclose(got, remapped, rtol=1e-10, err_msg=name)
        else:
            # the pre-fix loss: exactly the dry-cell share is discarded, and it
            # is not negligible on this coastline
            np.testing.assert_allclose(got, _integral(r_field, area_o, wet),
                                       rtol=1e-10, err_msg=name)
            assert abs(remapped - got) > 1e-3 * abs(remapped), name


# ---------------------------------------------------------------------------
# F38: segment-mean handback incl. stress
# ---------------------------------------------------------------------------
def _response(shape, k):
    from legoesm.core.coupling_fields import SurfaceToAtm
    vals = {f: jnp.full(shape, 1.0 + k) for f in SurfaceToAtm._fields}
    vals["T_sfc"] = vals["T_rad"] = jnp.full(shape, 280.0 + k)
    vals["emissivity"] = jnp.full(shape, 0.97)
    vals["tau_x"] = jnp.full(shape, -0.1 * (k + 1))
    vals["tau_y"] = jnp.full(shape, 0.05 * (k + 1))
    return SurfaceToAtm(**vals)


def _hook_stub(shape, n_sub, handback=True):
    from legoesm.driver.coupled_config import CoupledConfig
    calls = {"k": 0}

    def step_surface(state, forcing, tiles, **kw):
        r = _response(shape, calls["k"])
        calls["k"] += 1
        return state, r

    z = jnp.zeros(shape)
    ident = types.SimpleNamespace(a2o=None, o2a=None, identity=True)
    atm = types.SimpleNamespace(_calendar_for_radiation=lambda d: (1, 0.0),
                                get_sfc_flux_override=None)
    stub = types.SimpleNamespace(
        coupled_cfg=CoupledConfig(couple_surface_fluxes=handback,
                                  coupling_dt=3600.0),
        _atm=atm, _grid_remapper=ident, _sfc_state=None, _tile_config=None,
        _last_sfc_response=None, _sfc_flux_handback=None,
        _build_atm_forcing=lambda day: types.SimpleNamespace(precip_total=z),
        _step_ocean=lambda *a, **k: None,
        _ocean_surface_KuvC=lambda: (z, z, z),
        _step_surface=step_surface,
        _step_co2_tracer=lambda dt: None,
        _update_co2_radiation=lambda: None,
        _log_coupled_diag=lambda day, dt: None,
    )
    return stub


def test_f38_handback_is_segment_mean_and_closes_the_integral():
    shape, n_sub = (3, 4), 6
    stub = _hook_stub(shape, n_sub)
    CoupledESMDriver._override_sfc_fluxes(stub)
    assert stub._atm.get_sfc_flux_override(0.0) == (None,) * 5
    CoupledESMDriver._segment_hook(stub, None, 1.0, n_sub * 3600.0)
    sh, lh, ev, tx, ty = stub._atm.get_sfc_flux_override(1.0)
    sent = {n: sum(float(getattr(_response(shape, k), n)[0, 0]) * 3600.0
                   for k in range(n_sub))
            for n in ("shflx", "lhflx", "surface_mass_flux", "tau_x", "tau_y")}
    seg = n_sub * 3600.0
    for name, got in zip(sent, (sh, lh, ev, tx, ty)):
        np.testing.assert_allclose(float(got[0, 0]) * seg, sent[name],
                                   rtol=1e-12, err_msg=name)
    # non-vacuity: the last sub-step (the pre-fix handback) differs
    assert float(_response(shape, n_sub - 1).shflx[0, 0]) != float(sh[0, 0])


def test_f38_refuses_sub_step_unequal_to_coupler_window():
    """The coupler emits a window MEAN on the closing sub-step; a handback mean
    over emitted responses is exact only when window == sub-step."""
    from legoesm.coupler.config import CouplerConfig
    stub = _hook_stub((2, 2), 4)
    stub._coupler_cfg = CouplerConfig(coupling_dt=7200.0)
    with pytest.raises(ValueError, match="flux window"):
        CoupledESMDriver._segment_hook(stub, None, 1.0, 4 * 3600.0)


def test_f38_no_accumulation_when_flag_off():
    stub = _hook_stub((2, 2), 3, handback=False)
    CoupledESMDriver._segment_hook(stub, None, 1.0, 3 * 3600.0)
    assert stub._sfc_flux_handback is None


def test_f38_compiled_lane_packs_stress_into_the_atmosphere():
    """Real C8 run: segment 2 packs the handed-back stress as the
    atmosphere's sfc_taux/tauy_override."""
    from legoesm.driver import compiled_segments as cs
    drv = _cube_driver(days=2, couple_surface_fluxes=True)
    seen = []
    real = cs.pack_forcing

    def spy(*a, **k):
        seen.append((k.get("sfc_taux_override"), k.get("sfc_tauy_override"),
                     drv._sfc_flux_handback))
        return real(*a, **k)

    with mock.patch.object(cs, "pack_forcing", spy):
        assert drv.run() == "COMPLETED"
    assert seen[0][0] is None                       # nothing coupled yet
    later = [s for s in seen[1:] if s[0] is not None]
    assert later, "no segment packed the stress handback"
    tx, ty, hb = later[0]
    np.testing.assert_array_equal(np.asarray(tx), np.asarray(hb[3]))
    np.testing.assert_array_equal(np.asarray(ty), np.asarray(hb[4]))
    assert float(jnp.max(jnp.abs(tx))) > 0.0


def test_f38_refused_on_a_lane_that_never_reads_it():
    from legoesm.driver.model_driver import ModelDriver
    md = types.SimpleNamespace(get_sfc_flux_override=lambda d: None)
    with pytest.raises(NotImplementedError, match="couple_surface_fluxes"):
        ModelDriver._reject_unconsumed_flux_override(md, "the MPAS column loop")
    ModelDriver._reject_unconsumed_flux_override(
        types.SimpleNamespace(get_sfc_flux_override=None), "x")


# ---------------------------------------------------------------------------
# F35: segment-mean surface radiation under diurnal_cycle
# ---------------------------------------------------------------------------
def _cube_driver(days=2, diurnal=False, couple_surface_fluxes=False,
                 land_mask_path="", slab_land_active=False):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    from legoesm.driver.coupled_config import PRESETS
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic"),
        output=OutputConfig(diag_days=1), radiation="gray", days=days,
        diurnal_cycle=diurnal, land_mask_path=land_mask_path,
        slab_land_active=slab_land_active)
    from legoesm.land.config import LandConfig
    from legoesm.land.surface_scheme import SimpleSEBConfig
    # SimpleSEB land: on this machine's CPU XLA the two-leaf canopy solver
    # aborts at compile (pre-existing, reproduced on clean main), and these
    # checks do not depend on the land surface scheme.
    cfg = PRESETS["slab_simple"](
        couple_surface_fluxes=couple_surface_fluxes,
        land_config=LandConfig(surface_scheme=SimpleSEBConfig()))
    drv = CoupledESMDriver(atm, cfg)
    drv.setup()
    return drv


def test_f35_surfaces_get_segment_mean_not_one_sun_position():
    drv = _cube_driver(days=1, diurnal=True)
    assert drv.run() == "COMPLETED"
    aux = drv._atm._carry_aux
    held = np.asarray(aux["held_sw_net_sfc"])
    mean = np.asarray(aux["seg_sw_net_sfc"])
    lat = np.rad2deg(np.asarray(drv._atm._grid_lat))
    trop = np.abs(lat) < 20.0
    # one UTC instant: night columns in the tropics; the day mean has none
    assert np.any(held[trop] <= 0.0)
    assert np.all(mean[trop] > 0.0)
    f = drv._build_atm_forcing(1.0)
    assert np.all(np.asarray(f.sw_down)[trop] > 0.0)
    # a lane without the means is refused under diurnal_cycle
    aux.pop("seg_sw_net_sfc")
    aux.pop("seg_lw_net_sfc")
    with pytest.raises(NotImplementedError, match="diurnal_cycle"):
        drv._build_atm_forcing(1.0)


# ---------------------------------------------------------------------------
# F36: coupled land mask sets the coupler's land, not the atm slab land
# ---------------------------------------------------------------------------
def _mask_file(tmp_path):
    import xarray as xr
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.linspace(0.5, 359.5, 360)
    land = np.where((np.abs(lat)[:, None] < 30.0)
                    & (lon[None, :] > 0.0) & (lon[None, :] < 60.0), 100.0, 0.0)
    p = tmp_path / "sftlf.nc"
    xr.Dataset({"sftlf": (("lat", "lon"), land)},
               coords={"lat": lat, "lon": lon}).to_netcdf(p)
    return str(p)


def test_f36_land_mask_feeds_coupler_land_not_atm_slab_land(tmp_path):
    drv = _cube_driver(days=1, land_mask_path=_mask_file(tmp_path))
    f_atm = np.asarray(drv._atm._f_land)
    assert f_atm.max() > 0.5                         # real land arrived
    np.testing.assert_array_equal(
        np.asarray(drv._tile_config.f_land), f_atm)  # coupler uses it
    assert drv._atm.physics.slab_land_active is False


def test_f36_explicit_atm_slab_land_refused_in_coupled_run(tmp_path):
    with pytest.raises(ValueError, match="second land model"):
        _cube_driver(days=1, land_mask_path=_mask_file(tmp_path),
                     slab_land_active=True)
