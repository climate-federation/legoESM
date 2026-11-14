"""P5c K4: the coupled driver when the atmosphere owns land and open water.

The ocean is forced by the window the atmosphere exported (open-water heat,
shortwave, stress, evaporation, open-water + ice precipitation, land runoff)
plus the previous exchange's ice/lake->ocean channels; the coupler steps ice
and lake with the radiation the atmosphere booked and hands them back.
Covered: the ice/lake->ocean channels of the owned exchange (against
blend_tiles), the conservative remap + dry->wet route of every export (shared
and cross grid) and its runtime tripwire, the slab's per-water-area flux, the
window wiring, the refusals and the checkpoint keys.
"""
from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.coupler.coupler import (  # noqa: E402
    OwnedOceanExchange, SurfaceHandback, initial_handback,
)
from legoesm.driver.coupled_config import CoupledConfig  # noqa: E402
from legoesm.driver.coupled_esm_driver import CoupledESMDriver  # noqa: E402

# Conservation error of the lat-lon 8 -> 16 conservative remap itself on
# these fields (measured: up to 2.8e-6 relative, the remapper's, not K4's).
REMAP_RTOL_8_TO_16 = 1.0e-5

EXPORTS = ("seg_ow_sw_net", "seg_ow_lw_net", "seg_shflx_ow", "seg_lhflx_ow",
           "seg_evap_ow", "seg_taux_ow", "seg_tauy_ow", "seg_precip_ow",
           "seg_precip_ice", "seg_runoff_land")


# ---------------------------------------------------------------------------
# the owned exchange hands the ocean what blend_tiles would
# ---------------------------------------------------------------------------
def test_owned_exchange_ocean_channels_match_blend_tiles():
    from tests.unit.test_coupler_atm_owns import (
        DT, ICE_CFG, LAKE_CFG, SST, TILES, ZU, _booked_from, _forcing,
        _owned_coupler, _state,
    )
    from legoesm.coupler.lake.two_layer_lake import step_lake
    from legoesm.coupler.tile_fractions import (
        blend_tiles, compute_tile_fractions,
    )
    from legoesm.ice.sea_ice import step_sea_ice
    s0 = _state()
    sw, lw = 250.0, 290.0
    _, _, to_o = _owned_coupler()(s0, _forcing(17.0, 3.0), TILES, SST, ZU, ZU,
                                  DT, owned_forcing=_booked_from(s0, sw, lw))
    assert isinstance(to_o, OwnedOceanExchange)
    gross = _forcing(sw, lw)
    _, ice_r = step_sea_ice(s0.ice, gross, SST, ZU, ZU, ICE_CFG,
                            CouplerConfig_U_min(), DT)
    _, lake_r = step_lake(s0.lake, gross, LAKE_CFG, CouplerConfig_U_min(), DT)
    zero = jax.tree.map(jnp.zeros_like, lake_r)
    fr0 = compute_tile_fractions(TILES, s0.ice.concentration.data)
    want = blend_tiles(zero, ice_r, zero, lake_r, fr0)
    for k in OwnedOceanExchange._fields:
        np.testing.assert_allclose(np.asarray(getattr(to_o, k)),
                                   np.asarray(getattr(want, k)),
                                   rtol=1e-13, atol=1e-18, err_msg=k)
    # live channels, so the comparison can fail
    for k in ("ice_lake_freshwater_flux", "ocean_heat_extraction",
              "ocean_stress_x"):
        assert np.max(np.abs(np.asarray(getattr(to_o, k)))) > 0.0, k


def CouplerConfig_U_min():
    from legoesm.coupler.config import CouplerConfig
    return CouplerConfig().U_min


# ---------------------------------------------------------------------------
# ocean forcing from the exports: integrals, signs, lagged channels
# ---------------------------------------------------------------------------
def _exports(shape, rng):
    """Random per-cell exports of a plausible size and sign."""
    out = {}
    for k in EXPORTS:
        if "tau" in k:
            x = rng.normal(0.0, 0.1, shape)
        elif "precip" in k or "evap" in k or "runoff" in k:
            x = rng.uniform(1e-6, 1e-5, shape)
        else:
            x = rng.uniform(10.0, 100.0, shape)
        out[k] = jnp.asarray(x)
    return out


def _prev(shape, rng):
    return OwnedOceanExchange(*(jnp.asarray(rng.normal(0.0, s, shape))
                                for s in (1e-6, 5.0, 1e-7, 0.02, 0.01)))


def _stub(atm_grid, ocean_grid=None, wet=None, route=True):
    from legoesm.coupler.grid_remap import make_grid_remapper
    d = CoupledESMDriver.__new__(CoupledESMDriver)
    d.coupled_cfg = CoupledConfig(atm_owns_surface=True)
    d._atm = types.SimpleNamespace(grid=atm_grid, _carry_aux={})
    og = atm_grid if ocean_grid is None else ocean_grid
    d._ocean_grid = og
    d._ocean_area_w = og.grid_area
    d._ocean_land_mask = None if wet is None else jnp.asarray(wet)
    d._grid_remapper = (None if ocean_grid is None
                        else make_grid_remapper(atm_grid, ocean_grid))
    if route and wet is not None:
        d._build_dry_to_wet_map()
    else:
        d._dry_to_wet_map = None
    return d


def _I(x, area, wet=None):
    x = np.asarray(x, np.float64)
    if wet is not None:
        x = x * np.asarray(wet, np.float64)
    return float(np.sum(x * np.asarray(area, np.float64)))


def _check_integrals(d, ex, prev, wet, rtol, remap_rtol=0.0):
    """Every export's integral over the wet ocean equals what was sent:
    the atm-grid integral, or across grids the integral of the remapped
    field (the remap's own conservation error, ``remap_rtol``, is the
    remapper's, measured separately; the route and mask must add none)."""
    from legoesm.coupler.grid_remap import remap_field
    a2o = None if d._grid_remapper is None else d._grid_remapper.a2o
    d._atm._carry_aux = dict(ex)
    d._owned_to_ocean = prev
    sf, fw = d._assemble_owned_ocean_forcing()
    aa, ao = d._atm.grid.grid_area, d._ocean_area_w
    q = (ex["seg_ow_sw_net"] + ex["seg_ow_lw_net"] - ex["seg_shflx_ow"]
         - ex["seg_lhflx_ow"])
    pairs = {
        "q_net": (q - prev.ocean_heat_extraction, sf.q_net),
        "sw": (ex["seg_ow_sw_net"], sf.sw_down),
        "tau_x": (ex["seg_taux_ow"] - prev.ocean_stress_x, sf.tau_x),
        "tau_y": (ex["seg_tauy_ow"] - prev.ocean_stress_y, sf.tau_y),
        "salt": (prev.salt_flux, sf.salt_flux),
        "precip": (ex["seg_precip_ow"] + ex["seg_precip_ice"], fw.precip),
        "evap": (ex["seg_evap_ow"], fw.evap),
        "runoff": (ex["seg_runoff_land"], fw.runoff),
        "ice_fw": (prev.ice_lake_freshwater_flux, fw.ice_fw),
    }
    for k, (a, o) in pairs.items():
        sent, got = _I(a, aa), _I(o, ao, wet)
        assert abs(sent) > 0.0, k
        if a2o is not None:
            remapped = _I(remap_field(a, a2o), ao)
            print("K4MEASURE remap rel err", k, abs(remapped - sent) / abs(sent))
            assert abs(remapped - sent) <= remap_rtol * abs(sent), k
            sent = remapped
        np.testing.assert_allclose(got, sent, rtol=rtol, err_msg=k)
    q_cell = (ex["seg_ow_sw_net"] + ex["seg_ow_lw_net"] - ex["seg_shflx_ow"]
              - ex["seg_lhflx_ow"])
    sent = {"heat": q_cell, "sw": ex["seg_ow_sw_net"],
            "evap": ex["seg_evap_ow"],
            "precip": ex["seg_precip_ow"] + ex["seg_precip_ice"],
            "runoff": ex["seg_runoff_land"]}
    res = d._last_owned_exchange_residual
    assert set(res) == set(sent)
    for k, x in sent.items():
        assert abs(res[k]) <= (rtol + remap_rtol) * abs(_I(x, aa)), (k, res[k])
    if wet is not None:
        for f in (sf.q_net, fw.precip, fw.runoff):
            assert np.all(np.asarray(f)[np.asarray(wet) == 0.0] == 0.0)
    return sf, fw


def test_shared_grid_exports_reach_the_wet_ocean_with_their_integral():
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(8)
    shape = tuple(g.grid_shape_2d)
    wet = np.ones(shape)
    wet[2:5, 3:9] = 0.0
    wet[0, :] = 0.0
    rng = np.random.default_rng(1)
    ex, prev = _exports(shape, rng), _prev(shape, rng)
    _check_integrals(_stub(g, wet=wet), ex, prev, wet, rtol=1e-12)


def test_cross_grid_exports_keep_their_integral():
    from legoesm.grids.latlon import create_latlon_grid
    ga, go = create_latlon_grid(8), create_latlon_grid(16)
    so = tuple(go.grid_shape_2d)
    wet = np.ones(so)
    wet[5:10, 6:18] = 0.0
    rng = np.random.default_rng(2)
    sa = tuple(ga.grid_shape_2d)
    ex, prev = _exports(sa, rng), _prev(sa, rng)
    _check_integrals(_stub(ga, go, wet=wet), ex, prev, wet, rtol=1e-12,
                     remap_rtol=REMAP_RTOL_8_TO_16)


def test_ocean_forcing_is_differentiable_in_the_exports():
    """The assembly (tripwire included) traces: d(sum q_net)/d(ow SW) is the
    integral weight the remap + route give each atm cell (1 on the shared
    grid with all-wet cells)."""
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(8)
    shape = tuple(g.grid_shape_2d)
    rng = np.random.default_rng(6)
    ex, prev = _exports(shape, rng), _prev(shape, rng)
    d = _stub(g, wet=np.ones(shape))
    d._owned_to_ocean = prev

    def loss(sw):
        d._atm._carry_aux = {**ex, "seg_ow_sw_net": sw}
        sf, _ = d._assemble_owned_ocean_forcing()
        return jnp.sum(sf.q_net)
    gr = jax.grad(loss)(ex["seg_ow_sw_net"])
    np.testing.assert_allclose(np.asarray(gr), 1.0, rtol=0, atol=0)


def test_tripwire_sees_a_dropped_dry_cell_share():
    """Planted violation: without the dry->wet route the ocean's wet mask
    drops the runoff on dry cells, and the tripwire reports the loss."""
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(8)
    shape = tuple(g.grid_shape_2d)
    wet = np.ones(shape)
    wet[2:5, 3:9] = 0.0
    rng = np.random.default_rng(3)
    ex, prev = _exports(shape, rng), _prev(shape, rng)
    d = _stub(g, wet=wet, route=False)
    d._atm._carry_aux = dict(ex)
    d._owned_to_ocean = prev
    d._assemble_owned_ocean_forcing()
    lost = _I(ex["seg_runoff_land"], g.grid_area, 1.0 - wet)
    assert lost > 0.0
    np.testing.assert_allclose(d._last_owned_exchange_residual["runoff"],
                               lost, rtol=1e-12)


# ---------------------------------------------------------------------------
# slab ocean
# ---------------------------------------------------------------------------
def test_slab_takes_the_open_water_heat_per_water_area():
    from legoesm.coupler.config import TileConfig
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig, init_slab_state, make_ocean,
    )
    shape = (1, 3)
    d = CoupledESMDriver.__new__(CoupledESMDriver)
    d._tile_config = TileConfig(f_land=jnp.array([[0.0, 0.5, 1.0]]),
                                f_lake=jnp.zeros(shape))
    q = jnp.array([[40.0, 30.0, 0.0]])
    z = jnp.zeros(shape)
    d._atm = types.SimpleNamespace(_carry_aux={
        "seg_ow_sw_net": q + 10.0, "seg_ow_lw_net": z - 4.0,
        "seg_shflx_ow": z + 2.0, "seg_lhflx_ow": z + 4.0})
    flux = d._owned_slab_surface_flux()
    np.testing.assert_allclose(np.asarray(flux), [[40.0, 60.0, 0.0]],
                               rtol=1e-15)
    cfg = SimpleOceanConfig(mode="slab")
    st0 = init_slab_state(shape, T_sfc_init=290.0)
    forcing = AtmToSurface(
        sw_down=z + 200.0, lw_down=z + 300.0, precip_total=z, precip_snow=z,
        T_lowest=z + 285.0, q_lowest=z + 0.006, u_lowest=z + 5.0,
        v_lowest=z, p_lowest=z + 1.0e5, p_surface=z + 1.01e5,
        rho_lowest=z + 1.2, cos_zenith=z + 0.5, co2_ppmv=z + 415.0,
        has_radiation=jnp.array(1.0), has_precipitation=jnp.array(1.0))
    dt = 3600.0
    st1, *_ = make_ocean(cfg)(st0, forcing, dt, q_flux=z,
                              surface_flux=flux)
    C = cfg.rho_ocean * cfg.c_ocean * cfg.h_mix
    np.testing.assert_allclose(
        np.asarray(C * (st1.T_sfc.data - 290.0) / dt), np.asarray(flux),
        rtol=1e-9)
    # without it the slab runs its own bulk + radiation (byte-identical)
    a, *_ = make_ocean(cfg)(st0, forcing, dt)
    b, *_ = make_ocean(cfg)(st0, forcing, dt, surface_flux=None)
    np.testing.assert_array_equal(np.asarray(a.T_sfc.data),
                                  np.asarray(b.T_sfc.data))


# ---------------------------------------------------------------------------
# window wiring and refusals
# ---------------------------------------------------------------------------
def _hb(shape, v):
    one = jnp.ones(shape)
    return SurfaceHandback(*(v * one for _ in SurfaceHandback._fields))


def _segment_stub(shape):
    d = CoupledESMDriver.__new__(CoupledESMDriver)
    d.coupled_cfg = CoupledConfig(atm_owns_surface=True, coupling_dt=1800.0)
    z = jnp.zeros(shape)
    aux = {f"seg_booked_{k}": z + i for i, k in enumerate(
        ("sw_net_ice", "lw_abs_ice", "sw_net_lake", "lw_abs_lake"), 1)}
    d._atm = types.SimpleNamespace(
        _carry_aux=aux, _calendar_for_radiation=lambda day: (5, 0.0))
    d._grid_remapper = types.SimpleNamespace(a2o=None, o2a=None, identity=True)
    d._tile_config = "tiles"
    d._sfc_state = "s0"
    order = []
    d._build_atm_forcing = lambda day: types.SimpleNamespace(precip_total=z)
    d._step_ocean = lambda f, dt, q_flux=None: order.append(("ocean", dt))
    d._ocean_surface_KuvC = lambda: (z + 280.0, z, z)
    seen = {}

    def step(state, forcing, tiles, **kw):
        order.append(("surface", kw["dt"]))
        seen.update(kw, state=state, tiles=tiles)
        return "s1", _hb(shape, 7.0), "to_ocean"
    d._step_surface = step
    d._log_coupled_diag = lambda day, dt: order.append(("diag", dt))
    return d, order, seen


def test_window_steps_ocean_then_ice_lake_with_the_booked_radiation():
    shape = (4,)
    d, order, seen = _segment_stub(shape)
    CoupledESMDriver._segment_hook(d, None, 1.0, 1800.0)
    assert order == [("ocean", 1800.0), ("surface", 1800.0), ("diag", 1800.0)]
    of = seen["owned_forcing"]
    for i, k in enumerate(of._fields, 1):
        np.testing.assert_array_equal(np.asarray(getattr(of, k)),
                                      np.full(shape, float(i)))
    assert seen["doy"] == 5.0 and seen["state"] == "s0"
    assert d._sfc_state == "s1" and d._owned_to_ocean == "to_ocean"
    assert float(d._surface_handback.shflx[0]) == 7.0
    with pytest.raises(ValueError, match="ONE coupling window"):
        CoupledESMDriver._segment_hook(d, None, 1.0, 3600.0)


@pytest.mark.parametrize("flag", ["couple_surface_fluxes",
                                  "couple_surface_radiation", "co2_tracer",
                                  "carbon_active"])
def test_install_refuses_what_the_owned_mode_cannot_run(flag):
    d = CoupledESMDriver.__new__(CoupledESMDriver)
    d.coupled_cfg = CoupledConfig(atm_owns_surface=True, **{flag: True})
    d._grid_remapper = None
    with pytest.raises(ValueError, match=flag):
        d._install_surface_handback(None, None)


def test_install_refuses_a_slab_off_the_atmosphere_grid():
    d = CoupledESMDriver.__new__(CoupledESMDriver)
    d.coupled_cfg = CoupledConfig(atm_owns_surface=True)
    d._is_dynamic_ocean = False
    d._grid_remapper = types.SimpleNamespace(identity=False)
    with pytest.raises(ValueError, match="slab ocean"):
        d._install_surface_handback(None, None)


def test_install_serves_the_window0_handback_and_zero_ocean_channels():
    from tests.unit.test_coupler_atm_owns import (
        ICE_CFG, LAKE_CFG, TILES, _state,
    )
    d = CoupledESMDriver.__new__(CoupledESMDriver)
    d.coupled_cfg = CoupledConfig(atm_owns_surface=True)
    d._is_dynamic_ocean = True
    d._grid_remapper = None
    d._sfc_state = _state()
    d._tile_config = TILES
    d._atm = types.SimpleNamespace(get_surface_handback=None,
                                   _f_land=TILES.f_land)
    d._install_surface_handback(ICE_CFG, LAKE_CFG)
    want = initial_handback(d._sfc_state, TILES, ICE_CFG, LAKE_CFG)
    got = d._atm.get_surface_handback()
    for k in SurfaceHandback._fields:
        np.testing.assert_array_equal(np.asarray(getattr(got, k)),
                                      np.asarray(getattr(want, k)), err_msg=k)
    for k in OwnedOceanExchange._fields:
        assert np.all(np.asarray(getattr(d._owned_to_ocean, k)) == 0.0)
    d._surface_handback = _hb(TILES.f_land.shape, 3.0)
    assert float(d._atm.get_surface_handback().evap.reshape(-1)[0]) == 3.0


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def test_install_refuses_two_land_partitions():
    from tests.unit.test_coupler_atm_owns import ICE_CFG, LAKE_CFG, TILES
    for atm_fl in (None, TILES.f_land + 1e-12):
        d = CoupledESMDriver.__new__(CoupledESMDriver)
        d.coupled_cfg = CoupledConfig(atm_owns_surface=True)
        d._is_dynamic_ocean = True
        d._grid_remapper = None
        d._tile_config = TILES
        d._atm = types.SimpleNamespace(_f_land=atm_fl)
        with pytest.raises(ValueError, match="land fraction is not"):
            d._install_surface_handback(ICE_CFG, LAKE_CFG)


def _owned_run(tmp_path, monkeypatch, n_steps):
    """Coupled MPAS (level 3, rrtmgp, multilayer land on the patched
    half-land mask, slab ocean, slab ice seeded at 30 % of the water) with
    the atmosphere owning land + open water; windows of 6 steps."""
    from tests.unit import test_mpas_multilayer_land_port as lp
    from legoesm.coupler.coupler import initial_handback
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    lp._patch_land_loaders(monkeypatch)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=lp.MPAS_RES,
                        nlev=lp.MPAS_NLEV, vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=lp.DT),
        output=OutputConfig(output_dir="", diag_days=0,
                            checkpoint_days=(n_steps * lp.DT + 1.0) / 86400.0),
        days=(n_steps * lp.DT + 1.0) / 86400.0, dataset="analytical",
        radiation="rrtmgp", convection="none", turbulence="louis",
        precision="fp64", land_mask_path="synthetic.nc",
        use_multilayer_land=True, multilayer_n_layers=4,
        multilayer_soil_depth=2.0, land_beta_soil=True,
        land_surface_scheme="simple_seb", land_params_refresh=False,
        land_update_seconds=900.0)
    cc = PRESETS["aquaplanet"]()._replace(
        coupling_dt=1800.0, atm_owns_surface=True, f_land_mode="analytical")
    d = CoupledESMDriver(cfg, cc, output_dir=str(tmp_path))
    d.setup()
    ice = d._sfc_state.ice
    seeded = ice._replace(
        h_ice=ice.h_ice.replace(data=jnp.full_like(ice.h_ice.data, 0.8)),
        concentration=ice.concentration.replace(
            data=jnp.full_like(ice.concentration.data, 0.3)))
    d._sfc_state = d._sfc_state._replace(ice=seeded)
    d._surface_handback = initial_handback(
        d._sfc_state, d._tile_config, d._ice_config or _ice_default(),
        d._lake_config or _lake_default())
    return d


def _ice_default():
    from legoesm.ice.config import SeaIceConfig
    return SeaIceConfig()


def _lake_default():
    from legoesm.coupler.lake.config import LakeConfig
    return LakeConfig()


def _owned_final(d):
    s = d._atm.state
    out = {f"atm_{n}": np.asarray(getattr(s, n).data) for n in ("u", "T", "p_s")}
    out.update({f"trc_{k}": np.asarray(getattr(v, "data", v))
                for k, v in (s.tracers or {}).items()})
    out.update({f"land_{i}": np.asarray(x) for i, x in enumerate(
        jax.tree.leaves(d._atm._land_ml_state))})
    out.update({k: np.asarray(v)
                for k, v in d.coupled_checkpoint_arrays().items()})
    out.update({f"hb_{k}": np.asarray(getattr(d._surface_handback, k))
                for k in SurfaceHandback._fields})
    return out


@pytest.mark.usefixtures("allow_version_drift")
@pytest.mark.parametrize("n_first", [8, 6])
def test_owned_restart_is_bitwise_across_exchanges(tmp_path, monkeypatch,
                                                   n_first):
    """12 steps (two exchanges) unbroken == n_first steps + checkpoint + the
    rest, bit for bit: atmosphere, land, ocean, ice, the ice/lake->ocean
    channels and the handback.  n_first=8 checkpoints mid-window after an
    exchange; 6 exactly on the exchange."""
    import glob
    import os
    cont = _owned_run(tmp_path / "c", monkeypatch, 12)
    assert cont.run() == "COMPLETED"
    ref = _owned_final(cont)
    # the exchange really moved ice and handed fluxes back
    assert np.max(np.abs(ref["hb_shflx"])) > 0.0
    assert any(k.startswith("owned2ocean_") and np.max(np.abs(v)) > 0.0
               for k, v in ref.items())
    a = _owned_run(tmp_path / "a", monkeypatch, n_first)
    assert a.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(str(tmp_path / "a"),
                                         "checkpoint_day_*.npz")))[-1]
    b = _owned_run(tmp_path / "b", monkeypatch, 12 - n_first)
    step, day = b._atm.load_checkpoint(ckpt)
    assert step == n_first
    b.load_coupled_checkpoint(day - b.atm_config.start_day,
                              checkpoint_dir=str(tmp_path / "a"),
                              required=True, atm_step=step)
    assert b.run(start_step=step, start_day=day) == "COMPLETED"
    got = _owned_final(b)
    assert sorted(got) == sorted(ref)
    for k in ref:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)
    # a resume without the ocean channels is refused
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(tmp_path / "a", bad)
    cp = sorted(bad.glob("coupled_day_*.npz"))[-1]
    with np.load(cp) as z:
        keep = {k: z[k] for k in z.files if not k.startswith("owned2ocean_")}
    np.savez(cp, **keep)
    r = _owned_run(tmp_path / "r", monkeypatch, 12 - n_first)
    step, day = r._atm.load_checkpoint(str(bad / os.path.basename(ckpt)))
    with pytest.raises(ValueError, match="owned2ocean_"):
        r.load_coupled_checkpoint(day - r.atm_config.start_day,
                                  checkpoint_dir=str(bad), required=True,
                                  atm_step=step)
