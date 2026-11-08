"""P5c K2: the MPAS column loop consumes the coupler's ice/lake handback.

The atmosphere owns land and open water; the coupler hands back the ice and
lake fluxes, emission, albedo and emissivity (K1).  Covered here: the
radiative surface boundary (plan L2'), the window-0 handback (P2), the
turbulence blend, the radiation keys, and the driver wiring.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.coupler.coupler import SurfaceHandback, initial_handback
from legoesm.forcing.surface_utils import handback_surface_boundary

SB = constants.sigma_sb
EPS = 0.98


def _bnd_inputs():
    # cells: mixed, half ice, ALL ICE (window 0: no emission), land+lake
    return dict(
        f_land=jnp.array([0.3, 0.0, 0.0, 0.2]),
        f_ice=jnp.array([0.2, 0.5, 1.0, 0.0]),
        f_lake=jnp.array([0.1, 0.0, 0.0, 0.3]),
        T_land=jnp.array([290.0, 280.0, 280.0, 300.0]),
        sst=jnp.array([275.0, 271.5, 271.5, 285.0]),
        emissivity=EPS,
        lw_emit_ice_lake=jnp.array([50.0, 120.0, 0.0, 80.0]),
        emissivity_ice=jnp.array([0.97, 0.97, 0.97, 0.97]),
        emissivity_lake=jnp.array([0.95, 0.95, 0.95, 0.95]),
        albedo_land=jnp.array([0.2, 0.2, 0.2, 0.25]),
        albedo_ocean=0.06,
        albedo_ice=jnp.array([0.6, 0.55, 0.7, 0.6]),
        albedo_lake=jnp.array([0.1, 0.1, 0.1, 0.12]),
    )


def test_boundary_emits_and_reflects_the_tile_sum():
    a = _bnd_inputs()
    b = handback_surface_boundary(**a)
    f_ow = 1.0 - a["f_land"] - a["f_ice"] - a["f_lake"]
    np.testing.assert_allclose(b["f_ow"], f_ow, rtol=0, atol=1e-15)
    emit = (EPS * SB * (a["f_land"] * a["T_land"] ** 4 + f_ow * a["sst"] ** 4)
            + a["lw_emit_ice_lake"])
    eps = (EPS * (a["f_land"] + f_ow) + a["f_ice"] * a["emissivity_ice"]
           + a["f_lake"] * a["emissivity_lake"])
    T = b["T_sfc_rad"]
    np.testing.assert_allclose(b["sfc_emissivity"], eps, rtol=1e-15)
    np.testing.assert_allclose(eps * SB * T ** 4, emit, rtol=1e-12)
    assert float(T[2]) == 0.0           # all ice, window 0: nothing emitted
    # What a solver given (eps, T) sends up = each tile's emission plus its
    # own reflection (1 - eps_i) * L of the downward longwave L.
    L = 300.0
    reflect = ((1 - EPS) * (a["f_land"] + f_ow)
               + a["f_ice"] * (1 - a["emissivity_ice"])
               + a["f_lake"] * (1 - a["emissivity_lake"])) * L
    np.testing.assert_allclose(eps * SB * T ** 4 + (1 - eps) * L,
                               emit + reflect, rtol=1e-12)
    alb = (a["f_land"] * a["albedo_land"] + f_ow * 0.06
           + a["f_ice"] * a["albedo_ice"] + a["f_lake"] * a["albedo_lake"])
    np.testing.assert_allclose(b["sfc_albedo"], alb, rtol=1e-15)


def test_boundary_gradients_finite_including_zero_emission():
    a = _bnd_inputs()

    def t_rad(sst, T_land, lw, f_ice):
        return handback_surface_boundary(
            **{**a, "sst": sst, "T_land": T_land, "lw_emit_ice_lake": lw,
               "f_ice": f_ice})["T_sfc_rad"]

    args = (a["sst"], a["T_land"], a["lw_emit_ice_lake"], a["f_ice"])
    # cells with emission: derivative checked against finite differences
    live = jnp.array([0, 1, 3])
    check_grads(lambda *x: t_rad(*x)[live], args, order=1, modes=["rev"],
                eps=1e-6, rtol=1e-5)
    g = jax.grad(lambda *x: jnp.sum(t_rad(*x)), argnums=(0, 1, 2, 3))(*args)
    for gi in g:
        assert np.all(np.isfinite(np.asarray(gi)))
    assert float(g[2][2]) == 0.0        # the zero-emission cell


def _k1():
    from tests.unit.test_coupler_atm_owns import (
        ICE_CFG, LAKE_CFG, SST, TILES, ZU, _booked_from, _forcing,
        _owned_coupler, _state,
    )
    return ICE_CFG, LAKE_CFG, SST, TILES, ZU, _booked_from, _forcing, \
        _owned_coupler, _state


def test_initial_handback_is_zero_exchange_with_the_tiles_own_optics():
    ICE, LAKE, SST, TILES, ZU, booked, forcing, owned, state = _k1()
    s0 = state()
    h = initial_handback(s0, TILES, ICE, LAKE)
    for k in ("shflx", "lhflx", "evap", "tau_x", "tau_y"):
        assert np.all(np.asarray(getattr(h, k)) == 0.0), k
    # emission = what exchange 0 debits: the initial state, explicit (V15)
    from legoesm.coupler.tile_fractions import compute_tile_fractions
    fr = compute_tile_fractions(TILES, s0.ice.concentration.data)
    SBc = constants.sigma_sb
    np.testing.assert_allclose(
        np.asarray(h.lw_emit),
        np.asarray(fr.f_ice * ICE.emissivity_ice * SBc * s0.ice.T_ice.data ** 4
                   + fr.f_lake * LAKE.emissivity_lake * SBc
                   * s0.lake.T_epi.data ** 4), rtol=1e-14)
    # Optics are exactly what an owned exchange reports for the state it
    # leaves: the next exchange's start state.
    s1, h1 = owned()(s0, forcing(), TILES, SST, ZU, ZU, 1800.0,
                     owned_forcing=booked(s0, 250.0, 290.0))
    h0_of_s1 = initial_handback(s1, TILES, ICE, LAKE)
    for k in ("f_ice", "f_lake", "albedo_ice", "albedo_lake",
              "emissivity_ice", "emissivity_lake", "lw_emit"):
        np.testing.assert_array_equal(np.asarray(getattr(h0_of_s1, k)),
                                      np.asarray(getattr(h1, k)), err_msg=k)
    assert np.any(np.asarray(s1.ice.T_ice.data)
                  != np.asarray(s0.ice.T_ice.data))   # the state really moved


def test_handback_stress_replaces_the_ice_lake_bulk_share():
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        handback_into_surface_flux,
    )
    n = 3
    sf = (jnp.array([0.1, 0.2, -0.1]), jnp.array([0.0, -0.05, 0.02]),
          jnp.full(n, 10.0), jnp.full(n, 80.0), jnp.full(n, 0.3))
    btx, bty = jnp.array([0.12, 0.3, -0.1]), jnp.array([0.01, -0.06, 0.02])
    f = jnp.array([0.0, 0.4, 1.0])
    htx, hty = jnp.array([0.0, 0.05, -0.2]), jnp.array([0.0, 0.01, 0.03])
    rho = jnp.full(n, 1.2)
    out = handback_into_surface_flux(sf, btx, bty, f, htx, hty, rho)
    tx = sf[0] - f * btx + htx
    ty = sf[1] - f * bty + hty
    np.testing.assert_allclose(out[0], tx, rtol=1e-15)
    np.testing.assert_allclose(out[1], ty, rtol=1e-15)
    np.testing.assert_array_equal(out[2], sf[2])
    np.testing.assert_array_equal(out[3], sf[3])
    # ustar rebuilt from the blended stress in every column (vote V14)
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        prescribed_into_surface_flux,
    )
    us = prescribed_into_surface_flux(sf, rho, tau_x=tx, tau_y=ty)[4]
    np.testing.assert_allclose(out[4], us, rtol=1e-14)
    assert np.all(np.asarray(out[4]) != np.asarray(sf[4]))


# --------------------------------------------------------------------------
# MPAS turbulence and radiation factories
# --------------------------------------------------------------------------

def _mpas_case():
    from legoesm.core.field import Field
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    nlev = 16
    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
    ncol = mesh.nCells
    sigma = create_sigma_coordinate(nlev)
    T = jnp.broadcast_to(jnp.linspace(300.0, 220.0, nlev)[None, :], (ncol, nlev))
    q = jnp.broadcast_to(jnp.linspace(0.017, 1e-5, nlev)[None, :], (ncol, nlev))
    state = MPASHydrostaticState(
        u=Field(jnp.full((mesh.nEdges, nlev), 4.0)), T=Field(T),
        p_s=Field(jnp.full((ncol,), 1.0e5)),
        phis=Field(jnp.zeros((ncol,))), tracers={"q_v": Field(q)})
    return mesh, sigma, state, ncol


def _turb(f_land, forcing, case):
    from legoesm.atmosphere.physics.turbulence import integration as ti
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    mesh, sigma, state, _ = case
    tc = ti.materialize_sub_config(TurbulenceConfig(scheme="louis"))
    fn = ti.make_turbulence_physics(tc, model_type="mpas", dt=600.0,
                                    f_land=f_land)
    out = fn(state, mesh, sigma, forcing=forcing)
    return out[0] if isinstance(out, tuple) else out


def test_turbulence_bulk_covers_open_water_only_and_adds_the_handback():
    case = _mpas_case()
    ncol = case[3]
    # binary-exact fractions and fluxes: the bridge works in the lane dtype
    f_land = np.zeros(ncol); f_land[:5] = 1.0; f_land[5:10] = 0.5
    sst = jnp.full((ncol,), 301.0)
    base = {"T_sfc": sst, "T_sfc_ocean": sst,
            "shflx_land": jnp.full((ncol,), 40.0),
            "lhflx_land": jnp.full((ncol,), 60.0),
            "evap_land": jnp.full((ncol,), 2.0 ** -15)}
    A = _turb(f_land, base, case)
    f_ow = jnp.asarray((1.0 - f_land) * 0.5)     # half the sea is ice/lake
    hb = {"f_ow": f_ow, "shflx_hb": jnp.full((ncol,), 7.0),
          "lhflx_hb": jnp.full((ncol,), 11.0),
          "evap_hb": jnp.full((ncol,), 2.0 ** -18),
          "taux_hb": jnp.full((ncol,), 0.05),
          "tauy_hb": jnp.full((ncol,), -0.02)}
    B = _turb(f_land, {**base, **hb}, case)
    sea = f_land < 1.0
    fl = jnp.asarray(f_land)
    for name, land, h in (("shflx_sfc", 40.0, 7.0), ("lhflx_sfc", 60.0, 11.0),
                          ("evap_sfc", 2.0 ** -15, 2.0 ** -18)):
        a = np.asarray(getattr(A, name).data)
        b = np.asarray(getattr(B, name).data)
        bulk = (a - fl * land) / jnp.where(sea, 1.0 - fl, 1.0)
        want = f_ow * bulk + fl * land + h
        np.testing.assert_allclose(b[sea], np.asarray(want)[sea], rtol=1e-6,
                                   err_msg=name)
    # the handback stress reaches the winds
    C = _turb(f_land, {**base, **hb, "taux_hb": jnp.zeros((ncol,)),
                       "tauy_hb": jnp.zeros((ncol,))}, case)
    assert np.max(np.abs(np.asarray(B.du_dt.data)
                         - np.asarray(C.du_dt.data))) > 0.0


def test_turbulence_refuses_a_handback_without_land_fluxes():
    case = _mpas_case()
    ncol = case[3]
    with pytest.raises(ValueError, match="shflx_hb"):
        _turb(np.zeros(ncol), {"T_sfc": jnp.full((ncol,), 300.0),
                               "shflx_hb": jnp.zeros((ncol,))}, case)


class _Stop(Exception):
    pass


def _radiation_kwargs(monkeypatch, scheme, extra):
    from legoesm.atmosphere.physics.radiation import integration as ri
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    seen = {}

    def spy(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(ri, "_call_radiation_backend", spy)
    mesh, sigma, state, ncol = _mpas_case()
    fn = ri.make_radiation_physics(RadiationConfig(scheme=scheme),
                                   model_type="mpas")
    forcing = {"T_sfc": jnp.full((ncol,), 300.0), **extra(ncol)}
    with pytest.raises(_Stop):
        fn(state, mesh, sigma, forcing=forcing)
    return seen


def test_radiation_takes_the_tile_emissivity_and_T_rad(monkeypatch):
    eps = lambda n: jnp.linspace(0.95, 0.99, n)     # noqa: E731
    trad = lambda n: jnp.linspace(250.0, 290.0, n)  # noqa: E731
    seen = _radiation_kwargs(monkeypatch, "rrtmgp", lambda n: {
        "sfc_emissivity": eps(n), "T_sfc_rad": trad(n)})
    n = seen["sfc_temperature"].shape[0]
    np.testing.assert_array_equal(np.asarray(seen["sfc_temperature"]),
                                  np.asarray(trad(n)))
    np.testing.assert_array_equal(np.asarray(seen["sfc_emissivity_override"]),
                                  np.asarray(eps(n)))
    plain = _radiation_kwargs(monkeypatch, "rrtmgp", lambda n: {})
    assert plain["sfc_emissivity_override"] is None
    np.testing.assert_array_equal(np.asarray(plain["sfc_temperature"]), 300.0)


@pytest.mark.parametrize("scheme,keys,match", [
    ("gray", ("sfc_emissivity", "T_sfc_rad"), "rrtmgp"),
    ("rrtmgp", ("sfc_emissivity",), "only one"),
])
def test_radiation_refuses_an_incomplete_or_unusable_boundary(
        monkeypatch, scheme, keys, match):
    with pytest.raises(ValueError, match=match):
        _radiation_kwargs(monkeypatch, scheme,
                          lambda n: {k: jnp.full((n,), 0.97 if k[4] == "e"
                                                 else 280.0) for k in keys})


# --------------------------------------------------------------------------
# Driver: the MPAS column loop
# --------------------------------------------------------------------------

def _driver(tmp_path, monkeypatch, radiation="rrtmgp", days=None, **extra):
    from tests.unit import test_mpas_multilayer_land_port as lp
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    from legoesm.driver.model_driver import ModelDriver
    lp._patch_land_loaders(monkeypatch)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=lp.MPAS_RES,
                        nlev=lp.MPAS_NLEV, vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=lp.DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=1),
        days=lp.TWO_STEPS_DAYS if days is None else days, dataset="analytical",
        radiation=radiation,
        convection="none", turbulence="louis", precision="fp64",
        land_mask_path="synthetic.nc", use_multilayer_land=True,
        multilayer_n_layers=4, multilayer_soil_depth=2.0,
        land_beta_soil=True, land_surface_scheme="simple_seb",
        land_params_refresh=False, **extra)
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    return d


def _couple(d, H, window_s=None):
    """What the coupled driver installs: a window callback, the window
    length, and the handback hook."""
    import math
    from tests.unit import test_mpas_multilayer_land_port as lp
    if window_s is None:
        k = math.lcm(int(1800 / lp.DT), max(1, d.config.rad_update_steps),
                     max(1, d.config.physics_update_steps))
        window_s = k * lp.DT
    d._coupling_dt_s = window_s
    d.get_surface_handback = lambda: H
    return lambda *a: None          # the window callback run() installs


def _H(ncol, f_ice=0.2, f_lake=0.1, sh=5.0):
    one = np.ones(ncol)
    return SurfaceHandback(
        shflx=sh * one, lhflx=9.0 * one, evap=3.0e-6 * one,
        tau_x=0.04 * one, tau_y=-0.01 * one, lw_emit=90.0 * one,
        f_ice=f_ice * one, f_lake=f_lake * one,
        albedo_ice=0.6 * one, albedo_lake=0.1 * one,
        emissivity_ice=0.97 * one, emissivity_lake=0.95 * one)


def test_driver_feeds_the_handback_from_the_first_step(tmp_path, monkeypatch):
    from tests.unit import test_mpas_multilayer_land_port as lp
    seen, _ = lp._record_turbulence_forcing_keys(monkeypatch)
    off = _driver(tmp_path / "off", monkeypatch)
    assert off.run() == "COMPLETED"
    assert seen and not any("shflx_hb" in s for s in seen)
    seen.clear()
    on = _driver(tmp_path / "on", monkeypatch)
    ncol = int(np.asarray(on.state.p_s.data).size)
    cb = _couple(on, _H(ncol))    # f_land 0.5 -> f_ow 0.2
    assert on.run(segment_callback=cb) == "COMPLETED"
    assert seen and all({"shflx_hb", "f_ow", "T_sfc_ocean"} <= s
                        for s in seen)
    assert np.max(np.abs(np.asarray(on.state.T.data)
                         - np.asarray(off.state.T.data))) > 0.0


def test_driver_refuses_mismatched_partitions(tmp_path, monkeypatch):
    d = _driver(tmp_path, monkeypatch)
    ncol = int(np.asarray(d.state.p_s.data).size)
    cb = _couple(d, _H(ncol, f_ice=0.45, f_lake=0.1))
    with pytest.raises(ValueError, match="same partition"):
        d.run(segment_callback=cb)


def test_driver_refuses_the_handback_without_rrtmgp(tmp_path, monkeypatch):
    d = _driver(tmp_path, monkeypatch, radiation="gray")
    ncol = int(np.asarray(d.state.p_s.data).size)
    cb = _couple(d, _H(ncol))
    with pytest.raises(ValueError, match="rrtmgp radiation"):
        d.run(segment_callback=cb)


def test_driver_refuses_a_window_off_the_radiation_cadence(tmp_path,
                                                            monkeypatch):
    from tests.unit import test_mpas_multilayer_land_port as lp
    d = _driver(tmp_path, monkeypatch, days=lp.FOUR_STEPS_DAYS,
                rad_update_steps=4)
    ncol = int(np.asarray(d.state.p_s.data).size)
    cb = _couple(d, _H(ncol), window_s=1800.0)  # 6 steps, radiation every 4
    with pytest.raises(ValueError, match="multiple of the radiation"):
        d.run(segment_callback=cb)
    d2 = _driver(tmp_path / "b", monkeypatch)
    d2.get_surface_handback = lambda: _H(ncol)  # no coupling window at all
    with pytest.raises(ValueError, match="coupling window"):
        d2.run(segment_callback=lambda *a: None)


def test_driver_refuses_the_handback_off_the_column_loop(tmp_path):
    """Only ``_run_column`` reads the hook; the lat-lon per-step lane would
    keep its own bulk fluxes over ice and lakes, so setting it there raises
    before any step."""
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    from legoesm.driver.model_driver import ModelDriver
    dt = 100.0
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=dt),
        output=OutputConfig(output_dir=str(tmp_path), diag_days=0,
                            checkpoint_days=0),
        days=dt / 86400.0, dataset="analytical", held_suarez_forcing=False,
        turbulence="louis", radiation="none", convection="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        distributed=False, n_devices=1)
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    d.get_surface_handback = lambda: None
    with pytest.raises(NotImplementedError, match="only the MPAS column loop"):
        d.run()


# --------------------------------------------------------------------------
# K3: what the column loop exports per coupling window, and its restart
# --------------------------------------------------------------------------

def _split(a, S, L, sw_net, lw_net):
    from legoesm.forcing.surface_utils import handback_tile_radiation
    return handback_tile_radiation(
        S, L, sw_net, lw_net, a["f_land"], a["f_ice"], a["f_lake"],
        a["albedo_land"], a["T_land"], a["emissivity"], a["lw_emit_ice_lake"],
        a["albedo_ice"], a["albedo_lake"], a["emissivity_ice"],
        a["emissivity_lake"])


def test_tile_radiation_split_is_the_open_water_share_of_the_net():
    """Fed the net a solver given ``handback_surface_boundary`` computes (SW:
    down x (1 - albedo); LW: eps x down - eps sigma T_rad^4), the open-water
    residual is exactly f_ow x its own absorption (SW) and absorption minus
    emission at the SST (LW); the ice/lake bookings are per tile area.  A
    direct-beam open-water albedo change moves only the open-water share."""
    a = _bnd_inputs()
    b = handback_surface_boundary(**a)
    S = jnp.array([400.0, 250.0, 120.0, 600.0])
    L = jnp.array([300.0, 260.0, 220.0, 350.0])
    eps, T = b["sfc_emissivity"], b["T_sfc_rad"]
    sw_net = S * (1.0 - b["sfc_albedo"])
    lw_net = eps * L - eps * SB * T ** 4
    t = _split(a, S, L, sw_net, lw_net)
    f_ow = b["f_ow"]
    np.testing.assert_allclose(t["ow_sw_net"], f_ow * S * (1.0 - 0.06),
                               rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(t["ow_lw_net"],
                               f_ow * EPS * (L - SB * a["sst"] ** 4),
                               rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(t["booked_sw_net_ice"],
                               S * (1.0 - a["albedo_ice"]), rtol=1e-15)
    np.testing.assert_allclose(t["booked_lw_abs_lake"],
                               a["emissivity_lake"] * L, rtol=1e-15)
    # direct beam: 60 % of S sees an open-water albedo 0.02 lower
    d_sw = f_ow * 0.6 * S * 0.02
    t2 = _split(a, S, L, sw_net + d_sw, lw_net)
    np.testing.assert_allclose(t2["ow_sw_net"] - t["ow_sw_net"], d_sw,
                               rtol=1e-12, atol=1e-10)
    for k in ("booked_sw_net_ice", "booked_sw_net_lake"):
        np.testing.assert_array_equal(t2[k], t[k])


_K3_KW = dict(land_update_seconds=900.0)     # land every 3 steps, window 6
# Where there is no open water the radiation residual is RRTMGP's net minus
# the land/ice/lake shares built from the same boundary.  Measured
# 2026-11-08 (6 steps, 642 cells, x64): SW 1.46e-5, LW 1.99e-2 W/m2 -- the
# solver's surface emission is not exactly eps sigma T_rad^4 (PLAUSIBLE:
# band-integrated Planck).  The residual hands it to the open water, so the
# tiles still sum to what the atmosphere applied (vote K3-V13).
K3_RAD_RESIDUAL_TOL = {"seg_ow_sw_net": 1e-4, "seg_ow_lw_net": 5e-2}


def _k3_run(tmp_path, monkeypatch, n_steps, H, **kw):
    d = _driver(tmp_path, monkeypatch, days=(n_steps * 300.0 + 1.0) / 86400.0,
                **_K3_KW, **kw)
    cb = _couple(d, H)
    return d, cb


def _rain(d, ncol, rate=(1e-5, 3e-5), seen=None):
    """The analytical IC is dry: publish a per-cell surface precip after
    every model step so the window has rain to split.  ``seen`` collects the
    land runoff rates the step finds (those of the latest land call)."""
    from legoesm.core.field import Field
    step = d.model.step
    P = Field(jnp.linspace(*rate, ncol), name="precip", dims=("cell",),
              units="kg/m^2/s")

    from legoesm.driver.model_driver import _sfc_diag_slot
    S = P.replace(data=0.25 * P.data, name="precip_frozen")

    def wet(*a, **k):
        if seen is not None:
            ls = d._land_ml_state
            seen.append(np.asarray(ls.runoff_surface + ls.runoff_subsurface))
        out = step(*a, **k)
        sd = list(d.model._sfc_diag)
        sd[2], sd[_sfc_diag_slot("precip_frozen")] = P, S
        d.model._sfc_diag = tuple(sd)
        return out
    d.model.step = wet


def _exports(d):
    return {k: np.asarray(v) for k, v in d._carry_aux.items()
            if k.startswith("seg_")}


def test_window_exports_open_water_share_and_tile_precip(tmp_path,
                                                         monkeypatch):
    """One window (6 steps).  With no open water (ice + lake fill what land
    leaves) every open-water export is exactly zero while the booked ice
    radiation is not; with open water they are not; the tile precip sums to
    the window precip."""
    probe = _driver(tmp_path / "p", monkeypatch)
    ncol = int(np.asarray(probe.state.p_s.data).size)
    f_land = 0.5                    # the patched loader: half land everywhere
    out = {}
    for tag, f_ice in (("closed", 1.0 - f_land - 0.1), ("open", 0.2)):
        d, cb = _k3_run(tmp_path / tag, monkeypatch, 6,
                        _H(ncol, f_ice=f_ice, f_lake=0.1))
        _rain(d, ncol)
        assert d.run(segment_callback=cb) == "COMPLETED"
        out[tag] = _exports(d)
    c, o = out["closed"], out["open"]
    for k in ("seg_shflx_ow", "seg_lhflx_ow", "seg_evap_ow", "seg_taux_ow",
              "seg_tauy_ow", "seg_precip_ow", "seg_ow_sw_net",
              "seg_ow_lw_net"):
        print("K3MEASURE", k, float(np.max(np.abs(c[k]))),
              float(np.max(np.abs(o[k]))))
    for k in ("seg_shflx_ow", "seg_lhflx_ow", "seg_evap_ow", "seg_taux_ow",
              "seg_tauy_ow", "seg_precip_ow"):
        assert np.max(np.abs(c[k])) < 1e-12, k
        assert np.max(np.abs(o[k])) > 0.0, k
    # No open water: the radiation residual is RRTMGP's own net minus the
    # land, ice and lake shares -- zero iff the solver applied exactly the
    # boundary's albedo, emissivity and T_rad (the assumption of the split).
    for k in ("seg_ow_sw_net", "seg_ow_lw_net"):
        assert np.max(np.abs(c[k])) < K3_RAD_RESIDUAL_TOL[k], k
        assert np.max(np.abs(o[k])) > 1.0, k
    assert np.min(c["seg_booked_sw_net_ice"]) >= 0.0
    assert np.max(c["seg_booked_sw_net_ice"]) > 0.0
    assert np.max(c["seg_booked_lw_abs_ice"]) > 0.0
    for e in (c, o):
        tot = sum(e[f"seg_precip_{t}"] for t in ("ow", "ice", "lake", "land"))
        np.testing.assert_allclose(tot, e["seg_precip"], rtol=1e-12,
                                   atol=0)
        assert np.min(e["seg_precip"]) > 0.0
        np.testing.assert_allclose(e["seg_precip_ice"],
                                   (0.4 if e is c else 0.2) * e["seg_precip"],
                                   rtol=1e-12)


def test_land_runoff_is_booked_in_the_window_the_land_debits_it(
        tmp_path, monkeypatch):
    """Window 6, land every 3: a one-sample bootstrap call at step 0 and a
    three-sample call at step 3.  The window's runoff export is f_land x
    (rate_0 x 1 + rate_3 x 3) / 6, the rates the two calls left."""
    probe = _driver(tmp_path / "p", monkeypatch)
    ncol = int(np.asarray(probe.state.p_s.data).size)
    d, cb = _k3_run(tmp_path / "r", monkeypatch, 6, _H(ncol))
    seen = []
    _rain(d, ncol, rate=(0.2, 0.3), seen=seen)   # ponds past pond_max
    assert d.run(segment_callback=cb) == "COMPLETED"
    ls = d._land_ml_state
    r0 = seen[1]
    r3 = np.asarray(ls.runoff_surface + ls.runoff_subsurface)
    np.testing.assert_array_equal(seen[4], r3)   # no call at steps 4, 5
    print("K3MEASURE runoff max", float(r0.max()), float(r3.max()))
    assert r3.max() > 0.0, "no runoff: the test cannot see the impulse"
    want = 0.5 * (r0 * 1 + r3 * 3) / 6
    np.testing.assert_allclose(_exports(d)["seg_runoff_land"], want,
                               rtol=1e-12, atol=0)


def test_no_hook_exports_nothing_new(tmp_path, monkeypatch):
    d = _driver(tmp_path, monkeypatch)
    assert d.run(segment_callback=None) == "COMPLETED"
    assert not any(k.startswith("seg_ow_") or k.startswith("seg_booked_")
                   for k in (d._carry_aux or {}))


def _leaves(d):
    st = [np.asarray(x) for x in jax.tree_util.tree_leaves(d.state)]
    land = [np.asarray(x) for x in
            jax.tree_util.tree_leaves(d._land_ml_state)]
    win = {k: np.asarray(v) for k, v in (d._cpl_window or {}).items()}
    return st, land, win


def _changing(d, ncol, H0):
    """Hook + window callback of a coupler whose handback changes at every
    exchange (H_k from the exchange's absolute day)."""
    cur = [H0]
    d._coupling_dt_s = 1800.0
    d.get_surface_handback = lambda: cur[0]

    def cb(drv, day, dt):
        k = int(round(day * 86400.0 / 1800.0)) % 5 + 1
        cur[0] = _H(ncol, f_ice=0.2 + 0.01 * k, sh=5.0 + k)
    return cb


@pytest.mark.usefixtures("allow_version_drift")
@pytest.mark.parametrize("n_first", [4, 6])
def test_restart_is_bitwise_mid_window_and_on_the_boundary(
        tmp_path, monkeypatch, n_first):
    """n_first steps (window 6, land every 3) + checkpoint + the rest of 8
    == 8 unbroken steps, bit for bit: state, land, window sums, exports, and
    the handback the checkpoint carries -- with a handback that changes at
    every exchange, so a checkpoint written on the boundary must carry the
    NEXT window's handback."""
    import glob
    import os
    probe = _driver(tmp_path / "p", monkeypatch)
    ncol = int(np.asarray(probe.state.p_s.data).size)
    days = lambda n: (n * 300.0 + 1.0) / 86400.0  # noqa: E731
    dA = _driver(tmp_path / "a", monkeypatch, days=days(n_first), **_K3_KW)
    cbA = _changing(dA, ncol, _H(ncol))
    assert dA.run(segment_callback=cbA) == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(str(tmp_path / "a"),
                                         "checkpoint_day_*.npz")))[-1]
    with np.load(ckpt) as z:
        assert int(z["cpl_win_n"]) == n_first % 6
        assert ("cpl_win_sum_shflx_ow" in z.files) == (n_first % 6 > 0)
        assert "cpl_win_hb_shflx" in z.files
    dC = _driver(tmp_path / "c", monkeypatch, days=days(8), **_K3_KW)
    cbC = _changing(dC, ncol, _H(ncol))
    assert dC.run(segment_callback=cbC) == "COMPLETED"
    dB = _driver(tmp_path / "b", monkeypatch, days=days(8 - n_first),
                 **_K3_KW)
    step, day = dB.load_checkpoint(ckpt)
    assert step == n_first
    Hck = dB.surface_handback_from_checkpoint()
    assert Hck is not None
    if n_first == 6:     # the post-exchange handback, not window 1's
        assert float(np.asarray(Hck.shflx)[0]) != 5.0
    cbB = _changing(dB, ncol, Hck)
    assert dB.run(start_step=step, start_day=day,
                  segment_callback=cbB) == "COMPLETED"
    sB, lB, wB = _leaves(dB)
    sC, lC, wC = _leaves(dC)
    for a, b in zip(sB + lB, sC + lC):
        np.testing.assert_array_equal(a, b)
    assert wB.keys() == wC.keys()
    for k in wB:
        np.testing.assert_array_equal(wB[k], wC[k])
    if n_first < 6:      # B closed window 1 itself
        eB, eC = _exports(dB), _exports(dC)
        assert "seg_shflx_ow" in eC and eB.keys() == eC.keys()
        for k in eC:
            np.testing.assert_array_equal(eB[k], eC[k])


def test_restart_refuses_a_handback_window_without_its_sums(tmp_path,
                                                            monkeypatch):
    import glob
    import os
    probe = _driver(tmp_path / "p", monkeypatch)
    ncol = int(np.asarray(probe.state.p_s.data).size)
    H = _H(ncol)
    dA, cbA = _k3_run(tmp_path / "a", monkeypatch, 4, H)
    assert dA.run(segment_callback=cbA) == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(str(tmp_path / "a"),
                                         "checkpoint_day_*.npz")))[-1]
    with np.load(ckpt) as z:
        stripped = {k: z[k] for k in z.files
                    if not k.startswith("cpl_win_sum_")
                    or k[len("cpl_win_sum_"):] in (
                        "sw_net_sfc", "lw_net_sfc", "precip",
                        "precip_frozen")}
        full = {k: z[k] for k in z.files}
    old = str(tmp_path / "old_checkpoint_day_0000.npz")
    np.savez(old, **stripped)
    dB, cbB = _k3_run(tmp_path / "b", monkeypatch, 4, H)
    step, day = dB.load_checkpoint(old)
    with pytest.raises(ValueError, match="cannot be resumed"):
        dB.run(start_step=step, start_day=day, segment_callback=cbB)
    # ... and a handback window resumed by a run with no handback
    dN = _driver(tmp_path / "n", monkeypatch, **_K3_KW)
    dN._coupling_dt_s = 1800.0
    step, day = dN.load_checkpoint(ckpt)
    with pytest.raises(ValueError, match="no handback is set"):
        dN.run(start_step=step, start_day=day,
               segment_callback=lambda *a: None)
    # ... and a boundary checkpoint (no pending sums, a pending handback)
    # resumed with no handback and no coupling window at all
    dE, cbE = _k3_run(tmp_path / "e", monkeypatch, 6, H)
    assert dE.run(segment_callback=cbE) == "COMPLETED"
    ckE = sorted(glob.glob(os.path.join(str(tmp_path / "e"),
                                        "checkpoint_day_*.npz")))[-1]
    with np.load(ckE) as z:
        assert int(z["cpl_win_n"]) == 0 and "cpl_win_hb_shflx" in z.files
    dM = _driver(tmp_path / "m", monkeypatch, **_K3_KW)
    step, day = dM.load_checkpoint(ckE)
    with pytest.raises(ValueError, match="pending coupler ice/lake handback"):
        dM.run(start_step=step, start_day=day)
    # a partial handback set is refused by the reader
    from legoesm.driver.model_driver import surface_handback_from_payload
    part = {k: v for k, v in full.items() if k != "cpl_win_hb_lw_emit"}
    with pytest.raises(ValueError, match="partial coupler handback"):
        surface_handback_from_payload(part)
