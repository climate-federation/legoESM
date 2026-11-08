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
