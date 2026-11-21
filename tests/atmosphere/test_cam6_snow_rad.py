"""CAM6 snow in the radiation (CloudConfig.cam6_snow_in_rad).

Oracle for the snow fields: CAM cam_cesm2_1_rel_60 micro_mg_cam.F90
L2309-2321 (cldfsnow, icswp), L2576-2583 and L2639 (snow diameter) and
clubb_intr.F90 L2512 (concld), verbatim includes with micro_mg_utils.F90
compiled unmodified (scripts/validate/oracle/cam6_snow_rad), 240 points.
The optics combination is checked against CAM's fraction-weighted formula
(radiation.F90:969-993, 1057-1067) with the Mitchell port.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    cam6_snow_for_radiation, compute_cloud_properties)
from legoesm.atmosphere.physics.radiation import integration as radint
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cloud_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import mitchell_ice_optics as mio
from tests.unit import _cam6_rad_columns as cc

jax.config.update("jax_enable_x64", True)

FIX = os.path.join(os.path.dirname(__file__), "oracle_fixtures", "cam6_snow_rad.npz")
REAL = os.environ.get("LEGOESM_MITCHELL_ICEOPTICS")


def _rel(got, want):
    got = np.asarray(got)
    scale = np.where(want != 0, np.abs(want), 1.0)
    return np.max(np.abs(got - want) / scale)


def test_snow_fields_match_the_fortran():
    fx = dict(np.load(FIX))
    cldfsnow, icswp, des = cam6_snow_for_radiation(
        *(jnp.asarray(fx[k]) for k in ("cld", "deepcu", "qliq", "qs", "ns", "T", "p", "pdel")))
    np.testing.assert_array_equal(np.asarray(cldfsnow), fx["cldfsnow"])
    assert _rel(icswp, fx["icswp"]) <= 1e-12
    assert _rel(des, fx["des"]) <= 1e-12
    np.testing.assert_array_equal(np.asarray(des) == 0, fx["des"] == 0)
    # every branch reached
    assert np.any(fx["cldfsnow"] == 0.25)
    assert np.any((fx["cldfsnow"] == 0) & (fx["cld"] > 1e-4))
    assert np.any((fx["des"] == 0) & (fx["qs"] > 0)) and np.any(fx["des"] > 0)


def test_zero_snow_number_gives_no_snow_optics():
    one = jnp.ones(2)
    _, _, des = cam6_snow_for_radiation(0.5 * one, 0 * one, 1e-4 * one,
                                        jnp.array([1e-4, 1e-4]), jnp.array([0.0, 1e4]),
                                        250 * one, 5e4 * one, 1e3 * one)
    assert float(des[0]) == 0.0 and float(des[1]) > 0.0


def _synthetic_table():
    """Mitchell-like magnitudes: geometric extinction 3/(rho_ice D), nearly
    conservative scattering except in the near-IR rows, absorption about
    half the extinction in the LW."""
    r = np.random.default_rng(0)
    d = np.linspace(2.0, 370.0, 40)
    ext = 3.0 / (917.0 * d * 1e-6)
    sw_ext = ext[None] * r.uniform(0.95, 1.05, (14, 1))
    absorb = np.r_[r.uniform(1e-4, 5e-4, 5), np.full(9, 1e-7)][:, None]
    sw_ssa = np.clip(1.0 - absorb * d[None], 0.5, 1.0)
    sw_asm = np.clip(0.75 + 2e-4 * d[None] * r.uniform(0.5, 1.5, (14, 1)), 0.0, 0.95)
    lw_abs = 0.5 * ext[None] * r.uniform(0.8, 1.2, (16, 1))
    return mio.MitchellIceOptics(*(jnp.asarray(a) for a in (d, sw_ext, sw_ssa, sw_asm, lw_abs)))


def test_band_helpers_are_rows_of_the_full_routines():
    t = _synthetic_table()
    r = np.random.default_rng(1)
    path = jnp.asarray(r.uniform(0, 0.2, (3, 5)) * (r.random((3, 5)) < 0.7))
    d = jnp.asarray(r.uniform(0, 500, (3, 5)) * (r.random((3, 5)) < 0.8))
    full_sw = mio.mitchell_sw_optics(t, path, d)
    full_lw = mio.mitchell_lw_abs_od(t, path, d)
    for b in (0, 7, 13):
        for got, want in zip(mio.mitchell_band_sw(t, b, path, d), full_sw[:3]):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want[b]))
    for b in (0, 15):
        np.testing.assert_array_equal(np.asarray(mio.mitchell_band_lw_abs_od(t, b, path, d)),
                                      np.asarray(full_lw[b]))


@pytest.fixture(scope="module")
def solver():
    return cc.solver()


def test_band_optics_follow_cams_fraction_weighting(solver):
    """With paths weighted cld/cldfprime and cldfsnow/cldfprime, one band's
    tau, tau*w, tau*w*g equal CAM's (cldfsnow snow + cld cloud)/cldfprime."""
    t = _synthetic_table()
    r = np.random.default_rng(2)
    sh = (4, 1, 6)
    cld = r.uniform(0, 1, sh)
    cfs = np.where(r.random(sh) < 0.5, cld, 0.25)
    cfp = np.maximum(cld, cfs)
    liq, ice, swp = (r.uniform(0, 0.1, sh) for _ in range(3))
    # half the snow layers deep enough that CAM's in-cloud LW cap of 50 binds
    swp = np.where(r.random(sh) < 0.5, swp, swp * 1.0e5)
    rl, ri = r.uniform(5e-6, 15e-6, sh), r.uniform(20e-6, 60e-6, sh)
    des = r.uniform(100, 400, sh)
    lib = solver.optics_lib
    for is_lw, look in ((False, lib.cloud_optics_sw), (True, lib.cloud_optics_lw)):
        ib = 3
        comb = cloud_optics.compute_optical_properties(
            look, jnp.asarray(liq * cld / cfp), jnp.asarray(ice * cld / cfp),
            jnp.asarray(rl), jnp.asarray(ri), ib, absorption_only=is_lw,
            cloud_snow=(jnp.asarray(swp * cfs / cfp), jnp.asarray(des), t,
                        jnp.asarray(cfs / cfp)))
        cl = cloud_optics.compute_optical_properties(
            look, jnp.asarray(liq), jnp.asarray(ice), jnp.asarray(rl), jnp.asarray(ri),
            ib, absorption_only=is_lw)
        if is_lw:
            s = mio.mitchell_band_lw_abs_od(t, ib, jnp.asarray(swp), jnp.asarray(des))
            want = (cfs * np.asarray(s) + cld * np.asarray(cl["optical_depth"])) / cfp
            np.testing.assert_allclose(np.asarray(comb["optical_depth"]), want, rtol=1e-12)
            continue
        s_tau, s_w, s_wg = mio.mitchell_band_sw(t, ib, jnp.asarray(swp), jnp.asarray(des))
        c_tau = np.asarray(cl["optical_depth"])
        c_w = c_tau * np.asarray(cl["ssa"])
        c_wg = c_w * np.asarray(cl["asymmetry_factor"])
        tau = (cfs * np.asarray(s_tau) + cld * c_tau) / cfp
        tw = (cfs * np.asarray(s_w) + cld * c_w) / cfp
        twg = (cfs * np.asarray(s_wg) + cld * c_wg) / cfp
        np.testing.assert_allclose(np.asarray(comb["optical_depth"]), tau, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(comb["ssa"]), tw / tau, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(comb["asymmetry_factor"]), twg / tw, rtol=1e-12)


def _backend(cfg, c, solver):
    return radint._call_radiation_backend(
        radiation_config=cc.radiation_config(cfg),
        T=c["T"], p_full=c["p_full"], p_half=c["p_half"],
        sfc_temperature=c["ts"], lat=c["lat"], q_v=c["q_v"],
        insolation=jnp.full(c["T"].shape[0], 700.0),
        q_cloud=c["q_c"], q_ice=c["q_i"], n_ice=c["n_i"],
        cloud_fraction_override=c["alst"],
        conv_mass_flux_up=c["conv_mass_flux_up"], conv_icwmr=c["conv_icwmr"],
        clubb_config=cc.CLUBB, rrtmgp_solver=solver,
        q_snow=c["q_s"], n_snow=c["n_s"])


def test_snow_brightens_snowy_columns_only(solver, monkeypatch):
    table = radint._snow_optics_table(REAL) if REAL else _synthetic_table()
    monkeypatch.setattr(radint, "_snow_optics_table", lambda path: table)
    c = cc.columns(4)
    off = _backend(cc.cloud_config(), c, solver)
    on = _backend(cc.cloud_config(cam6_snow_in_rad=True, cam6_snow_optics_file="x"), c, solver)
    up_off, up_on = (np.asarray(o.sw_flux_up)[:, 0] for o in (off, on))
    # column 1 carries no snow: identical; the snowy columns reflect more
    np.testing.assert_allclose(up_on[1], up_off[1], rtol=0, atol=1e-9)
    snowy = [i for i in range(c["T"].shape[0]) if i != 1]
    assert np.all(up_on[snowy] > up_off[snowy] + 0.5)
    olr_off, olr_on = (np.asarray(o.lw_flux_up)[:, 0] for o in (off, on))
    assert np.all(olr_on[snowy] < olr_off[snowy])


def test_snow_needs_mcica_tracers_and_a_file(solver):
    c = cc.columns(5)
    with pytest.raises(ValueError, match="McICA"):
        _backend(cc.cloud_config(cam6_snow_in_rad=True, cam6_snow_optics_file="x",
                                 cloud_vertical_overlap_optics="none"), c, solver)
    c2 = dict(c, q_s=None)
    with pytest.raises(ValueError, match="q_snow, n_snow"):
        _backend(cc.cloud_config(cam6_snow_in_rad=True), c2, solver)
    with pytest.raises(ValueError, match="iceoptics_c080917"):
        radint._snow_optics_table("")


def test_deck_keys_reach_the_cloud_config():
    from legoesm.driver.model_driver import _standalone_cloud_config
    from tests.unit.test_cam6_cloud_fraction import _exp_cfg
    cfg = _exp_cfg(cloud_vertical_overlap_optics="mcica", cloud_cam6_snow_in_rad=True,
                   cloud_cam6_snow_optics_file="/x/iceoptics_c080917.nc")
    cfg.validate_strict()
    got = _standalone_cloud_config(cfg, "cam6_clubb")
    assert got.cam6_snow_in_rad and got.cam6_snow_optics_file.endswith("c080917.nc")
    with pytest.raises(ValueError, match="cloud_cam6_snow_in_rad=True needs"):
        _exp_cfg(cloud_cam6_snow_in_rad=True).validate_strict()




@pytest.mark.skipif(not REAL, reason="real CAM6 iceoptics_c080917.nc not configured")
def test_real_table_loads_through_the_backend_loader():
    t = radint._snow_optics_table(REAL)
    assert t.sw_ext.shape[0] == 14 and t.lw_abs.shape[0] == 16
