"""RRTMGP gas optics on the ecRad lane: band maps, units and clear-sky fluxes.

The clear-sky reference is the lane-independent RRTMGP solve, itself held to the
RTE Fortran in test_rrtmgp_rte_oracle.py, on that test's RFMIP columns.  Both share
the gas tables, so the gaps pinned here are ecRad's solvers and gas preparation.
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ecrad
from legoesm.atmosphere.physics.radiation.ecrad_ifs import gas_optics as go
from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_solvers as ms
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).parent / "oracle_fixtures" / "rrtmgp_rte_clear.npz"
_LW_EDGES = (10, 250, 500, 630, 700, 820, 980, 1080, 1180, 1390, 1480, 1800, 2080, 2250, 2380,
             2600, 3250)


def _rrtmgp_limits(lookup):
    return np.asarray(lookup.bnd_lims_wn), np.asarray(lookup.g_point_to_bnd)


def test_each_gpoint_lands_in_the_matching_rrtmg_band():
    lib = go._optics()
    lims, g2b = _rrtmgp_limits(lib.gas_optics_sw)
    rr = np.stack([ms._SW_WAVENUMBER1_BAND, ms._SW_WAVENUMBER2_BAND], 1)
    lo, hi = lims[g2b, 0], lims[g2b, 1]
    band = go.band_from_g_sw()
    overlap = np.minimum(hi, rr[band, 1]) - np.maximum(lo, rr[band, 0])
    assert np.all(overlap / (hi - lo) > 0.95)
    lims, g2b = _rrtmgp_limits(lib.gas_optics_lw)
    edges = np.asarray(_LW_EDGES, float)
    band = go.band_from_g_lw()
    lo, hi = lims[g2b, 0], lims[g2b, 1]
    overlap = np.minimum(hi, edges[band + 1]) - np.maximum(lo, edges[band])
    # 2390-2680 against RRTMG's 2380-2600 is the worst band.
    assert np.all(overlap / (hi - lo) > 0.7)


def _gas(ncol, nlev, temperature=280.0):
    p = jnp.tile(jnp.geomspace(100.0, 1.0e5, nlev + 1), (ncol, 1))
    t = jnp.full_like(p, temperature)
    mmr = jnp.zeros((ncol, nlev, 12)).at[..., go.ICO2].set(6e-4)
    return go.gas_optics(p, t, mmr, jnp.full((ncol,), temperature),
                         jnp.zeros((ncol, go.band_from_g_lw().size)), 1361.0,
                         constants.g_ecrad, 1e-15)


def test_planck_terms_sum_to_a_blackbody_and_sun_to_the_constant():
    g = _gas(2, 6)
    sigma_t4 = constants.sigma_sb * 280.0**4
    assert float(jnp.sum(g.lw_emission[0])) == pytest.approx(sigma_t4, rel=2e-3)
    assert float(jnp.sum(g.planck_hl[0, -1])) == pytest.approx(sigma_t4, rel=2e-3)
    assert float(jnp.sum(g.planck_hl[0, 0])) == pytest.approx(sigma_t4, rel=2e-3)
    assert float(jnp.sum(g.incoming_sw[0])) == pytest.approx(1361.0, rel=1e-12)


def test_level_sources_follow_the_level_temperature_top_down():
    p = jnp.tile(jnp.geomspace(100.0, 1.0e5, 5), (1, 1))
    t = jnp.linspace(200.0, 300.0, 5)[None, :]
    g = go.gas_optics(p, t, jnp.zeros((1, 4, 12)), jnp.full((1,), 300.0),
                      jnp.zeros((1, go.band_from_g_lw().size)), 1361.0, constants.g_ecrad, 1e-15)
    level = np.asarray(jnp.sum(g.planck_hl[0], axis=-1))
    np.testing.assert_allclose(level, constants.sigma_sb * np.asarray(t[0])**4, rtol=0.1)


def test_tables_follow_a_precision_change():
    jax.config.update("jax_enable_x64", False)
    try:
        assert go._optics().gas_optics_lw.kmajor.dtype == jnp.float32
    finally:
        jax.config.update("jax_enable_x64", True)
    assert go._optics().gas_optics_lw.kmajor.dtype == jnp.float64


def test_gas_optics_shapes_are_column_first_top_down():
    g = _gas(3, 5)
    n_lw, n_sw = go.band_from_g_lw().size, go.band_from_g_sw().size
    assert g.od_lw.shape == (3, 5, n_lw) and g.planck_hl.shape == (3, 6, n_lw)
    assert g.od_sw.shape == g.ssa_sw.shape == (3, 5, n_sw)
    # The bottom layer holds most of the mass.
    assert float(g.ssa_sw[0, -1].max()) > 0 and float(g.od_sw[0, -1].sum()) > float(g.od_sw[0, 0].sum())


def test_tables_are_the_ones_the_gaps_were_measured_on():
    root = Path(ms.__file__).parents[1] / "rrtmgp/optics/rrtmgp_data"
    digest = {f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16]
              for f in ("rrtmgp-gas-lw-g128.nc", "rrtmgp-gas-sw-g112.nc")}
    print("TABLES", digest)
    assert digest == _TABLES


_TABLES = {'rrtmgp-gas-lw-g128.nc': '77c39fc74e5cddf3', 'rrtmgp-gas-sw-g112.nc': '74f699d147cd5074'}
# [W/m2], worst level of 20 columns, measured 2026-11-06.  Solvers, gas preparation
# and level temperature all differ here; not attributed between them.  LW peaks one
# level above the surface (0.60 at TOA), SW up at TOA, SW down at the surface.
_GAPS = dict(lw_up=3.7, lw_dn=4.5, sw_up=0.72, sw_dn=0.86)


@pytest.fixture(scope="module")
def clear():
    z = np.load(_FIX)
    ghg = json.loads(str(z["ghg"]))
    a = lambda k: jnp.asarray(z[k])  # noqa: E731
    ph, t, q, o3 = a("p_half"), a("T"), a("q"), a("o3")
    pf = 0.5 * (ph[:, 1:] + ph[:, :-1])
    th = jnp.concatenate([t[:, :1], 0.5 * (t[:, 1:] + t[:, :-1]), a("tsfc")[:, None]], axis=1)
    tf = 0.5 * (th[:, 1:] + th[:, :-1])
    cfg = RRTMGPConfig(S_0=float(z["s0"]), include_clouds=False, overhead_layer=False,
                       gpoint_batch_size=0, gpoint_checkpoint=False, use_optimal_angle=False,
                       use_scan=True, co2_ppmv=ghg["co2"] * 1e6, ch4_ppbv=ghg["ch4"] * 1e9,
                       n2o_ppbv=ghg["n2o"] * 1e9)
    ref = RRTMGP.from_legoesm_config(cfg).solve_columns(
        T=tf, p_full=pf, p_half=ph, sfc_temperature=a("tsfc"), q_v=q, cos_zenith=a("mu0"),
        sfc_albedo=a("alb"), sfc_emissivity=a("emis"), o3_vmr=o3, ghg_vmr_override=ghg)
    gas = jnp.zeros(q.shape + (12,))
    h2o_vmr = q / (1.0 - q) * (constants.M_dry / constants.M_h2o)
    for name, i, value in (("h2o", go.IH2O, h2o_vmr), ("o3", go.IO3, o3), ("co2", go.ICO2, ghg["co2"]),
                           ("n2o", go.IN2O, ghg["n2o"]), ("ch4", go.ICH4, ghg["ch4"])):
        gas = gas.at[..., i].set(value * go._MOLAR_MASS[name][1] / go._AIR_MOLAR_MASS)
    ncol, nlev = q.shape
    zero = jnp.zeros_like(q)
    ulw, usw = ecrad.draw_uniforms(jnp.arange(ncol), nlev)
    out = ecrad.radiation(
        pressure_hl=ph, temperature_hl=th, gas_mmr=gas, skin_temperature=a("tsfc"), cos_sza=a("mu0"),
        solar_irradiance=float(z["s0"]), sw_albedo=jnp.tile(a("alb")[:, None], (1, 6)),
        sw_albedo_direct=jnp.tile(a("alb")[:, None], (1, 6)),
        lw_emissivity=jnp.tile(a("emis")[:, None], (1, 2)), cloud_fraction=zero, q_liquid=zero,
        q_ice=zero, re_liquid=zero + 1e-5, re_ice=zero + 4e-5,
        overlap_param=jnp.ones((ncol, nlev - 1)), fractional_std=zero,
        uniforms_lw=ulw, uniforms_sw=usw, sharpness=np.inf)
    return out, ref


@pytest.mark.parametrize("name,ecrad_key,ref_key", [
    ("lw_up", "flux_up_lw", "lw_flux_up"), ("lw_dn", "flux_dn_lw", "lw_flux_down"),
    ("sw_up", "flux_up_sw", "sw_flux_up"), ("sw_dn", "flux_dn_sw", "sw_flux_down")])
def test_clear_sky_gap_to_rrtmgp_is_as_recorded(clear, name, ecrad_key, ref_key):
    out, ref = clear
    d = np.asarray(out[ecrad_key]) - np.asarray(getattr(ref, ref_key))
    gap = float(np.abs(d).max())
    k = int(np.unravel_index(np.abs(d).argmax(), d.shape)[1])
    print(f"GAP {name} max {gap:.4e} level {k}/{d.shape[1]-1} rms {np.sqrt((d**2).mean()):.4e}")
    assert gap <= _GAPS[name]
