"""les_suite/cloudbench_radiation.py: the CloudBench runs' radiation on a column."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing, validate_forcing
from legoesm.atmosphere.les_suite import cloudbench_radiation as cr

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_PARAMS = {"sst": 298.0, "zenith": 0.9, "irrad": 500.0}
_SOUNDING = {"p": np.array([1.0e5, 8.0e4, 5.0e4, 2.6e4, 5.5e3]),
             "T": np.array([298.0, 288.0, 265.0, 230.0, 215.0]),
             "q_t": np.array([0.015, 0.008, 0.002, 1.0e-4, 3.0e-6])}


def _column(nlev=8, cloud=True):
    p_half = np.linspace(4.8e4, 1.0e5, nlev + 1)
    p_full = 0.5 * (p_half[1:] + p_half[:-1])
    T = np.interp(p_full, [4.8e4, 1.0e5], [262.0, 297.0])
    q_v = np.interp(p_full, [4.8e4, 1.0e5], [0.002, 0.015])
    q_c = np.zeros(nlev)
    if cloud:
        q_c[-3] = 4.0e-4
    rho = p_full / (constants.R_d * T)
    return tuple(jnp.asarray(a) for a in (T, q_v, q_c, p_full, p_half, rho))


def test_liu_hallett_radius_is_swirl_lms_formula():
    rho, q_c = 1.1, 5.0e-4
    want = (4.0 / 3.0 * np.pi * 1000.0 * 0.8) ** (-1.0 / 3.0) * (rho * q_c / 1.0e8) ** (1.0 / 3.0)
    assert float(cr.liu_hallett_r_eff(rho, q_c)) == pytest.approx(want, rel=1e-14)
    assert 5e-6 < want < 20e-6                    # a cloud-droplet radius


def test_extension_takes_the_sounding_above_the_column_top():
    case = cr.case_radiation(_PARAMS, _SOUNDING, 4.8e4)
    assert np.allclose(case.ext_p_full, [5.5e3, 2.6e4])
    assert np.allclose(case.ext_p_half, [5.5e3 * np.sqrt(5.5e3 / 2.6e4),
                                         np.sqrt(5.5e3 * 2.6e4), 4.8e4])
    assert case.cos_zenith == pytest.approx(np.cos(0.9)) and case.irrad == 500.0
    with pytest.raises(ValueError, match="two levels"):
        cr.case_radiation(_PARAMS, _SOUNDING, 2.0e4)


def test_column_heating_closes_on_the_boundary_fluxes():
    case = cr.case_radiation(_PARAMS, _SOUNDING, 4.8e4)
    T, q_v, q_c, pf, ph, rho = _column()
    rec = cr.column_radiation(case, T, q_v, q_c, pf, ph, rho)
    dp = np.diff(np.asarray(ph))
    heat = constants.c_pd / constants.g * np.sum(dp * np.asarray(rec.dT_dt))
    supply = float(rec.net_down_top - rec.net_down_sfc)
    gross = constants.c_pd / constants.g * np.sum(dp * np.abs(np.asarray(rec.dT_dt)))
    assert abs(heat - supply) <= 1e-13 * gross
    # the cloud layer's top cools in the longwave
    clear = cr.column_radiation(case, T, q_v, q_c, pf, ph, rho, clouds=False)
    assert float(rec.dT_dt[-3]) < float(clear.dT_dt[-3]) - 1.0 / 86400.0


def test_the_column_must_end_at_the_extension():
    case = cr.case_radiation(_PARAMS, _SOUNDING, 4.7e4)
    with pytest.raises(ValueError, match="top interface"):
        cr.column_radiation(case, *_column())


def test_the_runs_gases_and_surface_reach_the_solver():
    cfg = cr.rrtmgp_config(500.0)
    assert cfg.include_clouds and cfg.S_0 == 500.0
    assert cfg.co2_ppmv == pytest.approx(400.0) and cfg.sfc_albedo == 0.07
    assert cfg.sfc_emissivity == 0.98 and cr.CB_GAS_VMR["co2"] == 4.0e-4


def test_two_radiations_on_one_column_are_refused():
    with pytest.raises(ValueError, match="exclusive"):
        validate_forcing(SCMForcing(rad_T_tendency=lambda t: 0.0,
                                    rad_column_fn=lambda s, c, t: None))


# --- liquid/ice split (RULE 5: smooth default, exact at width 0, gap recorded) ---

def _swirl_liquid_fraction(T):
    """swirl-lm water.liquid_fraction_from_temperature (t_icenuc 233 K default)."""
    T = np.asarray(T, dtype=np.float64)
    return np.where(T > constants.T_freeze, 1.0,
                    np.where(T <= 233.0, 0.0, (T - 233.0) / (constants.T_freeze - 233.0)))


#: fixture columns: (column top [Pa], T at the top [K], cloud layer index) -- cloud at
#: the upper kink, on the ramp away from both kinks, and at the lower kink
_MIXED = {"upper_kink": (4.8e4, 262.0, 2), "on_ramp": (4.8e4, 262.0, 0),
          "lower_kink": (3.0e4, 229.2, 0)}


def _mixed_column(top_p, T_top, k_cloud, nlev=8):
    p_half = np.linspace(top_p, 1.0e5, nlev + 1)
    p_full = 0.5 * (p_half[1:] + p_half[:-1])
    T = np.interp(p_full, [top_p, 1.0e5], [T_top, 297.0])
    q_v = np.interp(p_full, [top_p, 1.0e5], [0.001, 0.015])
    q_c = np.zeros(nlev)
    q_c[k_cloud] = 3.0e-4
    rho = p_full / (constants.R_d * T)
    return tuple(jnp.asarray(a) for a in (T, q_v, q_c, p_full, p_half, rho))


def _net(case, col, **kw):
    out, n_ext = cr.column_fluxes(case, *col, **kw)
    return np.asarray((out.lw_flux_down + out.sw_flux_down - out.lw_flux_up
                       - out.sw_flux_up)[0][n_ext:])


def test_width_zero_is_swirl_lms_liquid_fraction():
    T = np.array([200.0, 233.0, 233.0 + 1e-9, 250.0, 273.0, constants.T_freeze,
                  constants.T_freeze + 1e-9, 300.0])
    assert np.array_equal(np.asarray(cr.liquid_fraction(jnp.asarray(T), 0.0)),
                          _swirl_liquid_fraction(T))
    rho, q = 0.6, 2.0e-5
    want = (4.0 / 3.0 * np.pi * 917.0 * 0.8) ** (-1.0 / 3.0) * (rho * q / 1.0e8) ** (1.0 / 3.0)
    assert float(cr.liu_hallett_r_eff(rho, q, constants.rho_ice)) == pytest.approx(want, rel=1e-14)


@pytest.mark.parametrize("name", sorted(_MIXED))
def test_width_zero_column_is_the_les_split(name):
    """At width 0 the column's fluxes are those of RRTMGP fed swirl-lm's exact split.
    The reference shares the solver, gas table and radius helper with the code under
    test (pinned separately above); it checks the split, its order and plumbing."""
    top_p, T_top, k = _MIXED[name]
    case = cr.case_radiation(_PARAMS, _SOUNDING, top_p, ice_ramp_width_k=0.0)
    col = _mixed_column(top_p, T_top, k)
    T, q_v, q_c, pf, ph, rho = (np.asarray(a) for a in col)
    f = _swirl_liquid_fraction(T)
    n_ext = case.ext_p_full.shape[0]
    z = np.zeros(n_ext)
    dpg = np.diff(ph) / constants.g
    r = lambda q, rc: np.concatenate([np.full(n_ext, 1e-6), np.maximum(  # noqa: E731
        np.asarray(cr.liu_hallett_r_eff(rho, q, rc)), 1e-6)])
    out = cr.rrtmgp_radiation(
        jnp.asarray(np.concatenate([case.ext_T, T]))[None],
        jnp.asarray(np.concatenate([case.ext_p_full, pf]))[None],
        jnp.asarray(np.concatenate([case.ext_p_half[:-1], ph]))[None],
        jnp.asarray([case.sst]), jnp.asarray(np.concatenate([case.ext_q, q_v]))[None],
        jnp.asarray([case.cos_zenith]), cr.rrtmgp_config(case.irrad),
        sfc_albedo_override=cr.CB_SFC_ALBEDO, sfc_emissivity_override=cr.CB_SFC_EMISSIVITY,
        o3_vmr=cr.mls_ozone_layer_vmr(
            jnp.asarray(np.concatenate([case.ext_p_half[:-1], ph]))[None])[0],
        cloud_path_liq=jnp.asarray(np.concatenate([z, q_c * f * dpg]))[None],
        cloud_r_eff_liq=jnp.asarray(r(q_c * f, constants.rho_water))[None],
        cloud_path_ice=jnp.asarray(np.concatenate([z, q_c * (1 - f) * dpg]))[None],
        cloud_r_eff_ice=jnp.asarray(r(q_c * (1 - f), constants.rho_ice))[None],
        ghg_vmr_override=cr.CB_GAS_VMR)
    want = np.asarray((out.lw_flux_down + out.sw_flux_down - out.lw_flux_up
                       - out.sw_flux_up)[0][n_ext:])
    np.testing.assert_allclose(_net(case, col), want, rtol=1e-12, atol=1e-10)


#: measured net-downward-flux gap [W/m2] of the smooth default (0.5 K) against the
#: exact ramp, per fixture column: (column top, surface, max |.| over interfaces)
#: (measured 2026-11-18, x64, CPU)
ICE_RAMP_GAPS = {"lower_kink": (-0.20412168993641444, -0.08747299837258993, 0.20412168993641444),
                 "on_ramp": (-6.909345984240645e-09, -6.098996863102002e-09, 6.9312431349999315e-09),
                 "upper_kink": (-0.6543497562616096, -0.5901353498023258, 0.6673252998806731)}


@pytest.mark.parametrize("name", sorted(_MIXED))
def test_smooth_default_gap_is_the_recorded_one(name):
    top_p, T_top, k = _MIXED[name]
    col = _mixed_column(top_p, T_top, k)
    smooth = _net(cr.case_radiation(_PARAMS, _SOUNDING, top_p), col)
    exact = _net(cr.case_radiation(_PARAMS, _SOUNDING, top_p, ice_ramp_width_k=0.0), col)
    d = smooth - exact
    got = (float(d[0]), float(d[-1]), float(np.max(np.abs(d))))
    print("ICE_RAMP_GAP", name, repr(got))
    np.testing.assert_allclose(got, ICE_RAMP_GAPS[name], rtol=1e-6, atol=1e-12)


def test_ice_split_gradients_at_the_default():
    from jax.test_util import check_grads
    T = jnp.asarray([232.6, 233.0, 233.4, 250.0, 272.9, constants.T_freeze, 273.5])
    check_grads(lambda t: cr.liquid_fraction(t, cr.CB_ICE_RAMP_WIDTH_K), (T,), order=2)
    top_p, T_top, k = _MIXED["on_ramp"]
    case = cr.case_radiation(_PARAMS, _SOUNDING, top_p)
    T0, q_v, q_c, pf, ph, rho = _mixed_column(top_p, T_top, k)
    g = jax.grad(lambda t: cr.column_radiation(case, t, q_v, q_c, pf, ph, rho).net_down_top)(T0)
    assert np.all(np.isfinite(np.asarray(g)))
    assert abs(float(g[k])) > 0.0
    gq = jax.grad(lambda q: cr.column_radiation(case, T0, q_v, q, pf, ph, rho).net_down_top)(q_c)
    assert np.all(np.isfinite(np.asarray(gq))) and abs(float(gq[k])) > 0.0
