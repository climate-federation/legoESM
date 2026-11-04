"""SI3 surface ports: smooth default vs NEMO hard mode, and gradients across every smoothed switch.

User rule 2026-10-27: every switch/floor/cap in a port is smooth by default (C1 ramps of width
``switch_width x scale``, core/smooth.py); ``switch_width=0.0`` is NEMO's hard form (oracle tests).
Fixtures are the tier-4 NEMO 5.0.1 ORCA1 extracts used by the oracle tests
(``fixtures/orca1_t4/si3_land_mask.npz`` and ``si3_qlead_501_cuts.npz``).

* Deviation: max |smooth - hard| per output on the fixture cells must stay within twice the value
  measured when the widths were chosen (vote W9; measured on the full 32-rank record: see PR body).
* Gradients: at a point inside each ramp, jax.grad matches a central finite difference.
The snow-free test (h_s == 0) stays hard in both modes (flagged; see sea_ice._si3_snow_free).
"""
from __future__ import annotations

import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIXDIR = pathlib.Path(__file__).parent / "fixtures" / "orca1_t4"
# measured max |smooth - hard| on the fixture cells (x2 applied below); 0.0 = bitwise equal
MEASURED = {
    "blk2:qns_tot": 0.0, "blk2:qprec_ice": 0.0, "blk2:qemp_oce": 0.0, "blk2:qemp_ice": 0.0,
    "blk2:qtr_ice_top": 0.0, "blk2:qsr_ice": 1.6485e-02, "blk2:albedo": 9.7755e-05,
    "alb:alb_ice": 2.6874e-05,
    "flx:qsb_ice_bot": 1.1522e-03, "flx:fhld": 0.0, "flx:qlead": 0.0,
    "s4:qsb_ice_bot": 1.4534e-03, "s4:fhld": 0.0, "s4:qlead": 0.0,
}


@pytest.fixture(autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _load(name):
    z = np.load(FIXDIR / name)
    return {k: z[k] for k in z.files}


def _sec(fx, prefix):
    return {k.split(":", 1)[1]: jnp.asarray(v) for k, v in fx.items() if k.startswith(prefix + ":")}


def _blk(fx, width=None, **over):
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
    from legoesm.ice.sea_ice import _nemo_si3_blk_ice_2
    i = _sec(fx, "blk2_in"); i.update(over)
    kw = {} if width is None else {"switch_width": width}
    return _nemo_si3_blk_ice_2(
        T_surface=i["t_su"], h_ice=i["h_i"], h_snow=i["h_s"], cloud_fraction=i["cloud_fra"],
        theta_air=i["theta_air_zt"], q_air=i["q_air_zt"], p_surface=i["slp"], lw_down=i["qlw"],
        precip=i["precip"], snow=i["snow"], qsr=i["qsr"], rho_air=i["rhoa"], wind=i["wndm_ice"],
        Ch=i["Ch_ice"], Ce=i["Ce_ice"], ice_fraction_before=i["at_i_b"], category_fraction_before=i["a_i_b"],
        sst_celsius=i["sst_m"], qns_ocean=i["qns_oce"], qsr_ocean=i["qsr_oce"], emp_ocean_raw=i["emp"],
        land_mask=i["smask0"], precip_factor=1.0, evap_factor=1.0, snow_blow_exponent=0.66,
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG, **kw)


def _alb(fx, width=None, **over):
    from legoesm.ice.sea_ice import _nemo_si3_ice_albedo
    i = _sec(fx, "alb_in"); i.update(over)
    kw = {} if width is None else {"switch_width": width}
    return _nemo_si3_ice_albedo(i["t_su"], i["h_i"], i["h_s"], i["cloud_fra"], land_mask=i["smask0"], **kw)


def _flx(fx, width=None, **over):
    from legoesm.ice.sea_ice import _nemo_si3_ice_flx_other
    i = _sec(fx, "flx_in"); i.update(over)
    s = {k: float(v) for k, v in _sec(fx, "flx_scalar").items()}
    kw = {} if width is None else {"switch_width": width}
    return _nemo_si3_ice_flx_other(
        ice_fraction=i["at_i"], ice_fraction_before=i["at_i_b"], ice_volume=i["vt_i"], u_ice=i["u_ice"],
        u_ice_west=i["u_ice_w"], v_ice=i["v_ice"], v_ice_south=i["v_ice_s"], u_ocean=i["ssu_m"],
        u_ocean_west=i["ssu_m_w"], v_ocean=i["ssv_m"], v_ocean_south=i["ssv_m_s"], drag_io=s["rn_Cd_io"],
        frq=i["frq_m"], qsr_ocean=i["qsr_oce"], qns_ocean=i["qns_oce"], qemp_ocean=i["qemp_oce"],
        ocean_layer_thickness=i["e3t_m"], sst_celsius=i["sst_m"], T_bottom=i["t_bo"], dt=s["rDt_ice"],
        rho_ocean=s["rho0"], c_ocean=s["rcp"], T0=s["rt0"], ice_epsilon=1.0e-10,
        max_ice_fraction=i["rn_amax_2d"], land_mask=i["smask0"], lead_cut_version="5.0.1", **kw)


def _deviations():
    lm, s4 = _load("si3_land_mask.npz"), _load("si3_qlead_501_cuts.npz")
    out = {}
    sm, hd = _blk(lm), _blk(lm, 0.0)
    for k in ("qns_tot", "qprec_ice", "qemp_oce", "qemp_ice", "qtr_ice_top", "qsr_ice", "albedo"):
        out[f"blk2:{k}"] = (np.asarray(sm[k]), np.asarray(hd[k]))
    out["alb:alb_ice"] = (np.asarray(_alb(lm)), np.asarray(_alb(lm, 0.0)))
    for tag, fx in (("flx", lm), ("s4", s4)):
        sm, hd = _flx(fx), _flx(fx, 0.0)
        for k in ("qsb_ice_bot", "fhld", "qlead"):
            out[f"{tag}:{k}"] = (np.asarray(sm[k]), np.asarray(hd[k]))
    return out


def test_smooth_default_stays_close_to_nemo_hard_mode():
    dev = _deviations()
    report = {k: float(np.max(np.abs(a - b))) for k, (a, b) in dev.items()}
    if os.environ.get("SI3_SMOOTH_REPORT"):
        print("MEASURED", report)
    for k, (a, b) in dev.items():
        assert np.all(np.isfinite(a)), k
        assert report[k] <= 2.0 * MEASURED[k], (k, report[k], MEASURED[k])


def _fd_check(f, x0, h, tol=1e-4):
    g = float(jax.grad(f)(jnp.asarray(x0)))
    fd = (float(f(jnp.asarray(x0 + h))) - float(f(jnp.asarray(x0 - h)))) / (2.0 * h)
    assert np.isfinite(g) and g != 0.0, (g, fd)
    assert abs(g - fd) <= tol * abs(fd), (g, fd)


def _cell(fx, group, prefix, pick=None):
    idx = np.flatnonzero(fx[f"{prefix}_group"] == group)
    c = idx[0] if pick is None else idx[pick(fx, idx)]
    return {k: (v[c] if v.ndim else v) for k, v in fx.items() if k.startswith(prefix)}, c


def test_gradients_inside_every_albedo_and_blk_ramp():
    from legoesm import constants
    lm = _load("si3_land_mask.npz")
    sub, _ = _cell(lm, "wet_ice", "alb", lambda fx, i: int(np.argmax(fx["alb_in:h_s"][i] == 0.0)))
    sub["alb_in:h_s"] = np.asarray(0.0)
    tf = constants.T_freeze
    # melt flag ramp (T in [Tf - 0.01 K, Tf])
    _fd_check(lambda T: _alb(sub, t_su=T), tf - 0.005, 1e-5)
    # thickness pieces (h in [0.05, 0.0505] and [pivot, 1.01 pivot])
    for h0 in (constants.albedo_ice_thin_break_nemo, constants.albedo_ice_pivot_orca1):
        _fd_check(lambda h: _alb(sub, t_su=jnp.asarray(tf - 1.0), h_i=h), h0 * 1.005, h0 * 1e-6)
    b, _ = _cell(lm, "wet_ice", "blk2", lambda fx, i: int(np.argmax((fx["blk2_in:h_s"][i] == 0.0)
                                                                    & (fx["blk2_in:qsr"][i] > 10.0))))
    b["blk2_in:h_s"] = np.asarray(0.0)
    b["blk2_in:snow"] = np.asarray(max(float(b["blk2_in:snow"]), 1e-5))
    # min(theta_air, rt0) ramp
    _fd_check(lambda t: _blk(b, theta_air_zt=t)["qprec_ice"], tf - 0.005, 1e-5)
    # thin-ice transmission ramp (h in [0.099, 0.1])
    _fd_check(lambda h: _blk(b, h_i=h)["qtr_ice_top"], 0.1 * 0.995, 1e-8)


def test_goff_floor_ramp_gradient():
    from legoesm import constants
    from legoesm.thermo import nemo_si3_saturation_over_ice
    f = lambda T: nemo_si3_saturation_over_ice(T, jnp.asarray(1.0e5))[0]  # noqa: E731
    _fd_check(f, constants.T_goff_floor_nemo + 0.005, 1e-5)
    # the returned analytic dq/dT carries the ramp slope (codex round 2)
    T = jnp.asarray(constants.T_goff_floor_nemo + 0.0025)
    dq = float(nemo_si3_saturation_over_ice(T, jnp.asarray(1.0e5))[1])
    np.testing.assert_allclose(dq, float(jax.grad(f)(T)), rtol=1e-10)
    hard = nemo_si3_saturation_over_ice(jnp.asarray(250.0), jnp.asarray(1.0e5), floor_width_K=0.0)
    soft = nemo_si3_saturation_over_ice(jnp.asarray(250.0), jnp.asarray(1.0e5))
    assert float(hard[0]) == float(soft[0]) and float(hard[1]) == float(soft[1])


def test_gradients_inside_every_flx_other_ramp():
    from legoesm import constants
    s4 = _load("si3_qlead_501_cuts.npz")
    sub, _ = _cell(s4, "s4", "flx")
    eps = 1.0e-10
    # ice switch ramp (at_i in [0.99 epsi10, epsi10]): qsb
    _fd_check(lambda a: _flx(sub, at_i=a)["qsb_ice_bot"], eps * (1.0 - 0.005), 1e-16)
    # packed cut ramp: at_i in [amax - 0.01 - 1e-5, amax - 0.01], ice at rest
    zero = jnp.asarray(0.0)
    rest = dict(u_ice=zero, u_ice_w=zero, v_ice=zero, v_ice_s=zero)
    floor = float(sub["flx_in:rn_amax_2d"]) - constants.ice_landfast_margin_nemo501
    _fd_check(lambda a: _flx(sub, at_i=a, **rest)["qlead"], floor - 0.5e-5, 1e-10)
    # landfast speed ramp on the squared speed, gradient finite at rest too
    stop = constants.ice_landfast_speed_stop_nemo
    u0 = 2.0 * stop * np.sqrt(1.005)   # speed^2 = 0.25*u0^2 = 1.005 stop^2, inside [stop^2, 1.01 stop^2]
    a_packed = jnp.asarray(floor + 1e-3)
    _fd_check(lambda u: _flx(sub, at_i=a_packed, u_ice=u, u_ice_w=zero, v_ice=zero, v_ice_s=zero)["qlead"],
              u0, 1e-9)
    g = jax.grad(lambda u: _flx(sub, u_ice=u, u_ice_w=zero, v_ice=zero, v_ice_s=zero)["qlead"])(zero)
    assert np.isfinite(float(g))
    # supercool stop ramps (cell without a lead cut): dT in [0, 0.01 K] with vt >= 20 m, vt in [19.8, 20] m
    w, _ = _cell(s4, "wet_ice", "flx", lambda fx, i: int(np.argmax(fx["flx_nemo:qlead"][i] < 0.0)))
    assert float(w["flx_nemo:qlead"]) < 0.0
    T0 = float(w["flx_scalar:rt0"])
    tb = float(w["flx_in:t_bo"])
    big_v = jnp.asarray(25.0)
    _fd_check(lambda x: _flx(w, sst_m=x, vt_i=big_v)["qlead"], tb - T0 - 0.005, 1e-6)
    vstop = constants.ice_supercool_volume_stop_nemo
    _fd_check(lambda v: _flx(w, sst_m=jnp.asarray(tb - T0 - 0.5), vt_i=v)["qlead"], vstop * 0.995, 1e-5)
    # full-cover ramp: at_i in [1 - epsi10 - 1e-12, 1 - epsi10] (packed cut moved out of reach)
    _fd_check(lambda a: _flx(w, at_i=a, rn_amax_2d=jnp.asarray(2.0), vt_i=jnp.asarray(1.0))["qlead"],
              1.0 - eps - 0.5e-12, 2e-14, tol=1e-2)   # FD limited by fp64 spacing (1.1e-16) at x ~ 1


def _band(f_soft, f_hard, x0, h):
    """At a point inside a C1 floor/cap band the smooth value differs from the hard one, and grad == FD."""
    soft, hard = float(f_soft(jnp.asarray(x0))), float(f_hard(jnp.asarray(x0)))
    assert soft != hard, (soft, hard)
    _fd_check(f_soft, x0, h)


def _freezing_cell():
    s4 = _load("si3_qlead_501_cuts.npz")
    # an ice cell above freezing (freeze energy < 0) with a positive basal flux
    w, _ = _cell(s4, "wet_ice", "flx", lambda fx, i: int(np.argmax(
        (fx["flx_nemo:qsb_ice_bot"][i] > 0.0)
        & (fx["flx_in:t_bo"][i] < fx["flx_in:sst_m"][i] + float(fx["flx_scalar:rt0"])))))
    assert float(w["flx_nemo:qsb_ice_bot"]) > 0.0
    g = {k.split(":", 1)[1]: float(v) for k, v in w.items() if k.startswith(("flx_in:", "flx_scalar:"))}
    freeze = g["rho0"] * g["rcp"] * g["e3t_m"] * (g["t_bo"] - (g["sst_m"] + g["rt0"]))
    lead0 = g["rDt_ice"] * ((1.0 - g["at_i_b"]) * (g["qsr_oce"] * g["frq_m"] + g["qns_oce"]) + g["qemp_oce"])

    def qns_for_lead(lead):   # lead energy is affine in qns_oce
        return g["qns_oce"] + (lead - lead0) / (g["rDt_ice"] * (1.0 - g["at_i_b"]))
    return w, g, freeze, qns_for_lead


def test_qlead_min_zero_is_smooth():
    w, g, freeze, qns_for_lead = _freezing_cell()
    q = qns_for_lead(freeze - 0.005 * g["rDt_ice"])           # lead - freeze inside (-0.01 dt, 0)
    _band(lambda x: _flx(w, qns_oce=x)["qlead"], lambda x: _flx(w, 0.0, qns_oce=x)["qlead"], q, 1e-6)


def test_fhld_max_zero_and_ice_floor_are_smooth():
    w, g, freeze, qns_for_lead = _freezing_cell()
    assert freeze < 0.0
    at = jnp.asarray(0.5)
    q = qns_for_lead(0.005 * g["rDt_ice"] * 0.5)              # (lead - 0)/dt/at_i inside (0, 0.01) W/m2
    _band(lambda x: _flx(w, qns_oce=x, at_i=at)["fhld"], lambda x: _flx(w, 0.0, qns_oce=x, at_i=at)["fhld"],
          q, 1e-9)
    q = qns_for_lead(10.0 * g["rDt_ice"])                     # warming lead; fhld ~ 1/max(at_i, epsi10)
    eps = 1.0e-10
    _band(lambda a: _flx(w, qns_oce=jnp.asarray(q), at_i=a)["fhld"],
          lambda a: _flx(w, 0.0, qns_oce=jnp.asarray(q), at_i=a)["fhld"], eps * 1.005, 1e-16)


def test_qsb_cap_and_freeze_parts_are_smooth():
    w, g, freeze, qns_for_lead = _freezing_cell()
    raw = float(_flx(w, 0.0, e3t_m=jnp.asarray(1e6))["qsb_ice_bot"])    # cap far away -> uncapped qsb
    assert raw > 0.0
    # cap = -freeze/dt/at_i is linear in e3t: place cap - raw inside (0, 0.01) W/m2 (min_c1 band)
    cap_per_e3t = -freeze / g["e3t_m"] / g["rDt_ice"] / max(g["at_i"], 1e-10)
    e3t = (raw + 0.005) / cap_per_e3t
    _band(lambda e: _flx(w, e3t_m=e)["qsb_ice_bot"], lambda e: _flx(w, 0.0, e3t_m=e)["qsb_ice_bot"], e3t, 1e-9 * e3t)
    # positive/negative parts of the freezing energy: freeze inside (0, 0.01 dt) (tiny supercooling)
    k = g["rho0"] * g["rcp"] * g["e3t_m"]
    sst = g["t_bo"] - g["rt0"] - 0.005 * g["rDt_ice"] / k
    qn = jnp.asarray(qns_for_lead(10.0 * g["rDt_ice"]))      # warming lead: fhld = (lead - freeze_pos)/dt/at_i
    _band(lambda x: _flx(w, sst_m=x, qns_oce=qn)["fhld"], lambda x: _flx(w, 0.0, sst_m=x, qns_oce=qn)["fhld"],
          sst, 1e-5 * g["rDt_ice"] / k)
