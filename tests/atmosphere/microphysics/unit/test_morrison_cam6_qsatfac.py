"""CAM6 do_subgrid_growth: the sub-grid saturation factor qsatfac
(cldfrc2m.F90:1116-1117) and its use as MG2's qsfm (micro_mg2_0.F90:880,
922-927, 933, 1118), against the RUNNING CAM6 Fortran.

Fixture ``cam6_qsatfac.npz`` from ``scripts/validate/oracle/cam6_qsatfac``
(make_fixture.py; unmodified wv_sat_methods / micro_mg_utils / micro_mg2_0,
verbatim cldfrc2m and MG2 lines in qsatfac_driver.F90).  x64.

* qsatfac (ramp term from the aist call, completed by morrison.mg2_qsfm on
  MG2's Goff-Gratch liquid curve): on CAM's curves it matches the Fortran
  to 1e-12, troposphere and legoESM's own stratosphere switch; on the aist
  curves (Goff liquid, Flatau ice, the documented aist departure) the
  measured gap is pinned.  Above CLUBB's top the ramp blends to 1.
* the scaled qvl/qvi, the nucleation humidity factor and every MG2 kernel
  that reads them match the Fortran at the hard limit (1e-12, as the
  unscaled kernel oracle).
* in the scheme, qsfm reaches exactly those kernels, only below T_freeze;
  a ramp of 1 is bitwise the qsfm = 1 path.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.clouds import cloud_fraction as CF  # noqa: E402
from legoesm.atmosphere.physics.clouds.config import CloudConfig  # noqa: E402
from legoesm.atmosphere.physics.microphysics import morrison as M  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402
from legoesm.thermo import (  # noqa: E402
    saturation_specific_humidity_goff_gratch as qsl,
    saturation_specific_humidity_ice_goff_gratch as qsi,
    saturation_vapor_pressure_goff_gratch as esl_gg,
    saturation_vapor_pressure_ice_goff_gratch as esi_gg,
)

from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_ice as I  # noqa: E402
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/cam6_qsatfac.npz"
J = jnp.asarray
NO_TOP = dict(trop_cloud_top_press=0.0, trop_cloud_taper_lnp_width=0.15)
QSATFAC_CURVE_GAP = 0.005476139307012856


def _fx():
    return dict(np.load(FIXTURE))


def _cam_curves(monkeypatch):
    """CAM's Goff-Gratch curves in place of the aist curves (liquid Goff,
    ice Flatau: the documented aist curve departure)."""
    monkeypatch.setattr(CF, "saturation_vapor_pressure_ice_flatau", esi_gg)
    monkeypatch.setattr(CF, "saturation_vapor_pressure_goff", esl_gg)
    monkeypatch.setattr(CF, "saturation_mixing_ratio_goff",
                        lambda T, p: qsl(T, p) / (1.0 - qsl(T, p)))


def _ramp(fx, sel, top=NO_TOP):
    """legoESM ramp term at the selected fixture points (one level per column),
    with the production CloudConfig: the stratospheric points (rhminis =
    rhmaxis = 1 in the fixture) get a layer top above the equatorial
    tropopause, so legoESM's own stratosphere switch selects them."""
    T, p = J(fx["in_T"][sel])[:, None], J(fx["in_p"][sel])[:, None]
    strat = J(fx["in_rhmini"][sel] == 1.0)[:, None]
    p_top = jnp.where(strat, 5000.0, p)
    _, ramp = CF.cam6_ice_stratus_fraction(
        J(fx["in_q"][sel])[:, None], T, p, J(fx["in_qi"][sel])[:, None],
        jnp.zeros(T.shape[0]), CloudConfig(scheme="cam6_clubb"), p_top,
        return_qsatfac=True, **top)
    return ramp


def _qsatfac(fx, sel, monkeypatch=None, width=0.0):
    """legoESM qsfm (ramp from the aist call, completed by MG2's mg2_qsfm on
    MG2's Goff-Gratch liquid curve) at cold fixture points."""
    if monkeypatch is not None:
        _cam_curves(monkeypatch)
    T, p = J(fx["in_T"][sel])[:, None], J(fx["in_p"][sel])[:, None]
    qf = M.mg2_qsfm(T, J(fx["in_q"][sel])[:, None], qsl(T, p), _ramp(fx, sel), width)
    return np.asarray(qf[:, 0])


def _rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


def test_fixture_coverage():
    fx = _fx()
    tro = fx["in_rhmini"] == 0.8
    cold = fx["in_T"] < C.T_freeze
    q, qf = fx["out_qsatfac"], fx["in_q"] / (fx["out_qvl"] / fx["out_qsatfac"])
    assert ((fx["out_qsatfac"] < 1.0) & tro & cold).sum() >= 100       # scaling active
    assert ((q > 0.8) & (q < 1.0) & tro).sum() >= 50                     # inside the ramp
    assert (np.isclose(q, qf, rtol=1e-12) & (q < 1.0) & tro & cold).any()  # qv/qs branch
    assert ((q == 1.0) & tro).any()                                       # capped at 1
    assert (q[~tro] == 1.0).all()                                         # stratospheric step


@pytest.mark.parametrize("strat", [False, True])
def test_qsatfac_matches_cam6_on_cams_curves(strat, monkeypatch):
    fx = _fx()
    sel = (fx["in_T"] < C.T_freeze) & ((fx["in_rhmini"] == 1.0) == strat)
    got = _qsatfac(fx, sel, monkeypatch)
    err = _rel(got, fx["out_qsatfac"][sel])
    print("qsatfac strat" if strat else "qsatfac trop", err)
    assert err < 1e-12


def test_qsatfac_curve_gap_is_the_documented_aist_departure():
    """KNOWN DIFFERENCE (aist docstring): legoESM's aist ramp uses the shared
    Goff liquid and Flatau ice curves, each < 0.5 % from CAM's Goff-Gratch
    over 200-273 K; the ramp term inherits both through rhi (the min(qv/qs, 1)
    branch is on MG2's Goff-Gratch curve, as in CAM).  The measured gap on
    the fixture (max relative, measured 2026-11-03) is pinned."""
    fx = _fx()
    sel = (fx["in_T"] < C.T_freeze) & (fx["in_rhmini"] == 0.8)
    got = _qsatfac(fx, sel)
    gap = np.abs(got - fx["out_qsatfac"][sel]) / fx["out_qsatfac"][sel]
    print("qsatfac curve gap", repr(gap.max()))
    assert gap.max() == pytest.approx(QSATFAC_CURVE_GAP, rel=1e-6)


def test_ramp_blends_to_one_above_the_clubb_top():
    """Above CLUBB's top (where CAM's MG2 does not run) the ramp term goes to
    1 with the aist taper weight w: ramp_top = 1 + w (ramp - 1)."""
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    fx = _fx()
    sel = (fx["in_T"] < C.T_freeze) & (fx["in_rhmini"] == 0.8)
    p = J(fx["in_p"][sel])[:, None]
    width = CLUBBConfig().trop_cloud_taper_lnp_width
    top_p = 2.0 * float(np.percentile(fx["in_p"][sel], 20))
    free = _ramp(fx, sel)
    topped = _ramp(fx, sel, dict(trop_cloud_top_press=top_p,
                                 trop_cloud_taper_lnp_width=width))
    w = CF.trop_cloud_top_taper(p, top_p, width)
    np.testing.assert_array_equal(np.asarray(topped), np.asarray(1.0 + w * (free - 1.0)))
    above = np.asarray(p[:, 0]) < top_p * 0.5
    assert above.any() and (np.asarray(w)[above, 0] < 0.02).all()
    assert (np.asarray(free)[above, 0] < 0.99).any()


def test_scaled_saturation_and_kernels_match_cam6(monkeypatch):
    fx = _fx()
    qf = J(fx["out_qsatfac"])

    def scaled(T, p):
        f = jnp.where(T >= C.T_freeze, 1.0, qf)
        qvl = qsl(T, p)
        return qvl * f, jnp.where(T >= C.T_freeze, qvl, qsi(T, p)) * f

    monkeypatch.setattr(I, "_sat", scaled)
    got = I._points({k: v for k, v in fx.items() if not k.startswith("in_rhm")})
    for k in ("qvl", "qvi", "vap_dep", "ice_sublim", "berg", "bergs", "prds"):
        w = fx[f"out_{k}"]
        err = _rel(got[k], w)
        print("qsfm", k, err)
        assert err < 1e-12, k
        assert ((got[k] == 0) == (w == 0)).all(), k
    # the naai humidity test (scheme form, read only below tmelt - 5 K)
    cold = fx["in_T"] < C.T_freeze
    T = J(fx["in_T"])
    qvl, _ = scaled(T, J(fx["in_p"]))
    gate = np.asarray(J(fx["in_q"]) / jnp.clip(qvl, 1.0e-12) * esl_gg(T) / esi_gg(T))
    assert cold.sum() > 500
    assert _rel(gate[cold], fx["out_gate"][cold]) < 1e-12


def _run(qsatfac_ramp, monkeypatch=None, spies=()):
    """The cold mixed-phase columns of the ice test, with rain below freezing."""
    fx = A._fx()
    c, T, hyd = I._cold(fx)
    rho = J(fx["col_rho"])
    hyd = dict(hyd, q_r=jnp.where(c, 2e-5, 0.0), N_r=jnp.where(c, 2e-5 * rho / 6.5e-8, 0.0))
    seen = {n: I._spy(monkeypatch, n) for n in spies}
    g = {k: J(fx[f"col_{k}"]) for k in
         ("q_v", "q_c", "cf", "q_r", "N_r", "p_full", "p_half", "rho", "nc_kg", "relvar", "npccn")}
    dz = (g["p_half"][:, 1:] - g["p_half"][:, :-1]) / (C.g * g["rho"])
    z = jnp.zeros_like(T)
    kw = dict(q_c=g["q_c"], q_r=g["q_r"], q_i=z, q_s=z, q_g=z, N_c=g["nc_kg"] * g["rho"],
              N_r=g["N_r"], N_i=z, N_s=z)
    kw.update(hyd)
    out = M.morrison_microphysics(
        T, g["q_v"], HydrometeorState(**kw), g["p_full"], g["p_half"], g["rho"], dz,
        float(fx["dt"]), _cfg(do_snow_to_graupel=False), cloud_fraction=g["cf"],
        relvar=g["relvar"], npccn=g["npccn"] * g["rho"], qsatfac_ramp=qsatfac_ramp)
    if monkeypatch is not None:
        monkeypatch.undo()    # a later run must not record through these spies
    return out, {n: dict(s) for n, s in seen.items()}, T, c


# (kernel, positions of qvl and qvi in its arguments)
KERNELS = {"rain_evaporation_mg2": (1, None), "mg2_ice_deposition_sublimation": (7, 8),
           "mg2_bergeron_snow": (3, 4), "mg2_snow_sublimation": (1, 2),
           "mg2_deposition_limiter": (None, 2)}


def test_qsfm_reaches_every_mg2_kernel_below_freezing_only(monkeypatch):
    _, base, T, c = _run(None, monkeypatch, KERNELS)
    ramp = jnp.full(T.shape, 0.85)
    _, sc, _, _ = _run(ramp, monkeypatch, KERNELS)
    q_v, q_sat = base["rain_evaporation_mg2"]["in"][:2]
    fac = np.asarray(M.mg2_qsfm(T, q_v, q_sat, ramp, _cfg().qsfm_freeze_ramp_K))
    cold = np.asarray(T) < C.T_freeze
    assert (~cold).any() and np.asarray(c).any()
    assert (fac[cold] == 0.85).any() and ((fac[cold] > 0.85) & (fac[cold] < 1.0)).any()
    assert (fac[~cold] == 1.0).all()
    for name, pos in KERNELS.items():
        for k in pos:
            if k is None:
                continue
            b, s = np.asarray(base[name]["in"][k]), np.asarray(sc[name]["in"][k])
            assert np.array_equal(s, b * fac), (name, k)
            assert not np.array_equal(s, b), (name, k)


def test_unit_qsatfac_is_bitwise_qsfm_one(monkeypatch):
    a, _, T, _ = _run(None, monkeypatch)
    b, _, _, _ = _run(jnp.ones_like(T), monkeypatch)
    for fld in a._fields:
        x, y = getattr(a, fld), getattr(b, fld)
        if isinstance(x, jax.Array):
            assert np.array_equal(np.asarray(x), np.asarray(y)), fld


def test_scaling_changes_the_tendencies(monkeypatch):
    a, _, T, _ = _run(None, monkeypatch)
    b, _, _, _ = _run(jnp.full_like(T, 0.85), monkeypatch)
    assert not np.array_equal(np.asarray(a.dq_v_dt), np.asarray(b.dq_v_dt))


def test_qsatfac_refused_outside_mg2_mode():
    fx = A._fx()
    T = J(fx["col_T"])
    z = jnp.zeros_like(T)
    hyd = HydrometeorState(q_c=J(fx["col_q_c"]), q_r=z, q_i=z, q_s=z, q_g=z,
                           N_c=None, N_r=None, N_i=None, N_s=None)
    with pytest.raises(ValueError, match="do_subgrid_growth"):
        M.morrison_microphysics(
            T, J(fx["col_q_v"]), hyd, J(fx["col_p_full"]), J(fx["col_p_half"]),
            J(fx["col_rho"]), jnp.ones_like(T), 300.0, M.MorrisonConfig(),
            qsatfac_ramp=jnp.ones_like(T))


def test_deck_key_reaches_the_cloud_config():
    from legoesm.driver.model_driver import _standalone_cloud_config
    from tests.unit.test_cam6_cloud_fraction import _exp_cfg
    cfg = _exp_cfg(cloud_cam6_do_subgrid_growth=True, microphysics="morrison")
    cfg.validate_strict()
    assert _standalone_cloud_config(cfg, "cam6_clubb").cam6_do_subgrid_growth
    assert not _standalone_cloud_config(_exp_cfg(), "cam6_clubb").cam6_do_subgrid_growth
    with pytest.raises(ValueError, match="cloud_cam6_do_subgrid_growth=True needs"):
        _exp_cfg(cloud_cam6_do_subgrid_growth=True).validate_strict()   # sundqvist


def test_run_amip_flag_round_trip():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_amip", ROOT / "scripts/run/run_amip.py")
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    p = ra.build_arg_parser()
    assert p.parse_args(["--cloud-cam6-do-subgrid-growth"]).cloud_cam6_do_subgrid_growth
    assert not p.parse_args([]).cloud_cam6_do_subgrid_growth


# RULE 5: MG2's qsfm switch at T_freeze, smoothed over MorrisonConfig.
# qsfm_freeze_ramp_K.  Cases: temperatures just below freezing, qv/qs = 0.7,
# ramp 0.85 (so qsatfac = 0.85); the hard switch gives qsfm = 0.85 below
# T_freeze.  Gap = |qsfm(default) - qsfm(hard)| per case, measured 2026-11-21.
QSFM_FREEZE_CASES_DT_K = (0.001, 0.025, 0.05, 0.075, 0.1, 0.15)
QSFM_FREEZE_GAPS = (0.1499553, 0.1265625, 0.075, 0.0234375, 0.0, 0.0)


def _qsfm_cases(width):
    T = C.T_freeze - jnp.asarray(QSFM_FREEZE_CASES_DT_K)
    q_sat = jnp.full_like(T, 1.0e-3)
    return np.asarray(M.mg2_qsfm(T, 0.7 * q_sat, q_sat, jnp.full_like(T, 0.85), width))


def test_qsfm_freeze_switch_hard_limit_is_mg2():
    got = _qsfm_cases(0.0)
    np.testing.assert_array_equal(got, np.full(got.shape, 0.85))
    T = jnp.asarray([C.T_freeze, C.T_freeze + 1.0])
    one = M.mg2_qsfm(T, jnp.full_like(T, 7e-4), jnp.full_like(T, 1e-3),
                     jnp.full_like(T, 0.85), 0.0)
    np.testing.assert_array_equal(np.asarray(one), [1.0, 1.0])


def test_qsfm_freeze_switch_default_gap_per_case():
    w = M.MorrisonConfig().qsfm_freeze_ramp_K
    gap = np.abs(_qsfm_cases(w) - _qsfm_cases(0.0))
    print("qsfm freeze gaps", repr(tuple(gap)))
    np.testing.assert_allclose(gap, QSFM_FREEZE_GAPS, rtol=1e-6, atol=1e-12)


def test_qsfm_freeze_switch_gradient_at_default():
    from jax.test_util import check_grads
    w = M.MorrisonConfig().qsfm_freeze_ramp_K

    def f(T):
        return M.mg2_qsfm(T, jnp.full_like(T, 7e-4), jnp.full_like(T, 1e-3),
                          jnp.full_like(T, 0.85), w)

    T = C.T_freeze - jnp.asarray([0.02, 0.05, 0.08])
    check_grads(f, (T,), order=2, modes=["fwd", "rev"])
    g = jax.grad(lambda t: jnp.sum(f(t)))(T)
    assert bool(jnp.all(jnp.isfinite(g))) and bool(jnp.all(g != 0.0))


@pytest.mark.parametrize("on", [False, True])
def test_mpas_lane_hands_the_ramp_to_mg2_only_when_on(on, monkeypatch):
    """Through the MPAS macmic lane: with cam6_do_subgrid_growth the ramp of
    the aist call reaches morrison_microphysics as qsatfac_ramp; without it
    nothing is passed (qsfm = 1)."""
    import sys
    sys.path.insert(0, "tests/unit")
    from legoesm.atmosphere.physics.microphysics import integration as mi
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from test_morrison_warm_rain_incloud import _mpas_physics
    from test_physics_macmic import _moist_setup
    real_micro, real_aist = mi.morrison_microphysics, mi.cam6_ice_stratus_fraction
    seen, ramps = [], []

    def aist(*a, **k):
        out = real_aist(*a, **k)
        if k.get("return_qsatfac"):
            ramps.append(np.asarray(out[1]))
        return out

    def micro(*a, qsatfac_ramp=None, **k):
        seen.append(None if qsatfac_ramp is None else np.asarray(qsatfac_ramp))
        return real_micro(*a, qsatfac_ramp=qsatfac_ramp, **k)
    monkeypatch.setattr(mi, "morrison_microphysics", micro)
    monkeypatch.setattr(mi, "cam6_ice_stratus_fraction", aist)
    mesh, sigma, state = _moist_setup()
    tr = dict(state.tracers)
    tr["q_i"] = state.T.replace(data=jnp.full_like(state.T.data, 1.0e-6), name="q_i",
                                units="kg/kg")
    tr["N_s"] = state.T.replace(data=jnp.zeros_like(state.T.data), name="N_s",
                                units="1/kg")
    state = state._replace(tracers=tr)
    _, cfg = _mpas_physics(True, n=2)
    rc = cfg.radiation._replace(cloud_config=cfg.radiation.cloud_config._replace(
        cam6_do_subgrid_growth=on))
    # the spies read concrete values: no module remat (it traces its body)
    cfg = cfg._replace(radiation=rc, module_checkpoint=False)
    from legoesm.atmosphere.physics.combined import make_physics
    fn = make_physics(cfg, model_type="mpas", dt=600.0, cld_macmic_num_steps=2)
    ps = init_physics_state(*state.T.data.shape, cfg)
    ps = ps._replace(cloud_fraction=jnp.full(state.T.data.shape, 0.123))
    fn(state, mesh, sigma, phys_state=ps)
    assert len(seen) == 2
    if on:
        assert len(ramps) == 2
        for s, r in zip(seen, ramps):
            np.testing.assert_array_equal(s, r.reshape(s.shape))
    else:
        assert seen == [None, None] and not ramps
