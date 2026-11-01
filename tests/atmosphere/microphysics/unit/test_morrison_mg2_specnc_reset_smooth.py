"""RULE 5 tests of the smoothed CAM6 MG2 early droplet size reset on the
SPECIFIED droplet-number path (prognostic_droplet_number=False, the
production deck): size_dist_param_liq (micro_mg2_0.F90:1296; in-cloud mean
diameter held in [2, 50] um, ncic <= qcic/1e-20), which the port evaluates
inside the kk2000_cam6 autoconversion and again for the MG2 snow riming
(#2216, user 2026-11-01: the default moves to the smooth form).

``MorrisonConfig.mg2_drop_early_reset_log_width`` (the prognostic path's
field, reused: the same MG2 call; codex + GLM-5.2 2026-11-01) is the
LogSumExp width (ln space) of both limits, default 0.01; 0 = MG2's hard clip
(in the oracle tests' HARD_LIMIT).  Gap: per specified-number oracle fixture,
per quantity its whole-call oracle test compares, per column, the default's
departure from that hard limit (every other switch at its hard limit),
relative to the column's largest MG2 value of that quantity.  Same format as
test_morrison_mg2_drop_early_reset_smooth.py.  x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402

from tests.atmosphere.microphysics.unit import test_mg2_ice_snow_oracle as S  # noqa: E402
from tests.atmosphere.microphysics.unit import test_mg2_warm_rain_oracle as R  # noqa: E402
from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import HARD_LIMIT, TOL, _rel  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
WIDTH = 0.01
F = "mg2_drop_early_reset_log_width"
CASES = ("warm_rv1", "warm_rv2", "warm_rv10", "snow")
# GAP: the reset alone (every other switch at its hard limit), default vs
# hard, per compared quantity and column.  Measured 2026-11-01; re-measure on
# any change of the reset or its width.
GAP = {
    "warm_rv1": {
        "prc": [2.89768593450626e-15, 7.181791319721783e-15, 7.482415118239819e-15, 6.932362431996127e-15, 1.4240321504480756e-15, 0.0],
        "pra": [0.0, 0.0, 0.0, 1.776327218538536e-16, 0.0, 0.0],
        "evap": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dq_r": [3.69716723408812e-15, 5.207145661689298e-16, 6.497937481157081e-16, 2.318120770659974e-15, 4.665338987345205e-16, 0.0],
        "dN_r": [3.126889751738716e-15, 6.853601514674669e-15, 5.285370959703066e-15, 5.024956612834105e-15, 6.092158821261943e-15, 0.0],
        "precip": [3.5536608738274703e-16, 0.0, 1.3320967854499058e-15, 1.862834161444325e-16, 1.839752368570503e-15, 0.0],
    },
    "warm_rv2": {
        "prc": [2.849782882836811e-15, 7.0234984235191765e-15, 7.46116196794395e-15, 7.132630359132924e-15, 1.4682164009251898e-15, 0.0],
        "pra": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "evap": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dq_r": [3.179778096995811e-15, 3.982671688209683e-16, 0.0, 1.4570538606005556e-15, 6.073432680137545e-19, 0.0],
        "dN_r": [3.1727002667871814e-15, 6.521059831435386e-15, 3.3924286504659936e-15, 4.9340712981074445e-15, 3.427322333550672e-16, 0.0],
        "precip": [4.1901963183959703e-16, 3.1863701864838814e-16, 0.0, 1.6652153564177127e-16, 1.6022583754905952e-15, 0.0],
    },
    "warm_rv10": {
        "prc": [3.015767856296434e-15, 7.180176961874657e-15, 7.523822227193236e-15, 7.197861697928993e-15, 1.511339846884232e-15, 0.0],
        "pra": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "evap": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dq_r": [6.650138065121179e-15, 5.420632959726834e-16, 0.0, 3.4248036794269e-15, 6.07360393700565e-19, 0.0],
        "dN_r": [4.912649961464009e-15, 3.4217425941765036e-15, 1.879989232553513e-15, 5.101075021370852e-15, 1.7830602501964849e-16, 0.0],
        "precip": [7.048583614389361e-16, 1.5753028364942432e-16, 0.0, 1.474149416212366e-16, 2.267289762859636e-15, 0.0],
    },
    "snow": {
        "dq_s": [3.618238205801238e-16, 2.312662051392827e-16, 0.0, 0.0, 0.0],
        "dN_s": [8.00969933774361e-16, 5.987380324296415e-16, 0.0, 0.0, 0.0],
        "dq_r": [4.2353354785534496e-16, 5.770886433495029e-16, 0.0, 0.0, 8.70902439363995e-22],
        "dN_r": [1.910820683492658e-15, 7.257009660934787e-16, 0.0, 0.0, 3.9359627147666846e-18],
        "psacws": [1.5274118267694708e-15, 1.5406961857197224e-16, 0.0, 0.0, 1.2590056938206493e-15],
        "bergs": [0.0, 3.4150251307454595e-16, 0.0, 0.0, 1.1076017902010745e-15],
        "snow_sfc": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rain_sfc": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
}
# POINT_GAP: the MG2 kernel points of mg2_warm_rain.npz (reset + KK2000
# autoconversion, no other switch), default vs the running MG2: (first, stop,
# relative gap) of the one run of points that moves (qcic = 5e-3 kg/kg, the
# 50 um side); every other point within round-off (<= 1e-12).
POINT_GAP = {"ncic_adj": (3600, 3960, 9.229213299880632e-09),
             "prc": (3600, 3960, 1.0152133662136641e-08),
             "nprc": (3600, 3960, 1.0152133491373965e-08),
             "nprc1": (3600, 3960, 9.229205191117857e-10)}


def _case(name):
    """(fixture, hard-limit config, run, {quantity: (port values, MG2 values)})."""
    if name == "snow":
        fx = S._fx()
        cfg = S._cfg(fx)

        def q(out):
            qb, rho = out.qc_budget, fx["col_rho"]
            snow = np.asarray(out.precip_frozen)
            return {"dq_s": (out.dq_s_dt, fx["ora_qstend"]),
                    "dN_s": (out.dN_s_dt, fx["ora_nstend"]),
                    "dq_r": (out.dq_r_dt, fx["ora_qrtend"]),
                    "dN_r": (np.asarray(out.dN_r_dt) / rho, fx["ora_nrtend"]),
                    "psacws": (-np.asarray(qb["riming_snow"]), fx["ora_psacwstot"]),
                    "bergs": (-np.asarray(qb["bergeron_snow"]), fx["ora_bergstot"]),
                    "snow_sfc": (snow, fx["ora_sflx_sfc"]),
                    "rain_sfc": (np.asarray(out.precipitation) - snow, fx["ora_rflx_sfc"])}
        return fx, cfg, lambda c: S._run_columns(fx, c), q
    rv = float(name[len("warm_rv"):])
    fx = R._fx()
    cfg = R._production_morrison()._replace(**HARD_LIMIT, warm_rain_scheme="kk2000_cam6",
                                            kk2000_cam6_relvar=rv)
    p = f"ora_rv{rv:g}_"

    def q(out):
        qb, rho = out.qc_budget, fx["col_rho"]
        return {"prc": (-np.asarray(qb["autoconversion"]), fx[p + "prctot"]),
                "pra": (-np.asarray(qb["accretion"]), fx[p + "pratot"]),
                "evap": (np.asarray(out.dq_v_dt) + np.asarray(qb["condensation"]),
                         fx[p + "nevapr"]),
                "dq_r": (out.dq_r_dt, fx[p + "qrtend"]),
                "dN_r": (np.asarray(out.dN_r_dt) / rho, fx[p + "nrtend"]),
                "precip": (out.precipitation, fx[p + "rain_sfc_flux"])}
    return fx, cfg, lambda c: R._run_columns(fx, c), q


def _spy_reset_inputs(name, monkeypatch, width=0.0):
    """Run at ``width``, recording the arguments (and widths) of every
    specified-path reset call: the autoconversion's (inside _warm_rain) and
    the riming's (morrison)."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    fx, cfg, run, q = _case(name)
    seen = {"autoconversion": [], "riming": []}
    widths = {"autoconversion": [], "riming": []}
    real_b, real_d = W.mg2_size_bounded_nc, M.mg2_size_dist_liq

    def spy_b(*a, **k):
        seen["autoconversion"].append(a)
        widths["autoconversion"].append(k)
        return real_b(*a, **k)

    def spy_d(*a, **k):
        seen["riming"].append(a)
        widths["riming"].append(k)
        return real_d(*a, **k)

    monkeypatch.setattr(W, "mg2_size_bounded_nc", spy_b)
    monkeypatch.setattr(M, "mg2_size_dist_liq", spy_d)
    out = run(cfg._replace(**{F: width}))
    monkeypatch.setattr(W, "mg2_size_bounded_nc", real_b)
    monkeypatch.setattr(M, "mg2_size_dist_liq", real_d)
    if width != 0.0:
        return widths
    return fx, cfg, run, q, out, seen


def _bounds_bind(args):
    """Cells where MG2's reset moves the number, split by which bound: the
    hard port's mean diameter (pgam+1)/lambda sits on 2 um / 50 um."""
    qc, n, rho = args
    active, pgam, lam, nc_kg = W.mg2_droplet_psd(qc, n, rho)
    d_mean = np.asarray((pgam + 1.0) / lam)
    moved = np.asarray(active) & ~np.isclose(np.asarray(nc_kg * rho), np.asarray(n),
                                              rtol=1e-12, atol=0.0)
    return (moved & np.isclose(d_mean, 2.0e-6, rtol=1e-12, atol=0.0),
            moved & np.isclose(d_mean, 50.0e-6, rtol=1e-12, atol=0.0))


def _warm_points(width):
    fx = R._fx()
    i = {k[6:]: jnp.asarray(v) for k, v in fx.items() if k.startswith("pt_in_")}
    o = {k[7:]: v for k, v in fx.items() if k.startswith("pt_out_")}
    rho = i["rho"]
    w = dict(number_cap_log_width=width, slope_log_width=width)
    prc, nr_au, x_c = W.autoconversion_kk2000_cam6(i["qcic"], i["ncic"] * rho, rho,
                                                   i["relvar"], **w)
    return {"ncic_adj": (W.mg2_size_bounded_nc(i["qcic"], i["ncic"] * rho, rho, **w) / rho,
                         o["ncic_adj"]),
            "prc": (prc, o["prc"]), "nprc": (nr_au / rho, o["nprc"]),
            "nprc1": (jnp.where(prc > 0, prc / x_c, 0.0), o["nprc1"])}, i, o


def _snow_points(width):
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.morrison import resolve_morrison_flavor
    fx = S._fx()
    i = {k[6:]: jnp.asarray(v) for k, v in fx.items() if k.startswith("pt_in_")}
    o = {k[7:]: v for k, v in fx.items() if k.startswith("pt_out_")}
    cfg = resolve_morrison_flavor(MorrisonConfig(morrison_flavor="mg", ice_snow_d_auto=500.0e-6,
                                                 melt_sharpness=1.0e9))
    rho, T, p, pf = i["rho"], i["T"], i["p"], i["precip_frac"]
    qsic, nsic, lams, n0s = W.mg2_inprecip_snow(i["qs"], i["ns"], pf, cfg, psd_sharpness=1.0e9)
    _, uns = W.mg2_snow_fall_speeds(lams, rho, cfg)
    nc_adj, pgam, lamc = W.mg2_size_dist_liq(i["qcic"], i["ncic"] * rho, rho,
                                             number_cap_log_width=width, slope_log_width=width)
    _, mu = W.mg2_snow_vent_eps(n0s, lams, T, p, rho, cfg)
    psacws, npsacws = W.mg2_accrete_cloud_water_snow(
        T, rho, mu, uns, i["qcic"], nc_adj / rho, qsic, pgam, lamc, lams, n0s, cfg, 1.0e-9)
    return {"ncic_adj": (nc_adj / rho, o["ncic_adj"]), "pgam": (pgam, o["pgam"]),
            "lamc": (lamc, o["lamc"]), "psacws": (psacws, o["psacws"]),
            "npsacws": (npsacws, o["npsacws"])}, i, o




def test_defaults_and_hard_limit():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    assert MorrisonConfig().mg2_drop_early_reset_log_width == WIDTH
    assert ExperimentConfig._field_defaults[f"morrison_{F}"] == WIDTH
    assert HARD_LIMIT[F] == 0.0
    # the deck runs the specified droplet number: the path these tests cover
    cfg = R._production_morrison()
    assert not cfg.prognostic_droplet_number and cfg.warm_rain_incloud
    assert cfg.mg2_drop_early_reset_log_width == WIDTH


@pytest.mark.parametrize("pts", [_warm_points, _snow_points])
def test_hard_limit_matches_mg2_kernel_points(pts):
    """Reset + autoconversion (warm) and reset + snow riming (snow) vs the
    RUNNING MG2's micro_mg_utils.F90 at every fixture point, width 0; at the
    default the warm points' comparison fails (the test can fail)."""
    q, i, o = pts(0.0)
    for k, (a, b) in q.items():
        assert float(_rel(a, b).max()) <= TOL, k
    assert np.any(o["ncic_adj"] != np.asarray(i["ncic"]))      # the reset moves points
    if pts is _warm_points:
        soft, _, _ = pts(WIDTH)
        assert float(_rel(*soft["prc"]).max()) > 10.0 * TOL


def test_default_point_gap_is_the_recorded_one():
    q, _, _ = _warm_points(WIDTH)
    for k, (a, b) in q.items():
        e = _rel(a, b)
        lo, hi, v = POINT_GAP[k]
        want = np.zeros_like(e)
        want[lo:hi] = v
        print(k, "point gap", float(e.max()))
        np.testing.assert_allclose(e, want, rtol=1e-6, atol=1e-12, err_msg=k)
    q, _, _ = _snow_points(WIDTH)      # no snow point is near a bound: no gap
    for k, (a, b) in q.items():
        assert float(_rel(a, b).max()) <= 1e-12, k


@pytest.mark.parametrize("name", CASES)
def test_hard_limit_matches_mg2_whole_call_through_both_sites(name, monkeypatch):
    """Whole call vs the RUNNING MG2 at width 0, with the specified-number
    reset reached at the autoconversion (all cases) and the riming (snow),
    and the 2 um bound binding at a fixture cell there."""
    fx, cfg, run, q, out, seen = _spy_reset_inputs(name, monkeypatch)
    assert len(seen["autoconversion"]) == 1
    assert len(seen["riming"]) == (1 if name == "snow" else 0)
    for calls in seen.values():
        for a in calls:
            assert _bounds_bind(a)[0].sum() >= 1
    for k, (a, b) in q(out).items():
        assert float(_rel(a, b).max()) <= TOL, k


@pytest.mark.parametrize("name", CASES)
def test_default_gap_per_column_is_the_recorded_one(name):
    fx, cfg, run, q = _case(name)
    hard, soft = q(run(cfg._replace(**{F: 0.0}))), q(run(cfg._replace(**{F: WIDTH})))
    for k in hard:
        d = np.abs(np.asarray(soft[k][0], np.float64) - np.asarray(hard[k][0], np.float64))
        scale = np.abs(np.asarray(hard[k][1], np.float64))
        if d.ndim == 1:
            d, scale = d[:, None], scale[:, None]
        gap = d.max(axis=1) / np.maximum(scale.max(axis=1), 1e-300)
        print(name, k, "gap per column", [float(x) for x in gap])
        np.testing.assert_allclose(gap, GAP[name][k], rtol=1e-6, atol=1e-12, err_msg=k)
        # a gap below the 1e-12 floor must not vanish (or appear) unnoticed
        assert np.array_equal(gap > 0.0, np.asarray(GAP[name][k]) > 0.0), k


def _near_50um_bound(u):
    """Autoconversion inputs whose mean diameter sits ``3*u*WIDTH`` in ln N
    from the 50 um bound (u < 0: too few droplets, the side MG2 clips)."""
    q = 1.0e-4 * jnp.ones_like(u)
    one = jnp.ones_like(u)
    n_b = W.mg2_size_bounded_nc(q, 1.0e3 * one, one)
    return q, n_b * jnp.exp(3.0 * WIDTH * u), one


def test_gradients_through_the_autoconversion_reset():
    u = jnp.asarray([-2.0, -0.5, 0.5, 2.0])
    w = dict(number_cap_log_width=WIDTH, slope_log_width=WIDTH)

    def f(x, **k):
        q, n, rho = _near_50um_bound(x)
        prc, nr, x_c = W.autoconversion_kk2000_cam6(q, n, rho, 2.0, **k)
        return jnp.stack([jnp.log(prc), jnp.log(x_c)])

    check_grads(lambda x: f(x, **w), (u,), order=2, modes=["rev"], eps=1e-3, atol=1e-6,
                rtol=1e-3)
    g = np.asarray(jax.grad(lambda x: jnp.sum(f(x, **w)[0]))(u))
    hard = np.asarray(jax.grad(lambda x: jnp.sum(f(x)[0]))(u))
    assert np.all(np.isfinite(g)) and np.all(g != 0.0)
    # MG2's clip has no derivative of the rate on its clipped side; the rounded one does
    assert np.all(np.abs(hard[u < 0]) < 1e-2 * np.abs(g[u < 0]))


def test_whole_call_gradient_finite_at_the_default():
    fx, cfg, run, q = _case("snow")
    cfg = cfg._replace(**{F: WIDTH})

    def f(nc):
        g = dict(fx)
        g["col_N_c"] = nc
        out = S._run_columns(g, cfg)
        return jnp.sum(out.dq_c_dt) + jnp.sum(out.dq_s_dt)

    g = np.asarray(jax.grad(f)(jnp.asarray(fx["col_N_c"])))
    assert np.isfinite(g).all() and np.abs(g).max() > 0.0


def test_flat_field_validation_and_production_deck():
    from scripts.run.run_amip import build_arg_parser

    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.run_config_yaml import load_yaml_config
    f = f"morrison_{F}"
    # read on the specified path: kk2000_cam6 (the mg flavour's law) or MG2 snow
    ExperimentConfig(microphysics="morrison", **{f: 0.0}).validate_strict()
    for kw in (dict(microphysics="morrison", morrison_flavor="sam"),
               dict(microphysics="morrison", morrison_warm_rain_scheme="kk2000"),
               dict(microphysics="kessler")):
        with pytest.raises(ValueError, match=f"{f}=0.0 is read only .* silently inert"):
            ExperimentConfig(**kw, **{f: 0.0}).validate_strict()
    amip = ROOT / "config/amip"
    prod = load_yaml_config(amip / "amip_production.yaml", build_arg_parser())
    assert prod[f] == WIDTH                      # named explicitly (RULE 3)
    assert prod["morrison_prognostic_droplet_number"] is False


def test_default_width_reaches_both_sites(monkeypatch):
    """The configured width is what both specified-path reset calls receive."""
    w = _spy_reset_inputs("snow", monkeypatch, width=WIDTH)
    want = dict(number_cap_log_width=WIDTH, slope_log_width=WIDTH)
    assert w["autoconversion"] == [want] and w["riming"] == [want]
