"""RULE 5 tests of the KF-Eta port's local and source-geometry switches (#2280, PRs 2-3/7).

Smoothed here, each by one ``KFSmoothing`` width (0 = the oracle):

* ``lut_knot_width``: F1 (ALU log table) and F2 (TTAB/QSTAB and THE0K):
  knots and edge clamps of the piecewise-linear lookups;
* ``tpmix_width_kgkg``: D7a (TPMIX2 condense / evaporate / deficit split);
* ``condload_sqrt_width_m2s2``: D7b and the CONDLOAD square roots (E11 there);
* ``wtw_floor_width_m2s2``: D7d (the |WTW| < 1e-4 floor);
* ``prof5_width``: D7e (PROF5's two polynomial branches);
* PR 3/7, source geometry: ``lcl_gather_width_m`` (C-KLCLg, TENV/QENV/WKL/QSLCL/
  PLCL at ZLCL), ``l5_gather_width_pa`` (C-L5g, the L5 wind speed),
  ``q0_clip_width_kgkg`` / ``q0_floor_width_kgkg`` (E1) and
  ``tlcl_cap_width_k`` (E2).  The level indices themselves stay hard, and so
  do the mixed-layer averages (their source-layer extraction profile is hard;
  smoothing one without the other breaks the water budget, so both go with
  PR 7).

The three tests per switch: the hard limit is the oracle (the oracle suite in
test_kain_fritsch_oracle.py runs ``KF_HARD``); the per-case gap at the default,
each width alone and all together, stored per fixture arm, column and output in
``oracle_fixtures/kf_eta_rule5_gaps.json`` (1e-6 relative, 1e-12 absolute
floor); and ``check_grads`` through each smoothed switch at the default.

Regenerate the stored gaps (fp64), one case per process, with
``JAX_ENABLE_X64=1 python tests/atmosphere/test_kf_eta_rule5_local.py --write CASE``
for CASE in each ``KFSmoothing`` field and ``all``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402
from legoesm.atmosphere.physics.convection import _kf_eta as kfe  # noqa: E402
from legoesm.atmosphere.physics.convection.config import KainFritschConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.kain_fritsch_wrf import _lego_constants  # noqa: E402

from legoesm import constants  # noqa: E402
from tests.atmosphere.test_kain_fritsch_oracle import (  # noqa: E402
    _FIX,
    _OUT,
    _arms,
    _consts,
    _run_core,
)

_GAPS = Path(__file__).parent / "oracle_fixtures" / "kf_eta_rule5_gaps.json"
_DEFAULT = kfe.kf_smoothing(KainFritschConfig())
_FIELDS = kfe.KFSmoothing._fields
# each width alone at its default (the others hard), then all together
_CASES = {f: kfe.KF_HARD._replace(**{f: getattr(_DEFAULT, f)}) for f in _FIELDS}
_CASES["all"] = _DEFAULT
_CASES["hard"] = kfe.KF_HARD  # the port's own round-off gap, the baseline of the others
_OUTS = _OUT + ("pratec",)


def _sweep(fix, sm):
    """Per sweep column, the outputs of the port with widths ``sm``."""
    out = {k: np.zeros_like(fix[f"sweep_{k}"]) for k in _OUTS}
    for ci in np.unique(fix["sweep_combo"]):
        cs, trig, warm, fqi, fqs = fix["sweep_combos"][ci]
        sel = np.where(fix["sweep_combo"] == ci)[0]
        kc = _consts(fix, cs)

        def one(u, v, t, q, p, dz, w, rh, dt, dx):
            o = kfe.kf_eta_para(u[None], v[None], t[None], q[None], p[None], dz[None], w[None],
                              rh[None], dt=dt, dx=dx, constants=kc, trigger=int(trig),
                              warm_rain=warm == "1", f_qi=fqi == "1", f_qs=fqs == "1",
                              cudt=float(fix["cudt"]), smoothing=sm)
            return {k: o[k][0] for k in _OUTS}

        args = [jnp.asarray(fix[f"sweep_{k}"][sel]) for k in
                ("u", "v", "T", "q", "p", "dz", "w", "rhoe", "dt", "dx")]
        o = jax.jit(jax.vmap(one))(*args)
        for k in _OUTS:
            out[k][sel] = np.asarray(o[k])
    return out


def _colmax(ours, ref):
    d = np.abs(np.asarray(ours, np.float64) - np.asarray(ref, np.float64))
    return d.reshape(d.shape[0], -1).max(axis=1)


def measure_case(fix, case):
    """{arm or "sweep": {output: [max over levels |port - oracle| per column]}}."""
    sm = _CASES[case]
    g = {}
    for arm in _arms():
        o = _run_core(fix, arm, smoothing=sm)
        g[arm] = {k: _colmax(o[k], fix[f"{arm}_{k}"]).tolist() for k in _OUTS}
    o = _sweep(fix, sm)
    g["sweep"] = {k: _colmax(o[k], fix[f"sweep_{k}"]).tolist() for k in _OUTS}
    return g


def _provenance():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parent).stdout.strip()
    return dict(fixture_sha256=hashlib.sha256(_FIX.read_bytes()).hexdigest(),
                widths=_DEFAULT._asdict(), x64=bool(jax.config.jax_enable_x64),
                generated_at_commit=sha, metric="max over levels of |port - WRF oracle|")


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


@pytest.fixture(scope="module")
def stored():
    return json.loads(_GAPS.read_text())


def test_stored_gaps_are_for_this_fixture_and_these_widths(stored):
    p = stored["provenance"]
    assert p["fixture_sha256"] == hashlib.sha256(_FIX.read_bytes()).hexdigest()
    assert p["widths"] == _DEFAULT._asdict()
    assert p["x64"]
    assert set(stored["gaps"]) == set(_CASES)


@pytest.mark.parametrize("case", list(_CASES))
def test_default_gap_per_case_matches_record(fix, stored, case):
    """KNOWN DIFFERENCE, per fixture arm/column/output, at the default widths."""
    got = measure_case(fix, case)
    assert set(got) == set(stored["gaps"][case])
    for arm, per_out in stored["gaps"][case].items():
        for k, want in per_out.items():
            np.testing.assert_allclose(got[arm][k], want, rtol=1e-6, atol=1e-12,
                                       err_msg=f"{case} {arm} {k}")


# Widths whose switch no fixture column reaches (their gap equals the hard gap
# everywhere): the CONDLOAD root bridge needs 0 < G1 < 1e-4 m2/s2 and the floor
# ramp |WTW| near 1e-4, both only at an updraft's last level.  Covered instead by
# the synthetic unit tests below (both sides of each switch).  Likewise the Q0
# clip / floor and L5 widths leave every fixture gap at the hard gap (measured).
_UNEXERCISED = {"condload_sqrt_width_m2s2", "wtw_floor_width_m2s2", "q0_clip_width_kgkg",
                "q0_floor_width_kgkg", "l5_gather_width_pa"}


def _moved(stored, case):
    h = stored["gaps"]["hard"]
    return any(abs(a - b) > 1e-6 * abs(b) + 1e-14
               for arm, po in stored["gaps"][case].items() for k, v in po.items()
               for a, b in zip(v, h[arm][k]))


def test_fixture_coverage_of_each_width_is_as_recorded(stored):
    """A width counts as exercised by the fixture when its gap differs from the
    hard limit's in at least one arm/column/output."""
    assert {f for f in _FIELDS if not _moved(stored, f)} == _UNEXERCISED


# --- unit level: the smooth forms, their joins and gradients --------------------

def _tables(sm):
    return kfe._kf_tables(_lego_constants(), jnp.float64, sm)


_LUT_S, _LUT_H = _tables(_DEFAULT), _tables(kfe.KF_HARD)
_LUT_TP = _tables(kfe.KF_HARD._replace(tpmix_width_kgkg=_DEFAULT.tpmix_width_kgkg))


def _knot_a1(i, frac):
    return kfe._ALU_ASTRT + (i + frac) * kfe._ALU_AINC


def test_lut_smooth_is_exact_away_from_knots_and_c1_across_them():
    w = _DEFAULT.lut_knot_width
    for frac in (w + 0.05, 0.5, 1.0 - w - 0.05):
        a1 = _knot_a1(40, frac)
        np.testing.assert_allclose(kfe._tlog_lookup(_LUT_S, a1), kfe._tlog_lookup(_LUT_H, a1),
                                   rtol=1e-13)
    g = jax.grad(lambda a: kfe._tlog_lookup(_LUT_S, a))
    gh = jax.grad(lambda a: kfe._tlog_lookup(_LUT_H, a))
    k = _knot_a1(40, 0.0)
    eps = 1e-9 * kfe._ALU_AINC
    assert abs(float(g(k - eps) - g(k + eps))) < 1e-5 * abs(float(g(k)))
    assert abs(float(gh(k - eps) - gh(k + eps))) > 1e-3 * abs(float(gh(k)))
    np.testing.assert_allclose(g(_knot_a1(40, 0.5)), gh(_knot_a1(40, 0.5)), rtol=1e-12)


def test_lut_edges_extrapolate_then_hold_like_the_oracle():
    for i in (-0.5, -3.0, kfe._ALU_N + 5):
        a1 = _knot_a1(i, 0.0)
        np.testing.assert_allclose(kfe._tlog_lookup(_LUT_S, a1), kfe._tlog_lookup(_LUT_H, a1),
                                   rtol=1e-13)


def test_tab_interp_smooth_matches_bilinear_away_from_knots():
    p = float(_LUT_S.plutop) + 100.5 / float(_LUT_S.rdpr)
    bth = float(kfe._interp_c1(_LUT_S.the0k_pad, jnp.asarray(100.5), _DEFAULT.lut_knot_width))
    thes = bth + 30.5 / float(_LUT_S.rdthk)
    np.testing.assert_allclose(kfe._tpmix2dd(_LUT_S, p, thes), kfe._tpmix2dd(_LUT_H, p, thes),
                               rtol=1e-13)


def test_lut_check_grads_at_knots():
    check_grads(lambda a: kfe._tlog_lookup(_LUT_S, a), (_knot_a1(40, 0.1),), order=1,
                modes=("fwd", "rev"), eps=1e-7)
    p0 = float(_LUT_S.plutop) + 100.1 / float(_LUT_S.rdpr)
    for i in (0, 1):
        def f(p, t, i=i):
            return kfe._tpmix2dd(_LUT_S, p, t)[i]
        check_grads(f, (p0, 330.0), order=1, modes=("fwd", "rev"), eps=1e-3)
        assert float(jnp.abs(jax.grad(f, argnums=1)(p0, 330.0))) > 0.0


_P, _THES = 7.0e4, 335.0


def _tp2(lut, qu, qliq, qice):
    return kfe._tpmix2(lut, _P, _THES, qu, qliq, qice, constants.L_v, 0.0)


def _qs():
    return float(kfe._tpmix2dd(_LUT_H, _P, _THES)[1])


def test_tpmix2_smooth_conserves_water_and_is_exact_outside_band():
    qs, w = _qs(), _DEFAULT.tpmix_width_kgkg
    for dq, qtot in ((-5 * w, 1e-4), (5 * w, 1e-3), (1e-3, 5 * w), (0.3 * w, 0.1 * w)):
        qu, ql, qi = qs - dq, 0.7 * qtot, 0.3 * qtot
        _, qu2, ql2, qi2, qnl, _ = _tp2(_LUT_TP, qu, ql, qi)
        # closed budget: vapour + condensate + new condensate (qnewlq) in = out
        np.testing.assert_allclose(qu2 + ql2 + qi2 + qnl, qu + ql + qi, rtol=1e-14, atol=0)
        if abs(dq) > w and abs(dq - qtot) > w:
            # bound: everything the smooth form does differently sits at the
            # oracle's 1e-10 kg/kg guard (KNOWN DIFFERENCE in _tpmix2)
            np.testing.assert_allclose(_tp2(_LUT_TP, qu, ql, qi)[:5],
                                       _tp2(_LUT_H, qu, ql, qi)[:5], rtol=1e-9,
                                       atol=kfe._TPMIX_QEPS)


@pytest.mark.parametrize("dq,qtot,out", [(-0.1, 100.0, 4), (100.0, 99.9, 0)])
def test_tpmix2_check_grads_inside_bands(dq, qtot, out):
    """In the condensation band (output qnewlq) and the deficit band (output tu)."""
    w = _DEFAULT.tpmix_width_kgkg
    dq, qtot = dq * w, qtot * w

    def f(qu):
        return _tp2(_LUT_TP, qu, 0.7 * qtot, 0.3 * qtot)[out]

    check_grads(f, (_qs() - dq,), order=1, modes=("fwd", "rev"), eps=1e-9)
    assert float(jax.grad(f)(_qs() - dq)) != 0.0


def _cl(sm, wtw, g1):
    """CONDLOAD with no condensate: g1 = wtw + boterm; returns the 5 outputs."""
    return kfe._condload(0.0, 0.0, wtw, 100.0, g1 - wtw, 0.0, 0.0, 0.0, constants.g, sm)


def test_condload_smooth_is_exact_outside_bands():
    ws = _DEFAULT.condload_sqrt_width_m2s2
    for wtw, g1 in ((1.0, 2.0), (4.0, 3 * ws)):
        np.testing.assert_allclose(_cl(_DEFAULT, wtw, g1), _cl(kfe.KF_HARD, wtw, g1),
                                   rtol=1e-13)


def test_condload_check_grads_in_sqrt_bridge_and_floor_ramp():
    ws, wf = _DEFAULT.condload_sqrt_width_m2s2, _DEFAULT.wtw_floor_width_m2s2

    load = 2.0 * constants.g * 100.0 * 0.5e-3 / kfe._KE_DIV  # g1's loading term

    def fall(g1):  # the root bridge of g1, seen through the fallout qlqout
        return kfe._condload(1e-3, 0.0, 1.0, 100.0, g1 - 1.0 + load, 0.0, 0.0, 0.0,
                             constants.g, _DEFAULT)[3]

    check_grads(fall, (0.3 * ws,), order=1, modes=("fwd", "rev"), eps=1e-8)
    assert float(jax.grad(fall)(0.3 * ws)) != 0.0
    # the WTW floor: the jump at -1e-4 is a ramp of width wf below it
    def wtw_out(x):  # boterm sets the new WTW to x
        return kfe._condload(0.0, 0.0, 1.0, 100.0, x - 1.0, 0.0, 0.0, 0.0, constants.g,
                           _DEFAULT)[2]
    x = -kfe._WTW_FLOOR - 0.5 * wf
    check_grads(wtw_out, (x,), order=1, modes=("fwd", "rev"), eps=1e-7)
    assert float(jax.grad(wtw_out)(x)) != 0.0


def test_prof5_smooth_blends_branches():
    w = _DEFAULT.prof5_width
    eq0 = kfe._P5_Y_OFF / kfe._P5_Y_SCALE  # y = 0
    for eq in (eq0 + 2 * w / kfe._P5_Y_SCALE, eq0 - 2 * w / kfe._P5_Y_SCALE, 0.9, 0.1):
        np.testing.assert_allclose(kfe.prof5(jnp.asarray(eq), w), kfe.prof5(jnp.asarray(eq)),
                                   rtol=1e-12)
    for i in (0, 1):
        def f(e, i=i):
            return kfe.prof5(e, w)[i]
        check_grads(f, (jnp.asarray(eq0 + 1e-3),), order=1, modes=("fwd", "rev"), eps=1e-6)
        g = jax.grad(f)
        assert abs(float(g(eq0 - 1e-9)) - float(g(eq0 + 1e-9))) < 1e-6 * abs(float(g(eq0)))


def test_prof5_fractions_stay_nonnegative_up_to_the_width_bound():
    """Inside the blend band the fractions stay >= 0 (outside it they are the
    oracle's, whose polynomial is -2e-11 at eq = 0)."""
    eq = jnp.linspace(0.0, 1.0, 20001)
    y = kfe._P5_Y_SCALE * eq - kfe._P5_Y_OFF
    for w in (_DEFAULT.prof5_width, kfe._PROF5_WIDTH_MAX):
        ee, ud = kfe.prof5(eq, w)
        band = jnp.abs(y) < w
        assert int(band.sum()) > 10
        assert float(jnp.min(ee[band])) >= 0.0 and float(jnp.min(ud[band])) >= 0.0, w


def test_tpmix2_smooth_keeps_vapour_and_condensate_nonnegative():
    """Cold saturated air with no condensate: nothing condenses at qu = qs."""
    qs, w = _qs(), _DEFAULT.tpmix_width_kgkg
    for qu, qtot in ((qs, 0.0), (qs + 0.3 * w, 0.0), (qs - 0.3 * w, 0.2 * w),
                     (qs - 0.7 * w, 0.1 * w), (qs - 0.5 * w, 0.4 * w),
                     (qs - 0.02 * w, 0.75 * w)):
        tu, qu2, ql2, qi2, qnl, _ = _tp2(_LUT_TP, qu, 0.7 * qtot, 0.3 * qtot)
        assert min(float(qu2), float(ql2), float(qi2), float(qnl)) >= 0.0
        assert float(qnl) <= max(qu - qs, 0.0) + 1e-18
        assert float(ql2) <= 0.7 * qtot + 1e-21  # never more condensate than given
    # supersaturated with a little condensate: nothing evaporates
    _, _, ql2, qi2, _, _ = _tp2(_LUT_TP, qs + 5 * w, 0.35 * w, 0.15 * w)
    assert float(ql2) == 0.35 * w and float(qi2) == 0.15 * w


# --- source geometry (PR 3) ------------------------------------------------------

def _col(n=12):
    """1-based padded column: heights rising, pressures falling, one smooth field."""
    rng = np.random.default_rng(3)
    dz = np.concatenate([[200.0], 300.0 + 1200.0 * rng.random(n - 1)])
    z = np.concatenate([[0.0], np.cumsum(dz) - 0.5 * dz, [0.0]])
    z[-1] = z[-2]
    p = 1.0e5 * np.exp(-z / 8000.0)
    kk = jnp.arange(n + 2)
    valid = (kk >= 1) & (kk <= n)
    f = 290.0 - 6.5e-3 * z + 2.0 * np.sin(z / 700.0)
    return jnp.asarray(z), jnp.asarray(p), kk, valid, jnp.asarray(f)


def _hard_gather(f, z0, z, lc):
    """The oracle's KLCL search and interpolation (concrete z)."""
    hit = (z <= np.asarray(z0[1:-1])) & (np.arange(1, len(z0) - 1) >= lc)
    k = int(np.argmax(hit)) + 1 if hit.any() else len(z0) - 2
    return f[k - 1] + (f[k] - f[k - 1]) * (z - z0[k - 1]) / (z0[k] - z0[k - 1])


def test_cap_c1_is_one_sided_exact_outside_band_and_differentiable():
    w = 1.0e-6
    x = jnp.linspace(-5 * w, 5 * w, 1001)
    c = kfe._cap_c1(x, 0.0, w)
    assert float(jnp.max(c)) <= 0.0
    np.testing.assert_array_equal(c[x <= -w], x[x <= -w])
    np.testing.assert_array_equal(kfe._cap_c1(x, 0.0, 0.0), jnp.minimum(x, 0.0))
    check_grads(lambda v: kfe._cap_c1(v, 0.0, w), (-0.3 * w,), order=1, modes=("fwd", "rev"),
                eps=1e-9)
    g = jax.grad(lambda v: kfe._cap_c1(v, 0.0, w))
    assert 0.0 < float(g(-0.3 * w)) < 1.0 and float(g(0.5 * w)) == 0.0


def test_gather_weights_sum_to_one_reproduce_lines_and_match_oracle_off_levels():
    z0, _, kk, valid, f = _col()
    w, lc = _DEFAULT.lcl_gather_width_m, 3
    g = lambda fld, z: kfe._gather_at_height(fld, z0, z, kk, valid, lc, w)  # noqa: E731
    for z in np.linspace(-500.0, float(z0[-2]) + 500.0, 157):
        np.testing.assert_allclose(g(jnp.ones_like(f), z), 1.0, rtol=1e-14)
        np.testing.assert_allclose(g(3.0 + 0.01 * z0, z), 3.0 + 0.01 * z, rtol=1e-13)
        if np.min(np.abs(z - np.asarray(z0[lc:-1]))) > w / 2:
            np.testing.assert_allclose(g(f, z), _hard_gather(f, z0, z, lc), rtol=1e-13)


def test_gather_from_the_first_level_extrapolates_like_the_oracle():
    z0, _, kk, valid, f = _col()
    w = _DEFAULT.lcl_gather_width_m
    for z in (0.0, 0.5 * float(z0[1]), float(z0[1]) + 0.6 * w, float(z0[3]) - 0.6 * w):
        np.testing.assert_allclose(kfe._gather_at_height(f, z0, z, kk, valid, 1, w),
                                   _hard_gather(f, z0, z, 1), rtol=1e-13)


def test_gather_is_c1_across_a_level_where_the_oracle_kinks():
    z0, _, kk, valid, f = _col()
    w, lc, zk = _DEFAULT.lcl_gather_width_m, 3, float(z0[6])
    g = jax.grad(lambda z: kfe._gather_at_height(f, z0, z, kk, valid, lc, w))
    gh = jax.grad(lambda z: jnp.interp(z, z0[lc - 1:-1], f[lc - 1:-1]))  # oracle, inside
    eps = 1e-6
    assert abs(float(g(zk - eps) - g(zk + eps))) < 1e-6 * abs(float(g(zk)))
    assert abs(float(gh(zk - eps) - gh(zk + eps))) > 1e-3 * abs(float(gh(zk)))
    check_grads(lambda z, fld: kfe._gather_at_height(fld, z0, z, kk, valid, lc, w),
                (zk + 0.2 * w, f), order=1, modes=("fwd", "rev"), eps=1e-4)


def test_l5_weights_one_hot_off_band_sum_to_one_and_differentiable():
    _, p, kk, valid, _ = _col()
    w = _DEFAULT.l5_gather_width_pa
    wt = kfe._l5_weights(p, kk, valid, w)
    pc = kfe._L5_FRAC * float(p[1])
    l5 = int(np.max(np.where(np.asarray(valid) & (np.asarray(p) >= pc), np.asarray(kk), 0)))
    np.testing.assert_allclose(float(wt.sum()), 1.0, rtol=1e-14)
    if np.min(np.abs(np.asarray(p[1:-1]) - pc)) > w / 2:
        np.testing.assert_array_equal(wt, (np.asarray(kk) == l5).astype(float))
    # move the level just above L5 into the band: two levels share the weight
    p2 = p.at[l5 + 1].set(pc - 0.1 * w)
    wt2 = kfe._l5_weights(p2, kk, valid, w)
    assert 0.0 < float(wt2[l5 + 1]) < 1.0
    np.testing.assert_allclose(float(wt2.sum()), 1.0, rtol=1e-14)
    check_grads(lambda q: kfe._l5_weights(q, kk, valid, w)[l5 + 1], (p2,), order=1,
                modes=("fwd", "rev"), eps=1e-2)


def test_widths_are_validated():
    with pytest.raises(ValueError, match="lut_knot_width"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(lut_knot_width=1.5))
    with pytest.raises(ValueError, match="prof5_width"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(prof5_width=3.0))
    with pytest.raises(ValueError, match="tpmix_width_kgkg"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(tpmix_width_kgkg=-1.0))
    with pytest.raises(ValueError, match="prof5_width"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(prof5_width=True))
    kfe.validate_kf_smoothing(kfe.KF_HARD)


def test_surrogate_refuses_wrf_widths():
    from legoesm.atmosphere.physics.convection.kain_fritsch import kain_fritsch_convection
    t = jnp.full((1, 4), 280.0)
    with pytest.raises(ValueError, match="wrf_prof5_width"):
        kain_fritsch_convection(t, t * 0, t * 0 + 5e4, jnp.full((1, 5), 5e4), t * 0, t * 0,
                                300.0, KainFritschConfig(wrf_prof5_width=0.1))


if __name__ == "__main__":
    # one case per process: all cases in one process exhaust LLVM code memory
    if len(sys.argv) != 3 or sys.argv[1] != "--write" or sys.argv[2] not in _CASES:
        sys.exit(f"usage: JAX_ENABLE_X64=1 python {Path(__file__).name} --write CASE; "
                 f"CASE in {list(_CASES)}")
    data = json.loads(_GAPS.read_text()) if _GAPS.exists() else {"gaps": {}}
    data["provenance"] = _provenance()
    data["gaps"][sys.argv[2]] = measure_case(dict(np.load(_FIX)), sys.argv[2])
    _GAPS.write_text(json.dumps(data, indent=0) + "\n")
    print("wrote", sys.argv[2], "to", _GAPS)
