"""RULE 5 tests of the KF-Eta port's local switches (#2280, PR 2/7).

Smoothed here, each by one ``KFSmoothing`` width (0 = the oracle):

* ``lut_knot_width``: F1 (ALU log table) and F2 (TTAB/QSTAB and THE0K):
  knots and edge clamps of the piecewise-linear lookups;
* ``tpmix_width_kgkg``: D7a (TPMIX2 condense / evaporate / deficit split);
* ``condload_sqrt_width_m2s2``: D7b and the CONDLOAD square roots (E11 there);
* ``wtw_floor_width_m2s2``: D7d (the |WTW| < 1e-4 floor);
* ``prof5_width``: D7e (PROF5's two polynomial branches).

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
# the synthetic unit tests below (both sides of each switch).
_UNEXERCISED = {"condload_sqrt_width_m2s2", "wtw_floor_width_m2s2"}


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
