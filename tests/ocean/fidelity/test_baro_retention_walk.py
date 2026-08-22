"""Direct test for #1455's barotropic retention walk.

The module's job is to keep two things apart that look identical in a single
arm: a LINEAR RETENTION of an injected barotropic increment, and CHAOTIC
DIVERGENCE of two trajectories.  Only the first may be multiplied by a
per-step injection rate to predict an accumulation.  Every assertion below is
written so it FAILS if that discrimination is removed or inverted.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "scripts", "validate", "ocean_fidelity", "dino_1226",
    "baro_retention_walk.py")


def _load():
    spec = importlib.util.spec_from_file_location("_baro_ret", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_baro_ret"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _arm(acc, scale, key="dU_avg"):
    return {"path": "<synthetic>", "acc": np.asarray(acc, dtype=np.float64),
            "gate": np.asarray(acc, dtype=np.float64),
            "scale": float(scale), "key": key}


# --------------------------------------------------------------- retention ---
def test_retention_recovers_a_planted_linear_response():
    """A response that is exactly f(t) * injection must give R(t) = f(t)."""
    inj = 5.0e-3
    f = np.array([0.5, 0.25, 0.1, 0.0])
    ctrl = _arm(np.full(4, 60.0), 0.0)
    pert = _arm(60.0 + f * inj, 1.0)
    r = M.retention(pert, ctrl, inj)
    assert r["R"] == pytest.approx(f, rel=1e-12)
    # and it must scale out: the SAME f at 100x injection gives the SAME R
    pert100 = _arm(60.0 + f * inj * 100.0, 100.0)
    r100 = M.retention(pert100, ctrl, inj)
    assert r100["R"] == pytest.approx(f, rel=1e-12)


def test_retention_refuses_a_zero_injection_arm():
    ctrl = _arm(np.full(4, 60.0), 0.0)
    with pytest.raises(SystemExit):
        M.retention(_arm(np.full(4, 60.0), 0.0), ctrl, 5.0e-3)


# --------------------------------------------------------------- linearity ---
def test_linearity_passes_a_planted_linear_pair_and_fails_a_chaotic_one():
    """The load-bearing discrimination, both arms of it.

    LINEAR: response proportional to injection => R identical => ratio 1.
    CHAOTIC: the large-amplitude arm saturates instead of scaling, so its R
    is far smaller and the ratio leaves the tolerance band.
    """
    inj = 5.0e-3
    big = 40.0 * M.FLOOR_SV        # comfortably readable
    f = np.array([1.0, 0.8, 0.6, 0.5]) * big / inj
    ctrl = _arm(np.full(4, 60.0), 0.0)

    a1 = M.retention(_arm(60.0 + f * inj, 1.0), ctrl, inj)
    a100_lin = M.retention(_arm(60.0 + f * inj * 100.0, 100.0), ctrl, inj)
    L = M.linearity(a100_lin, a1)
    assert L["n"] == 4, "all four days must be readable in this fixture"
    assert L["median_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert L["pass"] is True

    # chaotic: the 100x arm saturates at 3x the 1x response, not 100x.
    a100_sat = M.retention(_arm(60.0 + f * inj * 3.0, 100.0), ctrl, inj)
    Lc = M.linearity(a100_sat, a1)
    assert Lc["median_ratio"] == pytest.approx(0.03, rel=1e-9)
    assert Lc["pass"] is False, (
        "a saturating (chaotic) response must NOT be certified linear -- "
        "multiplying it by a per-step rate reports a Lyapunov exponent as a "
        "retention factor")


def test_linearity_rejects_a_decorrelated_pair_the_median_ratio_would_pass():
    """THE defect this gate was rebuilt for, planted and asserted.

    A sign-oscillating divergence has a heavy-tailed pointwise ratio
    distribution whose MEDIAN sits near 1 by accident.  The first version of
    this gate passed exactly such a pair -- real arms whose response series
    correlate at 0.06 -- and printed "LINEAR".  The correlation bar is what
    catches it, so it gets a planted case that the ratio bar alone cannot.
    """
    inj = 5.0e-3
    big = 40.0 * M.FLOOR_SV
    rng = np.random.default_rng(0)
    n = 60
    ctrl = _arm(np.full(n, 60.0), 0.0)
    # two INDEPENDENT sign-oscillating series, scaled so the median pointwise
    # ratio is ~1 while the curves are unrelated.
    ra = rng.standard_normal(n) * big
    rb = rng.standard_normal(n) * big
    a = M.retention(_arm(60.0 + ra, 1.0), ctrl, inj)
    b = M.retention(_arm(60.0 + rb, 1.0), ctrl, inj)
    L = M.linearity(a, b)
    assert abs(L["corr"]) < M.LINEAR_CORR_MIN, "fixture is not decorrelated"
    assert L["corr_ok"] is False
    assert L["pass"] is False, (
        "two unrelated curves must never be certified as a linear response, "
        "however close their median pointwise ratio lands to 1")


# ---------------------------------------------------------- linear window ---
def test_linear_window_is_contiguous_and_stops_at_the_first_disagreement():
    """It must stop at a real disagreement, and NOT stop at a quiet day.

    Both halves are load-bearing and both were wrong in a first version:
    scoring every day on which the arms happened to agree inflated the sum
    sevenfold, and breaking at the first day the reference arm dipped below
    the floor threw away four days of genuine agreement.
    """
    inj = 5.0e-3
    big = 40.0 * M.FLOOR_SV
    ctrl = _arm(np.full(6, 60.0), 0.0)
    # days 1-3 agree; day 4 is BELOW the floor for the 1x arm only (quiet, no
    # information); day 5 agrees again; day 6 DISAGREES by 2x.
    f1 = np.array([1.0, 0.8, 0.6, 0.001, 0.5, 0.4])
    f2 = np.array([1.0, 0.8, 0.6, 0.001, 0.5, 0.8])
    a1 = M.retention(_arm(60.0 + f1 * big, 1.0), ctrl, inj)
    a2 = M.retention(_arm(60.0 + f2 * big * 100.0, 100.0), ctrl, inj)
    w = M.linear_window({"a1": a1, "a2": a2}, ["a1", "a2"])
    assert w == [0, 1, 2, 3, 4], (
        f"expected the window to span days 1-5 and stop at the day-6 "
        f"disagreement, got days {[i+1 for i in w]}")
    # and it must NOT extend past a real disagreement
    assert 5 not in w


def test_linearity_accepts_a_perfect_negative_scale_arm():
    """A -100x arm's PERFECT linear response must read LINEAR, not anti-linear.

    R already divides by the SIGNED injection, so a perfect response gives the
    SAME R at every scale.  Correlating the raw responses instead would give
    -1 here and print it beside a LINEAR verdict, reading as its opposite.
    """
    inj = 5.0e-3
    big = 40.0 * M.FLOOR_SV
    f = np.array([1.0, 0.8, 0.6, 0.5]) * big / inj
    ctrl = _arm(np.full(4, 60.0), 0.0)
    a1 = M.retention(_arm(60.0 + f * inj, 1.0), ctrl, inj)
    am100 = M.retention(_arm(60.0 - f * inj * 100.0, -100.0), ctrl, inj)
    L = M.linearity(am100, a1)
    assert L["corr"] == pytest.approx(1.0, rel=1e-9)
    assert L["pass"] is True


def test_linear_window_needs_two_distinct_amplitudes():
    """A +100x/-100x pair alone must NOT certify a day.

    They agree on R whenever the response is merely sign-antisymmetric, which
    says nothing about the amplitude linearity the window exists to certify.
    """
    inj = 5.0e-3
    big = 40.0 * M.FLOOR_SV
    ctrl = _arm(np.full(3, 60.0), 0.0)
    f = np.array([1.0, 0.9, 0.8])
    ap = M.retention(_arm(60.0 + f * big * 100.0, 100.0), ctrl, inj)
    am = M.retention(_arm(60.0 - f * big * 100.0, -100.0), ctrl, inj)
    assert M.linear_window({"ap": ap, "am": am}, ["ap", "am"]) == [], (
        "two arms of the SAME magnitude must not certify amplitude linearity")


def test_linear_window_stops_immediately_on_a_day_one_disagreement():
    inj = 5.0e-3
    big = 40.0 * M.FLOOR_SV
    ctrl = _arm(np.full(3, 60.0), 0.0)
    a1 = M.retention(_arm(60.0 + np.full(3, big), 1.0), ctrl, inj)
    a2 = M.retention(_arm(60.0 + np.full(3, big * 100.0 * 3.0), 100.0), ctrl, inj)
    assert M.linear_window({"a1": a1, "a2": a2}, ["a1", "a2"]) == []


def test_linearity_reports_no_readable_day_rather_than_inventing_one():
    """Below the floor there is no information, and the code must say so."""
    inj = 5.0e-3
    tiny = 0.1 * M.FLOOR_SV
    ctrl = _arm(np.full(3, 60.0), 0.0)
    a = M.retention(_arm(60.0 + np.full(3, tiny), 1.0), ctrl, inj)
    b = M.retention(_arm(60.0 + np.full(3, tiny * 100), 100.0), ctrl, inj)
    L = M.linearity(a, b)
    assert L["n"] == 0 and L["pass"] is False
    assert "readable" in L["reason"]


# -------------------------------------------------------------- accounting ---
def test_accumulate_includes_the_age_zero_bin():
    """R(0)=1 must be prepended, and the trapezoid must match by hand.

    Both reviews caught a plain sum over R(1..90) dropping the steps younger
    than one day -- up to 37% of the target gap, always in the direction that
    makes a suspect look too small.  This pins the fix.
    """
    R = np.array([0.5, 0.25, 0.0])
    d = 4.55e-3
    # trapezoid over ages [0,1,2,3] with R(0)=1: 0.75 + 0.375 + 0.125 = 1.25
    assert M.accumulate(R, d) == pytest.approx(d * M.STEPS_PER_DAY * 1.25)
    # a plain sum of the daily samples would give 0.75 -- 40% smaller.
    assert M.accumulate(R, d) > d * M.STEPS_PER_DAY * float(R.sum())


def test_accumulate_is_fatal_on_a_non_finite_retention():
    with pytest.raises(SystemExit):
        M.accumulate(np.array([0.1, np.nan, 0.0]), 4.55e-3)


def test_accumulate_decayed_retention_cannot_pay_the_gap():
    """The physical statement the campaign needs, asserted on the real code."""
    R = np.concatenate([[0.108, -0.004], np.zeros(88)])
    assert abs(M.accumulate(R, 4.55e-3)) < 0.5 * abs(M.GAP_FULL_SECTION)


# ----------------------------------------------------------------- verdict ---
@pytest.mark.parametrize("s_coh,s_all,need,expect", [
    # positive S against a NEGATIVE gap: sign refutes regardless of size
    (0.02, 0.16, -0.401, "EXONERATED (wrong sign AND too small)"),
    (0.50, 0.60, -0.401, "REFUTED BY SIGN (right size, wrong sign)"),
    # right sign, too small
    (-0.01, -0.02, -0.401, "EXONERATED (right sign, too small)"),
    # right sign and size
    (-0.30, -0.40, -0.401, "CANDIDATE (right sign and size)"),
    # exactly at the half-gap bar counts as big enough
    (-0.2005, -0.2005, -0.401, "CANDIDATE (right sign and size)"),
])
def test_verdict_table(s_coh, s_all, need, expect):
    """Every branch, including the two the sign discriminator decides.

    An inline version of this had its sign test inverted by mutation and the
    whole suite stayed green.
    """
    assert M.verdict(s_coh, s_all, need) == expect


def test_verdict_sign_discriminator_is_not_vacuous():
    """Inverting the sign changes the answer -- the property mutation checks."""
    assert M.verdict(0.5, 0.6, -0.401) != M.verdict(-0.5, -0.6, -0.401)


# --------------------------------------------------------------- lyapunov ---
def test_lyapunov_growth_separates_growth_from_decay():
    """The discriminator the amplitude ladder cannot provide.

    Tangent-linear chaotic growth is exactly proportional to the injection
    amplitude, so an amplitude ladder agrees under BOTH hypotheses.  The shape
    in time is what separates them.
    """
    t = np.arange(1, 91, dtype=float)
    grow = 100 * M.FLOOR_SV * np.exp(t / 20.0)
    decay = 100 * M.FLOOR_SV * np.exp(-t / 20.0)
    g = M.lyapunov_growth(grow, M.FLOOR_SV)
    assert g["growing"] is True
    assert g["e_folding_days"] == pytest.approx(20.0, rel=1e-6)
    d = M.lyapunov_growth(decay, M.FLOOR_SV)
    assert d["growing"] is False
    # a series entirely below the floor carries no information and must say so
    n = M.lyapunov_growth(np.full(90, 0.01 * M.FLOOR_SV), M.FLOOR_SV)
    assert n["n"] == 0 and n["growing"] is False


# ------------------------------------------------------------------- loader ---
def test_load_arm_rejects_an_unstable_arm_and_a_nan_arm(tmp_path):
    good = dict(acc_dep_daily=np.zeros(3), acc_gate_daily=np.zeros(3),
                perturb_baro_scale=np.float64(1.0),
                perturb_baro_key=np.str_("dU_avg"),
                stable=np.array(True), blew_up_at_step=-1)
    p = tmp_path / "ok.npz"
    np.savez(p, **good)
    assert M.load_arm(str(p))["scale"] == 1.0

    bad = dict(good); bad["stable"] = np.array(False)
    p2 = tmp_path / "unstable.npz"; np.savez(p2, **bad)
    with pytest.raises(SystemExit):
        M.load_arm(str(p2))

    nan = dict(good); nan["acc_dep_daily"] = np.array([0.0, np.nan, 0.0])
    p3 = tmp_path / "nan.npz"; np.savez(p3, **nan)
    with pytest.raises(SystemExit):
        M.load_arm(str(p3))

    missing = {k: v for k, v in good.items() if k != "acc_dep_daily"}
    p4 = tmp_path / "missing.npz"; np.savez(p4, **missing)
    with pytest.raises(SystemExit):
        M.load_arm(str(p4))


def test_module_actually_uses_its_own_linearity_function():
    """A source assertion: main() must call the tested function.

    Without this the tests above could pass while main() re-implemented the
    arithmetic inline -- the exact defect this branch blocked on twice.
    """
    import inspect
    src = inspect.getsource(M.main)
    assert "linearity(" in src, "main() does not call linearity()"
    assert "linear_window(" in src, "main() does not call linear_window()"
    assert "verdict(" in src, "main() does not call verdict()"
    assert "lyapunov_growth(" in src, "main() does not call lyapunov_growth()"
    assert "accumulate(" in src, "main() does not call accumulate()"
    assert "retention(" in src, "main() does not call retention()"
