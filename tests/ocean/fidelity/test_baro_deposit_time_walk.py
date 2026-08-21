"""Direct test for #1455's barotropic deposit time-walk driver.

Covers the two pieces that decide what the walk concludes: the PRE-REGISTERED
scoring arithmetic and the parsers that lift the three deposit numbers out of
the committed instrument's stdout.  Each assertion is written so it FAILS if
the logic is removed or inverted (no vacuous checks).
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
    "baro_deposit_time_walk.py")


def _load():
    spec = importlib.util.spec_from_file_location("_baro_walk", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_baro_walk"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


def test_states_cover_the_ninety_day_window():
    days = [s[0] for s in M.STATES]
    kts = [s[1] for s in M.STATES]
    assert days == list(range(180, 271, 10))
    # every kt must be its day: kt * 2700 s == day * 86400 s
    assert all(kt * M.DT == day * 86400.0 for day, kt in zip(days, kts))
    # 320 steps per 10-day window, nine windows over the 2880-step campaign
    assert kts[-1] - kts[0] == 9 * M.STEPS_PER_WINDOW == M.N_STEPS_90D


def test_score_recovers_a_planted_constant_accumulator():
    """A constant deposit equal to the target rate must integrate to the row."""
    b = np.full(9, M.TARGET_PER_STEP)
    d = np.full(10, M.TARGET_PER_STEP)
    sc = M.score(d, b)
    assert sc["integral_90d"] == pytest.approx(M.BARO_ROW_90D, rel=1e-12)
    assert sc["sign_agree"] == 9
    assert sc["coherence"] == pytest.approx(1.0)
    assert sc["A1_in_band"] and sc["A2_sign_7of9"]
    assert not sc["E1_incoherent"] and not sc["E2_small_integral"]


def test_score_exonerates_a_pure_oscillation():
    """Alternating sign, large amplitude: big instantaneously, zero integrated."""
    b = np.full(9, M.TARGET_PER_STEP)
    d = np.array([+5e-3, -5e-3] * 5)
    sc = M.score(d, b)
    assert abs(sc["integral_90d"]) < 1e-9        # trapezoid of +/- cancels
    assert sc["coherence"] < 0.5                 # E1
    assert sc["E1_incoherent"] and sc["E2_small_integral"]
    assert not sc["A1_in_band"]


def test_score_band_is_a_factor_two_either_side_and_signed():
    b = np.full(9, M.TARGET_PER_STEP)
    # exactly 2x the row -> at the far edge, inside; 2.5x -> outside
    assert M.score(np.full(10, 2.0 * M.TARGET_PER_STEP), b)["A1_in_band"]
    assert not M.score(np.full(10, 2.5 * M.TARGET_PER_STEP), b)["A1_in_band"]
    assert M.score(np.full(10, 0.5 * M.TARGET_PER_STEP), b)["A1_in_band"]
    assert not M.score(np.full(10, 0.4 * M.TARGET_PER_STEP), b)["A1_in_band"]
    # a POSITIVE integral of the right magnitude must NOT pass: the row is
    # negative, and a sign-blind band would crown the wrong survivor.
    assert not M.score(np.full(10, -M.TARGET_PER_STEP), b)["A1_in_band"]


def test_score_sign_agreement_follows_the_window_curve():
    b = np.array([+1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]) * 1e-4
    d = np.full(10, -1e-4)          # always negative -> agrees on 8 of 9
    assert M.score(d, b)["sign_agree"] == 8
    assert M.score(d, b)["A2_sign_7of9"]
    assert M.score(np.full(10, +1e-4), b)["sign_agree"] == 1


def test_score_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        M.score(np.zeros(9), np.zeros(9))


def test_committed_budget_curve_matches_its_own_ninety_day_total():
    """The per-window table must add up to the row the walk has to pay for."""
    cum = 2.0 * M._BARO_CUM_HALVED
    assert cum[-1] == pytest.approx(M.BARO_ROW_90D, abs=5e-3)
    assert M.TARGET_PER_STEP == pytest.approx(M.BARO_ROW_90D / 2880.0)


def test_output_parsers_lift_the_three_numbers():
    out = (
        "  seasonal clock: t_seconds=15557400.0 s (day 180.06); ...\n"
        "  FORCING: wind_through_step=True tau_x[Pa] range=[-0.1999,0.1000]\n"
        "    velocity-average deposit diff (lego - NEMO)  = +5.5555e-03 "
        "Sv/step   =  -1984.1% of the -2.80e-4 Sv/step budget row\n"
        "    deposit with NEMO's frozen forcing = +2.0198e-03 Sv/step  "
        "(36.4% of it survives)\n"
        "  PLANT (NEMO shifted +1 substep): wall max = 1.0e-01 "
        "(shift-sensitivity ratio 4.40)\n")
    assert M._RE_TOTAL.findall(out) == ["+5.5555e-03"]
    assert M._RE_INLOOP.findall(out) == ["+2.0198e-03"]
    assert M._RE_CLOCK.findall(out) == ["15557400.0"]
    assert M._RE_TAU.findall(out) == [("-0.1999", "0.1000")]
    assert M._RE_PLANT.findall(out) == ["4.40"]
    # the transport-route line must NOT be mistaken for the velocity route
    other = ("    transport-average deposit diff (un_adv route) = "
             "+2.9505e-03 Sv/step\n")
    assert M._RE_TOTAL.findall(out + other) == ["+5.5555e-03"]
