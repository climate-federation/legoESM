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
    # and the OWN-forcing line must NOT be mistaken for the frozen-forcing one
    # (this is the decoy that actually sits next to it in the real output)
    decoy = ("    deposit with legoESM's own forcing = +5.5555e-03 Sv/step\n"
             "    -> IN-LOOP share (assumption-free) = +2.0198e-03 Sv/step\n")
    assert M._RE_INLOOP.findall(out + decoy) == ["+2.0198e-03"]
    assert M._RE_TOTAL.findall(out + decoy) == ["+5.5555e-03"]


def test_consecutive_catalog_is_five_steps_in_a_row():
    kts = [s[1] for s in M.CONSEC_STATES]
    assert kts == [5760, 5761, 5762, 5763, 5764]
    # the 10-day grid samples every 320th step -- an EVEN offset, which is
    # exactly why a period-2 mode is invisible on it.
    grid = [s[1] for s in M.STATES]
    assert all((b - a) == 320 and (b - a) % 2 == 0
               for a, b in zip(grid, grid[1:]))


def test_power_check_fires_on_the_real_numbers_and_can_also_not_fire():
    """The pre-registered POWER branch fired in the committed run.  This tests
    the FUNCTION main() calls, both arms, so deleting or inverting the check
    fails here."""
    d = np.array([5.5555e-3, 3.9401e-3, 2.9627e-3, 4.2694e-3, 4.1532e-3,
                  4.4172e-3, 4.8150e-3, 5.1232e-3, 5.6129e-3, 5.9099e-3])
    got = M.power_check(d)
    assert got["underpowered"] is True
    assert got["sem"] == pytest.approx(2.86e-4, rel=2e-2)
    assert got["target"] == pytest.approx(abs(M.TARGET_PER_STEP))
    # and a tight series must NOT trip it, or the branch is a constant
    tight = np.full(10, M.TARGET_PER_STEP) + 1e-9 * np.arange(10)
    assert M.power_check(tight)["underpowered"] is False


def test_power_check_is_the_one_main_uses():
    import inspect
    src = inspect.getsource(M.main)
    assert "power_check(series[\"total\"])" in src, (
        "main() must call power_check, or the test above pins nothing")


def test_entry_state_control_reports_the_inode_identity(tmp_path, monkeypatch):
    """The control must SAY when both sides resolve to one file, because there
    it is an identity and proves nothing.  Regression for a false claim that
    shipped in a commit message."""
    import netCDF4 as nc

    lane = tmp_path / "LANE"
    lane.mkdir()
    tiles = tmp_path / "TILES"
    tiles.mkdir()
    real = tmp_path / "real.nc"
    with nc.Dataset(real, "w") as ds:
        ds.createDimension("x", 3)
        ds.createDimension("y", 2)
        ds.createDimension("t", 1)
        v = ds.createVariable("sshn", "f8", ("t", "y", "x"))
        v[:] = np.arange(6.0).reshape(1, 2, 3)
        # the in-memory stitcher reads the DOMAIN_* decomposition attributes
        ds.DOMAIN_size_global = np.array([3, 2], dtype="i4")
        ds.DOMAIN_position_first = np.array([1, 1], dtype="i4")
        ds.DOMAIN_position_last = np.array([3, 2], dtype="i4")
        ds.DOMAIN_halo_size_start = np.array([0, 0], dtype="i4")
        ds.DOMAIN_halo_size_end = np.array([0, 0], dtype="i4")
    base = "DINO_00000001_restart"
    os.symlink(real, lane / f"{base}.nc")
    os.symlink(real, tiles / f"{base}_0000.nc")
    (lane / "namelist_cfg").write_text(f'  cn_ocerst_in = "{base}"\n')
    monkeypatch.setattr(M, "_DINO", str(tmp_path))
    got = M._assert_same_entry_state(1, 1, "LANE", "TILES")
    assert got["entry_state_maxdiff"] == 0.0
    assert got["entry_same_inode"] is True

    # and the FALSE arm: a byte-identical COPY at a different inode must still
    # pass the bit-comparison but must NOT be reported as an identity, or a
    # hardcoded True would satisfy the test above.
    import shutil
    lane2 = tmp_path / "LANE2"
    lane2.mkdir()
    shutil.copy(real, lane2 / f"{base}.nc")
    (lane2 / "namelist_cfg").write_text(f'  cn_ocerst_in = "{base}"\n')
    got2 = M._assert_same_entry_state(1, 1, "LANE2", "TILES")
    assert got2["entry_state_maxdiff"] == 0.0
    assert got2["entry_same_inode"] is False


def test_consecutive_stats_separates_a_planted_period2_from_a_planted_null():
    """The criterion carries the whole epistemic weight of the aliasing arm.
    This calls the PROBE'S OWN function, so zeroing either statistic inside it
    fails here -- the previous version re-implemented the arithmetic inline
    and passed with the whole block deleted."""
    grid = M.GRID320_IN_LOOP_SPREAD
    base = 2.0e-3

    # planted PERIOD-2 signal, amplitude 20x the grid spread
    p2 = base * (1.0 + 20.0 * grid * np.array([+1.0, -1.0, +1.0, -1.0, +1.0]))
    got = M.consecutive_stats(p2)
    assert got["spread"] / grid > 5.0            # lands in the ALIASED arm
    assert got["parity_split"] > 10.0 * grid     # and the parity statistic sees it
    assert got["diff_signs"] == "-+-+"

    # planted NULL.  It must not be constant (a tautology) AND must not be a
    # pure linear trend either: at odd n a straight line has EXACTLY zero
    # parity split by construction, so that assertion would again be satisfied
    # by a machine zero rather than by the statistic working.  Use a trend
    # with a small non-linear wobble, so the parity split is genuinely small
    # rather than structurally zero.
    null = base * (1.0 + 1e-5 * np.arange(5.0)
                   + 2e-6 * np.array([0.0, 0.3, -0.2, 0.4, 0.1]))
    got0 = M.consecutive_stats(null)
    assert 0.0 < got0["spread"] / grid <= 2.0
    assert 0.0 < got0["parity_split"] < grid      # small, but NOT zero
    assert got0["diff_signs"] == "++++"

    # leave-one-out is reported and is not a copy of the full-sample number
    assert got["parity_split_drop_first"] != got["parity_split"]


def test_consecutive_stats_is_the_one_the_report_uses():
    import inspect
    src = inspect.getsource(M._report_consecutive)
    assert "consecutive_stats(d)" in src, (
        "_report_consecutive must call consecutive_stats, or the test above "
        "pins nothing")


def test_consecutive_stats_rejects_too_few_samples():
    with pytest.raises(ValueError):
        M.consecutive_stats(np.array([1.0, 2.0]))


# --- the reconciliation-halves control (baro_recon_halves.py) ---------------

def _load_halves():
    path = os.path.join(os.path.dirname(_MOD_PATH), "baro_recon_halves.py")
    spec = importlib.util.spec_from_file_location("_baro_halves", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_baro_halves"] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load_halves()


def test_read3d_strips_exactly_the_two_cell_halo(tmp_path):
    jpi, jpj, nlev = 9, 7, 3
    a = np.arange(nlev * jpj * jpi, dtype="<f8").reshape(nlev, jpj, jpi)
    f = tmp_path / "d.bin"
    a.tofile(f)
    got = H._read3d(str(f), jpi, jpj, nlev)
    assert got.shape == (nlev, jpj - 2 * H.HLS, jpi - 2 * H.HLS)
    # the strip must take the INTERIOR, not the corner
    assert np.array_equal(got, a[:, H.HLS:-H.HLS, H.HLS:-H.HLS])


def test_read3d_rejects_a_partial_level(tmp_path):
    f = tmp_path / "bad.bin"
    np.arange(9 * 7 * 3 - 1, dtype="<f8").tofile(f)
    with pytest.raises(SystemExit):
        H._read3d(str(f), 9, 7, 3)


def test_halves_lane_days_match_the_walk_states():
    assert H.DAYS == [s[0] for s in M.STATES]
