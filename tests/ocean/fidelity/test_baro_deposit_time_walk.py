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


def test_consecutive_catalog_is_four_steps_in_a_row():
    kts = [s[1] for s in M.CONSEC_STATES]
    assert kts == [5760, 5761, 5762, 5763]
    # the 10-day grid samples every 320th step -- an EVEN offset, which is
    # exactly why a period-2 mode is invisible on it.
    grid = [s[1] for s in M.STATES]
    assert all((b - a) == 320 and (b - a) % 2 == 0
               for a, b in zip(grid, grid[1:]))


def test_underpowered_branch_is_reachable_and_fires_on_the_real_numbers():
    """The pre-registered POWER branch fired in the committed run; a test must
    pin that, because a silently-skipped pre-registration is the failure the
    rule exists to stop."""
    d = np.array([5.5555e-3, 3.9401e-3, 2.9627e-3, 4.2694e-3, 4.1532e-3,
                  4.4172e-3, 4.8150e-3, 5.1232e-3, 5.6129e-3, 5.9099e-3])
    sem = float(d.std(ddof=1) / np.sqrt(d.size))
    assert sem > abs(M.TARGET_PER_STEP), (
        "the committed ten deposits must trip the underpowered criterion")
    assert sem == pytest.approx(2.86e-4, rel=2e-2)


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
