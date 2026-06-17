"""Unit tests for the MITgcm %MON monitor-output parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.fidelity import mitgcm_monitor

# One header-ordered block exactly as MITgcm prints it (time_tsnumber is NOT
# first — the parser must segment by key repetition, not by a leading key).
_BLOCK_TEMPLATE = """\
(PID.TID 0000.0001) %MON ke_max                       =   {ke_max:.13E}
(PID.TID 0000.0001) %MON ke_mean                      =   {ke_mean:.13E}
(PID.TID 0000.0001) %MON time_tsnumber                =   {step:18d}
(PID.TID 0000.0001) %MON time_secondsf                =   {secs:.13E}
(PID.TID 0000.0001) %MON dynstat_eta_max              =   {eta:.13E}
"""


def _synthetic_log(n=3, dt=1200.0):
    blocks = []
    for i in range(n):
        blocks.append(
            _BLOCK_TEMPLATE.format(
                ke_max=1e-8 * (i + 1),
                ke_mean=5e-9 * (i + 1),
                step=i * 10,
                secs=i * 10 * dt,
                eta=4.5e-4 * (i + 1),
            )
        )
    return "".join(blocks)


def test_segments_blocks_by_key_repetition():
    mon = mitgcm_monitor.parse_monitor_text(_synthetic_log(n=3))
    assert mon.n_records == 3
    np.testing.assert_array_equal(mon.steps, [0, 10, 20])
    np.testing.assert_allclose(mon.times_s, [0.0, 12000.0, 24000.0])
    np.testing.assert_allclose(mon.series["ke_mean"], [5e-9, 1e-8, 1.5e-8])


def test_final_returns_last_value():
    mon = mitgcm_monitor.parse_monitor_text(_synthetic_log(n=3))
    np.testing.assert_allclose(mon.final("ke_mean"), 1.5e-8)
    np.testing.assert_allclose(mon.final("dynstat_eta_max"), 4.5e-4 * 3)


def test_final_unknown_key_raises():
    mon = mitgcm_monitor.parse_monitor_text(_synthetic_log(n=1))
    with pytest.raises(KeyError, match="no monitor key"):
        mon.final("nope")


def test_fortran_d_exponent_parsed():
    text = (
        "(PID.TID 0000.0001) %MON ke_mean = 5.92083D-09\n"
        "(PID.TID 0000.0001) %MON time_tsnumber = 10\n"
        "(PID.TID 0000.0001) %MON time_secondsf = 1.2D+04\n"
    )
    mon = mitgcm_monitor.parse_monitor_text(text)
    np.testing.assert_allclose(mon.series["ke_mean"], [5.92083e-09])
    np.testing.assert_allclose(mon.times_s, [1.2e4])


def test_missing_key_becomes_nan():
    text = (
        "(PID.TID 0000.0001) %MON ke_mean = 1.0E-09\n"
        "(PID.TID 0000.0001) %MON time_tsnumber = 1\n"
        "(PID.TID 0000.0001) %MON ke_mean = 2.0E-09\n"  # 2nd record lacks tsnumber
    )
    mon = mitgcm_monitor.parse_monitor_text(text)
    assert mon.n_records == 2
    np.testing.assert_allclose(mon.series["ke_mean"], [1e-9, 2e-9])
    assert np.isnan(mon.series["time_tsnumber"][1])


def test_empty_text_yields_no_records():
    mon = mitgcm_monitor.parse_monitor_text("no monitor lines here\n")
    assert mon.n_records == 0
    assert mon.steps.size == 0


def test_multitile_log_filtered_to_master_tile():
    """Per-tile monitor lines must not spuriously split a step into records."""
    text = (
        "(PID.TID 0000.0001) %MON ke_mean = 1.0E-09\n"
        "(PID.TID 0000.0001) %MON time_tsnumber = 1\n"
        "(PID.TID 0001.0001) %MON ke_mean = 9.9E-09\n"   # other tile — must skip
        "(PID.TID 0001.0001) %MON time_tsnumber = 1\n"
        "(PID.TID 0000.0001) %MON ke_mean = 2.0E-09\n"
        "(PID.TID 0000.0001) %MON time_tsnumber = 2\n"
    )
    mon = mitgcm_monitor.parse_monitor_text(text)
    assert mon.n_records == 2  # not 3
    np.testing.assert_array_equal(mon.steps, [1, 2])
    np.testing.assert_allclose(mon.series["ke_mean"], [1e-9, 2e-9])


def test_nan_and_inf_values_are_captured():
    """A blow-up (NaN/Inf) must parse, not silently become a missing key."""
    text = (
        "(PID.TID 0000.0001) %MON ke_max = NaN\n"
        "(PID.TID 0000.0001) %MON ke_mean = Inf\n"
        "(PID.TID 0000.0001) %MON time_tsnumber = 5\n"
    )
    mon = mitgcm_monitor.parse_monitor_text(text)
    assert np.isnan(mon.series["ke_max"][0])
    assert np.isinf(mon.series["ke_mean"][0])


def test_final_raises_on_nonfinite_by_default():
    text = (
        "(PID.TID 0000.0001) %MON ke_mean = NaN\n"
        "(PID.TID 0000.0001) %MON time_tsnumber = 5\n"
    )
    mon = mitgcm_monitor.parse_monitor_text(text)
    with pytest.raises(ValueError, match="non-finite"):
        mon.final("ke_mean")
    assert np.isnan(mon.final("ke_mean", require_finite=False))


def test_parse_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mitgcm_monitor.parse_monitor_file(tmp_path / "absent.txt")


# --- Validation against a REAL MITgcm log when one is reachable (never a hard
# dependency on an external tree). Confirms the parser matches genuine format. -
_REAL_LOGS = [
    "/swot/SUM01/spencer/MITgcm/verification/tutorial_barotropic_gyre/results/output.txt",
]


@pytest.mark.parametrize("log", _REAL_LOGS)
def test_parses_real_mitgcm_log_if_present(log):
    p = Path(log)
    if not p.exists():
        pytest.skip(f"real MITgcm log not present: {log}")
    mon = mitgcm_monitor.parse_monitor_file(p)
    # tutorial_barotropic_gyre runs 10 steps -> 11 monitor records (t=0 + 10).
    assert mon.n_records >= 2
    assert "ke_mean" in mon.series and "dynstat_eta_max" in mon.series
    # KE grows from rest; the final mean KE is the documented ~5.9e-9 m^2/s^2.
    assert mon.final("ke_mean") > 0.0
    assert np.all(np.diff(mon.steps) > 0)  # monotonically increasing timesteps
