"""The /proc parse is the instrument behind every CPU-occupancy claim on these
lanes, and the previous shell version of it read the wrong two fields. These
tests pin the field offsets and the unit conversion against a real process."""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "sample_cpu_cores.py"
_spec = importlib.util.spec_from_file_location("sample_cpu_cores", _SRC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_parse_picks_utime_and_stime_not_the_child_counters():
    """A process that has just burned CPU must report it. cutime/cstime -- what
    the old shell probes actually read -- are zero here, so a wrong offset shows
    up as a zero rather than as an error."""
    before = sum(_mod.parse_cpu_ticks(Path("/proc/self/stat").read_text()))
    t = time.time()
    while time.time() - t < 0.4:
        pass
    after = sum(_mod.parse_cpu_ticks(Path("/proc/self/stat").read_text()))
    burned = (after - before) / os.sysconf("SC_CLK_TCK")
    assert 0.2 < burned < 0.8, f"parsed {burned:.3f} s of CPU for a 0.4 s busy loop"


def test_parse_survives_a_comm_containing_spaces_and_parentheses():
    # state then fields 4..13, so utime (field 14) is the first of the range
    line = "1234 (my (odd) name) R 1 1 1 0 -1 0 0 0 0 0 " + " ".join(
        str(i) for i in range(100, 130))
    assert _mod.parse_cpu_ticks(line) == (100, 101)
    # splitting on the FIRST ')' instead would land three fields early
    assert _mod.parse_cpu_ticks(line)[0] != 0


def test_cores_busy_returns_cores_not_hundreds_of_percent():
    clk = os.sysconf("SC_CLK_TCK")
    # 4.5 cores for 5 s = 22.5 core-seconds
    assert abs(_mod.cores_busy(0, int(22.5 * clk), 5.0) - 4.5) < 1e-6


def test_zero_interval_raises_rather_than_dividing():
    try:
        _mod.cores_busy(0, 100, 0.0)
    except ValueError:
        return
    raise AssertionError("a zero interval must not silently produce a number")
