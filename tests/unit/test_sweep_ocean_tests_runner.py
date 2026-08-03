"""Direct test for scripts/validate/sweep_ocean_tests.py (#1387).

Every new .py gets a test; for a runner that shells out, the useful surface is
file collection, the thread-headroom probe, and the summary bookkeeping.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "sweep_ocean_tests", REPO / "scripts" / "validate" / "sweep_ocean_tests.py")
sweep = importlib.util.module_from_spec(SPEC)
sys.modules["sweep_ocean_tests"] = sweep
SPEC.loader.exec_module(sweep)


def test_collects_ocean_test_files():
    files = sweep.collect_files(["tests/ocean/unit"], None)
    assert files, "no ocean unit test files collected"
    assert all(f.name.startswith("test_") for f in files)


def test_pattern_filters_by_file_name():
    files = sweep.collect_files(["tests/ocean/unit"], "barotropic")
    assert files
    assert all("barotropic" in f.name for f in files)
    assert len(files) < len(sweep.collect_files(["tests/ocean/unit"], None))


def test_thread_headroom_reports_the_two_colliding_numbers():
    soft, cores = sweep.thread_headroom()
    assert cores >= 1
    assert soft == -1 or soft > 0  # -1 = RLIM_INFINITY


def test_run_one_reports_pytest_own_summary_not_just_exit_code(tmp_path):
    """An exit code is not evidence — the runner must keep the summary line."""
    t = REPO / "tests" / "unit" / "test_sweep_ocean_tests_runner.py"
    r = sweep.run_one(t, cores=2, k="test_collects_ocean_test_files",
                      timeout=600)
    assert r["file"].endswith("test_sweep_ocean_tests_runner.py")
    assert "passed" in r["summary"]
    assert r["aborted"] is False
