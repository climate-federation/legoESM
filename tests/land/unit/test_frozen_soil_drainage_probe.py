"""Direct test of the frozen-soil drainage probe's budget arithmetic."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_P = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "validate" / "frozen_soil_drainage.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("frozen_soil_drainage", _P)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_budget_closes_and_catches_a_missing_term(probe):
    probe._self_check()
    dz = np.array([0.1, 0.2])
    th0 = np.array([[0.3, 0.3]])
    th1 = np.array([[0.29, 0.3]])            # 1 mm left the soil
    s0 = probe.water_storage(th0, dz, np.zeros(1), 0.0, np.zeros(1))
    s1 = probe.water_storage(th1, dz, np.zeros(1), 0.0, np.zeros(1))
    dt = 1800.0
    r = probe.budget_residual(s0, s1, 0.0, 0.0, 0.0, np.array([1.0 / dt]), dt)
    assert abs(r[0]) < 1e-12
    r_bad = probe.budget_residual(s0, s1, 0.0, 0.0, 0.0, np.zeros(1), dt)
    assert abs(r_bad[0]) == pytest.approx(1.0)


def test_held_column_moves_no_water(probe):
    """A held column keeps a stale drainage field and zero response: counting
    that drainage (or fw - drainage as runoff) would fake a flux pair that
    cancels in the budget."""
    ro, dr = probe.step_runoff(np.array([2e-4, 2e-4]), np.array([3e-4, 0.0]),
                               np.array([False, True]))
    assert dr.tolist() == [2e-4, 0.0]
    assert ro[0] == pytest.approx(1e-4) and ro[1] == 0.0


def test_area_mean_weights(probe):
    assert probe.area_mean(np.array([1.0, 3.0]), np.array([3.0, 1.0])) == pytest.approx(1.5)


def test_usage_without_separator(probe):
    with pytest.raises(SystemExit, match="usage"):
        probe.main(["--days", "1"])
