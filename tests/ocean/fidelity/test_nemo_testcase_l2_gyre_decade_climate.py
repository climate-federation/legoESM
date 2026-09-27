"""Direct tests for the GYRE DECADE climate-tier scorer.

Every assertion is paired with a synthetic violation that must FAIL, so none of
them can pass vacuously.  The harness is loaded by path because it is a script,
not an installed module.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
HARNESS = (REPO / "scripts" / "validate" / "ocean_fidelity" / "testcases"
           / "nemo_testcase_l2_gyre_decade_climate.py")
ACQUISITION = (HARNESS.parent / "nemo_testcase_l2_gyre_decade_climate"
               / "run.sh")
PREREG = (REPO / "docs" / "ocean" / "fidelity"
          / "PREREG_nemo_testcases_l2_gyre_decade_climate.md")


@pytest.fixture(scope="module")
def harness():
    assert HARNESS.is_file(), HARNESS
    spec = importlib.util.spec_from_file_location("gyre_decade_climate",
                                                  HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_self_check_passes_as_a_subprocess():
    """The harness's own self-check is the gate; run it the way CI would."""
    result = subprocess.run([sys.executable, str(HARNESS), "--self-check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SELF-CHECK OK" in result.stdout


@pytest.mark.parametrize("plant", ["ratio-denominator-zero", "mld-unsorted",
                                   "trend-short", "month-shift"])
def test_every_plant_fails(plant):
    """A guard that cannot fire is not a guard."""
    result = subprocess.run([sys.executable, str(HARNESS), "--plant", plant],
                            capture_output=True, text=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "failed as required" in result.stdout


def test_the_cadence_follows_from_the_cards_timestep(harness):
    """Every step count in this comparison comes from rn_Dt = 14400 s.

    The round brief assumed four steps a day and therefore 14400 steps for the
    decade; the card's namelist says six and 21600.  This pins the arithmetic
    so the wrong pair cannot come back.
    """
    assert harness.DT_S == 14400.0
    assert harness.STEPS_PER_DAY == 6
    assert harness.MONTH_STEPS == 180
    assert harness.DECADE_MONTHS == 120
    assert harness.DECADE_STEPS == 21600
    assert 86400.0 / harness.DT_S == harness.STEPS_PER_DAY


def test_the_basin_mean_is_area_weighted_not_a_plain_mean(harness):
    """A global statistic on a non-uniform grid needs area weights."""
    mask = np.ones((2, 3), dtype=bool)
    weights = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    values = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
    assert harness._weighted_mean(values, weights, mask) == pytest.approx(3.0)
    # The violation: dropping the weights gives a different, wrong answer, so
    # the weighting is load-bearing rather than decorative.
    assert float(np.mean(values)) == pytest.approx(2.0)


def test_the_ratio_is_the_difference_over_the_fields_own_variability(harness):
    mask = np.ones((2, 3), dtype=bool)
    weights = np.ones((2, 3))
    reference = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    row = harness._ratio(np.full((2, 3), 0.01), reference, weights, mask)
    assert row["spatial_scale"] == pytest.approx(1.0)
    assert row["ratio"] == pytest.approx(0.01)
    assert row["bar"] == 1.0e-2
    # The violation: a constant reference has no spatial variability, so the
    # two-orders-of-magnitude bar is undefined and must raise rather than
    # divide by a number that happens to be tiny.
    with pytest.raises(harness.GateError):
        harness._ratio(np.ones((2, 3)), np.ones((2, 3)), weights, mask)


def test_the_mixed_layer_interpolates_between_cell_centres(harness):
    depth = np.broadcast_to(np.array([5.0, 15.0]), (1, 1, 2)).copy()
    wet = np.ones((1, 1, 2), dtype=bool)
    mld = harness._mixed_layer_depth(np.array([[[20.0, 19.0]]]), depth, wet)
    assert float(mld[0, 0]) == pytest.approx(7.0)
    # A fully mixed column reports its deepest wet cell, not a magic sentinel a
    # downstream mean would silently average in.
    mld = harness._mixed_layer_depth(np.array([[[20.0, 20.0]]]), depth, wet)
    assert float(mld[0, 0]) == pytest.approx(15.0)
    # The violation: a depth axis that is not increasing must raise, because
    # the bracketing search would otherwise return a plausible wrong depth.
    with pytest.raises(harness.GateError):
        harness._mixed_layer_depth(np.array([[[20.0, 19.0]]]), depth, wet,
                                   plant="mld-unsorted")


def test_the_zonal_mean_uses_wet_cells_only(harness):
    field = np.array([[[1.0], [3.0]], [[5.0], [9.0]]])
    wet = np.ones((2, 2, 1), dtype=bool)
    assert np.allclose(harness._zonal_mean(field, wet), [[2.0], [7.0]])
    # The violation: masking one column changes the answer, so the mask is
    # doing work rather than being carried along.
    wet[0, 1, 0] = False
    assert np.allclose(harness._zonal_mean(field, wet), [[1.0], [7.0]])


def test_the_trend_is_relative_and_refuses_too_few_points(harness):
    trend = harness._relative_trend(np.array([1.0, 2.0, 3.0, 4.0]))
    assert trend["slope_per_month"] == pytest.approx(1.0)
    assert trend["relative_trend_per_month"] == pytest.approx(1.0 / 2.5)
    assert trend["spike_ratio"] == pytest.approx(4.0 / 2.5)
    # The violation: two points always fit a line exactly, so a "trend" over
    # them is arithmetic with no evidence in it.
    with pytest.raises(harness.GateError):
        harness._relative_trend(np.array([1.0, 2.0]))


def test_the_acquisition_refuses_to_run_without_the_explicit_flag():
    """Ten model years must not start because someone opened the file."""
    assert ACQUISITION.is_file(), ACQUISITION
    result = subprocess.run(["bash", str(ACQUISITION)], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    assert "Re-run with --run" in result.stdout
    text = ACQUISITION.read_text()
    assert "NN_ITEND=21600" in text
    assert "NN_STOCK=180" in text
    # The admission check of operator's note AS: year 1 against round 132.
    assert "round132/oracle_daily_restarts" in text


def test_the_preregistration_states_the_bar_the_scorer_reports(harness):
    """The bar lives in the preregistration; the code only reports it."""
    text = PREREG.read_text()
    assert "arXiv:2608.01546" in text
    assert "1e-2" in text
    assert harness.BAR_RATIO == 1.0e-2
    assert harness.CLIM_FIRST_MONTH == 13
