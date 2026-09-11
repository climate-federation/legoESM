"""Direct tests for the GYRE from-rest YEAR harness.

Every assertion here is paired with a synthetic violation that must FAIL, so
none of them can pass vacuously.  The harness is loaded by path because it is
a script, not an installed module.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HARNESS = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
           / "ocean_fidelity" / "testcases"
           / "nemo_testcase_l2_gyre_year_fromrest.py")


@pytest.fixture(scope="module")
def harness():
    assert HARNESS.is_file(), HARNESS
    spec = importlib.util.spec_from_file_location("gyre_year_fromrest", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy(harness):
    depth = np.broadcast_to(np.array([10.0, 20.0]), (2, 2, 2)).copy()
    lat = np.broadcast_to(np.array([[30.0], [31.0]])[..., None],
                          (2, 2, 2)).copy()
    return depth, lat, np.ones_like(depth)


def test_self_check_passes_as_a_subprocess():
    """The harness's own self-check is the gate; run it the way CI would."""
    result = subprocess.run([sys.executable, str(HARNESS), "--self-check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SELF-CHECK OK" in result.stdout


def test_seed_zero_is_exactly_the_unperturbed_path(harness):
    depth, lat, mask = _toy(harness)
    assert np.count_nonzero(
        harness.nemo_istate_perturbation(depth, lat, mask, 0)) == 0
    # non-vacuity: a non-zero seed must NOT be zero
    assert np.count_nonzero(
        harness.nemo_istate_perturbation(depth, lat, mask, 1)) > 0


def test_fortran_nint_rounds_half_away_from_zero(harness):
    assert harness._nint(0.5) == 1.0
    assert harness._nint(-0.5) == -1.0
    assert harness._nint(2.5) == 3.0
    # the thing it exists to avoid: numpy rounds half to EVEN
    assert np.round(2.5) == 2.0
    assert harness._nint(2.5) != np.round(2.5)


def test_distinct_seeds_give_distinct_perturbations(harness):
    depth, lat, mask = _toy(harness)
    fields = [harness.nemo_istate_perturbation(depth, lat, mask, seed)
              for seed in (1, 2, 3)]
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            assert not np.array_equal(fields[i], fields[j])


@pytest.mark.parametrize("plant", ["perturbation-zero", "perturbation-relative"])
def test_perturbation_plants_turn_the_assertions_red(harness, plant):
    depth, lat, mask = _toy(harness)
    wet = mask.astype(bool)
    # the baseline passes, so a red plant is the plant and not the baseline
    harness.assert_perturbation_properties(
        harness.nemo_istate_perturbation(depth, lat, mask, 1), lat, wet)
    planted = harness.nemo_istate_perturbation(depth, lat, mask, 1, plant=plant)
    with pytest.raises(harness.GateError):
        harness.assert_perturbation_properties(planted, lat, wet)


def test_group_constancy_assertion_can_fail(harness):
    depth, lat, mask = _toy(harness)
    wet = mask.astype(bool)
    good = harness.nemo_istate_perturbation(depth, lat, mask, 1)
    harness.assert_perturbation_properties(good, lat, wet)
    broken = good.copy()
    broken[0, 0, 0] += 1e-12
    with pytest.raises(harness.GateError):
        harness.assert_perturbation_properties(broken, lat, wet)


def test_the_scalar_floor_carries_no_extra_sqrt2(harness):
    """The within-ensemble statistic is already a pairwise DIFFERENCE.

    Two independent reviews of the preregistration caught a sqrt(2) applied on
    top of it.  Pin it by source so it cannot creep back.
    """
    source = HARNESS.read_text()
    needle = "floor" + " * " + "SQRT" + "2"
    assert source.count(needle) == 0


def test_depth_bands_partition_the_column_exactly_once(harness):
    edges = [(lo, hi) for _, lo, hi in harness.DEPTH_BANDS]
    assert edges[0][0] == 0.0
    assert edges[-1][1] > 5000.0
    for (_, upper), (lower, _) in zip(edges[:-1], edges[1:]):
        assert upper == lower


def test_psi_is_signed_and_sees_the_weaker_gyre(harness):
    """max|PSI| reports the STRONGER cell only; the signed pair sees both.

    The first version of this test asserted that the signed pair distinguishes
    a spatial sign flip.  It does not -- psi = [+1,-1] and [-1,+1] have the
    same max AND the same min -- so the EXPECTATION was wrong, not the code.
    What the signed pair actually buys is the weaker gyre, which is what is
    checked here.
    """
    dz = np.full(1, 500.0)
    dy = np.full((1, 1), 1.0e5)

    def psi(subtropical, subpolar):
        return harness._psi_field_sv(
            np.array([[[subtropical]], [[subpolar]]]), dz, dy)

    base = psi(0.04, -0.10)          # psi = [+2, -3]
    weaker = psi(0.04, -0.11)        # only the SUBPOLAR cell changed
    assert abs(float(base.max()) - float(weaker.max())) < 1e-12
    assert abs(float(np.abs(base).max()) - float(np.abs(weaker).max())) > 1e-9
    assert float(base.min()) < 0.0 < float(base.max())
    assert abs(float(base.min()) - float(weaker.min())) > 1e-9


def test_cell_count_diagnostic_sees_one_switched_cell(harness):
    left = {"T": np.zeros((2, 2, 2))}
    right = {"T": np.zeros((2, 2, 2))}
    wet = np.ones((2, 2, 2), dtype=bool)
    assert harness._cells_over(left, right, wet) == 0
    right["T"][1, 0, 1] = 10.0 * harness.CELL_COUNT_THRESHOLD_K
    assert harness._cells_over(left, right, wet) == 1


def test_the_year_arithmetic_matches_the_namelist(harness):
    # rn_Dt = 14400 and nn_leapy = 30 -> 6 steps a day, 2160 steps a year
    assert harness.DT_S == 14400.0
    assert harness.STEPS_PER_DAY == 6
    assert harness.YEAR_STEPS == 2160
    assert harness.SNAP_STEPS == 180
    assert harness.SCORED_DAYS == tuple(range(30, 361, 30))


def test_vacuity_gate_refuses_a_zero_floor(harness):
    rows = {"T3D": {"days": {"30": {"floor": 1e-9}, "360": {"floor": 0.0}}}}
    gate = harness._vacuity(rows)
    assert not gate["phase1_may_run"]
    assert gate["zero_floor_days"] == ["360"]
    # non-vacuity: a positive floor everywhere must PASS
    rows["T3D"]["days"]["360"]["floor"] = 1e-9
    assert harness._vacuity(rows)["phase1_may_run"]


def test_floor_collapse_is_classified_rather_than_discovered_later(harness):
    def report(floor_30, floor_360):
        rows = {"T3D": {"days": {"30": {"floor": floor_30},
                                 "360": {"floor": floor_360}}}}
        return harness._expectations(rows, phase0_only=True,
                                     repro={"status": "UNMEASURED"})
    assert report(1e-9, 1e-12)["floor_shape"]["classification"] == "FLOOR_COLLAPSE"
    assert report(1e-12, 1e-9)["floor_shape"]["classification"] == "FLOOR_GROWING"


def test_preregistered_bounds_are_constants_not_judgement(harness):
    """A bound held in a variable can be relaxed at scoring time; pin them."""
    assert harness.P1_FLOOR_360_MAX_K == 1.0e-3
    assert harness.P2_FLOOR_GROWTH_MAX == 1.0e3
    assert harness.P3_GAP_MIN_K == 2.8e-3
    assert harness.VERDICT_FACTOR == 2.0
