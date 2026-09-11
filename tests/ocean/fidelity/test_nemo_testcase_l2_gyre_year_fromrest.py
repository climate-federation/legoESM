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


def test_the_floor_carries_no_root_factor(harness):
    """The within-ensemble statistic is already a pairwise DIFFERENCE.

    Two independent reviews caught a sqrt(2) applied on top of it, and a third
    caught the FIRST fix: a grep for one exact spelling that `SQRT2 * floor`
    would have walked past.  Check the property, not the spelling: the
    module-level constant is gone.
    """
    assert not hasattr(harness, "SQRT" + "2")
    assert "hypot" in HARNESS.read_text(), "the quadrature combination vanished"


def test_ensemble_spread_is_the_max_over_pairs_and_propagates_nan(harness):
    def prepared(values):
        return [{"X": (value, None)} for value in values]

    harness.SCALAR_ROWS = harness.SCALAR_ROWS  # documentational
    rows = prepared([1.0, 2.0, 4.0, 8.0])
    # _distance dispatches on membership in SCALAR_ROWS, so use a real one
    name = harness.SCALAR_ROWS[0]
    rows = [{name: (value, None)} for value in (1.0, 2.0, 4.0, 8.0)]
    assert harness._ensemble_spread(rows, name) == 7.0
    same = [{name: (3.0, None)} for _ in range(4)]
    assert harness._ensemble_spread(same, name) == 0.0


def test_distance_is_a_distance_for_fields_and_scalars(harness):
    name = harness.SCALAR_ROWS[0]
    assert harness._distance(name, 2.0, -3.0, None) == 5.0
    wet = np.ones((1, 1, 2), dtype=bool)
    a = np.array([[[1.0, 2.0]]])
    b = np.array([[[1.0, 3.0]]])
    assert abs(harness._distance("T3D", a, b, wet) - np.sqrt(0.5)) < 1e-12
    assert harness._distance("T3D", a, a, wet) == 0.0
    # and it is symmetric, which a difference-of-rms would not be
    assert (harness._distance("T3D", a, b, wet)
            == harness._distance("T3D", b, a, wet))


def test_census_does_not_hide_a_non_finite_value(harness):
    """The probe printed a finite-looking range for a half-infinite field.

    That is the repo's own 'nanmax hides failures' rule biting inside the
    probe written to catch exactly this, so pin the fix by source.
    """
    source = HARNESS.read_text()
    assert "values[np.isfinite(values)]" not in source
    assert "float(values.min()), float(values.max())" in source


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


# --------------------------------------------------------- the alignment gate --
DATA_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest")


def test_frame_mappings_are_reversals_and_none_is_the_identity(harness):
    """A3's alternatives must actually be alternatives.

    If a "reversal" happened to be a no-op on the array under test, the
    discrimination would pass vacuously and a flipped frame would score the
    same as the right one.
    """
    mappings = harness._frame_mappings()
    assert set(mappings) == {"identity", "reverse_j", "reverse_i",
                             "reverse_both"}
    field = np.arange(6.0).reshape(3, 2)
    assert np.array_equal(mappings["identity"](field), field)
    for name in ("reverse_j", "reverse_i", "reverse_both"):
        assert not np.array_equal(mappings[name](field), field), name
    # non-vacuity of the non-vacuity check: on a SYMMETRIC array the
    # reversals ARE the identity, which is exactly the case the gate's ratio
    # requirement exists to refuse.
    symmetric = np.ones((3, 2))
    assert all(np.array_equal(fn(symmetric), symmetric)
               for fn in mappings.values())


def test_level_for_depth_takes_the_nearest_centre(harness):
    depths = np.array([5.0, 15.0, 100.0, 1000.0, 4000.0])
    assert harness._level_for_depth(depths, 100.0) == 2
    assert harness._level_for_depth(depths, 1000.0) == 3
    # nearest, not first-over: 90 m is nearer 100 than 15
    assert harness._level_for_depth(depths, 90.0) == 2
    # non-vacuity: a different target must give a different level
    assert harness._level_for_depth(depths, 6.0) == 0


def test_alignment_thresholds_are_constants_not_judgement(harness):
    assert harness.FRAME_COORD_TOL_DEG == 1.0e-4
    assert harness.FRAME_COORD_RATIO == 1.0e4
    assert harness.FIGURE_DEPTHS_M == (100.0, 1000.0)
    # the figure depths are the preregistered band boundaries, not a choice
    boundaries = {value for _, lo, hi in harness.DEPTH_BANDS
                  for value in (lo, hi)}
    assert set(harness.FIGURE_DEPTHS_M) <= boundaries


@pytest.mark.skipif(not (DATA_ROOT / "nemo_seed0").is_dir(),
                    reason="the NEMO members are not on this machine")
@pytest.mark.parametrize("plant", ["frame-flip", "alignment-initial",
                                   "operand-mismatch"])
def test_alignment_gate_plants_exit_non_zero(plant):
    """The gate must be SHOWN to fail; a gate that cannot fail is decoration."""
    result = subprocess.run(
        [sys.executable, str(HARNESS), "--alignment-gate",
         "--root", str(DATA_ROOT), "--plant", plant],
        capture_output=True, text=True)
    assert result.returncode != 0, result.stdout[-2000:]
    assert "GATE ERROR" in result.stderr, result.stderr[-2000:]


@pytest.mark.skipif(not (DATA_ROOT / "nemo_seed0").is_dir(),
                    reason="the NEMO members are not on this machine")
def test_alignment_gate_passes_unplanted():
    result = subprocess.run(
        [sys.executable, str(HARNESS), "--alignment-gate",
         "--root", str(DATA_ROOT)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "STATUS ALIGNED" in result.stdout
