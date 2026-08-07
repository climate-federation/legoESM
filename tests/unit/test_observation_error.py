"""Unit tests for the structural-error floor model (P_e).

The two guard tests are the point of the file: they prove the tripwires FIRE on
synthetic versions of the two configurations that actually burned us —
a fractional floor on an arbitrary-zero statistic (tas, too loose) and a
fractional floor on a near-zero residual (net_toa, too tight).
"""
from __future__ import annotations

import pytest

from legoesm.training.observation_error import (
    DIAGNOSTIC_ONLY,
    FRACTIONAL_FLOOR,
    MIN_REFERENCE_TO_SCALE_RATIO,
    OBSERVATION_FLOORS,
    TAS_STRUCTURAL_SIGMA_K,
    FloorSpec,
    build_structural_covariance_diagonal,
    check_reference_against_scale,
    is_fitted,
    observation_statistics,
    structural_sigma,
    structural_variance,
    validate_all_floors,
    validate_floor_spec,
)

# Long-run references measured on cldopt10y (2 complete years, 1979-1980).
# TAS_REF is a global-mean AIR TEMPERATURE in kelvin, not a gas constant.
TAS_REF = 287.141  # const-ok: observed global-mean tas [K], not R_d
REFERENCES = {
    "rsut": 98.856, "rlut": 240.513, "pr": 3.086, "prw": 24.273,
    "tas": TAS_REF, "net_toa": 0.900,
}


# --------------------------------------------------------------------------
# the shipped table
# --------------------------------------------------------------------------

def test_shipped_table_is_self_consistent():
    validate_all_floors()
    validate_all_floors(references=REFERENCES)


def test_every_entry_has_a_reason():
    for name, spec in OBSERVATION_FLOORS.items():
        assert spec.reason.strip(), name


def test_four_statistics_keep_the_papers_five_percent_rule():
    for name in ("rsut", "rlut", "pr", "prw"):
        spec = OBSERVATION_FLOORS[name]
        assert spec.kind == "fractional", name
        assert spec.value == pytest.approx(FRACTIONAL_FLOOR), name
        assert spec.physical_zero is True, name


def test_unchanged_statistics_give_exactly_the_five_percent_floor():
    """Pure addition: these four must score exactly as before."""
    for name in ("rsut", "rlut", "pr", "prw"):
        ref = REFERENCES[name]
        assert structural_sigma(name, ref) == pytest.approx(0.05 * ref)
        assert structural_variance(name, ref) == pytest.approx((0.05 * ref) ** 2)


def test_specific_unchanged_floor_values():
    assert structural_sigma("rsut", 98.856) == pytest.approx(4.9428, abs=1e-4)
    assert structural_sigma("rlut", 240.513) == pytest.approx(12.0257, abs=1e-4)
    assert structural_sigma("pr", 3.086) == pytest.approx(0.1543, abs=1e-4)
    assert structural_sigma("prw", 24.273) == pytest.approx(1.2137, abs=1e-4)


# --------------------------------------------------------------------------
# FIX 1 — net_toa excluded
# --------------------------------------------------------------------------

def test_net_toa_is_excluded_from_the_fitted_vector():
    assert OBSERVATION_FLOORS["net_toa"].kind == "excluded"
    assert "net_toa" not in observation_statistics()
    assert is_fitted("net_toa") is False


def test_net_toa_is_still_a_reported_diagnostic():
    assert "net_toa" in DIAGNOSTIC_ONLY
    assert "net_toa" in OBSERVATION_FLOORS      # present, with its reason


def test_net_toa_exclusion_reason_warns_against_restoring_it():
    reason = OBSERVATION_FLOORS["net_toa"].reason
    assert "diagnostic" in reason.lower()
    assert "restore" in reason.lower()


def test_asking_for_net_toa_floor_raises():
    with pytest.raises(ValueError, match="excluded"):
        structural_sigma("net_toa", 0.9)


def test_excluded_entry_may_not_carry_a_value():
    bad = FloorSpec("excluded", 1.0, "W/m2", True, 340.0, "r")
    with pytest.raises(ValueError, match="excluded entries take None"):
        validate_floor_spec("x", bad)


def test_covariance_diagonal_omits_excluded_statistics():
    pe = build_structural_covariance_diagonal(REFERENCES)
    assert "net_toa" not in pe
    assert set(pe) == set(observation_statistics())


def test_covariance_diagonal_requires_a_reference_for_every_fitted_stat():
    partial = {k: v for k, v in REFERENCES.items() if k != "rsut"}
    with pytest.raises(KeyError, match="rsut"):
        build_structural_covariance_diagonal(partial)


# --------------------------------------------------------------------------
# FIX 2 — tas absolute floor
# --------------------------------------------------------------------------

def test_tas_uses_an_absolute_floor():
    spec = OBSERVATION_FLOORS["tas"]
    assert spec.kind == "absolute"
    assert spec.physical_zero is False
    assert spec.units == "K"


def test_tas_floor_is_independent_of_the_reference_value():
    """An absolute floor must not scale with an arbitrary-zero reference."""
    assert structural_sigma("tas", TAS_REF) == pytest.approx(
        TAS_STRUCTURAL_SIGMA_K)
    assert structural_sigma("tas", 14.0) == pytest.approx(
        TAS_STRUCTURAL_SIGMA_K)


def test_tas_floor_is_far_tighter_than_the_broken_fractional_rule():
    """5% of 287 K = 14.36 K would accept any plausible bias; ours must not."""
    broken = 0.05 * TAS_REF
    assert broken > 14.0
    assert structural_sigma("tas", TAS_REF) < 0.1 * broken


def test_tas_floor_marks_the_measured_bias_as_beyond_tolerance():
    """The model's -1.769 K bias must land OUTSIDE the floor."""
    assert abs(-1.769) > structural_sigma("tas", TAS_REF)


def test_tas_reason_records_why_the_fraction_fails():
    reason = OBSERVATION_FLOORS["tas"].reason.lower()
    assert "arbitrary zero" in reason
    assert "287" in reason or "14" in reason


# --------------------------------------------------------------------------
# THE GUARDS — proven to fire on synthetic violations
# --------------------------------------------------------------------------

def test_guard_fires_on_fractional_floor_with_near_zero_reference():
    """The net_toa trap: a residual of large terms, reference ~ 0.

    Synthetic statistic with a 340 W/m2 characteristic scale and a 0.9 W/m2
    reference — ratio 0.0026, far below the 0.1 limit.
    """
    spec = FloorSpec("fractional", 0.05, "W/m2", True, 340.0,
                     "synthetic near-zero residual")
    validate_floor_spec("synthetic_net", spec)      # spec itself is well formed
    with pytest.raises(ValueError, match="near-zero residual|TIGHT"):
        check_reference_against_scale("synthetic_net", spec, 0.9)


def test_guard_fires_on_fractional_floor_with_arbitrary_zero():
    """The tas trap: a fraction of a value measured from an arbitrary zero."""
    spec = FloorSpec("fractional", 0.05, "K", False, TAS_REF,
                     "synthetic arbitrary-zero statistic")
    with pytest.raises(ValueError, match="zero is not physical"):
        validate_floor_spec("synthetic_tas", spec)


def test_guard_passes_a_legitimate_fractional_statistic():
    """Non-vacuous: the guard must NOT fire on a well-posed statistic."""
    spec = OBSERVATION_FLOORS["rsut"]
    validate_floor_spec("rsut", spec)
    check_reference_against_scale("rsut", spec, 98.856)   # ratio 1.0


@pytest.mark.parametrize("ratio,fires", [(0.5, False), (0.2, False),
                                         (0.05, True), (0.001, True)])
def test_guard_threshold_is_where_it_says_it_is(ratio, fires):
    spec = FloorSpec("fractional", 0.05, "u", True, 100.0, "synthetic")
    ref = ratio * 100.0
    if fires:
        with pytest.raises(ValueError):
            check_reference_against_scale("s", spec, ref)
    else:
        check_reference_against_scale("s", spec, ref)
    assert (ratio < MIN_REFERENCE_TO_SCALE_RATIO) == fires


def test_fractional_floor_requires_a_characteristic_scale():
    spec = FloorSpec("fractional", 0.05, "u", True, None, "synthetic")
    with pytest.raises(ValueError, match="characteristic_scale"):
        validate_floor_spec("s", spec)


def test_missing_reason_is_rejected():
    with pytest.raises(ValueError, match="no reason"):
        validate_floor_spec("s", FloorSpec("absolute", 1.0, "u", True, None, ""))


def test_non_positive_floor_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        validate_floor_spec(
            "s", FloorSpec("absolute", 0.0, "u", True, None, "r"))


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown floor kind"):
        validate_floor_spec(
            "s", FloorSpec("bogus", 1.0, "u", True, 1.0, "r"))


def test_unknown_statistic_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="no floor spec"):
        structural_sigma("clt", 67.0)
    with pytest.raises(KeyError, match="no floor spec"):
        is_fitted("clt")


def test_adding_a_bad_entry_to_a_copy_of_the_table_is_caught():
    """End-to-end: the trap cannot be walked into silently a second time."""
    table = dict(OBSERVATION_FLOORS)
    table["new_residual"] = FloorSpec(
        "fractional", 0.05, "W/m2", True, 340.0, "someone adds a residual")
    refs = dict(REFERENCES, new_residual=0.5)
    with pytest.raises(ValueError, match="TIGHT|near-zero"):
        validate_all_floors(table, refs)
