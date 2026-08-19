"""Controls for the seed-agreement reporter.

It exists to say how much of the ranking is search noise, so a bug here would
either hide real churn or invent it.  Each test is a case whose answer is known
before the code runs.
"""

from __future__ import annotations

import csv
import json

import pytest

from scripts.validate import scm_rce_tq_seed_agreement as S


def _rows(pairs, noise=0.01):
    """pairs: {scheme: tuned_objective}."""
    return {
        scheme: {
            "scheme": scheme,
            "tuned_objective": str(value),
            "tuned_thermo_window_std": str(noise),
        }
        for scheme, value in pairs.items()
    }


def test_identical_seeds_show_no_churn():
    rows = _rows({"a": 0.1, "b": 0.2, "c": 0.3})
    out = S.compare_seeds(rows, rows, seed_a="1", seed_b="2")
    assert out["n_pair_swaps"] == 0
    assert out["max_rank_shift"] == 0
    assert out["n_differences_exceeding_noise"] == 0


def test_a_single_swap_is_counted_once():
    a = _rows({"x": 0.1, "y": 0.2, "z": 0.3})
    b = _rows({"x": 0.1, "y": 0.35, "z": 0.3})   # y and z swap
    out = S.compare_seeds(a, b, seed_a="1", seed_b="2")
    assert out["n_pair_swaps"] == 1
    assert sorted(out["pair_swaps"][0]) == ["y", "z"]
    assert out["max_rank_shift"] == 1


def test_a_full_reversal_swaps_every_pair():
    a = _rows({"p": 0.1, "q": 0.2, "r": 0.3})
    b = _rows({"p": 0.3, "q": 0.2, "r": 0.1})
    out = S.compare_seeds(a, b, seed_a="1", seed_b="2")
    assert out["n_pair_swaps"] == 3
    assert out["pair_swap_fraction"] == pytest.approx(1.0)


def test_a_difference_inside_the_noise_floor_is_not_flagged():
    a = _rows({"s": 0.100}, noise=0.05)
    b = _rows({"s": 0.130}, noise=0.05)
    out = S.compare_seeds(a, b, seed_a="1", seed_b="2")
    assert out["per_scheme"]["s"]["difference_exceeds_noise"] is False
    assert out["n_differences_exceeding_noise"] == 0


def test_a_difference_outside_the_noise_floor_is_flagged():
    a = _rows({"s": 0.10}, noise=0.01)
    b = _rows({"s": 0.40}, noise=0.01)
    out = S.compare_seeds(a, b, seed_a="1", seed_b="2")
    assert out["per_scheme"]["s"]["difference_exceeds_noise"] is True
    assert out["n_differences_exceeding_noise"] == 1


def test_a_nonfinite_objective_is_reported_as_a_failure_not_ranked():
    """A crashed run has no rank. Ordering it by scheme NAME and letting it take
    part in the swap count would manufacture churn out of two failures."""
    a = _rows({"good": 0.1, "alpha": float("nan"), "zulu": float("nan")})
    out = S.compare_seeds(a, a, seed_a="1", seed_b="2")
    assert out["ranking_seed_1"] == ["good"]
    assert out["n_schemes"] == 1
    assert out["schemes_non_finite_in_one_or_both"] == ["alpha", "zulu"]
    assert out["n_pair_swaps"] == 0


def test_a_scheme_missing_from_one_seed_is_reported_not_silently_dropped():
    """Ranking the full table and filtering afterwards leaves GAPS, so a scheme
    present in one seed only shifts every rank below it."""
    a = _rows({"x": 0.1, "gone": 0.15, "y": 0.2, "z": 0.3})
    b = _rows({"x": 0.1, "y": 0.2, "z": 0.3})
    out = S.compare_seeds(a, b, seed_a="1", seed_b="2")
    assert out["schemes_only_in_seed_1"] == ["gone"]
    assert out["schemes_only_in_seed_2"] == []
    # y and z must NOT read as shifted just because `gone` vanished.
    assert out["max_rank_shift"] == 0
    assert out["n_pair_swaps"] == 0


def test_the_noise_floor_of_a_difference_is_the_quadrature_sum():
    a = _rows({"s": 0.10}, noise=0.03)
    b = _rows({"s": 0.20}, noise=0.04)
    out = S.compare_seeds(a, b, seed_a="1", seed_b="2")
    assert out["per_scheme"]["s"]["window_noise"] == pytest.approx(0.05)
    # 0.10 difference against a 0.05 floor
    assert out["per_scheme"]["s"]["difference_exceeds_noise"] is True


def test_a_missing_noise_floor_is_ignored_not_propagated_as_nan():
    """`max(nan, x)` is NaN, which would report "unknown" even when one run has
    a perfectly usable spread."""
    a = _rows({"s": 0.10}, noise=float("nan"))
    b = _rows({"s": 0.50}, noise=0.02)
    out = S.compare_seeds(a, b, seed_a="1", seed_b="2")
    assert out["per_scheme"]["s"]["window_noise"] == pytest.approx(0.02)
    assert out["per_scheme"]["s"]["difference_exceeds_noise"] is True


def test_parameters_with_different_bounds_are_incomparable(tmp_path):
    """Two tuned values on different declared ranges are not on a common scale,
    so a range fraction would be a number with no meaning."""
    import json as _json

    arm_a, arm_b = tmp_path / "a", tmp_path / "b"
    for arm, hi in ((arm_a, 10.0), (arm_b, 20.0)):
        arm.mkdir()
        (arm / "tuned_parameters.json").write_text(_json.dumps({
            "sch": {"k.p": {"default": 5.0, "tuned": 1.0,
                            "bounds": [0.0, hi], "units": "1"},
                    "k.q": {"default": 1.0, "tuned": 2.0 if hi == 10.0 else 4.0,
                            "bounds": [0.0, 10.0], "units": "1"}}
        }))
    out = S.compare_parameters(arm_a, arm_b)
    assert out["sch"]["incomparable_bounds"] == ["k.p"]
    assert set(out["sch"]["per_parameter_range_fraction"]) == {"k.q"}


def test_parameter_disagreement_is_a_fraction_of_the_range(tmp_path):
    """Distant-but-tied parameter values are non-identifiability, and the
    reporter must express the distance in units of each parameter's own range."""
    arm_a, arm_b = tmp_path / "a", tmp_path / "b"
    for arm, tuned in ((arm_a, 1.0), (arm_b, 9.0)):
        arm.mkdir()
        (arm / "tuned_parameters.json").write_text(json.dumps({
            "sch": {"k.p": {"default": 5.0, "tuned": tuned,
                            "bounds": [0.0, 10.0], "units": "1"}}
        }))
    out = S.compare_parameters(arm_a, arm_b)
    assert out["sch"]["max_range_fraction"] == pytest.approx(0.8)


def test_zero_width_bounds_are_skipped_not_divided_by(tmp_path):
    arm_a, arm_b = tmp_path / "a", tmp_path / "b"
    for arm in (arm_a, arm_b):
        arm.mkdir()
        (arm / "tuned_parameters.json").write_text(json.dumps({
            "sch": {"k.p": {"default": 1.0, "tuned": 1.0,
                            "bounds": [1.0, 1.0], "units": "1"}}
        }))
    assert S.compare_parameters(arm_a, arm_b) == {}


def test_one_seed_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="at least two"):
        S.main(["--arm-dir", str(tmp_path), "--seeds", "1"])


def test_empty_csv_raises(tmp_path):
    path = tmp_path / "x.csv"
    with path.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=["scheme"]).writeheader()
    with pytest.raises(ValueError, match="no data rows"):
        S._read_csv(path)
