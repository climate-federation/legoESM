import numpy as np

from scripts.validate.validate_duo_land_device_split import leaf_rel


def test_leaf_rel_relative_to_peak_and_refusals():
    ref = {"a": np.array([2.0, -4.0]), "m": np.array([True, False]),
           "z": np.zeros(2), "n": np.array([np.nan, 1.0])}
    got = {"a": np.array([2.0, -4.0 + 4e-9]), "m": np.array([True, False]),
           "z": np.array([0.0, 1e-30]), "n": np.array([0.5, 1.0])}
    r = leaf_rel(ref, got)
    assert abs(r["['a']"] - 1e-9) < 1e-15
    assert r["['m']"] == 0.0
    assert r["['z']"] == float("inf")      # nonzero diff on a zero-peak leaf
    assert r["['n']"] == float("inf")      # NaN placement differs
    assert leaf_rel(ref, ref)["['z']"] == 0.0


def test_infinities_never_pass_and_residuals_use_the_absolute_bound():
    from scripts.validate.validate_duo_land_device_split import (
        leaf_abs, over_tolerance)
    ref = {"x": np.array([np.inf, 1.0])}
    assert leaf_rel(ref, {"x": np.array([np.nan, 1.0])})["['x']"] == float("inf")
    assert leaf_rel(ref, {"x": np.array([-np.inf, 1.0])})["['x']"] == float("inf")
    assert leaf_rel(ref, ref)["['x']"] == 0.0
    rel = {"[3].residual_int": 2.0, "[0].T_soil": 1e-12}
    assert over_tolerance(rel, {"[3].residual_int": 3e-13, "[0].T_soil": 0.0}) == {}
    assert set(over_tolerance(rel, {"[3].residual_int": 1e-11,
                                    "[0].T_soil": 0.0})) == {"[3].residual_int"}
    assert set(over_tolerance({"[0].T_soil": 1e-9}, {"[0].T_soil": 0.0})) == {"[0].T_soil"}
    assert leaf_abs(ref, ref)["['x']"] == 0.0
