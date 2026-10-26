"""The per-cell probe drove a retraction, so its arithmetic gets a check.

Only the pure helpers are covered here: the dump/compare modes need the NEMO
oracle records, which are machine-local. What is checked is the part that can
silently produce a confident wrong number — the ulp distance, whose first
version reported 4.2e18 for a cell that was 4.8e-28 on one side and exactly
zero on the other.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_PROBE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "scripts" / "validate" / "ocean_fidelity" / "testcases"
    / "nemo_testcase_state_ulp_probe.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_state_ulp_probe", _PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load()


def test_ulp_distance_counts_representable_steps():
    x = np.float64(0.0622)
    steps = np.array([np.nextafter(x, np.inf)])
    assert probe.ulp_distance(steps, np.array([x]))[0] == 1
    three = x
    for _ in range(3):
        three = np.nextafter(three, np.inf)
    assert probe.ulp_distance(np.array([three]), np.array([x]))[0] == 3


def test_ulp_distance_is_zero_for_identical_values():
    values = np.array([1.0, -2.5, 1e-300, 0.0])
    assert not probe.ulp_distance(values, values.copy()).any()


def test_ulp_distance_crosses_zero_monotonically():
    """Sign-magnitude to two's complement: -0.0 and +0.0 are one step apart."""
    assert probe.ulp_distance(np.array([-0.0]), np.array([0.0]))[0] == 0
    below = np.nextafter(np.float64(0.0), -np.inf)
    above = np.nextafter(np.float64(0.0), np.inf)
    assert probe.ulp_distance(np.array([above]), np.array([below]))[0] == 2


def test_ulp_distance_against_zero_is_enormous_which_is_why_it_is_floored():
    """The number the floor exists to keep out of the report."""
    assert probe.ulp_distance(np.array([4.78e-28]), np.array([0.0]))[0] > 1e18


@pytest.mark.parametrize(
    "field,shape,expected",
    [("u", (3, 5, 2), (3, 4, 2)), ("v", (3, 5, 2), (2, 5, 2)),
     ("T", (3, 5, 2), (3, 5, 2)), ("eta", (3, 5), (3, 5))],
)
def test_gate_slice_matches_the_trajectory_gate_staggering(field, shape, expected):
    assert probe.gate_slice(field, np.zeros(shape)).shape == expected
