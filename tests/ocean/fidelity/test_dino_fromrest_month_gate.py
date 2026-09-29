"""Direct tests for the DINO from-rest month landing gate.

No GPU and no model run: the gate's decision is a pure function, and these
tests pin it, including the regression the gate was written to catch.
"""
from __future__ import annotations

import math

import pytest

from scripts.validate.ocean_fidelity.dino_1226 import dino_fromrest_month_gate as G


def test_certified_number_passes_and_the_2026_09_regression_fails():
    """Non-vacuity: the bar must reject the number that motivated the gate."""
    passed, line = G.verdict(G.CERTIFIED_T3D_K)
    assert passed and "PASS" in line
    regressed, line_bad = G.verdict(6.981690958e-03)
    assert not regressed and "FAIL" in line_bad


def test_bar_is_above_the_certified_number_and_below_the_regression():
    assert G.CERTIFIED_T3D_K < G.BAR_T3D_K < 6.981690958e-03


@pytest.mark.parametrize("bad", [float("nan"), -1.0])
def test_unusable_measurements_fail_closed(bad):
    passed, _ = G.verdict(bad)
    assert not passed


def test_a_measurement_exactly_at_the_bar_passes_and_one_ulp_above_does_not():
    assert G.verdict(G.BAR_T3D_K)[0]
    assert not G.verdict(math.nextafter(G.BAR_T3D_K, 1.0))[0]


def test_the_protocol_constants_are_the_certified_ones():
    """A silent protocol change (card, recipe, window, step) goes red here."""
    assert G.CARD == "scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml"
    assert G.RECIPE == "nemo_dino_kamm_mlf"
    assert (G.DAYS, G.DT_SECONDS, G.NEMO_KT) == (30, 2700.0, 960)


def test_self_test_entry_point_runs_clean():
    assert G.self_test() == 0
