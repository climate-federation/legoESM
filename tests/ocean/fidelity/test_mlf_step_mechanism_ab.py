"""Direct test for the M-01 ``stp_MLF`` mechanism A/B probe.

The probe compares two whole-step implementations of NEMO's ``stp_MLF`` and
reports the difference in ULPS OF THE FIELD'S OWN SCALE — the same
normalisation ``legoesm.ocean.fidelity.ulp_move_gate`` uses, so the number can
be read against that gate's 2-ulp bar.  Everything else in the probe needs
NEMO's restart; ``field_move`` is the pure part, and it is the part the
headline number comes from, so it is the part that gets pinned here against
cases whose answer is known without running it.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_PROBE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scripts", "validate",
    "ocean_fidelity", "dino_1226", "mlf_step_mechanism_ab.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "mlf_step_mechanism_ab", os.path.abspath(_PROBE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def test_identical_fields_report_no_move():
    """The identity case must be exactly zero, not merely small.

    The probe's own identity control (leapfrog vs leapfrog) is only meaningful
    if this returns 0.0 rather than a rounding-scale number.
    """
    a = np.array([[1.0, -2.5], [3.25, 0.0]])
    assert M.field_move(a, a.copy()) == (0.0, 0.0)


def test_one_ulp_at_field_scale_reads_as_one_ulp():
    """A move of exactly one ulp of the field's largest value reads 1.0."""
    a = np.array([1.0, 8.0])                      # field scale = 8.0
    b = a.copy()
    b[1] = np.nextafter(8.0, np.inf)              # exactly 1 ulp at 8.0
    diff, ulps = M.field_move(a, b)
    assert diff == pytest.approx(8.0 * 2.0 ** -52, rel=1e-12)
    assert ulps == pytest.approx(1.0, rel=1e-12)


def test_scale_is_the_field_maximum_not_the_moved_element():
    """A move on a SMALL element is normalised by the FIELD's scale.

    This is the property that makes the number comparable with the ulp gate's
    ``normalized_max_abs`` rows, and the one way the probe could quietly
    inflate a residual is by normalising each element by itself.
    """
    small, big = 1.0e-8, 1.0e3
    a = np.array([small, big])
    b = np.array([small + 1.0e-12, big])
    diff, ulps = M.field_move(a, b)
    assert diff == pytest.approx(1.0e-12, rel=1e-6)
    assert ulps == pytest.approx(1.0e-12 / (2.0 ** -52 * big), rel=1e-6)
    # per-element normalisation would have been ~1e4 x larger
    assert ulps < 1.0e-12 / (2.0 ** -52 * small)


def test_all_zero_fields_do_not_divide_by_zero():
    z = np.zeros((3, 4))
    assert M.field_move(z, z) == (0.0, 0.0)


def test_float32_input_is_measured_in_float64():
    """fp32 inputs must not silently set an fp32 ulp scale (skill Rule 1c)."""
    a = np.array([1.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 4.0 + 1.0e-6], dtype=np.float32)
    diff, ulps = M.field_move(a, b)
    assert diff > 0.0
    # normalised by the float64 ulp of 4.0, so a 1e-6 move is ~1e9 ulp, not ~1
    assert ulps > 1.0e8
