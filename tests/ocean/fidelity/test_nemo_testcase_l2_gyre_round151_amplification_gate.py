"""Controls for the Round-151 amplification-threshold gate."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


GATE = (Path(__file__).parents[3] / "scripts" / "validate" /
        "ocean_fidelity" / "testcases" /
        "nemo_testcase_l2_gyre_round151_amplification_gate.py")


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("round151_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_amplitude_registry_is_exact_and_closed(gate):
    assert gate.AMPLITUDES_K == (1.0e-8, 1.0e-6, 1.0e-4)
    assert tuple(gate.AMPLITUDE_TAGS) == gate.AMPLITUDES_K
    with pytest.raises(gate.GateError):
        gate._tag(1.0e-5)


def test_threshold_uses_exact_equality_and_adjacent_values(gate):
    threshold = gate.AMPLIFICATION_THRESHOLD_K
    empty = {tag: 0.0 for tag in gate.AMPLITUDE_TAGS.values()}

    def classify(value):
        rows = dict(empty)
        rows[gate.AMPLITUDE_TAGS[1.0e-4]] = value
        return gate._classify(rows)

    assert classify(np.nextafter(threshold, -np.inf)) != (
        "THRESHOLD_AMPLIFICATION")
    assert classify(threshold) == "THRESHOLD_AMPLIFICATION"
    assert classify(np.nextafter(threshold, np.inf)) == (
        "THRESHOLD_AMPLIFICATION")


def test_small_arm_crossing_prevents_threshold_label(gate):
    rows = {tag: 0.0 for tag in gate.AMPLITUDE_TAGS.values()}
    rows[gate.AMPLITUDE_TAGS[1.0e-8]] = gate.AMPLIFICATION_THRESHOLD_K
    rows[gate.AMPLITUDE_TAGS[1.0e-4]] = gate.AMPLIFICATION_THRESHOLD_K
    assert gate._classify(rows) == (
        "INCONCLUSIVE_UNDER_REGISTERED_DISCRIMINATOR")
