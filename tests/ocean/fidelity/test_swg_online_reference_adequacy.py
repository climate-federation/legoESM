"""Direct tests for the SWG online reference-adequacy gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity"
         / "testcases/nemo_testcase_l1_swg_online_reference_adequacy.py")
_SPEC = importlib.util.spec_from_file_location("swg_reference_adequacy", _PATH)
adequacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adequacy)


def _metrics(scale=1.0):
    return {
        "basin_mean_ke_m2_s2": np.asarray([1.0, 1.0]) * scale,
        "basin_mean_eke_m2_s2": np.asarray([0.5, 0.5]) * scale,
        "jet_y_m": np.asarray([1.0e6, 1.1e6]),
        "mean_kinetic_energy_spectrum_m3_s2": (
            np.asarray([0.0, 2.0, 1.0]) * scale),
    }


def test_reference_adequacy_gate_passes_identity_and_fails_ten_percent_plant():
    identical = adequacy.adequacy_metrics(
        _metrics(), _metrics(), basin_width_m=4.0e6)
    assert identical["adequate"]
    assert all(value == 0.0 for value in identical["scores"].values())
    planted = adequacy.adequacy_metrics(
        _metrics(scale=1.11), _metrics(), basin_width_m=4.0e6)
    assert not planted["adequate"]
    assert not planted["passes"]["kinetic_energy_relative_error"]
    assert not planted["passes"]["eddy_kinetic_energy_relative_error"]
