from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3] / "scripts" / "validate" /
         "ocean_fidelity" / "testcases" /
         "nemo_testcase_l1_swg_online_control_health.py")
_SPEC = importlib.util.spec_from_file_location("swg_online_control_health", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
health = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(health)


def test_summarize_year_reports_extrema_and_refuses_nonfinite():
    arrays = {
        "basin_mean_ke_m2_s2": np.array([1.0, 3.0]),
        "basin_mean_eke_m2_s2": np.array([0.5, 1.5]),
        "top_octave_ke_fraction": np.array([0.2]),
        "top_octave_enstrophy_fraction": np.array([0.7]),
        "native_grid_noise_index": np.array([0.01, 0.03]),
        "minimum_total_layer_thickness_m": np.array([499.0, 498.0]),
        "maximum_u_courant": np.array([0.1, 0.2]),
        "maximum_v_courant": np.array([0.3, 0.2]),
    }
    result = health.summarize_year(arrays)
    assert result["basin_mean_ke_m2_s2"] == 2.0
    assert result["grid_noise_index_max"] == 0.03
    assert result["minimum_total_layer_thickness_m"] == 498.0
    assert result["maximum_v_courant"] == 0.3

    arrays["native_grid_noise_index"][0] = np.nan
    with pytest.raises(Exception, match="non-finite"):
        health.summarize_year(arrays)
