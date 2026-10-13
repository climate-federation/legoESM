"""Direct tests for the growing-truth stationarity instrument."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity"
         / "testcases/nemo_testcase_l1_swg_online_stationarity.py")
_SPEC = importlib.util.spec_from_file_location("swg_stationarity", _PATH)
stationarity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stationarity)


def test_year_metrics_separates_mean_ke_from_eke_and_uses_wet_area():
    shape = (4, 4)
    wet = np.zeros(shape, dtype=bool)
    wet[1:3, 1:3] = True
    masks = {kind: wet.copy() for kind in ("T", "U", "V")}
    geometry = {
        "mask": masks,
        "area": {"T": np.ones(shape)},
    }
    frames = []
    for velocity in (1.0, 3.0):
        u = np.zeros(shape)
        u[wet] = velocity
        frames.append({"T": np.zeros(shape), "U": u, "V": np.zeros(shape)})
    metrics = stationarity.yearly_metrics(frames, geometry)
    assert metrics["mean_ke_m2_s-2"] > metrics["mean_eke_m2_s-2"] > 0.0
    planted = [dict(frame) for frame in frames]
    planted[0] = planted[0] | {"U": planted[0]["U"].copy()}
    planted[0]["U"][~wet] = 1.0e9
    planted_metrics = stationarity.yearly_metrics(planted, geometry)
    assert planted_metrics["mean_ke_m2_s-2"] == metrics["mean_ke_m2_s-2"]
    assert planted_metrics["mean_eke_m2_s-2"] == metrics["mean_eke_m2_s-2"]
