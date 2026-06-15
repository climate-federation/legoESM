"""Smoke test for the Silvestri baroclinic-jet driver (scripts/run/...).

Runs the canonical driver's ``run()`` for a tiny (scheme, resolution, days) case
and checks it integrates stably, emits the metric arrays, and saves the npz —
exercising the recipe → restore → Phase-2-diagnostics wiring end to end.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_DRIVER = (Path(__file__).resolve().parents[3]
           / "scripts" / "run" / "run_silvestri_baroclinic_jet.py")


def _load_driver():
    spec = importlib.util.spec_from_file_location("_silvestri_jet_driver", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("scheme", ["W9V", "SM2"])
def test_driver_runs_and_saves(scheme, tmp_path):
    drv = _load_driver()
    verdict = drv.run(scheme, "12x12", days=1, dt=900.0, out=str(tmp_path),
                      nlev=6, tag="_smoke")
    assert verdict == "STABLE"
    npz = list(tmp_path.glob(f"silvestri_jet_{scheme}_12x12_smoke.npz"))
    assert len(npz) == 1
    data = np.load(npz[0], allow_pickle=True)
    # The driver saves the Fig-8/9/10 metric arrays.
    for key in ("t", "tke", "eke", "ape", "L_d_km", "gridscale_frac",
                "k_energy", "P_energy", "zonal_mean_buoyancy", "zeta_surface"):
        assert key in data, key
    assert np.all(np.isfinite(data["tke"]))
    assert float(data["L_d_km"]) > 0.0
