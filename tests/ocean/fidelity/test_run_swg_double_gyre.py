"""Direct tests for the SWG double-gyre long-run driver."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

_PATH = (Path(__file__).resolve().parents[3]
         / "scripts/run/run_swg_double_gyre.py")
_SPEC = importlib.util.spec_from_file_location("run_swg_double_gyre", _PATH)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def test_resolved_maddison_clock_is_exactly_daily():
    namelist = driver.resolved_daily_namelist(3)
    assert namelist.rn_Dt_s == 50.0
    assert namelist.nn_itend == 3 * 1728
    assert namelist.nn_write == namelist.nn_stock == 1728


def test_tiny_cpu_driver_writes_daily_native_state_with_provenance(tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    # Two benign steps exercise the real card/production step/I/O path without
    # turning this direct test into a long physical integration.
    tiny = SWG_MADDISON_NAMELIST._replace(
        nn_SWG=1, rn_Dt_s=43_200.0, nn_itend=2, nn_write=2, nn_stock=2,
        rn_modified_grav_m_s2=1.0e-6, rn_tau_pa=0.0)
    output = tmp_path / "run"
    manifest = driver.run_long(
        tiny, output, days=1, mode="cpu-certified", commit="a" * 40)
    assert manifest["certified_policy"] is True
    assert manifest["deck_certification"] == "pending NEMO ladder acquisition"
    assert manifest["status"] == "complete"
    assert manifest["completed_days"] == 1
    assert manifest["transcendentals"] == "libm"
    saved_manifest = json.loads((output / "manifest.json").read_text())
    assert saved_manifest["steps_per_day"] == 2
    assert not (output / "manifest.pending.json").exists()
    assert (output / "daily.sha256").read_text().endswith("  day0001.npz\n")
    with np.load(output / "day0001.npz") as saved:
        assert saved["ssh_m"].shape == saved["u_m_s"].shape == saved["v_m_s"].shape
        assert saved["ssh_m"].shape == (44, 44)
        assert saved["ssh_m"].dtype == np.dtype("float64")
        assert saved["legoesm_git_sha"].item() == "a" * 40
        assert saved["step"].item() == 2
        assert saved["certified_policy"].item()
        assert all(np.all(np.isfinite(saved[name]))
                   for name in ("ssh_m", "u_m_s", "v_m_s"))
