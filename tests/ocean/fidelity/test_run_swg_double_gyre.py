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
    assert manifest["integration"] == "production-python-step-loop"
    assert manifest["profile"] is None
    assert manifest["scan_vs_python"] is None
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


def test_scanned_program_matches_nontrivial_python_steps_on_tiny_grid():
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
        swg_surface_wind,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    tiny = SWG_MADDISON_NAMELIST._replace(
        nn_SWG=1, rn_Dt_s=50.0, nn_itend=3_456,
        nn_write=1_728, nn_stock=1_728,
        rn_modified_grav_m_s2=1.0e-6)
    card = build_swg_card(tiny)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    tau_u, tau_v = swg_surface_wind(tiny)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64),
    )
    state = driver.seed_steady_forcing_carry(
        card.recipe.initial_state, forcing)
    python_state = driver.run_python_steps(
        model, state, card.dt_s, forcing, 3_456)
    scanned_day = driver.build_scanned_stepper(
        model, card.dt_s, forcing, 1_728)
    scan_state = scanned_day(scanned_day(state))
    jax.block_until_ready(scan_state.eta.data)

    differences = driver.prognostic_differences(python_state, scan_state)
    assert differences["within_atol"], json.dumps(differences, sort_keys=True)
    assert differences["atol"] == 1.0e-12
    assert np.any(np.asarray(python_state.u.data) != 0.0)
