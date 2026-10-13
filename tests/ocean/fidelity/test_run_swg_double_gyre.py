"""Direct tests for the SWG double-gyre long-run driver."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3]
         / "scripts/run/run_swg_double_gyre.py")
_SPEC = importlib.util.spec_from_file_location("run_swg_double_gyre", _PATH)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


class _Field(NamedTuple):
    data: object


class _ScanState(NamedTuple):
    eta: _Field
    u: _Field
    v: _Field


class _IncrementModel:
    def step(self, state, dt_s, *, surface_forcing):
        del dt_s, surface_forcing
        return _ScanState(*(
            _Field(field.data + 1.0) for field in state))


def test_resolved_maddison_clock_is_exactly_daily():
    namelist = driver.resolved_daily_namelist(3)
    assert namelist.rn_Dt_s == 50.0
    assert namelist.nn_itend == 3 * 1728
    assert namelist.nn_write == namelist.nn_stock == 1728


def test_resolved_dg50_clock_and_unknown_dispatch():
    namelist = driver.resolved_daily_namelist(20, deck="dg50")
    assert namelist.nn_SWG == 2
    assert namelist.rn_Dt_s == 400.0
    assert namelist.nn_itend == 20 * 216
    assert namelist.nn_write == namelist.nn_stock == 216
    assert namelist.rn_Lv_m == 10.0
    control = driver.resolved_daily_namelist(30, deck="dg50-nu80")
    assert control.nn_SWG == 2 and control.rn_Dt_s == 400.0
    assert control.rn_Lv_m == 80.0
    sensitivity = driver.resolved_daily_namelist(3, deck="dg3125")
    assert sensitivity.nn_SWG == 32 and sensitivity.rn_Dt_s == 25.0
    assert sensitivity.nn_itend == 3 * 3_456
    assert (sensitivity.rn_Dt_s / (sensitivity.rn_dx_m / sensitivity.nn_SWG)
            == namelist.rn_Dt_s / (namelist.rn_dx_m / namelist.nn_SWG))
    with pytest.raises(ValueError, match="unsupported double-gyre deck"):
        driver.resolved_daily_namelist(1, deck="typo")


def test_tiny_cpu_driver_writes_daily_native_state_with_provenance(tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    # Two benign steps exercise the real card/production step/I/O path without
    # turning this direct test into a long physical integration.
    tiny = SWG_MADDISON_NAMELIST._replace(
        nn_SWG=1, rn_Dt_s=43_200.0, nn_itend=2, nn_write=2, nn_stock=2,
        rn_modified_grav_m_s2=1.0e-6, rn_tau_pa=0.0)
    output = tmp_path / "run"
    manifest = driver.run_long(
        tiny, output, days=1, mode="cpu-certified", commit="a" * 40,
        physical_gpu=None)
    assert manifest["certified_policy"] is True
    assert manifest["deck_certification"] == (
        "NEMO ladder measured; see fidelity receipts")
    assert manifest["status"] == "complete"
    assert manifest["completed_days"] == 1
    assert manifest["transcendentals"] == "libm"
    assert manifest["integration"] == "production-python-step-loop"
    assert manifest["profile"] is None
    assert manifest["scan_vs_python"] is None
    assert manifest["numerical_health"]["all_finite"]
    assert manifest["numerical_health"]["minimum_total_layer_thickness_m"] > 0.0
    # This zero-wind/rest control must remain exactly zero; a sign/abs
    # tautology would also pass a broken nonzero diagnostic.
    assert manifest["numerical_health"]["maximum_u_courant"] == 0.0
    assert manifest["numerical_health"]["maximum_v_courant"] == 0.0
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
    raw_state = card.recipe.initial_state
    state = driver.seed_steady_forcing_carry(raw_state, forcing)
    raw_first = model.step(raw_state, card.dt_s, surface_forcing=forcing)
    seeded_first = model.step(state, card.dt_s, surface_forcing=forcing)
    first_differences = driver.prognostic_differences(
        raw_first, seeded_first)
    assert first_differences["all_finite"]
    assert all(first_differences[f"{name}_unequal"] == 0
               for name in ("ssh_m", "u_m_s", "v_m_s"))
    python_state = driver.run_python_steps(
        model, state, card.dt_s, forcing, 3_456)
    scanned_day = driver.build_scanned_stepper(
        model, card.dt_s, forcing, 1_728)
    scan_state = scanned_day(scanned_day(state))
    jax.block_until_ready(scan_state.eta.data)

    differences = driver.prognostic_differences(python_state, scan_state)
    assert differences["within_atol"], json.dumps(differences, sort_keys=True)
    assert differences["atol"] == 1.0e-15
    assert np.any(np.asarray(python_state.u.data) != 0.0)
    planted = scan_state._replace(eta=scan_state.eta.replace(
        data=scan_state.eta.data + 2.0e-15))
    assert not driver.prognostic_differences(
        python_state, planted)["within_atol"]


def test_compiled_day_mean_matches_post_step_production_states_on_tiny_grid():
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
        nn_SWG=1, rn_Dt_s=50.0, nn_itend=3, nn_write=3, nn_stock=3,
        rn_modified_grav_m_s2=1.0e-6)
    card = build_swg_card(tiny)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    tau_u, tau_v = swg_surface_wind(tiny)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64),
    )
    state = driver.seed_steady_forcing_carry(card.recipe.initial_state, forcing)
    samples = []
    reference = state
    for count in range(1, 4):
        reference = model.step(reference, card.dt_s, surface_forcing=forcing)
        fields = (reference.eta.data, reference.u.data[:, 1:, 0],
                  reference.v.data[1:, :, 0])
        if not samples:
            samples = [np.asarray(field) for field in fields]
        else:
            samples = [
                (mean * np.float64(count - 1) + np.asarray(field))
                / np.float64(count)
                for mean, field in zip(samples, fields, strict=True)
            ]
    scanned = driver.build_scanned_day_mean_stepper(
        model, card.dt_s, forcing, 3)
    final, means = scanned(state)
    jax.block_until_ready(final.eta.data)
    assert driver.prognostic_differences(reference, final)["within_atol"]
    for expected, actual in zip(samples, means, strict=True):
        assert np.allclose(expected, np.asarray(actual), rtol=0.0, atol=1.0e-18)
    # A final-state substitution cannot satisfy this daily-mean test.
    assert np.any(np.asarray(means[1]) != np.asarray(final.u.data[:, 1:, 0]))


def test_compiled_hourly_sampler_retains_right_endpoints_only():
    import jax
    import jax.numpy as jnp

    state = _ScanState(
        _Field(jnp.zeros((2, 2), dtype=jnp.float64)),
        _Field(jnp.zeros((2, 3, 1), dtype=jnp.float64)),
        _Field(jnp.zeros((3, 2, 1), dtype=jnp.float64)),
    )
    scanned = driver.build_scanned_hourly_target_stepper(
        _IncrementModel(), 1.0, None, 1,
        lambda eta, u, v: (eta, u, v))
    final, hourly = scanned(state)
    jax.block_until_ready(final.eta.data)
    assert np.array_equal(
        np.asarray(hourly[0])[:, 0, 0], np.arange(1.0, 25.0))
    assert float(np.asarray(final.eta.data)[0, 0]) == 24.0
    assert np.asarray(hourly[1]).shape == (24, 2, 2)
    assert np.asarray(hourly[2]).shape == (24, 2, 2)


def test_operator_device_is_explicit_and_matches_visibility():
    driver.validate_operator_device(
        "gpu-native", 0, environ={"CUDA_VISIBLE_DEVICES": "0"})
    driver.validate_operator_device(
        "gpu-native", 1, environ={"CUDA_VISIBLE_DEVICES": "1"})
    driver.validate_operator_device("cpu-certified", None, environ={})
    with pytest.raises(ValueError, match="explicit"):
        driver.validate_operator_device(
            "gpu-native", None, environ={"CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(RuntimeError, match="does not match"):
        driver.validate_operator_device(
            "gpu-native", 0, environ={"CUDA_VISIBLE_DEVICES": "1"})
    with pytest.raises(ValueError, match="only valid"):
        driver.validate_operator_device("cpu-scanned", 0, environ={})


def test_gpu_wrapper_requires_a_physical_device_argument():
    wrapper = _PATH.with_name("run_swg_double_gyre_gpu.sh")
    result = subprocess.run(
        ["bash", str(wrapper)], capture_output=True, text=True, check=False)
    assert result.returncode == 64
    assert "PHYSICAL_GPU" in result.stderr


def test_snapshot_admission_and_bitwise_comparison_have_live_plants(tmp_path):
    path = tmp_path / "day0732.npz"
    arrays = {
        "ssh_m": np.arange(4.0).reshape(2, 2),
        "u_m_s": np.arange(4.0).reshape(2, 2) + 10.0,
        "v_m_s": np.arange(4.0).reshape(2, 2) + 20.0,
    }
    np.savez(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "complete", "daily_sha256": {path.name: digest},
    }))
    admission = driver.admit_listed_snapshot(path)
    assert admission["snapshot_sha256"] == digest
    state = _ScanState(
        _Field(arrays["ssh_m"]),
        _Field(np.pad(arrays["u_m_s"][:, :, None], ((0, 0), (1, 0), (0, 0)))),
        _Field(np.pad(arrays["v_m_s"][:, :, None], ((1, 0), (0, 0), (0, 0)))),
    )
    assert driver.compare_native_state_bitwise(state, path)["all_equal"]
    planted = state._replace(eta=_Field(arrays["ssh_m"] + 1.0e-15))
    result = driver.compare_native_state_bitwise(planted, path)
    assert not result["all_equal"]
    assert result["ssh_m_unequal"] > 0
    np.savez(path, **(arrays | {"ssh_m": arrays["ssh_m"] + 1.0}))
    with pytest.raises(ValueError, match="hash differs"):
        driver.admit_listed_snapshot(path)


def test_daily_mean_output_refuses_certified_cpu_mode(tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    tiny = SWG_MADDISON_NAMELIST._replace(
        nn_SWG=1, rn_Dt_s=43_200.0, nn_itend=2, nn_write=2, nn_stock=2,
        rn_modified_grav_m_s2=1.0e-6, rn_tau_pa=0.0)
    with pytest.raises(ValueError, match="non-certified gpu-native"):
        driver.run_long(
            tiny, tmp_path / "run", days=1, mode="cpu-certified",
            commit="a" * 40, physical_gpu=None, daily_means=True)


def test_cpu_profile_records_production_step_cost(tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    tiny = SWG_MADDISON_NAMELIST._replace(
        nn_SWG=1, rn_Dt_s=43_200.0, nn_itend=2, nn_write=2, nn_stock=2,
        rn_modified_grav_m_s2=1.0e-6, rn_tau_pa=0.0)
    manifest = driver.run_long(
        tiny, tmp_path / "run", days=1, mode="cpu-certified",
        commit="a" * 40, physical_gpu=None, profile_steps=2)
    assert manifest["profile"]["program"] == "production-python-step-loop"
    assert manifest["profile"]["steps"] == 2
    assert manifest["profile"]["python_ms_per_step"] > 0.0


def test_profile_only_writes_no_trajectory(tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    tiny = SWG_MADDISON_NAMELIST._replace(
        nn_SWG=1, rn_Dt_s=43_200.0, nn_itend=2, nn_write=2, nn_stock=2,
        rn_modified_grav_m_s2=1.0e-6, rn_tau_pa=0.0)
    output = tmp_path / "profile"
    manifest = driver.run_long(
        tiny, output, days=1, mode="cpu-certified", commit="a" * 40,
        physical_gpu=None, profile_steps=2, profile_only=True)
    assert manifest["profile_only"] is True
    assert manifest["completed_days"] == 0
    assert list(output.iterdir()) == [output / "manifest.json"]
    with pytest.raises(ValueError, match="positive profile_steps"):
        driver.run_long(
            tiny, tmp_path / "invalid", days=1, mode="cpu-certified",
            commit="a" * 40, physical_gpu=None, profile_only=True)


def test_cpu_scanned_mode_uses_certified_policy_and_day_program(tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    tiny = SWG_MADDISON_NAMELIST._replace(
        nn_SWG=1, rn_Dt_s=43_200.0, nn_itend=2, nn_write=2, nn_stock=2,
        rn_modified_grav_m_s2=1.0e-6, rn_tau_pa=0.0)
    manifest = driver.run_long(
        tiny, tmp_path / "run", days=1, mode="cpu-scanned",
        commit="a" * 40, physical_gpu=None)
    assert manifest["certified_policy"] is True
    assert manifest["transcendentals"] == "libm"
    assert manifest["integration"] == "one-jitted-scan-per-day"


def test_restart_restores_native_stagger_and_fixed_seed_perturbation(tmp_path):
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
        swg_surface_wind,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    tiny = SWG_MADDISON_NAMELIST._replace(nn_SWG=1, rn_Dt_s=400.0)
    card = build_swg_card(tiny)
    tau_u, tau_v = swg_surface_wind(tiny)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64))
    shape = card.recipe.initial_state.eta.data.shape
    path = tmp_path / "restart.npz"
    np.savez(path, ssh_m=np.ones(shape), u_m_s=np.full(shape, 2.0),
             v_m_s=np.full(shape, 3.0), day=1095, step=1234, dt_s=400.0)
    left, provenance = driver.restart_state_from_npz(
        card.recipe.initial_state, forcing, path, dt_s=400.0,
        perturb_amplitude_m=1.0e-6)
    right, _ = driver.restart_state_from_npz(
        card.recipe.initial_state, forcing, path, dt_s=400.0,
        perturb_amplitude_m=1.0e-6)
    assert np.array_equal(left.eta.data, right.eta.data)
    assert np.any(np.asarray(left.eta.data) != 1.0)
    assert np.array_equal(np.asarray(left.u.data[:, 1:, 0]), np.full(shape, 2.0))
    assert np.array_equal(np.asarray(left.v.data[1:, :, 0]), np.full(shape, 3.0))
    assert np.all(np.asarray(left.u.data[:, :1, :]) == 0.0)
    assert np.all(np.asarray(left.v.data[:1, :, :]) == 0.0)
    assert provenance["source_day"] == 1095
    with pytest.raises(ValueError, match="timestep differs"):
        driver.restart_state_from_npz(
            card.recipe.initial_state, forcing, path, dt_s=50.0)


def test_balanced_vorticity_perturbation_is_smooth_and_nondivergent():
    from legoesm.ocean.diagnostics import relative_vorticity_cell_centre
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
    )
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        cgrid_geometry,
        native_face_transports,
        transport_divergence,
    )

    tiny = SWG_MADDISON_NAMELIST._replace(nn_SWG=1)
    card = build_swg_card(tiny)
    amplitude = 1.0e-10
    perturbed, provenance = driver.apply_balanced_vorticity_perturbation(
        card.recipe.initial_state, tiny, amplitude)
    geometry = cgrid_geometry(tiny)
    wet = geometry["mask"]["T"]
    zeta = np.asarray(relative_vorticity_cell_centre(
        perturbed.u.data[..., 0], perturbed.v.data[..., 0],
        geometry["grid"]))
    assert np.sqrt(np.mean(zeta[wet] ** 2)) == pytest.approx(
        amplitude, rel=2.0e-14)
    assert np.array_equal(
        np.asarray(perturbed.eta.data),
        np.asarray(card.recipe.initial_state.eta.data))
    fields = {
        "T": np.asarray(perturbed.eta.data),
        "U": np.asarray(perturbed.u.data[:, 1:, 0]),
        "V": np.asarray(perturbed.v.data[1:, :, 0]),
    }
    divergence = transport_divergence(
        native_face_transports(fields, geometry), geometry)
    assert np.max(np.abs(divergence[wet])) <= 1.0e-15
    assert provenance["amplitude_s-1"] == amplitude
    # A native-face plant must violate the same physical control.
    fields["U"] = fields["U"].copy()
    fields["U"][wet] += 1.0e-3
    planted = transport_divergence(
        native_face_transports(fields, geometry), geometry)
    assert np.max(np.abs(planted[wet])) > 1.0e-15
