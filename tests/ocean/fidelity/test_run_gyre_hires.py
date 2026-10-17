import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/run/run_gyre_hires.py"
SPEC = importlib.util.spec_from_file_location("run_gyre_hires", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def test_gpu_selection_is_explicit_exact_and_cpu_forbids_it():
    DRIVER.validate_physical_gpu("gpu-native", 0, {"CUDA_VISIBLE_DEVICES": "0"})
    DRIVER.validate_physical_gpu("gpu-native", 3, {"CUDA_VISIBLE_DEVICES": "3"})
    DRIVER.validate_physical_gpu("cpu-certified", None, {})
    with pytest.raises(ValueError, match="requires --physical-gpu"):
        DRIVER.validate_physical_gpu("gpu-native", None, {})
    with pytest.raises(RuntimeError, match="must equal"):
        DRIVER.validate_physical_gpu("gpu-native", 1, {"CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError, match="does not accept"):
        DRIVER.validate_physical_gpu("cpu-certified", 1, {})


def test_gpu_wrapper_has_no_default_device():
    wrapper = ROOT / "scripts/run/run_gyre_hires_gpu.sh"
    result = subprocess.run(
        ["bash", str(wrapper)], check=False, text=True, capture_output=True,
    )
    assert result.returncode == 64
    assert "--physical-gpu PHYSICAL_INDEX" in result.stderr


def test_resolved_daily_decks_keep_declared_resolution_and_cadence():
    shipped = DRIVER.resolved_daily_namelist("shipped", 2)
    r4 = DRIVER.resolved_daily_namelist("r4", 1095)
    smag = DRIVER.resolved_daily_namelist("r4_smag", 1095)
    assert (shipped.nn_GYRE, shipped.rn_Dt_s, shipped.nn_itend, shipped.nn_write) == (
        1,
        14400.0,
        12,
        6,
    )
    assert (r4.nn_GYRE, r4.rn_Dt_s, r4.nn_itend, r4.nn_write, r4.nn_stock) == (
        4,
        3600.0,
        26280,
        24,
        8760,
    )
    assert (smag.nn_GYRE, smag.ln_dynldf_blp, smag.nn_ahm_ijk_t, smag.rn_Uv_m_s, smag.rn_Lv_m) == (
        4,
        True,
        32,
        None,
        None,
    )
    with pytest.raises(ValueError, match="deck must"):
        DRIVER.resolved_daily_namelist("invented", 1)


def test_full_state_window_requires_observer_free_per_step_integration():
    assert DRIVER.parse_day_window("30:180", days=180) == (30, 180)
    assert DRIVER.parse_day_window(None, days=180) is None
    for value in ("30", "0:30", "90:30", "30:181"):
        with pytest.raises(ValueError, match="full-state window"):
            DRIVER.parse_day_window(value, days=180)
    with pytest.raises(ValueError, match="require --per-step"):
        DRIVER.validate_full_state_request(
            days=180, per_step=False, window=(30, 180))
    DRIVER.validate_full_state_request(
        days=180, per_step=True, window=(30, 180))


def test_full_state_native_faces_and_thickness_weighted_finalize(monkeypatch):
    field = lambda value: SimpleNamespace(data=np.asarray(value, dtype=np.float64))
    state = SimpleNamespace(
        T=field(np.ones((2, 3, 2))),
        S=field(2.0 * np.ones((2, 3, 2))),
        u=field(np.arange(16).reshape(2, 4, 2)),
        v=field(np.arange(18).reshape(3, 3, 2)),
        tke=field(3.0 * np.ones((2, 3, 2))),
    )
    class Vertical:
        @staticmethod
        def compute_layer_thickness(eta, bathymetry, z_coord):
            del eta, bathymetry, z_coord
            return 4.0 * np.ones((2, 3, 2))

    card = SimpleNamespace(recipe=SimpleNamespace(z_coord=object()))
    state.eta = field(np.zeros((2, 3)))
    state.H_bathy = field(np.ones((2, 3)))
    original = __import__(
        "legoesm.ocean.vertical", fromlist=["compute_layer_thickness"])
    monkeypatch.setattr(
        original, "compute_layer_thickness", Vertical.compute_layer_thickness)
    arrays = DRIVER._full_state_arrays(state, card)
    assert [array.shape for array in arrays] == [
        (2, 3, 2), (2, 3, 2), (2, 3, 2), (2, 3, 2), (2, 3, 2),
        (2, 3, 2)]
    np.testing.assert_array_equal(arrays[2], state.u.data[:, 1:, :])
    np.testing.assert_array_equal(arrays[3], state.v.data[1:, :, :])

    thickness = np.array([[[2.0, 4.0]]])
    finalized = DRIVER._finalize_full_means((
        np.array([[[20.0, 80.0]]]),
        np.array([[[70.0, 180.0]]]),
        np.ones((1, 1, 2)),
        2.0 * np.ones((1, 1, 2)),
        3.0 * np.ones((1, 1, 2)),
        thickness,
    ))
    np.testing.assert_array_equal(np.asarray(finalized[0]), [[[10.0, 20.0]]])
    np.testing.assert_array_equal(np.asarray(finalized[1]), [[[35.0, 45.0]]])
    np.testing.assert_array_equal(np.asarray(finalized[5]), thickness)


def test_full_state_means_are_an_observer_of_ordinary_steps(monkeypatch):
    from legoesm.ocean.fidelity import nemo_testcase_recipe
    from legoesm.ocean import vertical

    field = lambda value: SimpleNamespace(data=np.asarray(value, dtype=np.float64))

    def initial_state():
        return SimpleNamespace(
            eta=field(np.zeros((2, 3))),
            H_bathy=field(np.ones((2, 3))),
            T=field(np.ones((2, 3, 2))),
            S=field(2.0 * np.ones((2, 3, 2))),
            u=field(np.ones((2, 4, 2))),
            v=field(np.ones((3, 3, 2))),
            tke=field(np.ones((2, 3, 2))),
            uu_b=field(np.ones((2, 4))),
            vv_b=field(np.ones((3, 3))),
        )

    class Model:
        @staticmethod
        def step(state, dt, *, freshwater, surface_forcing):
            del dt, freshwater, surface_forcing
            values = vars(state).copy()
            for name in ("eta", "T", "S", "u", "v", "tke", "uu_b", "vv_b"):
                values[name] = field(getattr(state, name).data + 1.0)
            return SimpleNamespace(**values)

    monkeypatch.setattr(
        nemo_testcase_recipe, "gyre_surface_forcings",
        lambda card, state, kt: (None, None),
    )
    monkeypatch.setattr(
        vertical, "compute_layer_thickness",
        lambda eta, bathymetry, z_coord: np.ones((2, 3, 2)),
    )
    card = SimpleNamespace(dt_s=1.0, recipe=SimpleNamespace(z_coord=object()))
    plain, _ = DRIVER.run_python_day(
        Model(), card, initial_state(), first_kt=1, n_steps=2)
    observed, _, full_means = DRIVER.run_python_day_with_full_state_means(
        Model(), card, initial_state(), first_kt=1, n_steps=2)
    for name in ("eta", "T", "S", "u", "v", "tke", "uu_b", "vv_b"):
        np.testing.assert_array_equal(
            getattr(observed, name).data, getattr(plain, name).data)
    assert len(full_means) == len(DRIVER.FULL_STATE_FIELD_NAMES)


def test_full_state_window_starting_before_tke_is_seeded(monkeypatch):
    """The first model.step seeds state.tke (None -> field); day 1 must work."""
    from legoesm.ocean.fidelity import nemo_testcase_recipe
    from legoesm.ocean import vertical

    field = lambda value: SimpleNamespace(data=np.asarray(value, dtype=np.float64))
    state = SimpleNamespace(
        eta=field(np.zeros((2, 3))), H_bathy=field(np.ones((2, 3))),
        T=field(np.ones((2, 3, 2))), S=field(2.0 * np.ones((2, 3, 2))),
        u=field(np.ones((2, 4, 2))), v=field(np.ones((3, 3, 2))),
        tke=None, uu_b=field(np.ones((2, 4))), vv_b=field(np.ones((3, 3))),
    )

    class Model:
        @staticmethod
        def step(state, dt, *, freshwater, surface_forcing):
            del dt, freshwater, surface_forcing
            values = vars(state).copy()
            values["tke"] = field(np.ones((2, 3, 1)))
            return SimpleNamespace(**values)

    monkeypatch.setattr(
        nemo_testcase_recipe, "gyre_surface_forcings",
        lambda card, state, kt: (None, None))
    monkeypatch.setattr(
        vertical, "compute_layer_thickness",
        lambda eta, bathymetry, z_coord: np.ones((2, 3, 2)))
    card = SimpleNamespace(dt_s=1.0, recipe=SimpleNamespace(z_coord=object()))
    _, _, full_means = DRIVER.run_python_day_with_full_state_means(
        Model(), card, state, first_kt=1, n_steps=2)
    assert len(full_means) == len(DRIVER.FULL_STATE_FIELD_NAMES)
    np.testing.assert_array_equal(np.asarray(full_means[4]), np.ones((2, 3, 1)))
    with pytest.raises(ValueError, match="requires prognostic TKE"):
        DRIVER._full_state_arrays(state, card)


@pytest.mark.slow
def test_tiny_real_gyre_two_day_scan_matches_certified_loop():
    import jax
    from legoesm.core.precision import get_policy, set_policy

    if jax.default_backend() != "cpu":
        pytest.skip("real two-day parity is a CPU control")
    old = get_policy()
    try:
        namelist = DRIVER.resolved_daily_namelist("shipped", 2)
        report = DRIVER.cpu_scan_parity(namelist, days=2)
    finally:
        set_policy(old)
    assert report["days"] == 2
    assert report["steps"] == 12
    assert report["differences"]["all_finite"] is True
    assert report["forcing_transcription"]["clock_and_phase_bit_equal"] is True
    assert report["classification"] == "COMPOSITION"
    assert report["prefix_comparisons"][0]["model_step"] == 2
    assert report["prefix_comparisons"][0]["fields"] == []
    assert report["prefix_comparisons"][0]["differences"]["all_bit_equal"] is True
    assert report["prefix_comparisons"][1]["model_step"] == 3
    assert report["first_difference"]["step"] == 3
    assert report["first_difference"]["first_field"] == "ssh_m"
    # Measured refutation: enclosing the same production step in a day scan
    # changes XLA's fused floating-point program.  Keep the failed 1e-12 bar
    # visible rather than silently relaxing it after seeing the residual.
    assert report["differences"]["within_atol"] is False
    assert report["differences"]["surface_temperature_c_max_abs"] == (
        pytest.approx(1.8106049992638873e-10, rel=1.0e-6)
    )
