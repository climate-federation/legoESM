import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget_attribution.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_budget_attribution", SCRIPT)
ATTRIBUTION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ATTRIBUTION)


def arrays(shape=(151, 2, 5)):
    result = {
        ATTRIBUTION.BUDGET_DAY_KEY: np.arange(30, 181),
        ATTRIBUTION.BUDGET_MASK_KEY: np.ones(shape[1:], dtype=bool),
    }
    for key in ATTRIBUTION.BUDGET_TERM_KEYS:
        result[key] = np.zeros(shape, dtype=np.float64)
    return result


def test_registered_score_and_active_nonfinite_refusal():
    mask = np.array([[True, False]])
    nemo = np.array([[[2.0, np.nan]], [[2.0, np.nan]]])
    lego = np.array([[[3.0, np.nan]], [[3.0, np.nan]]])
    row = ATTRIBUTION.row_metrics(lego, nemo, mask)
    assert row["signed_legoesm_minus_nemo_mean_m_s"] == 1.0
    assert row["nemo_rms_m_s"] == 2.0
    assert row["registered_relative_score"] == 0.5

    planted = lego.copy()
    planted[0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        ATTRIBUTION.row_metrics(planted, nemo, mask)


def test_component_closure_plant_fires():
    lego = arrays()
    nemo = arrays()
    mask = lego[ATTRIBUTION.BUDGET_MASK_KEY]
    closure = ATTRIBUTION.component_closure(lego, nemo, mask)
    assert closure["passes"] is True
    lego["surface_v_advection_daily_mean"][0, 0, 0] = 2.0e-12
    closure = ATTRIBUTION.component_closure(lego, nemo, mask)
    assert closure["passes"] is False
    assert closure["max_abs_m_s"] == 2.0e-12


def test_boundary_cancellation_cannot_be_called_a_physics_owner(
        tmp_path, monkeypatch):
    lego = arrays()
    nemo = arrays()
    mask = lego[ATTRIBUTION.BUDGET_MASK_KEY]
    # Keep every NEMO denominator nonzero. The raw vertical/external split is
    # deliberately huge and cancelling, while vorticity is the largest row
    # after the source-matched pair is formed.
    for key in ATTRIBUTION.BUDGET_TERM_KEYS:
        nemo[key][:] = 1.0
        lego[key][:] = 1.0
    lego["surface_v_vertical_mixing_daily_mean"] += 0.4
    lego["surface_v_external_mode_correction_daily_mean"] -= 0.4
    lego["surface_v_total_vorticity_coriolis_daily_mean"] += 0.1
    lego["surface_v_total_increment_daily_mean"] += 0.1

    contracts = {
        "legoesm": {"producer_commit": "a" * 40,
                    "precision_flavour": "native-fp64-gpu",
                    "data_sha256": "1" * 64,
                    "closure_max_abs_m_s": 0.0},
        "nemo": {"producer_commit": "b" * 40,
                 "precision_flavour": "vector-math-conda",
                 "data_sha256": "2" * 64,
                 "closure_max_abs_m_s": 0.0},
    }

    twin = {key: value.copy() for key, value in nemo.items()}
    twin["surface_v_total_vorticity_coriolis_daily_mean"] += 0.01
    twin["surface_v_total_increment_daily_mean"] += 0.01

    def fake_load(root, provider, _flavour="vector-math-conda"):
        values = lego if provider == "legoesm" else (
            twin if root.name == "twin" else nemo)
        return ({"contract": contracts[provider]}, values)

    monkeypatch.setattr(ATTRIBUTION, "load_budget", fake_load)
    admission = tmp_path / "admission.json"
    admission.write_text('{"verdict":"PASS"}')
    envelope = tmp_path / "envelope.json"
    envelope.write_text("{}")
    monkeypatch.setattr(
        ATTRIBUTION, "state_envelope_context", lambda _path: {"scope": "plant"})
    monkeypatch.setattr(ATTRIBUTION, "sha256", lambda _path: "0" * 64)

    report = ATTRIBUTION.build_report(
        tmp_path / "lego", tmp_path / "nemo", admission, envelope)
    assert report["registered_numerical_carrier"] in (
        "vertical_mixing", "external_mode_correction")
    assert report["boundary_alignment"][
        "vertical_external_signed_difference_cancellation_fraction"] == (
            pytest.approx(1.0))
    assert report["first_source_aligned_candidate"] == (
        "total_vorticity_coriolis")
    assert report["physics_owner"] == "UNMEASURED"
    assert report["shared_path_change"] == "NONE"

    with_twin = ATTRIBUTION.build_report(
        tmp_path / "lego", tmp_path / "nemo", admission, envelope,
        tmp_path / "twin")
    assert with_twin["term_twin_envelope"]["status"] == "MEASURED"
    assert with_twin["term_twin_envelope"]["rows"][0][
        "candidate_to_twin_ratio"] == pytest.approx(10.0)
    assert with_twin["term_twin_envelope"]["rows"][0][
        "clears_four_times_twin"] is True


def test_provider_mask_mismatch_refuses(tmp_path, monkeypatch):
    lego = arrays()
    nemo = arrays()
    nemo[ATTRIBUTION.BUDGET_MASK_KEY] = np.zeros((2, 5), dtype=bool)
    contracts = {
        "producer_commit": "a" * 40,
        "precision_flavour": "plant",
        "data_sha256": "1" * 64,
        "closure_max_abs_m_s": 0.0,
    }
    monkeypatch.setattr(
        ATTRIBUTION, "load_budget",
        lambda _root, provider: (
            {"contract": contracts}, lego if provider == "legoesm" else nemo))
    admission = tmp_path / "admission.json"
    admission.write_text('{"verdict":"PASS"}')
    envelope = tmp_path / "envelope.json"
    envelope.write_text("{}")
    with pytest.raises(RuntimeError, match="masks differ"):
        ATTRIBUTION.build_report(
            tmp_path / "lego", tmp_path / "nemo", admission, envelope)
