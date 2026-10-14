"""Direct tests for the SWG online reference-adequacy gate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST
from legoesm.ocean.fidelity.swg_coarse_graining import cgrid_geometry

_PATH = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity"
         / "testcases/nemo_testcase_l1_swg_online_reference_adequacy.py")
_SPEC = importlib.util.spec_from_file_location("swg_reference_adequacy", _PATH)
adequacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adequacy)
_COARSE_TRUTH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/"
    "conservative_truth_dg50_1095d/snapshots")


def _metrics(scale=1.0):
    return {
        "basin_mean_ke_m2_s2": np.asarray([1.0, 1.0]) * scale,
        "basin_mean_eke_m2_s2": np.asarray([0.5, 0.5]) * scale,
        "jet_y_m": np.asarray([1.0e6, 1.1e6]),
        "spectrum_wavenumber_m-1": np.asarray([0.0, 1.0, 2.0]),
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


def test_early_common_path_reports_crossings_and_correlation_failure():
    wet = np.ones((4, 4), dtype=bool)
    y, x = np.mgrid[:4, :4]
    base = np.stack([np.sin(x + day) + np.cos(y - day)
                     for day in range(8)])
    truth = {
        "ssh": base,
        "relative_vorticity": 2.0 * base,
        "rotational_velocity": np.stack((base, -base), axis=1),
    }
    identity = adequacy.early_path_metrics(truth, truth, wet)
    assert identity["adequate"]
    planted = {name: value.copy() for name, value in truth.items()}
    planted["ssh"][2:] *= -1.0
    failed = adequacy.early_path_metrics(planted, truth, wet)
    assert not failed["adequate"]
    assert failed["fields"]["ssh"]["first_day_above"]["0.1"] == 3


def test_early_loader_uses_preregistered_rotational_velocity(monkeypatch,
                                                             tmp_path):
    shape = cgrid_geometry(SWG_DG50_NAMELIST)["mask"]["T"].shape
    hashes = {}
    for day in range(1, 9):
        path = tmp_path / f"day{day:04d}.npz"
        np.savez(path, ssh_m=np.zeros(shape), u_m_s=np.zeros(shape),
                 v_m_s=np.zeros(shape))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "complete", "daily_sha256": hashes,
    }))
    calls = []

    def rotational(zeta, mask, spacing):
        calls.append((zeta.shape, mask.shape, spacing))
        return np.full(shape, 7.0), np.full(shape, 8.0)

    monkeypatch.setattr(adequacy, "rotational_velocity_from_vorticity", rotational)
    fields = adequacy._load_early_fields(tmp_path)
    assert len(calls) == 8
    assert set(fields) == {"ssh", "relative_vorticity", "rotational_velocity"}
    assert np.all(fields["rotational_velocity"][:, 0] == 7.0)
    assert np.all(fields["rotational_velocity"][:, 1] == 8.0)


def test_reference_cli_exits_nonzero_on_inadequacy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        adequacy, "compare_references",
        lambda *args, **kwargs: {"adequate": False})
    assert adequacy.main([
        "--candidate-root", str(tmp_path / "a"),
        "--sensitivity-root", str(tmp_path / "b"),
        "--years", "1", "--json", str(tmp_path / "report.json")]) == 1


def test_spectrum_error_keeps_the_gravest_sine_mode():
    """The sine-transform spectrum has no k=0 bin; its first bin is a mode."""
    candidate = dict(_metrics(), **{
        "spectrum_wavenumber_m-1": np.asarray([1.0, 2.0, 3.0]),
        "mean_kinetic_energy_spectrum_m3_s2": np.asarray([3.0, 2.0, 1.0])})
    sensitivity = dict(candidate, **{
        "mean_kinetic_energy_spectrum_m3_s2": np.asarray([6.0, 2.0, 1.0])})
    scores = adequacy.adequacy_metrics(
        candidate, sensitivity, basin_width_m=4.0e6)["scores"]
    assert scores["ke_spectrum_relative_l1_error"] == 3.0 / 9.0
    with pytest.raises(ValueError, match="different wavenumber bins"):
        adequacy.adequacy_metrics(
            candidate, dict(sensitivity, **{
                "spectrum_wavenumber_m-1": np.asarray([0.0, 1.0, 2.0])}),
            basin_width_m=4.0e6)


def test_compare_references_selects_online_dg50_diagnostics(monkeypatch,
                                                           tmp_path):
    """Review EV-02: the registered default reads 6.25 km coordinates."""
    calls = []

    class StopError(Exception):
        pass

    def spy(provider, **kwargs):
        calls.append(kwargs)
        raise StopError

    monkeypatch.setattr(adequacy, "filtered_npz_provider",
                        lambda *args, **kwargs: object())
    monkeypatch.setattr(adequacy, "diagnose", spy)
    with pytest.raises(StopError):
        adequacy.compare_references(tmp_path, tmp_path, years=1)
    assert calls[0]["diagnostics"] == "online"
    assert calls[0]["namelist"] is SWG_DG50_NAMELIST


def test_compare_references_refuses_mixed_snapshot_and_daily_mean(monkeypatch,
                                                                 tmp_path):
    """A daily-mean sensitivity is a different quantity from snapshot truth."""
    from types import SimpleNamespace

    def provider(root, days, *, start_day, name):
        return SimpleNamespace(sampling="daily_mean" if "sensitivity" in name
                               else "daily_snapshot")

    monkeypatch.setattr(adequacy, "filtered_npz_provider", provider)
    monkeypatch.setattr(adequacy, "diagnose", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="sampling mismatch"):
        adequacy.compare_references(tmp_path, tmp_path, years=1)


@pytest.mark.skipif(not (_COARSE_TRUTH / "manifest.json").is_file(),
                    reason="registered dg50 coarse truth not mounted")
def test_reference_cli_fails_a_planted_500_km_jet_shift(tmp_path):
    """Review EV-02 end to end: the tip reported this plant adequate, exit 0.

    The sensitivity is the real coarse truth with the whole flow moved ten
    50 km rows north after day 8, so the early common path is identical and
    only the year-1 jet (0.125 basin widths away) can fail.
    """
    wet_rows = np.flatnonzero(
        cgrid_geometry(SWG_DG50_NAMELIST)["mask"]["T"].any(axis=1))
    south, north = int(wet_rows[0]), int(wet_rows[-1])
    shift = 10
    sensitivity = tmp_path / "shifted"
    sensitivity.mkdir()
    manifest = json.loads((_COARSE_TRUTH / "manifest.json").read_text())
    pins = {}
    for day in range(1, adequacy.DAYS_PER_YEAR + 1):
        with np.load(_COARSE_TRUTH / f"day{day:04d}.npz") as saved:
            arrays = {name: np.asarray(saved[name]) for name in saved.files}
        if day > adequacy.EARLY_DAYS:
            for name in ("ssh_m", "u_m_s", "v_m_s"):
                field = arrays[name]
                moved = field.copy()
                moved[south + shift:north + 1] = field[south:north + 1 - shift]
                moved[south:south + shift] = field[south]
                arrays[name] = moved
            arrays["v_m_s"][north] = 0.0  # the north wall face stays closed
        path = sensitivity / f"day{day:04d}.npz"
        np.savez(path, **arrays)
        pins[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.update(days=adequacy.DAYS_PER_YEAR,
                    completed_days=adequacy.DAYS_PER_YEAR,
                    output_daily_sha256=pins)
    (sensitivity / "manifest.json").write_text(json.dumps(manifest))
    report_path = tmp_path / "report.json"
    code = adequacy.main([
        "--candidate-root", str(_COARSE_TRUTH),
        "--sensitivity-root", str(sensitivity),
        "--years", "1", "--json", str(report_path)])
    report = json.loads(report_path.read_text())
    assert report["early_common_path"]["adequate"]
    annual = report["year1_statistics"]
    jet = "jet_position_rmse_over_basin_width"
    assert annual["scores"][jet] > annual["bars"][jet]
    assert not annual["passes"][jet]
    assert code == 1
