"""Direct tests for the SWG online reference-adequacy gate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST
from legoesm.ocean.fidelity.swg_coarse_graining import cgrid_geometry

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
