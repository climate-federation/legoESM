"""Direct tests for the ORCA2 round-345 Decision-188 tools."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round345_decision188_gate as decision188,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round345_sst_ulp_gate as ulp_gate,
)


def _pilot_report(*, ke_rms: float) -> dict:
    monthly = {
        key: {"bias": 0.0, "rms": 0.0}
        for key in decision188.FIXED_RMS_BARS
    }
    monthly.update({
        "KE": {"bias": 0.0, "rms": ke_rms},
        "PSI": {"bias": 0.0, "rms": 0.0, "pattern_corr": 1.0},
        "DRAKE": {
            "nemo_mean_sv": 100.0, "cand_mean_sv": 100.0,
            "nemo_std_sv": 0.0, "cand_std_sv": 0.0,
        },
    })
    return {
        "format": "orca2x-r1-pilot-metrics-v1",
        "monthly": monthly,
        "daily": [
            {"day": day, "KE": {"nemo_mean": 0.1}}
            for day in range(1, 31)
        ],
    }


def test_decision188_plants_fire() -> None:
    assert decision188.run_plants() == {
        "fixed_bar_plant": "PASS",
        "ke_inside_3x_envelope_plant": "PASS",
        "ke_floor_plant": "PASS",
        "zero_envelope_refusal_plant": "PASS",
        "partial_zero_envelope_refusal_plant": "PASS",
        "nonperturbed_envelope_refusal_plant": "PASS",
        "missing_unperturbed_control_refusal_plant": "PASS",
    }


def test_decision188_runs_on_orca2x_30_day_pilot_schema() -> None:
    result = decision188.evaluate_decision188(
        _pilot_report(ke_rms=0.002),
        _pilot_report(ke_rms=0.0007),
        decision188.minimal_envelope_admission(),
    )
    assert result["verdict"] == "PASS"
    assert result["months_evaluated"] == 1
    assert result["months"][0]["label"] == "PILOT_DAYS_1_30"
    assert result["months"][0]["status"] == "EVALUATED_PARTIAL_MONTH"
    assert result["months"][0]["metrics"]["KE"]["bar_fraction"] == pytest.approx(0.021)


def test_decision188_refuses_different_month_windows() -> None:
    candidate = decision188.minimal_year_report(sst=0.0, ke=0.0)
    envelope = decision188.minimal_year_report(sst=0.0, ke=0.0)
    envelope["months"][1]["metrics"]["KE"]["relative_rms_fraction"] = 0.001
    envelope["months"][0]["saved_kts"] = [16]
    with pytest.raises(decision188.GateError, match="saved steps differ"):
        decision188.evaluate_decision188(
            candidate, envelope, decision188.minimal_envelope_admission())


def test_decision188_refuses_zero_envelope_in_any_month() -> None:
    candidate = decision188.minimal_year_report(sst=0.0, ke=0.0)
    envelope = decision188.minimal_year_report(sst=0.0, ke=0.001)
    envelope["months"][7]["metrics"]["KE"]["relative_rms_fraction"] = 0.0
    with pytest.raises(decision188.GateError, match="must be > 0"):
        decision188.evaluate_decision188(
            candidate, envelope, decision188.minimal_envelope_admission())


def test_decision188_refuses_missing_restart_control() -> None:
    admission = decision188.minimal_envelope_admission()
    admission.pop("unperturbed_restart_control")
    with pytest.raises(decision188.GateError, match="restart-continuation control"):
        decision188.evaluate_decision188(
            decision188.minimal_year_report(sst=0.0, ke=0.0),
            decision188.minimal_year_report(sst=0.0, ke=0.001),
            admission,
        )


def _write_restart(path: Path, rank: int) -> np.ndarray:
    values = np.arange(2 * 2 * 3, dtype=np.float64).reshape(1, 2, 2, 3)
    values += np.float64(rank + 1)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.title = "synthetic restart"
        dataset.createDimension("time", 1)
        dataset.createDimension("z", 2)
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 3)
        variable = dataset.createVariable("tn", "f8", ("time", "z", "y", "x"))
        variable.long_name = "restart potential temperature"
        variable.units = "Celsius"
        variable[:] = values
        salinity = dataset.createVariable("sn", "f8", ("time", "z", "y", "x"))
        salinity[:] = values + 30.0
        kt = dataset.createVariable("kt", "f8")
        kt[...] = 8.0
    return values


def _write_domain(path: Path) -> np.ndarray:
    top = np.array([[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0]], dtype=np.int32)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 6)
        variable = dataset.createVariable("top_level", "i4", ("y", "x"))
        variable[:] = top
    return top > 0


def test_day1_restart_gets_exact_uniform_fp64_one_ulp_pattern(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    originals = []
    for rank in (0, 1):
        originals.append(_write_restart(
            source / f"ORCA2_00000008_restart_{rank:04d}.nc", rank
        ))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wet = _write_domain(run_dir / ulp_gate.DOMAIN_FILE)

    audit = ulp_gate.perturb_day1_restart_pair(run_dir, source)
    assert audit["changed_values"] == int(wet.sum())
    assert audit["exact_ulp_distance"] == 1
    assert ulp_gate.verify_perturbation_manifest(run_dir) == audit

    for rank in (0, 1):
        target = run_dir / f"ORCA2_00000008_restart_{rank:04d}.nc"
        with netCDF4.Dataset(target) as dataset:
            variable = dataset.variables["tn"]
            variable.set_auto_maskandscale(False)
            actual = np.asarray(variable[:], dtype=np.float64)
        expected = originals[rank].copy()
        rank_wet = wet[:, rank * 3:(rank + 1) * 3]
        expected[0, 0][rank_wet] = np.nextafter(expected[0, 0][rank_wet], np.inf)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    refused = tmp_path / "refused" / "retained"
    refused.parent.mkdir()
    shutil.move(run_dir, refused)
    assert ulp_gate.verify_perturbation_manifest(refused) == audit

    target = refused / "ORCA2_00000008_restart_0000.nc"
    with netCDF4.Dataset(target, "r+") as dataset:
        variable = dataset.variables["tn"]
        variable.set_auto_maskandscale(False)
        surface = np.asarray(variable[0, 0], dtype=np.float64)
        j, i = np.argwhere(wet[:, :3])[0]
        surface[j, i] = np.nextafter(surface[j, i], np.inf)
        variable[0, 0] = surface
    with pytest.raises(ulp_gate.GateError, match=r"exact \+1-fp64-ULP pattern"):
        ulp_gate.verify_perturbation_manifest(refused)


def test_unperturbed_restart_control_requires_every_daily_file_exact(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    control = tmp_path / "control"
    baseline.mkdir()
    control.mkdir()
    for kt in (8, 16):
        for rank in (0, 1):
            name = f"ORCA2_{kt:08d}_restart_{rank:04d}.nc"
            payload = f"restart {kt} rank {rank}\n".encode()
            (baseline / name).write_bytes(payload)
            (control / name).write_bytes(payload)
    result = ulp_gate.verify_restart_continuation_control(
        control, baseline_arm=baseline, expected_steps=16
    )
    assert result["all_daily_restarts_byte_identical"] is True
    assert result["restart_files_compared"] == 4
    (control / "ORCA2_00000016_restart_0001.nc").write_bytes(b"drift\n")
    with pytest.raises(ulp_gate.GateError, match="differs from arm A"):
        ulp_gate.verify_restart_continuation_control(
            control, baseline_arm=baseline, expected_steps=16
        )


def test_round345_document_git_shas_resolve_and_are_full_length() -> None:
    repo = Path(__file__).resolve().parents[3]
    round342_repo = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r342/repo"
    )
    docs = [
        repo / "docs/ocean/fidelity/testcases" /
        "PREREG_nemo_testcases_l4_orca2_round345_year_comparison_gate.md",
        repo / "docs/ocean/fidelity/testcases" /
        "nemo_testcases_l4_orca2_round345_year_comparison_gate_receipt.md",
    ]
    short_or_full = re.compile(r"`([0-9a-f]{7,40})`")
    for document in docs:
        shas = short_or_full.findall(document.read_text(encoding="utf-8"))
        assert shas, f"no git SHA found in {document}"
        assert all(len(sha) == 40 for sha in shas), f"non-full git SHA in {document}: {shas}"
        for sha in shas:
            local = subprocess.run(
                ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"], check=False
            )
            external = subprocess.run(
                ["git", "-C", str(round342_repo), "cat-file", "-e", f"{sha}^{{commit}}"],
                check=False,
            )
            assert local.returncode == 0 or external.returncode == 0, sha


def test_round345_document_file_shas_rehash(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    baseline = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/"
        "s4_year/orca2_k0_year1_a_np2"
    )
    perturbed = tmp_path / "perturbed"
    perturbed.mkdir()
    shutil.copy2(baseline / ulp_gate.DOMAIN_FILE, perturbed / ulp_gate.DOMAIN_FILE)
    ulp_gate.perturb_day1_restart_pair(perturbed, baseline)
    expected = {
        "835f03b99a3c96e80d23f35c415d099fde292678ae30572e9c5412be4b67e16d":
            baseline / "nemo",
        "3fbed2281e66f7d51cd34247e799d1b76c8a102d36d5b64c0a3b80aba56dd2e3":
            baseline / "ORCA2_00000008_restart_0000.nc",
        "a5660f4c44b70a7cf7d70500b2df8e2d2145f08a205b266301c7312517413023":
            baseline / "ORCA2_00000008_restart_0001.nc",
        "bb25179365e776ae73e6ead9d2bad42e222f2094f687c22ede3b0eb3bcae1f01":
            perturbed / "ORCA2_00000008_restart_0000.nc",
        "0279bb4021ef1b557e178ef01b5c4cac732479102404871b557b984433fff8cf":
            perturbed / "ORCA2_00000008_restart_0001.nc",
    }
    documents = (
        repo / "docs/ocean/fidelity/testcases" /
        "PREREG_nemo_testcases_l4_orca2_round345_year_comparison_gate.md",
        repo / "docs/ocean/fidelity/testcases" /
        "nemo_testcases_l4_orca2_round345_year_comparison_gate_receipt.md",
    )
    for document in documents:
        cited = set(re.findall(r"`([0-9a-f]{64})`", document.read_text(encoding="utf-8")))
        assert cited == set(expected), (document, cited)
        for digest in cited:
            assert hashlib.sha256(expected[digest].read_bytes()).hexdigest() == digest
