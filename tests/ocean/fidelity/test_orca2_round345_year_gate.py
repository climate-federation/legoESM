"""Direct tests for the ORCA2 round-345 Decision-188 tools."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
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


def _write_restart(
    path: Path,
    rank: int,
    *,
    kt_value: int = 8,
    include_delay_buffers: bool = False,
) -> np.ndarray:
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
        for name, offset in (("un", 40.0), ("vn", 50.0), ("sshn", 60.0)):
            state = dataset.createVariable(name, "f8", ("time", "z", "y", "x"))
            state[:] = values + offset
        for index in range(14):
            state = dataset.createVariable(
                f"synthetic_state_{index:02d}", "f8", ("time", "z", "y", "x")
            )
            state[:] = values + 100.0 + index
        kt = dataset.createVariable("kt", "f8")
        kt[...] = float(kt_value)
        if include_delay_buffers:
            dataset.createDimension("delay", 2)
            dataset.createVariable("DELAY_re8_fwb2", "f8", ("delay",))[:] = [1.0, 2.0]
            dataset.createVariable("DELAY_im8_fwb2", "f8", ("delay",))[:] = [3.0, 4.0]
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


def test_unperturbed_restart_control_requires_every_common_variable_data_exact(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    control = tmp_path / "control"
    baseline.mkdir()
    control.mkdir()
    for rank in (0, 1):
        name = f"ORCA2_00000016_restart_{rank:04d}.nc"
        _write_restart(
            baseline / name, rank, kt_value=16, include_delay_buffers=True
        )
        _write_restart(control / name, rank, kt_value=16)
    result = ulp_gate.verify_restart_continuation_control(
        control, baseline_arm=baseline, expected_steps=16
    )
    assert result["all_common_variable_data_bitwise_identical"] is True
    assert result["restart_files_compared"] == 2
    assert result["common_variable_data_comparisons"] == 40
    assert result["required_arm_a_only_variables"] == [
        "DELAY_im8_fwb2", "DELAY_re8_fwb2",
    ]

    target = control / "ORCA2_00000016_restart_0001.nc"
    pristine = tmp_path / "control_pristine.nc"
    shutil.copy2(target, pristine)
    with netCDF4.Dataset(target, "r+") as dataset:
        variable = dataset.variables["tn"]
        variable.set_auto_maskandscale(False)
        values = np.asarray(variable[:])
        values.view(np.uint8).reshape(-1)[0] ^= np.uint8(1)
        variable[:] = values
    with pytest.raises(ulp_gate.GateError, match="tn restart variable data differs"):
        ulp_gate.verify_restart_continuation_control(
            control, baseline_arm=baseline, expected_steps=16
        )
    shutil.copy2(pristine, target)

    with netCDF4.Dataset(target, "r+") as dataset:
        dataset.createVariable("unknown_restart_state", "f8").assignValue(1.0)
    with pytest.raises(ulp_gate.GateError, match="control_only=.*unknown_restart_state"):
        ulp_gate.verify_restart_continuation_control(
            control, baseline_arm=baseline, expected_steps=16
        )
    shutil.copy2(pristine, target)

    ulp_gate._copy_netcdf_excluding_variables(pristine, target, {"sn"})
    with pytest.raises(ulp_gate.GateError, match="arm_A_only=.*sn"):
        ulp_gate.verify_restart_continuation_control(
            control, baseline_arm=baseline, expected_steps=16
        )


def test_perturbed_restart_outputs_use_data_and_exact_variable_set(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    perturbed = tmp_path / "perturbed"
    baseline.mkdir()
    perturbed.mkdir()
    for rank in (0, 1):
        name = f"ORCA2_00000016_restart_{rank:04d}.nc"
        _write_restart(baseline / name, rank, kt_value=16, include_delay_buffers=True)
        _write_restart(perturbed / name, rank, kt_value=16)
    result = ulp_gate.verify_perturbed_restart_outputs(
        perturbed, baseline_arm=baseline, expected_steps=16
    )
    assert result["restart_files_validated"] == 2
    assert result["required_arm_a_only_variables"] == [
        "DELAY_im8_fwb2", "DELAY_re8_fwb2",
    ]

    target = perturbed / "ORCA2_00000016_restart_0001.nc"
    with netCDF4.Dataset(target, "r+") as dataset:
        variable = dataset.variables["tn"]
        values = np.asarray(variable[:])
        values.reshape(-1)[0] = np.nan
        variable[:] = values
    with pytest.raises(ulp_gate.GateError, match="tn contains non-finite values"):
        ulp_gate.verify_perturbed_restart_outputs(
            perturbed, baseline_arm=baseline, expected_steps=16
        )


def _stub_rank_env(name: str, rank: int) -> dict[str, str]:
    env = os.environ.copy()
    for rank_name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK"):
        env.pop(rank_name, None)
    env[name] = str(rank)
    return env


def test_stub_pipeline_uses_byte_identical_executables(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    run_sh = repo / (
        "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round345_sst_ulp_acquisition/run.sh"
    )
    stub_root = tmp_path / "stub_round345"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout = fake_bin / "timeout"
    timeout.write_text(
        "#!/usr/bin/env bash\n"
        "OMPI_COMM_WORLD_RANK=0 ./nemo && OMPI_COMM_WORLD_RANK=1 ./nemo\n",
        encoding="utf-8",
    )
    timeout.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        R345_CHECK_REPO_ROOT=str(repo),
        R345_REQUIRED_RAM_GIB="0",
        R345_REQUIRED_DISK_GIB="0",
    )
    result = subprocess.run(
        ["bash", str(run_sh), "--stub-run-dir", str(stub_root), "--stub-steps", "16"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    control = stub_root / ulp_gate.CONTROL_ARM_NAME / "nemo"
    perturbed = stub_root / ulp_gate.PERTURBED_ARM_NAME / "nemo"
    assert control.read_bytes() == perturbed.read_bytes()


@pytest.mark.parametrize("control", [True, False], ids=["control", "perturbed"])
def test_restart_stub_two_concurrent_rank_processes_preserve_input_and_ownership(
    tmp_path: Path,
    control: bool,
) -> None:
    """Exercise the generated stub executable without relying on mpirun."""
    run_dir = tmp_path / ("control" if control else "perturbed")
    run_dir.mkdir()
    stage = ulp_gate.stage_control_arm if control else ulp_gate.stage_perturbed_arm
    stage(run_dir, expected_steps=16)

    gate_path = Path(ulp_gate.__file__).resolve()
    stub = run_dir / "nemo"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(gate_path))} "
        "--act-as-nemo-binary\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    input_paths = [
        run_dir / f"ORCA2_{ulp_gate.RESTART_KT:08d}_restart_{rank:04d}.nc"
        for rank in (0, 1)
    ]
    input_before = [ulp_gate.common.sha256(path) for path in input_paths]

    # Rank 1 alone must not touch rank-0 or shared outputs.  This isolates file
    # ownership before the two actual stub processes are launched concurrently.
    shared_or_rank0 = [
        run_dir / "ocean.output",
        run_dir / "output.namelist.dyn",
        run_dir / "time.step",
        run_dir / "mesh_mask_0000.nc",
        run_dir / "ORCA2_00000016_restart_0000.nc",
    ]

    def snapshot(paths: list[Path]) -> dict[str, str | None]:
        return {
            path.name: ulp_gate.common.sha256(path) if path.is_file() else None
            for path in paths
        }

    before_rank1 = snapshot(shared_or_rank0)
    rank1_probe = subprocess.run(
        [str(stub)], cwd=run_dir,
        env=_stub_rank_env("PMIX_RANK", 1),
        capture_output=True, text=True, check=False,
    )
    assert rank1_probe.returncode == 0, rank1_probe.stderr
    assert rank1_probe.stderr == ""
    assert snapshot(shared_or_rank0) == before_rank1
    assert (run_dir / "mesh_mask_0001.nc").is_file()
    assert (run_dir / "ORCA2_00000016_restart_0001.nc").is_file()
    assert [ulp_gate.common.sha256(path) for path in input_paths] == input_before

    rank0 = subprocess.Popen(
        [str(stub)], cwd=run_dir,
        env=_stub_rank_env("OMPI_COMM_WORLD_RANK", 0),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    rank1 = subprocess.Popen(
        [str(stub)], cwd=run_dir,
        env=_stub_rank_env("PMIX_RANK", 1),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout0, stderr0 = rank0.communicate()
    stdout1, stderr1 = rank1.communicate()
    assert rank0.returncode == 0, stderr0
    assert rank1.returncode == 0, stderr1
    assert stderr0.strip() == "STOP 0"
    assert stderr1 == ""

    (run_dir / "mpirun.stdout.log").write_text(stdout0 + stdout1, encoding="utf-8")
    (run_dir / "mpirun.stderr.log").write_text(stderr0 + stderr1, encoding="utf-8")
    assert [ulp_gate.common.sha256(path) for path in input_paths] == input_before

    inventory = ulp_gate.r342.verify_no_unexpected_files(run_dir, expected_steps=16)
    assert inventory["expected_restart_file_count"] == 4
    completion = ulp_gate.r342.verify_nemo_completion_signal(run_dir, expected_steps=16)
    assert completion["time_step"] == 16
    expected_sizes = (
        ulp_gate.r342.R339_RESTART_0.stat().st_size,
        ulp_gate.r342.R339_RESTART_1.stat().st_size,
    )
    for kt in (8, 16):
        for local_rank, expected_size in enumerate(expected_sizes):
            if kt == 16:
                with netCDF4.Dataset(
                    run_dir / f"ORCA2_{kt:08d}_restart_{local_rank:04d}.nc"
                ) as dataset:
                    assert set(dataset.variables).isdisjoint(
                        ulp_gate.DELAY_RESTART_VARIABLES
                    )
                    assert int(np.asarray(dataset.variables["kt"][...])) == kt
                continue
            ulp_gate.r342._verify_single_restart_file(
                run_dir / f"ORCA2_{kt:08d}_restart_{local_rank:04d}.nc",
                kt,
                expected_size,
            )
    if control:
        continuation = ulp_gate.verify_restart_continuation_control(
            run_dir, expected_steps=16
        )
        assert continuation["all_common_variable_data_bitwise_identical"] is True
        assert continuation["restart_files_compared"] == 2
    else:
        continuation = ulp_gate.verify_perturbed_restart_outputs(
            run_dir, expected_steps=16
        )
        assert continuation["restart_files_validated"] == 2
        perturbation = ulp_gate.verify_perturbation_manifest(run_dir)
        assert perturbation["exact_ulp_distance"] == 1


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
