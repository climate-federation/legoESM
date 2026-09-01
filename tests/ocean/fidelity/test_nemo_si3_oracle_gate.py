"""Direct and planted controls for the SI3 lane-3 oracle gate."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest

GATE_PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/nemo_si3_oracle_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_oracle_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _minimal_run(root: Path, *, nx: int = 59, ny: int = 59, nz: int = 2) -> None:
    with netCDF4.Dataset(root / "mesh_mask.nc", "w") as ds:
        ds.createDimension("t", 1)
        ds.createDimension("x", nx)
        ds.createDimension("y", ny)
        ds.createDimension("nav_lev", nz)
        ds.createVariable("e1t", "f8", ("t", "y", "x"))[:] = 4.0
    with netCDF4.Dataset(root / "CASE_00000040_restart_ice.nc", "w") as ds:
        ds.createDimension("x", 1)
        ds.createVariable("state", "f8", ("x",))[:] = 1.0
    (root / "output.namelist.dyn").write_text("&namrun\n nn_itend = 40\n/\n")
    (root / "output.namelist.ice").write_text(
        "&nampar\n nlay_i = 3\n nlay_s = 3\n/\n"
        "&namthd_sal\n nn_icesal = 4\n/\n"
        "&namthd_pnd\n ln_pnd = .true.\n/\n"
    )
    (root / "ocean.output").write_text("run complete\n")


def test_frame_registry_is_complete_unique_and_sourced() -> None:
    assert len(gate.FRAME_REGISTRY) == 19
    assert len({row[0] for row in gate.FRAME_REGISTRY}) == len(gate.FRAME_REGISTRY)
    assert all(
        row[2] in {"STEP_ENTRY_CURRENT", "CARRIED_PREVIOUS_STEP", "CARRIED_PREVIOUS_BEFORE_LEVEL"}
        for row in gate.FRAME_REGISTRY
    )
    assert all("icestp.F90:154-171" in row[3] for row in gate.FRAME_REGISTRY)


def test_frame_round_trip_uses_fp64_and_fortran_order(tmp_path: Path) -> None:
    dims = {"jpi": 5, "jpj": 6, "jpl": 2, "nlay_i": 3, "nlay_s": 3}
    expected: dict[str, np.ndarray] = {}
    path = tmp_path / "oracle_ice_step_entry_kt00000001.bin"
    with path.open("wb") as fh:
        fh.write(gate.MAGIC.encode().ljust(16))
        fh.write(struct.pack("=9i", 1, 1, 5, 6, 2, 3, 3, 64, len(gate.FRAME_REGISTRY)))
        offset = 0
        for name, axes, _, _ in gate.FRAME_REGISTRY:
            shape = tuple(dims[axis] for axis in axes)
            values = np.arange(offset, offset + int(np.prod(shape)), dtype=np.float64).reshape(
                shape, order="F"
            )
            expected[name] = values
            values.ravel(order="F").tofile(fh)
            offset += values.size

    header, actual = gate.read_frame(path)
    assert header["storage_bits"] == 64
    for name in expected:
        assert actual[name].dtype == np.float64
        assert np.array_equal(actual[name], expected[name])


def test_restart_contract_disposes_active_and_inactive_appendix_a_fields() -> None:
    adv = gate.restart_contract("3.1", 3, 3, 4, True)
    rhg = gate.restart_contract("3.3", 3, 3, 4, True)
    assert adv["stress1_i"]["status"] == "WAIVED"
    assert rhg["stress1_i"]["status"] == "VERIFIED"
    assert rhg["sxsi_l03"]["status"] == "VERIFIED"
    assert rhg["sxsal"]["status"] == "WAIVED"
    assert rhg["sxvl"]["status"] == "VERIFIED"
    assert rhg["cnd_ice"]["status"] == "WAIVED"
    assert rhg["t_s_l03"]["status"] == "WAIVED"


def test_planted_unaccounted_mesh_array_goes_red(tmp_path: Path) -> None:
    _minimal_run(tmp_path)
    manifest = gate.disposition_template(tmp_path, "3.1")
    with pytest.raises(
        gate.GateError,
        match=r"missing=\['PLANTED_UNACCOUNTED_FILE_ARRAY'\].*extra=\[\]",
    ):
        gate.check_manifest(tmp_path, "3.1", manifest, plant_unaccounted=True)


def test_planted_geometry_field_perturbation_goes_red(tmp_path: Path) -> None:
    _minimal_run(tmp_path)
    with netCDF4.Dataset(tmp_path / "mesh_mask.nc", "a") as ds:
        for name in ("e1u", "e1v", "e1f", "e2t", "e2u", "e2v", "e2f"):
            ds.createVariable(name, "f8", ("t", "y", "x"))[:] = 4.0
        for name in ("ff_t", "ff_f"):
            ds.createVariable(name, "f8", ("t", "y", "x"))[:] = 0.0
        ds.createDimension("z", 1)
        for name in ("tmask", "umask", "vmask"):
            ds.createVariable(name, "i1", ("t", "z", "y", "x"))[:] = 1
    with pytest.raises(gate.GateError, match="e1t metric"):
        gate.geometry(tmp_path, "3.1", plant_field=True)


def test_manifest_contract_omission_and_unmeasured_mesh_go_red(tmp_path: Path) -> None:
    _minimal_run(tmp_path)
    manifest = gate.disposition_template(tmp_path, "3.1")
    manifest["restart_contract"].pop("t_s_l01")
    with pytest.raises(gate.GateError, match="exhaustive Appendix-A contract"):
        gate.check_manifest(tmp_path, "3.1", manifest)

    manifest = gate.disposition_template(tmp_path, "3.1")
    manifest["entries"]["mesh"]["e1t"]["status"] = "UNMEASURED"
    with pytest.raises(gate.GateError, match="UNMEASURED is forbidden"):
        gate.check_manifest(tmp_path, "3.1", manifest)


def test_phenomenology_maximum_preservation_control_goes_red() -> None:
    shape3 = (9, 9, 1)
    final_shape3 = (5, 5, 1)
    final_shape2 = (5, 5)
    initial = {
        "a_i": np.ones(shape3),
        "v_i": np.ones(shape3),
        "u_ice": np.zeros((9, 9)),
    }
    final = {
        "a_i": np.full(final_shape3, 0.5),
        "v_i": np.ones(final_shape3),
        "u_ice": np.zeros(final_shape2),
    }
    verdict = gate.phenomenology("3.2", initial, final)
    assert verdict["status"] == "REFUTE"
    claims = verdict["refuted_predicates"]
    assert any("maximum-concentration preservation refuted" in row for row in claims)


def test_native_conservation_violation_is_refuted(tmp_path: Path) -> None:
    _minimal_run(tmp_path)
    assert gate.conservation_diagnostics(tmp_path)["status"] == "CONFIRM"
    (tmp_path / "ocean.output").write_text("icedyn_adv : violation v_i < 0 = -1\n")
    verdict = gate.conservation_diagnostics(tmp_path)
    assert verdict["status"] == "REFUTE"
    assert verdict["violation_count"] == 1


RUN_ROOTS = {
    "3.1": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv1d/final"),
        Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/ICE_ADV1D_OMIP_L3"),
        "ice_adv1d_l3",
    ),
    "3.2": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d/final"),
        Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/ICE_ADV2D_OMIP_L3"),
        "ice_adv2d_l3",
    ),
    "3.3": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final"),
        Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/ICE_ADV2D_RHG_OMIP_L3"),
        "ice_adv2d_rhg_l3",
    ),
}
MANIFEST_DIR = GATE_PATH.parent / "manifests"


def _cli(rung: str, *extra: str) -> subprocess.CompletedProcess[str]:
    run_dir, case_dir, manifest = RUN_ROOTS[rung]
    return subprocess.run(
        [
            sys.executable,
            str(GATE_PATH),
            "--run-dir",
            str(run_dir),
            "--case-dir",
            str(case_dir),
            "--rung",
            rung,
            "--manifest",
            str(MANIFEST_DIR / f"{manifest}.json"),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("rung", sorted(RUN_ROOTS))
@pytest.mark.parametrize("control", ["--plant-unaccounted", "--plant-field"])
def test_cli_planted_controls_exit_nonzero(rung: str, control: str) -> None:
    """End-to-end: each preregistered plant must make the shipped CLI exit red."""
    run_dir, case_dir, _ = RUN_ROOTS[rung]
    if not run_dir.is_dir() or not case_dir.is_dir():
        pytest.skip(f"oracle run root absent: {run_dir}")
    assert _cli(rung, control).returncode != 0


def test_cli_clean_rung_is_green_and_refuted_rungs_are_red() -> None:
    """The controls only prove anything if the unplanted arms are not always red."""
    for rung, expected_green in (("3.3", True), ("3.1", False), ("3.2", False)):
        run_dir, case_dir, _ = RUN_ROOTS[rung]
        if not run_dir.is_dir() or not case_dir.is_dir():
            pytest.skip(f"oracle run root absent: {run_dir}")
        result = _cli(rung)
        assert (result.returncode == 0) is expected_green, result.stderr[-400:]
        payload = json.loads(result.stdout)
        assert payload["status"] == ("VERIFIED" if expected_green else "DEBT")
