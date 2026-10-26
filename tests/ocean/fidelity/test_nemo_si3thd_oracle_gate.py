"""Controls for the C1D_OMIP_L3 SI3 thermodynamics oracle gate."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import netCDF4
import numpy as np
import pytest


GATE_PATH = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_oracle_gate.py"
MY_SRC = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC"
SPEC = importlib.util.spec_from_file_location("nemo_si3thd_oracle_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _write_thd(path: Path, nsteps: int = 48) -> None:
    nx = ny = nc = 1
    ni = ns = 3
    with path.open("wb") as handle:
        for kt in range(1, nsteps + 1):
            thickness = 2.0 + (0.01 * kt if kt <= 24 else 0.48 - 0.01 * kt)
            for stage in range(8):
                payload = 0 if stage in {0, 6, 7} else 1
                npti = 1
                handle.write(b"NEMO_L3THD_001  ")
                handle.write(struct.pack("=11i", 1, kt, stage, payload, nx, ny, nc, ni, ns, npti, 64))
                if payload == 0:
                    data = np.ones(18, dtype=np.float64)
                    data[0] = 1.0
                    data[1] = thickness
                else:
                    data = np.ones(13, dtype=np.float64)
                data.tofile(handle)


def _write_exchange(path: Path, nsteps: int = 48) -> None:
    with path.open("wb") as handle:
        for kt in range(1, nsteps + 1):
            handle.write(b"NEMO_L3XCHG_001 ")
            handle.write(struct.pack("=6i", 1, kt, 1, 1, 1, 64))
            data = np.zeros(36, dtype=np.float64)
            data[-1] = 0.9
            data.tofile(handle)


def _write_restart(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 1)
        for name in sorted(gate.REQUIRED_RESTART):
            ds.createVariable(name, "f8", ("x",))[:] = 1.0


@pytest.mark.parametrize("token,wanted", [(".true.", True), ("F", False)])
def test_fortran_logical(token: str, wanted: bool) -> None:
    assert gate.parse_logical(token) is wanted


def test_registered_thermo_stream_and_planted_thickness(tmp_path: Path) -> None:
    path = tmp_path / "thermo.bin"
    _write_thd(path)
    _, result = gate.read_thd_frames(path, nsteps=48)
    assert result["frames"] == 384
    with pytest.raises(gate.GateError, match="HFN thickness bound"):
        gate.read_thd_frames(path, plant=True, nsteps=48)


def test_exchange_stream_and_planted_violation(tmp_path: Path) -> None:
    path = tmp_path / "exchange.bin"
    _write_exchange(path)
    assert gate.read_exchange_frames(path, nsteps=48)["frames"] == 48
    with pytest.raises(gate.GateError, match="exchange fr_i bound"):
        gate.read_exchange_frames(path, plant=True, nsteps=48)


def test_restart_enumeration_and_planted_unaccounted(tmp_path: Path) -> None:
    path = tmp_path / "restart_ice.nc"
    _write_restart(path)
    assert set(gate.check_restart(path)["verified"]) == gate.REQUIRED_RESTART
    with pytest.raises(gate.GateError, match="PLANTED_UNACCOUNTED_RESTART_ARRAY"):
        gate.check_restart(path, plant=True)


def test_restart_missing_required_fails(tmp_path: Path) -> None:
    path = tmp_path / "restart_ice.nc"
    _write_restart(path)
    with netCDF4.Dataset(path, "a") as ds:
        ds.renameVariable("v_i", "not_v_i")
    with pytest.raises(gate.GateError, match="restart required missing"):
        gate.check_restart(path)


def test_completion_log_requires_documented_end_and_rejects_error(tmp_path: Path) -> None:
    (tmp_path / "ocean.output").write_text(
        "fld_read: Y/M/D = 2018/12/31\n"
        "ice_rst_write : write ice restart file  kt =        8760\n"
    )
    (tmp_path / "run.stdout").write_text("")
    (tmp_path / "run.stderr").write_text(
        "Note: The following floating-point exceptions are signalling: "
        "IEEE_UNDERFLOW_FLAG IEEE_DENORMAL\nSTOP 0\n"
    )
    result = gate.check_completion_logs(tmp_path)
    assert result["nonfatal_ieee_flags"] == ["IEEE_UNDERFLOW_FLAG", "IEEE_DENORMAL"]
    (tmp_path / "ocean.output").write_text("E R R O R\n")
    with pytest.raises(gate.GateError, match="NEMO error block"):
        gate.check_completion_logs(tmp_path)


def test_oracle_streams_are_not_closed_and_replaced_at_nitend() -> None:
    """A repeated terminal call must append, not replace the full-year stream."""
    for source in (MY_SRC / "icethd.F90", MY_SRC / "icestp.F90"):
        text = source.read_text(encoding="utf-8")
        assert "STATUS='REPLACE'" in text
        assert "CLOSE(num_l3" not in text
        assert "IF( num_l3" not in text
        assert "IF( .NOT. ll_l3" in text
