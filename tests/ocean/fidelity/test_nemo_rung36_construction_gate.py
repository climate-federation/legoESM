"""Unit and binding-control tests for the rung-3.6 construction stop."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np

GATE = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_rung36_construction_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_rung36_construction_gate", GATE)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _fixture(root: Path, *, bathy: float = 1.0) -> None:
    ssh = -5.0 / 3.0
    e3 = bathy + ssh
    values = np.array(
        [
            0.0,
            0.0,
            -1.690032958984375,
            34.0,
            ssh,
            e3,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    with (root / "oracle_rung36_ssm_frames.bin").open("wb") as stream:
        stream.write(gate.MAGIC)
        stream.write(struct.pack("=7i", 1, 1, 1, 1, 0, 15, 64))
        values.tofile(stream)
    # Use nm itself as a small valid dynamic-symbol input for the fixture.
    (root / "nemo.exe").symlink_to("/usr/bin/true")
    (root / "C1D_OMIP_L3_COUPLED_init_v2.nc").write_bytes(b"fixture")
    (root / "namelist_cfg").write_text(f"rn_bathy={bathy}\nrn_ssh_init=0.0\n", encoding="utf-8")
    (root / "ocean.output").write_text(
        "volumic mass of reference rho0 = 0.6 kg/m^3\n"
        "1. / rho0 r1_rho0 = 1.6666666666666667 m^3/kg\n"
        "kt 36 |ssh| max 1.767 at\nkt 36 |U| max 1.1436E+06 at\n"
        "kt 36 |V| max 1.2422E+06 at\n",
        encoding="utf-8",
    )
    # Canonical exchange stream with snwice_mass_b=1: displacement=1/0.6.
    layout, record_bytes = gate.exchange_gate._layout(5, 5, 1)
    data = np.zeros((record_bytes - gate.exchange_gate.HEADER_BYTES) // 8)
    cursor = 0
    for spec, _start, count in layout:
        if spec.name == "snwice_mass_b":
            data[cursor + 12] = 1.0
        cursor += count
    with (root / "oracle_si3_exchange_frames.bin").open("wb") as stream:
        stream.write(gate.exchange_gate.MAGIC)
        stream.write(struct.pack("=6i", 1, 1, 5, 5, 1, 64))
        data.tofile(stream)


def test_real_failure_is_the_first_boundary(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = gate.evaluate(tmp_path)
    assert result["verdict"] == "CONSTRUCTION_DEBT"
    assert result["ordered_stop"] == "INITIAL_STATE.positive_wet_layer_thickness"
    assert result["run_abort"]["step"] == 36


def test_positive_geometry_passes_registered_rows(tmp_path: Path) -> None:
    _fixture(tmp_path, bathy=5.0)
    result = gate.evaluate(tmp_path)
    assert result["verdict"] == "AT_BAR"
    assert all(row["status"] == "AT_BAR" for row in result["rows"])


def test_each_plant_changes_its_scored_row(tmp_path: Path) -> None:
    _fixture(tmp_path)
    for plant in ("temperature", "salinity", "geometry"):
        result = gate.evaluate(tmp_path, plant=plant)
        assert result["plant_binding"]["changed"] is True
        assert result["plant_binding"]["before_bits"] != result["plant_binding"]["after_bits"]
        expected_row = {
            "temperature": "temperature_Kmm",
            "salinity": "salinity_Kmm",
            "geometry": "e3t_source_expansion",
        }[plant]
        row = next(item for item in result["rows"] if item["name"] == expected_row)
        assert row["status"] == "DEBT"
