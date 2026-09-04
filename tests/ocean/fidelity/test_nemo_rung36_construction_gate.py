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


def _fixture(root: Path, *, e3: float = -2.0 / 3.0) -> None:
    values = np.array(
        [
            0.0,
            0.0,
            -1.690032958984375,
            34.0,
            e3 - 1.0,
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
    (root / "ocean.output").write_text(
        "kt 36 |ssh| max 1.767 at\nkt 36 |U| max 1.1436E+06 at\nkt 36 |V| max 1.2422E+06 at\n",
        encoding="utf-8",
    )


def test_real_failure_is_the_first_boundary(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = gate.evaluate(tmp_path)
    assert result["verdict"] == "CONSTRUCTION_DEBT"
    assert result["ordered_stop"] == "INITIAL_STATE.positive_wet_layer_thickness"
    assert result["run_abort"]["step"] == 36


def test_positive_geometry_passes_registered_rows(tmp_path: Path) -> None:
    _fixture(tmp_path, e3=4.0)
    result = gate.evaluate(tmp_path)
    assert result["verdict"] == "AT_BAR"
    assert all(row["status"] == "AT_BAR" for row in result["rows"])


def test_each_plant_changes_its_scored_row(tmp_path: Path) -> None:
    _fixture(tmp_path)
    for plant in ("temperature", "salinity", "geometry"):
        result = gate.evaluate(tmp_path, plant=plant)
        assert result["plant_binding"]["changed"] is True
        assert result["plant_binding"]["before_bits"] != result["plant_binding"]["after_bits"]
