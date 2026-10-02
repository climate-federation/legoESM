"""Controls for the round-104 self-describing EEN operand record."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[3]
ACQ = ROOT / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round104_een_accum_acquisition"
REPAIR = ROOT / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round105_een_accum_acquisition"


def _gate():
    spec = importlib.util.spec_from_file_location("round104_gate", ACQ / "check_record.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_groups(handle, gate, values):
    for name, array in values.items():
        handle.write(name.encode("ascii").ljust(16, b" "))
        handle.write(gate.GROUP.pack(2, *array.shape, 1))
        handle.write(np.asfortranarray(array).tobytes(order="F"))


def _synthetic_tree(tmp_path: Path):
    gate = _gate()
    for rank, nimpp in ((0, 1), (1, 91)):
        shape = (90, 148)
        acc = {}
        final = {}
        scale = {}
        for index, (acc_name, scl_name, final_name) in enumerate(
                zip(gate.ACC, gate.SCL, gate.FINAL), start=1):
            value = np.zeros(shape, dtype=np.float64, order="F")
            value[1, 1] = index
            acc[acc_name] = value
            scale[scl_name] = np.ones(shape, dtype=np.float64, order="F")
            final[final_name] = value.copy(order="F")

        operand = tmp_path / f"oracle_r104_een_accum_rank{rank:04d}_kt00000001.bin"
        with operand.open("wb") as handle:
            handle.write(gate.MAGIC.encode("ascii").ljust(16, b" "))
            handle.write(gate.HEADER.pack(
                1, 1, rank, 94, 152, 31, nimpp, 1, 3, 3, 92, 150, 64, 16))
            _write_groups(handle, gate, acc | scale)

        final_path = tmp_path / f"oracle_r98_een_coeff_rank{rank:04d}_kt00000001.bin"
        with final_path.open("wb") as handle:
            handle.write(gate.FINAL_MAGIC.encode("ascii").ljust(16, b" "))
            handle.write(gate.FINAL_HEADER.pack(
                1, 1, 1, 3, rank, 94, 152, 31, nimpp, 1, 3, 3, 92, 150, 64, 8))
            _write_groups(handle, gate, final)

    for step in range(1, 11):
        for rank in range(2):
            (tmp_path / f"ORCA2_{step:08d}_restart_{rank:04d}.nc").write_bytes(
                struct.pack("=ii", step, rank))
    return gate


def test_synthetic_record_passes_and_every_plant_fires(tmp_path):
    gate = _synthetic_tree(tmp_path)
    result = gate.run(tmp_path, tmp_path, tmp_path, "none")
    assert result["status"] == "PASS_R104_EEN_ACCUM_ADMISSION"
    assert result["rank_coverage"] == "exactly-once"
    assert all(not any(row["reconstructed_final_bit_differences"].values())
               for row in result["records"])

    for plant in gate.PLANTS:
        if plant == "none":
            continue
        with pytest.raises(gate.Refusal, match="."):
            gate.run(tmp_path, tmp_path, tmp_path, plant)

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round98_coriolis_residual as residual,
    )
    assembled, census = residual.assemble_oracle_accumulators(tmp_path)
    assert census["coverage"] == "exactly-once"
    assert set(assembled) == set(gate.FIELDS)
    assert all(value.shape == (148, 180) for value in assembled.values())


def test_patch_is_additions_only_and_launcher_is_fail_closed():
    patch = (ACQ / "dynspg_ts_round104.patch").read_text(encoding="utf-8").splitlines()
    removed = [line for line in patch if line.startswith("-") and not line.startswith("---")]
    assert removed == []
    launcher = (ACQ / "run.sh").read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in launcher
    assert "REFUSE:" in launcher
    assert "/usr/bin/time" not in launcher


def test_round105_repair_initializes_one_absolute_rank_file_and_is_additions_only():
    patch = (REPAIR / "dynspg_ts_round105.patch").read_text(
        encoding="utf-8").splitlines()
    removed = [line for line in patch if line.startswith("-") and not line.startswith("---")]
    assert removed == []

    writer = (REPAIR / "l4_r105_een_accum.F90").read_text(encoding="utf-8")
    assert writer.count("GET_ENVIRONMENT_VARIABLE('ORCA2_R105_EEN_ACCUM_DIR'") == 1
    assert writer.count("output_dir(1:1) /= '/'") == 1
    assert writer.count("IF(dumped) RETURN") == 1
    assert writer.count("STATUS='NEW'") == 1
    assert writer.index("STATUS='NEW'") < writer.index("SUBROUTINE r105_een_accum_dump")
    assert writer.count('rank",I4.4') == 1

    launcher = (REPAIR / "run.sh").read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in launcher
    assert "REFUSE:" in launcher
    assert "export ORCA2_R105_EEN_ACCUM_DIR=$TARGET_RUN" in launcher
    assert "--plant-path" in launcher
    assert "--plant-duplicate" in launcher
    assert "--plant-rank-log" in launcher
    assert '"$root/ocean.output" "$root/run.user.stdout.log"' in launcher
    assert "ocean.output_0001" not in launcher
    assert "/usr/bin/time" not in launcher
