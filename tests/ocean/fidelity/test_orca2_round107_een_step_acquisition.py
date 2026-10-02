from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round107_een_step_acquisition import (
    check_record as gate,
)


ACQ = Path(
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round107_een_step_acquisition"
)
RESUME = Path(
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round108_een_step_resume/run.sh"
)


def _record() -> bytes:
    n1, n2, n3 = 2, 3, 4
    raw = bytearray(gate.MAGIC.ljust(16, b" "))
    raw.extend(gate.HEADER.pack(1, 1, 0, 6, 7, n3, 1, 1, 3, 3, 4, 5, 64, 8))
    for name in gate.FIELDS:
        raw.extend(name.encode("ascii").ljust(16, b" "))
        rank = 2 if name == "mbku" else 3
        depth = 1 if name == "mbku" else n3
        raw.extend(gate.GROUP.pack(rank, n1, n2, depth))
        value = np.ones((n1, n2, depth), dtype=np.float64)
        raw.extend(value.tobytes(order="F"))
    return bytes(raw)


def test_parser_uses_self_describing_field_headers(tmp_path):
    path = tmp_path / "record.bin"
    path.write_bytes(_record())
    row = gate.read_record(path)
    assert set(row["groups"]) == set(gate.FIELDS)
    assert row["groups"]["zpvo_nw"].shape == (2, 3, 4)
    assert row["groups"]["mbku"].shape == (2, 3, 1)


def test_parser_refuses_header_dimension_mutation(tmp_path):
    raw = bytearray(_record())
    offset = 16 + gate.HEADER.size + 16
    ndim, n1, n2, n3 = gate.GROUP.unpack_from(raw, offset)
    gate.GROUP.pack_into(raw, offset, ndim, n1 + 1, n2, n3)
    path = tmp_path / "record.bin"
    path.write_bytes(raw)
    try:
        gate.read_record(path)
    except gate.Refusal:
        pass
    else:
        raise AssertionError("dimension mutation stayed green")


def test_patch_is_additions_only_and_launcher_is_fail_closed():
    patch = (ACQ / "dynspg_ts_round107.patch").read_text(encoding="utf-8").splitlines()
    removed = [line for line in patch if line.startswith("-") and not line.startswith("---")]
    assert removed == []
    launcher = (ACQ / "run.sh").read_text(encoding="utf-8")
    assert "ORCA2_R107_EEN_STEP_DIR" in launcher
    assert "oracle_r107_een_step_rank????_kt00000001.bin" in launcher
    assert "--source-root" in launcher
    assert "/usr/bin/time" not in launcher


def test_resume_launcher_exports_both_inherited_recorder_directories():
    launcher = RESUME.read_text(encoding="utf-8")
    assert "export ORCA2_R105_EEN_ACCUM_DIR=$TARGET_RUN" in launcher
    assert "export ORCA2_R107_EEN_STEP_DIR=$TARGET_RUN" in launcher
    assert "orca2_rung0_een_step_ranked_resume_10step_np2" in launcher
    assert "mpirun -np 2 --oversubscribe ./nemo" in launcher
    assert "/usr/bin/time" not in launcher
