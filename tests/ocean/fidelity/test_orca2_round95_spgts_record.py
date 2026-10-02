"""Controls for the ORCA2 round-95 self-describing SPG record."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round95_spgts_acquisition import (
    check_record as record,
)


def write_group(payload: bytearray, name: str, rank: int) -> None:
    shape = (3, 2) if rank == 2 else (4,)
    n1, n2 = shape if rank == 2 else (shape[0], 1)
    payload.extend(name.encode("ascii").ljust(16, b" "))
    payload.extend(struct.pack("=4i", rank, n1, n2, 1))
    payload.extend(np.zeros(n1 * n2, dtype=np.float64).tobytes())


def synthetic_record(path: Path, *, icycle: int = 2) -> None:
    payload = bytearray(record.MAGIC.encode("ascii").ljust(16, b" "))
    payload.extend(struct.pack(
        "=18i", 1, 1, 1, 1, 3, 3, 0, 4, 4, 31, icycle,
        1, 1, 2, 2, 3, 3, 64,
    ))
    for name in record.ENTRY:
        write_group(payload, f"i000_{name}", 1 if name in {"wgtbtp1", "wgtbtp2", "entry_sc"} else 2)
    for substep in range(1, icycle + 1):
        for name in record.SUBSTEP:
            write_group(payload, f"j{substep:03d}_{name}", 1 if name in {"ext_coef", "bck_coef", "sum_coef"} else 2)
    for name in record.EXIT:
        write_group(payload, f"o000_{name}", 2)
    path.write_bytes(payload)


def test_record_derives_frames_and_payloads_from_its_header(tmp_path: Path) -> None:
    path = tmp_path / "oracle_r95_spg_rank0000_kt00000001.bin"
    synthetic_record(path)
    parsed = record.read_record(path)
    assert parsed["icycle"] == 2
    assert parsed["frames"] == 4
    assert parsed["groups"] == len(record.ENTRY) + 2 * len(record.SUBSTEP) + len(record.EXIT)


@pytest.mark.parametrize(
    "plant", ("header", "field-name", "field-dims", "truncation", "missing-frame"),
)
def test_record_plants_refuse(tmp_path: Path, plant: str) -> None:
    path = tmp_path / "oracle_r95_spg_rank0000_kt00000001.bin"
    synthetic_record(path)
    with pytest.raises(record.Refusal):
        record.read_record(path, plant)


def test_swapped_rank_plant_is_observable(tmp_path: Path) -> None:
    path = tmp_path / "oracle_r95_spg_rank0000_kt00000001.bin"
    synthetic_record(path)
    assert record.read_record(path, "swapped-rank")["rank"] == 1


def test_round96_launcher_stages_admitted_deck_before_decision83_patch() -> None:
    launcher = Path(
        "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round95_spgts_acquisition/run.sh"
    ).read_text(encoding="utf-8")
    stage = 'cp "$SOURCE_RUN/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"'
    pin = "pin \"$SOURCE_NML_SHA\" \"$TARGET_ROOT/EXP00/namelist_cfg\""
    patch = 'patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/EXP00" <"$DECISION83_PATCH"'
    assert launcher.index(stage) < launcher.index(pin) < launcher.index(patch)
    assert "readonly TARGET_CFG=ORCA2_OMIP_L4_R96SPG" in launcher
