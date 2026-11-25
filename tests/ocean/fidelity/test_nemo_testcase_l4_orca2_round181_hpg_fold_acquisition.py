from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round181_hpg_fold_acquisition import (
    check_record,
)

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / (
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round181_hpg_fold_acquisition/run.sh"
)


def _record_bytes() -> bytes:
    chunks = [check_record.MAGIC, struct.pack(
        "=16i", 1, 1, 1, 3, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 64,
        len(check_record.NAMES))]
    for index, name in enumerate(check_record.NAMES):
        chunks.extend((name.encode().ljust(16, b" "),
                       struct.pack("=5i", 0, 3, 1, 1, 1),
                       np.asarray([index], dtype="=f8").tobytes()))
    return b"".join(chunks)


def test_self_describing_record_parser(tmp_path):
    path = tmp_path / "record.bin"
    path.write_bytes(_record_bytes())
    result = check_record.read_record(path)
    assert tuple(result["fields"]) == check_record.NAMES
    assert result["rank"] == 0


@pytest.mark.parametrize("plant", ("header", "field-name", "field-dims", "truncation"))
def test_parser_plants_refuse(tmp_path, plant):
    path = tmp_path / "record.bin"
    path.write_bytes(_record_bytes())
    with pytest.raises(check_record.Refusal):
        check_record.read_record(path, plant)


def test_round182_launcher_uses_fresh_target_and_zero_fuzz_patch():
    text = RUNNER.read_text(encoding="utf-8")
    assert "ORCA2_OMIP_L4_R182HPGFOLD" in text
    assert "orca2_rounds/round182/acquisition" in text
    assert "PREREG_nemo_testcases_l4_orca2_round182.md" in text
    assert 'patch -s --fuzz=0 -p0 -d "$target_root/MY_SRC"' in text
    assert 'cmp -s "$scratch/dynhpg.F90" "$target_root/MY_SRC/dynhpg.F90"' in text
    assert "PATCH_SYMLINK_PARENT_PROOF_PASS" in text
    assert 'git apply --unsafe-paths -p0 --directory="$target_root/MY_SRC"' not in text
    assert "ORCA2_OMIP_L4_R181HPGFOLD" not in text
