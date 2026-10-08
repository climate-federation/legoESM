from __future__ import annotations

import struct

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round181_hpg_fold_acquisition import (
    check_record,
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
