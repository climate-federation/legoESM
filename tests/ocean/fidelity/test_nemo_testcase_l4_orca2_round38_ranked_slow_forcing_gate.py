"""Binding tests for the ORCA2 round-38 ranked slow-forcing reader."""

from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round38_ranked_slow_forcing_gate as gate,
)

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round20/acquisition/orca1ice_slow_forcing_ranked_np2")


def test_ranked_reader_admits_both_distinct_records():
    records = [gate.read_ranked(
        ROOT / f"oracle_slow_forcing_ranked_kt00000001_r{rank:04d}.bin", rank)
        for rank in range(2)]
    assert all(record["bytes"] == gate.RECORD_BYTES for record in records)
    assert records[0]["sha256"] != records[1]["sha256"]


def test_ranked_reader_rejects_swapped_rank():
    with pytest.raises(gate.GateError, match="header changed"):
        gate.read_ranked(
            ROOT / "oracle_slow_forcing_ranked_kt00000001_r0000.bin", 1)
