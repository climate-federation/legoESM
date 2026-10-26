from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round107_een_u_operand_walk as gate,
)


def test_score_distinguishes_signed_zero_from_magnitude():
    candidate = np.array([0.0, 2.0], dtype=np.float64)
    reference = np.array([-0.0, 3.0], dtype=np.float64)
    row = gate._score(candidate, reference)
    assert row["bit_unequal"] == 2
    assert row["signed_zero_only"] == 1
    assert row["magnitude_unequal"] == 1


def test_registered_round106_census_is_complete():
    assert set(gate.EXPECTED_CURRENT) == {
        f"acc_{face}_{corner}"
        for face in ("u", "v")
        for corner in ("nw", "ne", "sw", "se")
    }
    assert gate.EXPECTED_CURRENT["acc_u_nw"] == (3953, 438)


def test_probe_keeps_source_and_current_mask_arms_separate():
    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert "source_e3u = b(e3u0 * b(one + r3u" in source
    assert "current_e3u = b(source_e3u * umask)" in source
    assert "source_e3v = b(e3v0 * b(one + r3v" in source
    assert "current_e3v = b(source_e3v * vmask)" in source
