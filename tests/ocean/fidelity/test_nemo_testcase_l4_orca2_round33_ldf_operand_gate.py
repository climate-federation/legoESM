"""Binding tests for the ORCA2 round-33 LDF operand gate."""

from __future__ import annotations

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round33_ldf_operand_gate as gate,
)


def test_native_f_coefficient_maps_to_redundant_vertex_layout():
    native = np.arange(2 * 3 * 1, dtype=np.float64).reshape(2, 3, 1)
    got = gate._mapped_record_ahmf(native)
    assert got.shape == (3, 4, 1)
    np.testing.assert_array_equal(got[0], 0.0)
    np.testing.assert_array_equal(got[1:, 1:], native)
    np.testing.assert_array_equal(got[1:, 0], native[:, -1])
