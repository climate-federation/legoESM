import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round114_een_south_mask_walk as gate,
)


def test_southern_zero_fill_differs_from_cyclic_mask():
    mask = np.asarray([
        [[0.0], [1.0]],
        [[1.0], [0.0]],
        [[1.0], [1.0]],
    ])
    zero_filled = np.concatenate([np.zeros_like(mask[:1]), mask[:-1]], axis=0)
    cyclic = np.roll(mask, 1, axis=0)
    np.testing.assert_array_equal(zero_filled[0], 0.0)
    assert not np.array_equal(zero_filled, cyclic)


def test_round114_plants_and_complete_fraction_walk_are_registered():
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "scope-route")
    assert gate.r111.SOURCE_ORDER[-3:] == (
        "frac_south", "sum_west_center", "sum_all")
