"""Round 303: fold-pair arm driver -- mirror check and arm table."""

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round303_fold_pair_arms as gate,
)


def _mirrored(seed=303):
    v = np.random.default_rng(seed).normal(0.0, 0.1, (4, 6, 3))
    perm = np.array([5, 4, 3, 2, 1, 0])
    v[-1] = -v[-2][perm]
    return v, perm


def test_fold_image_passes_and_signed_zero_is_value_equal():
    v, perm = _mirrored()
    v[-2, 2, 1] = 0.0
    v[-1, 3, 1] = 0.0  # +0 vs the image's -0: bit differs, value does not
    row = gate.fold_mirror(v, perm, -1.0)
    assert row == {"value_unequal": 0, "bit_unequal": 1,
                   "bit_diffs_are_signed_zeros": True}


def test_one_ulp_on_the_top_row_is_caught():
    v, perm = _mirrored()
    v[-1, 0, 0] = np.nextafter(v[-1, 0, 0], np.inf)
    row = gate.fold_mirror(v, perm, -1.0)
    assert row["value_unequal"] == 1 and not row["bit_diffs_are_signed_zeros"]


def test_wrong_sign_is_caught():
    v, perm = _mirrored()
    assert gate.fold_mirror(v, perm, 1.0)["value_unequal"] == v[-1].size


def test_four_arms_are_the_two_statements_crossed():
    assert gate.ARMS == {"hooked": (False, False), "topv": (True, False),
                         "tmask": (False, True), "pair": (True, True)}
