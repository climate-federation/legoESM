import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round109_een_per_level_walk as r109,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round115_een_product_walk as gate,
)


def test_round115_source_order_and_plants_are_registered():
    assert gate.SOURCE_ORDER == (
        "mbku", "zpvo_nw", "e3u_live", "e3v_live", "neighbor_mask",
        "term_nw", "acc_before", "acc_after",
    )
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "scope-route")


def test_round115_score_detects_a_product_sign_bit():
    candidate = np.array([[[-0.0]]], dtype=np.float64)
    oracle = np.array([[[0.0]]], dtype=np.float64)
    assert r109._score(candidate, oracle) == {
        "bit_unequal": 1,
        "magnitude_unequal": 0,
        "signed_zero_only": 1,
        "first_bit_unequal_j_i_k": [0, 0, 0],
        "bit_unequal_j_values": [0],
        "bit_unequal_k_values": [0],
    }
