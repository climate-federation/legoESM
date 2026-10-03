import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round127_een_pair_walk as gate,
)


def test_round127_registry_stops_at_neighbor_thickness():
    assert gate.EXPECTED_FIRST == {"nw": "e3u", "ne": "e3u"}
    assert gate.EXPECTED_PREFIX["nw"]["e3u"] == (95, 95)
    assert gate.EXPECTED_PREFIX["ne"]["e3u"] == (91, 91)
    assert gate.PLANTS == ("none", "oracle-bit", "candidate-bit", "scope-route")


def test_round127_literal_term_preserves_factor_order():
    face = np.array([[[2.0, -0.0]]])
    neighbor = np.array([[[3.0, 4.0]]])
    mask = np.array([[[1.0, 1.0]]])
    quotient = np.array([[[5.0, -2.0]]])
    result = np.asarray(gate.literal_term(face, neighbor, mask, quotient))
    np.testing.assert_array_equal(result, np.array([[[30.0, 0.0]]]))
    assert not np.signbit(result[0, 0, 1])


def test_round127_candidate_bit_changes_score():
    candidate = np.zeros((1, 1, 1), dtype=np.float64)
    reference = candidate.copy()
    candidate.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
    score = gate.r109._score(candidate, reference)
    assert score["bit_unequal"] == 1
