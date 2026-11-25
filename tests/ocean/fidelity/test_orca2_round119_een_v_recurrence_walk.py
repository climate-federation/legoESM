import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round115_een_product_walk as r115,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round119_een_v_recurrence_walk as gate,
)


def test_round119_source_order_and_paths_are_registered():
    assert gate.LABELS == ("nw", "ne", "sw", "se")
    assert gate.SOURCE_ORDER == (
        "mbkv", "zpvo", "e3v", "e3u", "mask", "term", "before", "after")
    assert gate.PLANTS == ("none", "oracle-bit", "candidate-bit", "scope-route")
    assert gate.EXPECTED_FIRST == {
        "nw": "zpvo", "ne": "zpvo", "sw": "before", "se": "before"}
    assert gate.EXPECTED_BASELINE["nw"]["zpvo"] == (1431, 1431)
    assert gate.EXPECTED_BASELINE["ne"]["zpvo"] == (1431, 1431)
    assert gate.EXPECTED_BASELINE["sw"]["before"] == (2557, 0)
    assert gate.EXPECTED_BASELINE["se"]["after"] == (6647, 0)
    assert all(pair == (0, 0) for row in ("sw", "se")
               for pair in gate.EXPECTED_CANDIDATE[row].values())


def test_literal_helper_exposes_every_v_recurrence_operand_name():
    source = (gate.r107.Path(gate.r107.__file__).read_text(encoding="utf-8"))
    for family in ("zpvo_v_", "neighbor_e3u_v_", "neighbor_mask_v_"):
        assert family in source


def test_ieee_zero_add_is_not_an_unregistered_magnitude_change():
    jnp = pytest.importorskip("jax.numpy")
    positive = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    negative = jnp.asarray([-0.0, -1.0], dtype=jnp.float64)
    result = np.asarray(r115.host_ieee_zero_add(positive, negative))
    np.testing.assert_array_equal(result, np.asarray([0.0, 0.0]))
    assert not np.signbit(result).any()
