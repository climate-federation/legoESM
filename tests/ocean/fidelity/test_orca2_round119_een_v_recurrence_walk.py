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


def test_ieee_zero_add_is_not_an_unregistered_magnitude_change():
    jnp = pytest.importorskip("jax.numpy")
    positive = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    negative = jnp.asarray([-0.0, -1.0], dtype=jnp.float64)
    result = np.asarray(r115.host_ieee_zero_add(positive, negative))
    np.testing.assert_array_equal(result, np.asarray([0.0, 0.0]))
    assert not np.signbit(result).any()
