import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round115_een_product_walk as r115,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round117_een_u_recurrence_walk as gate,
)


def test_round117_source_order_paths_and_plants_are_registered():
    assert gate.LABELS == ("ne", "sw", "se")
    assert gate.SOURCE_ORDER == (
        "mbku", "zpvo", "e3u", "e3v", "mask", "term", "before", "after")
    assert gate.PLANTS == ("none", "oracle-bit", "candidate-bit", "scope-route")


def test_ieee_zero_add_preserves_negative_zero_only_for_two_negative_inputs():
    jnp = pytest.importorskip("jax.numpy")
    positive = jnp.asarray([0.0], dtype=jnp.float64)
    negative = jnp.asarray([-0.0], dtype=jnp.float64)
    assert np.signbit(np.asarray(r115.host_ieee_zero_add(negative, negative))[0])
    assert not np.signbit(np.asarray(r115.host_ieee_zero_add(positive, negative))[0])
    assert not np.signbit(np.asarray(r115.host_ieee_zero_add(negative, positive))[0])


def test_southern_vmask_arm_zero_fills_without_moving_inner_rows():
    jnp = pytest.importorskip("jax.numpy")
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _nemo_south_zero_fill

    mask = np.arange(12.0).reshape(3, 4, 1)
    filled = np.asarray(_nemo_south_zero_fill(jnp.asarray(mask)))
    np.testing.assert_array_equal(filled[0], 0.0)
    np.testing.assert_array_equal(filled[1:], mask[:-1])
