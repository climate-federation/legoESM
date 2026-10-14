"""Red-capable controls for the two G2 statements imported in round 276."""

import jax
import jax.numpy as jnp
import numpy as np


def test_g2_imported_selectors_default_off():
    from legoesm.ocean.state import LatLonCGridOceanConfig

    assert LatLonCGridOceanConfig._field_defaults[
        "zdf_tracer_nemo_rounded_thickness"] is False
    assert LatLonCGridOceanConfig._field_defaults[
        "barotropic_rhs_ascending_level_sum"] is False


def test_ascending_level_sum_is_written_order_and_control_can_fail():
    from legoesm.ocean.dynamics.barotropic_common import ascending_level_sum

    rng = np.random.default_rng(276)
    values = rng.standard_normal((400, 30))
    values *= np.exp(rng.uniform(-20.0, 0.0, values.shape))

    expected = np.zeros(values.shape[:-1], dtype=np.float64)
    for level in range(values.shape[-1]):
        expected = expected + values[..., level]

    ordered = np.asarray(jax.jit(ascending_level_sum)(jnp.asarray(values)))
    np.testing.assert_array_equal(ordered, expected)
    lanes = [values[:, level::4].sum(axis=1) for level in range(4)]
    four_lane = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    assert np.count_nonzero(four_lane != expected) > 0
