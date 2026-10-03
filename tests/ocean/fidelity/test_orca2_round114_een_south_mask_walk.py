import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round114_een_south_mask_walk as gate,
)


def test_southern_zero_fill_differs_from_cyclic_mask():
    jnp = pytest.importorskip("jax.numpy")
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_south_zero_fill,
    )

    mask = np.asarray([
        [[0.0], [1.0]],
        [[1.0], [0.0]],
        [[1.0], [1.0]],
    ])
    zero_filled = np.asarray(_nemo_south_zero_fill(jnp.asarray(mask)))
    cyclic = np.roll(mask, 1, axis=0)
    np.testing.assert_array_equal(zero_filled[0], 0.0)
    assert not np.array_equal(zero_filled, cyclic)


def test_literal_builder_executes_southern_zero_mask(monkeypatch):
    jnp = pytest.importorskip("jax.numpy")
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as bt
    from tests.ocean.fidelity.test_orca2_round113_een_south_e3f_walk import (
        test_literal_builder_executes_southern_mesh_thickness_copy,
    )

    called = []
    original = bt._nemo_south_zero_fill

    def observed(field):
        called.append(field.shape)
        return original(field)

    monkeypatch.setattr(bt, "_nemo_south_zero_fill", observed)
    test_literal_builder_executes_southern_mesh_thickness_copy(monkeypatch)
    assert called == [(3, 4, 2), (3, 4, 2)]


def test_round114_plants_and_complete_fraction_walk_are_registered():
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "scope-route")
    assert gate.r111.SOURCE_ORDER[-3:] == (
        "frac_south", "sum_west_center", "sum_all")
