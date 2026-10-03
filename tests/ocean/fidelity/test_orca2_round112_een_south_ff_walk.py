import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round112_een_south_ff_walk as gate,
)


def test_source_copy_differs_from_cyclic_south_association():
    jnp = pytest.importorskip("jax.numpy")
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_een_south_ff_copy,
    )

    field = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]])
    copied = np.asarray(_nemo_een_south_ff_copy(field))
    cyclic = np.roll(np.asarray(field), 1, axis=0)
    np.testing.assert_array_equal(copied, [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    assert not np.array_equal(copied, cyclic)


def test_round112_plants_are_registered():
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "scope-route")


def test_round112_card_scope_is_frozen():
    assert gate.EXPECTED_CARD_SCOPE == {
        "DINO-nemo_dino_kamm": True,
        "DINO-nemo_dino_kamm_mlf": True,
        "ORCA2-zps": True,
        "GYRE-zco": False,
        "LOCK_EXCHANGE-zco": False,
        "OVERFLOW-zps": False,
        "VORTEX-zco": True,
        "VORTEX_VEC-zco": True,
    }


def test_round112_reuses_admitted_source_order():
    assert gate.r111.SOURCE_ORDER[12:18] == (
        "south_ff", "south_e3f0", "south_r3f", "south_mask",
        "south_denom", "frac_south",
    )
