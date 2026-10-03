import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round122_een_north_ff_walk as gate,
)


def test_round122_frozen_registry_is_complete():
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "permutation-shift", "scope-route"
    )
    assert gate.EXPECTED_NEXT == {
        "ne": ("1_e3f0", 514),
        "nw": ("3_e3f0", 521),
    }


def test_round122_candidate_fields_preserve_literal_fraction_order():
    shape = (3, 4, 30)
    parts = {
        "een_ff": np.arange(12, dtype=np.float64).reshape(3, 4),
        "een_e3f0": np.ones(shape),
        "een_r3f": np.ones((3, 4)),
        "een_fmask": np.ones(shape),
        "een_denom": np.ones(shape),
        "een_q": np.arange(np.prod(shape), dtype=np.float64).reshape(shape),
    }
    fields = gate._candidate_fields(parts, np.ones(shape, dtype=bool))
    np.testing.assert_array_equal(
        fields["ne"]["1_ff"][0],
        np.broadcast_to(parts["een_ff"][1, :, None], (4, 30)),
    )
    np.testing.assert_array_equal(
        fields["nw"]["3_ff"][0, 0],
        np.broadcast_to(parts["een_ff"][1, -1], (30,)),
    )
