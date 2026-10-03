import numpy as np
from types import SimpleNamespace

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


def test_round122_output_names_the_northern_scope_not_the_inherited_southern_gate():
    source = gate.Path(gate.__file__).read_text(encoding="utf-8")
    assert '"executes_northern_een_ff_association"' in source


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


def test_round122_candidate_fields_replace_only_northern_ff_neighbor():
    shape = (3, 4, 30)
    parts = {
        "een_ff": np.arange(12, dtype=np.float64).reshape(3, 4),
        "een_e3f0": np.ones(shape),
        "een_r3f": np.ones((3, 4)),
        "een_fmask": np.ones(shape),
        "een_denom": np.ones(shape),
        "een_q": np.zeros(shape),
    }
    north = np.array([103.0, 102.0, 101.0, 100.0])
    fields = gate._candidate_fields(parts, np.ones(shape, dtype=bool), north)
    np.testing.assert_array_equal(fields["ne"]["1_ff"][-1, :, 0], north)
    np.testing.assert_array_equal(fields["nw"]["3_ff"][-1, :, 0], np.roll(north, 1))
    np.testing.assert_array_equal(
        fields["ne"]["1_ff"][:-1, :, 0], parts["een_ff"][1:]
    )


def test_round122_production_north_ff_uses_pivot_f_permutation():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _nemo_een_north_ff

    field = np.arange(5 * 4, dtype=np.float64).reshape(5, 4)
    fold = SimpleNamespace(
        is_active=True,
        fold_j=4,
        pivot_row_stored=True,
        perm_f=np.array([3, 2, 1, 0]),
        perm_v=np.array([0, 3, 2, 1]),
    )
    associated = np.asarray(_nemo_een_north_ff(field, SimpleNamespace(fold=fold)))
    np.testing.assert_array_equal(associated[:-1], field[1:])
    np.testing.assert_array_equal(associated[-1], field[-3, ::-1])
