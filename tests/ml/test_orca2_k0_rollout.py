"""Unit tests for the K0 rollout's control splitting (no ORCA2 decks needed).

The real-card equivalence (loop == step-by-step run) is in
tests/ocean/fidelity/test_orca2_k0_rollout_equivalence.py.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.sponge import SpongeForcing
from legoesm.training.orca2_k0_rollout import _controls_at, make_k0_rollout, split_controls


class Clock(NamedTuple):
    field: object
    isecsbc: object
    stride: object


def _controls(n, kt0=3):
    mask = jnp.ones((2, 2))
    gamma = jnp.full((2, 2, 3), 1e-8)
    out = []
    for i in range(n):
        out.append({
            "s0f": Clock(field=jnp.full((2, 2), float(i)), isecsbc=(kt0 + i) * 10800, stride=2),
            "s2_base": {"mask": mask, "flag": False, "opt": None},
            "sponge": SpongeForcing(gamma=jnp.array(gamma), T_ref=jnp.full((2, 2, 3), float(i)),
                                    S_ref=jnp.zeros((2, 2, 3)), nemo_kt=kt0 + i),
        })
    return out


def test_leaves_are_classified_static_shared_or_stacked():
    treedef, kinds, values = split_controls(_controls(4), kt0=3)
    ctrl = _controls_at(treedef, kinds, values, 2)
    assert ctrl["s0f"].stride == 2 and isinstance(ctrl["s0f"].stride, int)   # static stays Python
    assert ctrl["s2_base"]["flag"] is False and ctrl["s2_base"]["opt"] is None
    assert int(ctrl["s0f"].isecsbc) == 5 * 10800                             # varying int -> stacked
    np.testing.assert_array_equal(ctrl["s0f"].field, np.full((2, 2), 2.0))
    np.testing.assert_array_equal(ctrl["sponge"].T_ref, np.full((2, 2, 3), 2.0))
    assert ctrl["sponge"].nemo_kt is None                                    # re-stamped in the loop
    by_shape = {np.shape(v): k for k, v in zip(kinds, values) if k != "static"}
    assert by_shape[(2, 2)] == "shared"            # mask, equal every step: held once
    assert by_shape[(2, 2, 3)] == "shared"         # gamma: distinct objects, equal values
    assert by_shape[(4, 2, 2, 3)] == "stacked"     # T_ref varies


def test_stale_damping_target_is_refused():
    ctrls = _controls(3)
    ctrls[1] = dict(ctrls[1], sponge=ctrls[0]["sponge"])
    with pytest.raises(ValueError, match="step 4 has kt=3"):
        split_controls(ctrls, kt0=3)


def test_structure_change_is_refused():
    ctrls = _controls(2)
    ctrls[1] = dict(ctrls[1], s2_base={"mask": ctrls[1]["s2_base"]["mask"], "flag": False})
    with pytest.raises(ValueError, match="structure"):
        split_controls(ctrls, kt0=3)


def test_rollout_must_start_on_a_forcing_step():
    treedef, kinds, _ = split_controls(_controls(2), kt0=3)
    with pytest.raises(ValueError, match="odd"):
        make_k0_rollout(None, None, 1, 4, treedef, kinds)
