"""Guards for the LOCK slow-forcing owner probe.

The probe's numeric arms need a full model step, so they run as a script with
its planted control.  What is guarded here is the arithmetic association the
probe exists to discriminate -- ``x / H`` against NEMO's ``x * r1_hu_0`` -- and
the fail-closed dispatch on an unknown association name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TESTCASES = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
)
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_lock_slow_forcing_owner",
    TESTCASES / "nemo_testcase_lock_slow_forcing_owner.py",
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class _FlatZCoord:
    """A one-column ladder whose live thickness is its reference thickness."""

    def __init__(self, thickness):
        self.h_partial = np.asarray(thickness)


def _statement(association, du, h_column):
    """Call the probe's statement on a single wet column of unit area.

    x64 is enabled HERE rather than left to the caller's environment: without
    it the float64 request silently downcasts and this test compares f32
    rounding, which is the exact failure mode Rule 1c is about.  It passed
    standalone and failed inside a combined pytest run until this was fixed.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    du = jnp.asarray(du, dtype=jnp.float64)[None, None, :]
    h_u = jnp.asarray(h_column, dtype=jnp.float64)[None, None, :]
    pair = jnp.sum(jnp.stack([h_u, du * h_u], axis=-1), axis=-2)
    depth = jnp.maximum(pair[..., 0], 1e-10)
    if association == "model":
        return np.asarray(pair[..., 1] / depth)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import nemo_source_round
    return np.asarray(pair[..., 1] * nemo_source_round(1.0 / depth))


def test_divide_and_reciprocal_associations_differ_by_a_bit():
    """The whole point of the probe: x/H and x*(1/H) are not the same bits.

    A test that only asserted they are close would pass with the probe's two
    arms fused, which is exactly the defect it is meant to detect.
    """
    rng = np.random.default_rng(20260905)
    du = rng.normal(scale=1.0e-3, size=64)
    h = rng.uniform(1.0, 400.0, size=64)
    divide = _statement("model", du, h)
    reciprocal = _statement("nemo_reciprocal", du, h)
    assert np.isclose(divide, reciprocal, rtol=1.0e-15).all()
    assert divide.view(np.uint64) != reciprocal.view(np.uint64)


def _guard_call(**kwargs):
    """The shared statement, called with everything valid but one field.

    Round 28 made this the ONE depth-mean implementation and gave it a face
    selector, so it grew a ``grid`` argument between the config and the mask.
    The call is spelled out here rather than positionally so the next
    signature change fails loudly in one place instead of turning a dispatch
    guard's ValueError into a TypeError, which is how this test broke.
    """
    return probe.depth_mean_statement(
        np.zeros((1, 1, 1)), np.zeros((1, 1)), np.ones((1, 1)),
        _FlatZCoord(np.ones((1, 1, 1))), object(), None, np.ones((1, 1)),
        **kwargs)


def test_unknown_association_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown association"):
        _guard_call(association="not_a_real_association")


def test_unknown_face_raises_rather_than_defaulting():
    """Same dispatch-hardening rule for the face selector round 28 added.

    A silent fall-through here would score a V face with the U operator, which
    is a wrong ANSWER rather than an error.
    """
    with pytest.raises(ValueError, match="unknown face"):
        _guard_call(tag="w")
