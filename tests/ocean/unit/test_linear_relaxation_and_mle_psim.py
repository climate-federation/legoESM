"""Direct unit tests for the shared relaxation + MLE-magnitude helpers (#518 §9).

* ``linear_relaxation(field, target, tau)`` replaces the ``-(f - target)/tau``
  forms in ``restoring.py`` and the ``(target - field)/tau`` form in
  ``flux_feedback.py``.
* ``mle_streamfunction_magnitude(rc_f, H, width, dbm, cap)`` replaces the
  ``rc_f*H^2*width*dbm*cap`` product in ``mle_latlon_cgrid`` (psim_u/psim_v) and
  ``mle_mpas`` (psim_e).

Both are pinned byte-identical to the original inline expressions for arbitrary
(non-degenerate) inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.surface_forcing._shared import linear_relaxation
from legoesm.ocean.physics.lateral_mixing.mle import (
    mle_streamfunction_magnitude,
)

jax.config.update("jax_enable_x64", True)


def test_linear_relaxation_matches_flux_feedback_form():
    rng = np.random.default_rng(0)
    field = jnp.asarray(rng.standard_normal((4, 5)))
    target = jnp.asarray(rng.standard_normal((4, 5)))
    tau = 1.0e6
    got = linear_relaxation(field, target, tau)
    ref = (target - field) / tau  # flux_feedback's exact form
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_linear_relaxation_matches_restoring_form_byte_identical():
    rng = np.random.default_rng(1)
    field = jnp.asarray(rng.standard_normal((6,)))
    target = jnp.asarray(rng.standard_normal((6,)))
    tau = 3.0e5
    # restoring.py's exact form via the explicit numerator selector.
    got = linear_relaxation(field, target, tau, numerator="neg_field_minus_target")
    ref = -(field - target) / tau
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_linear_relaxation_numerator_sign_of_zero():
    """The two numerator forms differ only by sign-of-zero on field==target;
    each is byte-identical to its respective call site's prior inline form."""
    field = jnp.asarray([1.0, 2.0])
    target = jnp.asarray([1.0, 2.0])  # field == target -> +/-0.0 numerator
    tau = 1.0
    a = np.asarray(linear_relaxation(field, target, tau,
                                     numerator="target_minus_field"))
    b = np.asarray(linear_relaxation(field, target, tau,
                                     numerator="neg_field_minus_target"))
    # (target-field) -> +0.0 ; -(field-target) -> -0.0
    assert not np.signbit(a[0]) and np.signbit(b[0])


def test_linear_relaxation_rejects_unknown_numerator():
    import pytest
    with pytest.raises(ValueError):
        linear_relaxation(jnp.zeros(2), jnp.zeros(2), 1.0, numerator="bad")


def test_linear_relaxation_drives_toward_target():
    field = jnp.asarray([0.0, 5.0])
    target = jnp.asarray([2.0, 2.0])
    tau = 10.0
    out = np.asarray(linear_relaxation(field, target, tau))
    assert out[0] > 0.0  # below target -> warms
    assert out[1] < 0.0  # above target -> cools


def test_mle_magnitude_matches_inline_product():
    rng = np.random.default_rng(2)
    rc_f = 1.234e-4
    H = jnp.asarray(np.abs(rng.standard_normal((3, 7))) * 50.0)
    width = jnp.asarray(np.abs(rng.standard_normal((3, 7))) * 1e4)
    dbm = jnp.asarray(rng.standard_normal((3, 7)) * 1e-7)
    cap = jnp.asarray(np.abs(rng.standard_normal((3, 7))) * 1e5)
    got = mle_streamfunction_magnitude(rc_f, H, width, dbm, cap)
    ref = rc_f * H * H * width * dbm * cap  # exact pre-#518 order
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_mle_magnitude_differentiable():
    rng = np.random.default_rng(3)
    rc_f = 1e-4
    H = jnp.asarray(np.abs(rng.standard_normal((4,))) * 30.0)
    width = jnp.asarray(np.abs(rng.standard_normal((4,))) * 1e4)
    dbm = jnp.asarray(rng.standard_normal((4,)) * 1e-7)
    cap = jnp.asarray(np.abs(rng.standard_normal((4,))) * 1e5)
    g = jax.grad(
        lambda h: jnp.sum(mle_streamfunction_magnitude(rc_f, h, width, dbm, cap))
    )(H)
    assert np.all(np.isfinite(np.asarray(g)))
