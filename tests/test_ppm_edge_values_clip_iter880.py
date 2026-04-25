"""Iter-880: pin the iter-880 fix that removed the non-Fortran-
faithful edge-value clip in `_ppm_edge_values`.

Pre-iter-880, ``_ppm_edge_values`` (`src/legoesm/core/operators_fv.py`)
contained an extra clip step:

    q_hat = jnp.clip(q_hat,
                     min(q_1d[i], q_1d[i+1]),
                     max(q_1d[i], q_1d[i+1]))

This clamped each 4th-order edge value into the local [min, max]
range BEFORE the downstream CW84 constraint (``_ppm_limit``) could
see it.  Fortran's ``xppm`` (tp_core.F90:353-355) does NOT clip:

    do i=is1, ie3
       al(i) = p1*(q1(i-1)+q1(i)) + p2*(q1(i-2)+q1(i+1))
    enddo
    ! ... goes directly into the CW84 constraint via pert_ppm

Our extra clip step made the PPM scheme MORE diffusive than Fortran
by pre-flattening edge overshoots before the CW84 constraint could
process them.  Iter-880 removes the clip for Fortran fidelity.

Tests:

1. ``test_iter880_constant_field_unchanged``: a constant field
   produces a constant edge-value field with or without clip
   (sanity, no behavioural change).

2. ``test_iter880_linear_field_unchanged``: a linear ramp produces
   exact midpoint edges with or without clip (sanity).

3. ``test_iter880_high_frequency_field_differs_from_clipped``: a
   sharp/zigzag input produces edge values that overshoot the local
   [min, max] range; iter-880's unclipped output preserves these
   overshoots (which the downstream CW84 constraint then handles).
   A reference implementation with the pre-iter-880 clip step
   produces DIFFERENT edge values on this input — proving the
   iter-880 fix is firing.

4. ``test_iter880_source_no_jnp_clip_in_ppm_edge_values``: AST scan
   asserting `_ppm_edge_values` source contains no `jnp.clip`
   call (catches a regression that re-introduces the clip).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import ast
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.core.operators_fv import _ppm_edge_values


def _ppm_edge_values_with_clip(q_1d):
    """Pre-iter-880 reference: same 4th-order formula + the removed
    clip step.  Used to verify iter-880's fix produces measurably
    different output on inputs where the clip would have fired."""
    q_pad = q_1d
    M = q_pad.shape[-2]
    q_hat_inner = ((7.0 / 12.0) * (q_pad[..., 1:-2, :] + q_pad[..., 2:-1, :])
                   - (1.0 / 12.0) * (q_pad[..., :-3, :] + q_pad[..., 3:, :]))
    if M >= 7:
        q_os_lo = (
            15.0 * q_pad[..., 2:3, :]
            - 10.0 * q_pad[..., 3:4, :]
            + 3.0 * q_pad[..., 4:5, :]
        ) / 8.0
        q_os_hi = (
            15.0 * q_pad[..., -3:-2, :]
            - 10.0 * q_pad[..., -4:-3, :]
            + 3.0 * q_pad[..., -5:-4, :]
        ) / 8.0
        q_hat_inner = q_hat_inner.at[..., 0:1, :].set(
            0.5 * (q_hat_inner[..., 0:1, :] + q_os_lo))
        q_hat_inner = q_hat_inner.at[..., -1:, :].set(
            0.5 * (q_hat_inner[..., -1:, :] + q_os_hi))
    q_hat_lo = 0.5 * (q_pad[..., 0:1, :] + q_pad[..., 1:2, :])
    q_hat_hi = 0.5 * (q_pad[..., -2:-1, :] + q_pad[..., -1:, :])
    q_hat = jnp.concatenate([q_hat_lo, q_hat_inner, q_hat_hi], axis=-2)

    # The clip step that iter-880 REMOVED.
    q_lo = jnp.minimum(q_pad[..., :-1, :], q_pad[..., 1:, :])
    q_hi = jnp.maximum(q_pad[..., :-1, :], q_pad[..., 1:, :])
    q_hat = jnp.clip(q_hat, q_lo, q_hi)
    return q_hat


def test_iter880_constant_field_unchanged():
    """Constant field: edge values are constant regardless of clip
    (sanity)."""
    n = 8
    q = jnp.ones((6, n + 4, n)) * 7.0
    q_hat = _ppm_edge_values(q)
    np.testing.assert_allclose(np.asarray(q_hat), 7.0, atol=1e-12)


def test_iter880_linear_field_unchanged():
    """Linear ramp: 4th-order formula is exact, clip is a no-op
    (sanity).  Iter-880 must preserve this exactness."""
    n = 8
    x = jnp.arange(n + 4, dtype=jnp.float64)
    q = jnp.broadcast_to(x[None, :, None], (6, n + 4, n))
    q_hat = _ppm_edge_values(q)
    expected_inner = x[1:-2] + 0.5
    expected_lo = 0.5 * (x[0] + x[1])
    expected_hi = 0.5 * (x[-2] + x[-1])
    expected = jnp.concatenate(
        [jnp.array([expected_lo]), expected_inner, jnp.array([expected_hi])])
    expected = jnp.broadcast_to(expected[None, :, None], (6, n + 3, n))
    np.testing.assert_allclose(
        np.asarray(q_hat), np.asarray(expected), atol=1e-10)


def test_iter880_high_frequency_field_differs_from_clipped():
    """Sharp/zigzag input: 4th-order formula overshoots local
    [min, max], and the pre-iter-880 clip would flatten those
    overshoots.  Iter-880's unclipped output MUST differ from the
    clipped reference on this input — proving the fix is firing.

    The downstream CW84 constraint handles the unclipped overshoots
    correctly; pre-iter-880's pre-flatten step made the scheme
    MORE diffusive than Fortran by removing the overshoot before
    the constraint could see it.
    """
    n = 8
    # Zigzag: alternates between -10 and +10 every cell.
    x = jnp.array([(-1.0) ** i * 10.0
                   for i in range(n + 4)], dtype=jnp.float64)
    q = jnp.broadcast_to(x[None, :, None], (6, n + 4, n))

    q_hat_unclipped = _ppm_edge_values(q)
    q_hat_clipped = _ppm_edge_values_with_clip(q)

    diff = float(jnp.max(jnp.abs(q_hat_unclipped - q_hat_clipped)))
    assert diff > 1e-6, (
        f"Iter-880 fix invisible: `_ppm_edge_values` output bit-"
        f"matches the pre-iter-880 clipped reference on a zigzag "
        f"input (max |Δ|={diff:.3e}).  Either the fix was reverted "
        f"or the input doesn't trigger the clip-vs-no-clip "
        f"divergence.  The 4th-order PPM edge formula on a "
        f"high-frequency input MUST overshoot the local [min, max] "
        f"range; iter-880's unclipped output should differ from "
        f"the clipped reference on those overshoot cells.")


def test_iter880_source_no_jnp_clip_in_ppm_edge_values():
    """AST scan: ``_ppm_edge_values`` source MUST NOT contain a
    ``jnp.clip(q_hat, ...)`` call.  The pre-iter-880 clip step
    flattened edge overshoots before the CW84 constraint could
    process them, making the scheme more diffusive than Fortran.
    """
    src_path = (Path(__file__).resolve().parent.parent
                / "src" / "legoesm" / "core" / "operators_fv.py")
    tree = ast.parse(src_path.read_text())

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "_ppm_edge_values"),
        None,
    )
    assert fn is not None, "Could not find _ppm_edge_values"

    # Look for any ``jnp.clip(...)`` Call within the function body.
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute)
                    and f.attr == "clip"
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "jnp"):
                raise AssertionError(
                    f"`_ppm_edge_values` re-introduced a `jnp.clip` "
                    f"call (Call source: `{ast.unparse(node)}`).  "
                    f"Iter-880 removed this clip step for Fortran "
                    f"fidelity (xppm tp_core.F90:353-355 has no "
                    f"such clip).  If the clip is being re-added "
                    f"intentionally, document the rationale and "
                    f"update this test.")
