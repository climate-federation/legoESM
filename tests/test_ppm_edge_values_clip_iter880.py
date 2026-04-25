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


def _ppm_edge_values_with_clip(q_1d, blend_edges=False):
    """Pre-iter-880 reference: same `_ppm_edge_values` body but WITH
    the clip step that iter-880 removed.  Mirrors the production
    signature (`blend_edges=False` by default) so the only
    behavioural difference vs the iter-880 function is the clip
    step itself.

    Iter-880b (Codex iter-880 stop-time fix): pre-iter-880b this
    reference unconditionally applied the blend-edges branch (`if
    M >= 7:`) regardless of the `blend_edges` flag.  That confused
    the diff in `test_iter880_high_frequency_field_differs_from_clipped`
    — the divergence came from BOTH the blend (extra in reference)
    AND the clip (extra in reference).  iter-880b restores the
    Fortran-faithful gating ``if blend_edges and M >= 7:`` so the
    reference differs from the production function ONLY by the
    clip step.
    """
    q_pad = q_1d
    M = q_pad.shape[-2]
    q_hat_inner = ((7.0 / 12.0) * (q_pad[..., 1:-2, :] + q_pad[..., 2:-1, :])
                   - (1.0 / 12.0) * (q_pad[..., :-3, :] + q_pad[..., 3:, :]))
    if blend_edges and M >= 7:
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


@pytest.mark.parametrize("blend_edges", [False, True])
def test_iter880_overshoot_input_differs_from_clipped(blend_edges):
    """Input designed to trigger the 4th-order overshoot: `[0, 0,
    10, 0, 0, 10, 0, 0, 10, 0, 0, 10]` (every-third-cell impulse).

    The 4th-order edge formula `(7/12)*(q[i]+q[i+1]) -
    (1/12)*(q[i-1]+q[i+2])` produces edge values that overshoot the
    local ``[min(q[i], q[i+1]), max(q[i], q[i+1])]`` range whenever
    there's a non-monotonic feature in the 4-cell stencil.

    Iter-880b (Codex iter-880 stop-time fix): the original zigzag
    `[10, -10, 10, -10, ...]` input does NOT actually overshoot —
    the 4th-order formula at the (10, -10) edge reduces to
    ``(7/12)*0 - (1/12)*0 = 0`` which IS inside the [-10, 10]
    range.  iter-880b switches to an every-third-cell impulse
    pattern that DOES produce overshoots: at the edge between two
    zeros adjacent to an impulse, the formula picks up the impulse
    contribution via the `(1/12)*(q[i-1]+q[i+2])` term, yielding
    a non-zero edge that the clip step would flatten to zero.

    Both `blend_edges` modes are tested to ensure the iter-880
    fix fires regardless of edge-blending choice.
    """
    n = 8
    # Every-third-cell impulse: 0, 0, 10, 0, 0, 10, ... (period 3).
    x = jnp.array(
        [10.0 if (i % 3 == 2) else 0.0 for i in range(n + 4)],
        dtype=jnp.float64)
    q = jnp.broadcast_to(x[None, :, None], (6, n + 4, n))

    q_hat_unclipped = _ppm_edge_values(q, blend_edges=blend_edges)
    q_hat_clipped = _ppm_edge_values_with_clip(q, blend_edges=blend_edges)

    diff = float(jnp.max(jnp.abs(q_hat_unclipped - q_hat_clipped)))
    assert diff > 1e-6, (
        f"Iter-880 fix invisible (blend_edges={blend_edges}): "
        f"`_ppm_edge_values` output bit-matches the pre-iter-880 "
        f"clipped reference on an impulse input "
        f"(max |Δ|={diff:.3e}). Either the fix was reverted or the "
        f"input doesn't trigger the clip-vs-no-clip divergence.  "
        f"The 4th-order PPM edge formula at edges flanking an "
        f"isolated impulse MUST overshoot the local [min, max] "
        f"range; iter-880's unclipped output should differ from "
        f"the clipped reference on those overshoot cells, in BOTH "
        f"`blend_edges` modes.")


def test_iter880b_reference_isolates_clip_only_diff():
    """Iter-880b: on a SMOOTH input (linear ramp) the iter-880
    function and the iter-880b reference (with clip but otherwise
    identical) must agree exactly — because the clip step is a
    no-op on smooth data.  This pins that the reference correctly
    isolates the clip behaviour from any other source of
    divergence (such as the blend-edges branch).
    """
    n = 8
    x = jnp.arange(n + 4, dtype=jnp.float64)
    q = jnp.broadcast_to(x[None, :, None], (6, n + 4, n))

    for blend_edges in (False, True):
        q_hat_unclipped = _ppm_edge_values(q, blend_edges=blend_edges)
        q_hat_clipped = _ppm_edge_values_with_clip(q,
                                                   blend_edges=blend_edges)
        # On a linear ramp the 4th-order formula is exact, the
        # blend-edges 3rd-order extrapolation is also exact, and
        # the clip is a no-op (edge values are inside [min,max]).
        # So both functions must agree exactly.
        np.testing.assert_allclose(
            np.asarray(q_hat_unclipped), np.asarray(q_hat_clipped),
            rtol=1e-12, atol=1e-12,
            err_msg=(
                f"Iter-880b reference does NOT isolate the clip "
                f"step (blend_edges={blend_edges}): on a smooth "
                f"linear ramp where the clip is a no-op, the "
                f"iter-880 function and the reference disagree.  "
                f"This means the reference has SOME other "
                f"behavioural difference besides the clip — "
                f"audit the reference's `if blend_edges and M >= "
                f"7:` gate, the 4th-order formula, and the "
                f"boundary edges to ensure they match the "
                f"production function exactly except for the clip "
                f"step.  Without this isolation the iter-880 "
                f"high-frequency test could pass for the wrong "
                f"reason."))


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
