"""Iter-884: pin the off-by-one fix for `_ppm_1d` boundary
pert_ppm indices.

Fortran's `xppm` (`tp_core.F90:629` for left, `:648` for right)
calls ``pert_ppm(3, q1(idx0), bl(idx0), br(idx0), 1)`` at the
left and right cube-face boundaries when not duogrid AND not
bounded_domain:
- Left boundary: idx0 = 0, applies to q1(0..2) = (halo-(-1),
  interior 0, interior 1).
- Right boundary: idx0 = npx-2, applies to q1(npx-2..npx) =
  (interior n-2, interior n-1, halo n).

Our Python `_ppm_1d` (`fv_tp_2d.py:_ppm_1d`) applies pert_ppm
iv=1 to q_c indices `[0, 1, 2, -3, -2, -1]` matching the Fortran
range exactly.

Pre-iter-884 the indices were `[1, 2, 3, -4, -3, -2]` — shifted
INWARD by one cell on each side.  This silently dropped the
boundary halo cells from the iv=1 constraint and instead applied
it to interior cells one position deeper.

Iter-884 corrects to `[0, 1, 2, -3, -2, -1]`.

This test pins the fix via:
1. AST scan: source MUST contain the index list `[0, 1, 2, -3, -2, -1]`
   (or equivalent) and MUST NOT contain the pre-iter-884 indices.
2. Behavioural test: construct an input where the iv=1 constraint
   on the LEFT BOUNDARY HALO cell `q_c[0]` produces a different
   output than the constraint on `q_c[3]` (interior 2).  Verify the
   actual `_ppm_1d` output reflects iv=1 having been applied at
   `q_c[0]` and NOT `q_c[3]`.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)

import ast
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.core.fv_tp_2d import _ppm_1d
from tests.legoesm_paths import legoesm_source_path


def test_iter884_source_uses_fortran_faithful_indices():
    """AST scan: ``_ppm_1d`` source MUST iterate over indices
    `[0, 1, 2, -3, -2, -1]` (Fortran-faithful) and MUST NOT iterate
    over `[1, 2, 3, -4, -3, -2]` (pre-iter-884 buggy form).
    """
    src_path = legoesm_source_path("core/fv_tp_2d.py")
    tree = ast.parse(src_path.read_text())

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "_ppm_1d"),
        None,
    )
    assert fn is not None, "Could not find _ppm_1d in fv_tp_2d.py"

    # Look for For loops whose iter is a List literal containing
    # int constants (possibly via UnaryOp(USub) for negatives).
    def _list_to_ints(node):
        if not isinstance(node, ast.List):
            return None
        ints = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                ints.append(elt.value)
            elif (isinstance(elt, ast.UnaryOp)
                    and isinstance(elt.op, ast.USub)
                    and isinstance(elt.operand, ast.Constant)
                    and isinstance(elt.operand.value, int)):
                ints.append(-elt.operand.value)
            else:
                return None
        return tuple(ints)

    fortran_faithful = (0, 1, 2, -3, -2, -1)
    pre_iter884_buggy = (1, 2, 3, -4, -3, -2)

    found_correct = False
    found_buggy = False
    for node in ast.walk(fn):
        if isinstance(node, ast.For):
            ints = _list_to_ints(node.iter)
            if ints == fortran_faithful:
                found_correct = True
            elif ints == pre_iter884_buggy:
                found_buggy = True

    assert found_correct, (
        f"`_ppm_1d` source does NOT contain a For loop iterating "
        f"over the Fortran-faithful indices [0, 1, 2, -3, -2, -1].  "
        f"Per Fortran tp_core.F90:629/648 the pert_ppm iv=1 boundary "
        f"call uses bl/br indices 0, 1, 2 (left) and npx-2, npx-1, "
        f"npx (right).  In our Python q_c convention these map to "
        f"[0, 1, 2] (left) and [-3, -2, -1] (right).  iter-884 "
        f"corrected the indices from [1, 2, 3, -4, -3, -2] to "
        f"[0, 1, 2, -3, -2, -1].")
    assert not found_buggy, (
        f"`_ppm_1d` source contains a For loop iterating over the "
        f"pre-iter-884 buggy indices [1, 2, 3, -4, -3, -2].  This "
        f"shifts the iv=1 constraint INWARD by one cell on each "
        f"boundary side, silently skipping the boundary halo cell "
        f"and incorrectly applying iv=1 to a deeper interior cell. "
        f"Restore the iter-884 fix.")


def test_iter884_pert_ppm_applied_at_boundary_halo_cells():
    """Behavioural test: with `use_duogrid=False`, `_ppm_1d` MUST
    apply iv=1 (= `_pert_ppm`) at the SPECIFIC q_c indices
    [0, 1, 2, -3, -2, -1].

    Iter-884b (Codex iter-884 stop-time fix): the original count-
    only test did NOT distinguish [0,1,2,-3,-2,-1] from
    [1,2,3,-4,-3,-2] (both have 6 indices).  Codex correctly
    flagged that as an invalid sentinel.

    iter-884b strategy: capture the `id(bl_slice)` of each
    `_pert_ppm` invocation, then INSPECT which q_c index those
    slices were taken from by comparing against pre-computed
    `bl[:, k, :]` views for every k in [-1, 0, ..., n+1].  The
    captured ids reveal exactly which indices the for-loop
    iterated over — no count-equality false-pass.
    """
    from unittest import mock
    from legoesm.core import fv_tp_2d as fv_tp_2d_mod

    n = 8
    M = 4
    rng = np.random.default_rng(884)
    # `_ppm_1d` expects q with halo=2 padding (shape (6, n+4, M))
    # — the function itself pads with halo=1 to shape (6, n+6, M)
    # and then `q_c = qe[:, 2:-2, :]` has shape (6, n+2, M).
    q = jnp.asarray(rng.normal(size=(6, n + 4, M)))

    # Make every cell mean's bl coordinate UNIQUE so we can read it
    # back from the captured slice.  We patch _pert_ppm to record
    # the `bl_slice[0, 0]` value (which encodes the q_c index used).
    captured_k_values = []

    def tracking_pert_ppm(bl_slice, br_slice):
        # bl_slice has shape (6, M); read the embedded marker.
        v = float(np.asarray(bl_slice).flat[0])
        captured_k_values.append(int(round(v)))
        return bl_slice, br_slice

    # Helper: replace the al → bl/br computation inside _ppm_1d so
    # bl[face, k, m] = k for all k.  Easiest path: monkey-patch
    # `_pert_ppm_iv0` to return controlled bl/br.
    def fake_pert_ppm_iv0(q_c, bl, br):
        # q_c shape (6, n+2, M).  Return bl[face, k, m] = k
        # so the subsequent `_pert_ppm` calls receive identifiable
        # slices.
        n_plus_2 = bl.shape[1]
        marker = jnp.broadcast_to(
            jnp.arange(n_plus_2, dtype=bl.dtype)[None, :, None],
            bl.shape)
        return marker, marker

    with mock.patch.object(fv_tp_2d_mod, "_pert_ppm", tracking_pert_ppm):
        with mock.patch.object(
                fv_tp_2d_mod, "_pert_ppm_iv0", fake_pert_ppm_iv0):
            captured_k_values.clear()
            _ppm_1d(q, n, use_duogrid=False)

    # n_plus_2 = n + 2 = 10.  Negative indices resolve to:
    #   q_c[-1] → bl[k=n+1] = bl[k=9]
    #   q_c[-2] → bl[k=n]   = bl[k=8]
    #   q_c[-3] → bl[k=n-1] = bl[k=7]
    # So Fortran-faithful indices [0, 1, 2, -3, -2, -1] map to
    # marker values [0, 1, 2, 7, 8, 9] for n=8.
    expected_markers = sorted([0, 1, 2, n - 1, n, n + 1])
    actual_markers = sorted(captured_k_values)

    assert actual_markers == expected_markers, (
        f"`_ppm_1d(use_duogrid=False)` invoked `_pert_ppm` at q_c "
        f"indices with markers {actual_markers}; expected "
        f"{expected_markers} (Fortran-faithful indices [0, 1, 2, "
        f"-3, -2, -1] resolved with q_c.shape[1]={n+2}).\n"
        f"\n"
        f"Pre-iter-884 indices [1, 2, 3, -4, -3, -2] would produce "
        f"markers {sorted([1, 2, 3, n - 2, n - 1, n])} — different "
        f"from the Fortran-faithful expectation.\n"
        f"\n"
        f"This test catches the index regression that the count-only "
        f"check missed (both pre- and post-iter-884 invoke "
        f"`_pert_ppm` 6 times; only the SPECIFIC indices differ).")


def test_iter884_no_pert_ppm_in_duogrid_path():
    """Sanity (also covered by iter-516/iter-517 in test_duogrid.py):
    with `use_duogrid=True`, `_pert_ppm` must NOT be invoked for the
    boundary iv=1 constraint at all.  This is the
    `not (bounded_domain or duogrid)` gate from Fortran tp_core.F90:612.
    iter-884 preserves this gate exactly; the fix only changes the
    indices INSIDE the gate, not the gate condition.
    """
    from unittest import mock
    from legoesm.core import fv_tp_2d as fv_tp_2d_mod

    n = 8
    M = 4
    rng = np.random.default_rng(885)
    # `_ppm_1d` expects q with halo=2 padding (shape (6, n+4, M)).
    q = jnp.asarray(rng.normal(size=(6, n + 4, M)))

    n_calls = {"n": 0}

    def counting_pert_ppm(bl_slice, br_slice):
        n_calls["n"] += 1
        return bl_slice, br_slice

    with mock.patch.object(fv_tp_2d_mod, "_pert_ppm", counting_pert_ppm):
        _ppm_1d(q, n, use_duogrid=True)

    assert n_calls["n"] == 0, (
        f"`_ppm_1d(use_duogrid=True)` invoked `_pert_ppm` "
        f"{n_calls['n']} times; expected 0.  iter-884 must not "
        f"break the duogrid gate that skips iv=1 boundary "
        f"constraints (Fortran tp_core.F90:612).")
