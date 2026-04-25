"""Iter-869: pure-helper test for the Fortran d_sw4 cube-vertex KE fix.

Pins the structural arithmetic of `_apply_legacy_d_sw4_corner_ke_fix`
(`fv3_sw_core.py`).  Fortran ``sw_core.F90:1442-1465`` overrides KE
at the four cube-vertex corners with a special formula:

::

    dt6 = dt / 6.
    if (sw_corner) ke(1, 1) = dt6 * (
        (ut(1, 1) + ut(1, 0)) * u(1, 1) +
        (vt(1, 1) + vt(0, 1)) * v(1, 1) +
        (ut(1, 1) + vt(1, 1)) * u(0, 1) )

(and similar for SE/NE/NW with sign tweaks).

The fix is gated by Fortran's ``.not. bounded_domain .or. .not.
duogrid``.  In legoESM (where bounded_domain ↔ duogrid is True for
duogrid mode) this collapses to ``not bounded_domain``: skip in
duogrid; apply in legacy global cubed sphere.

Iter-869 ports the fix as a pure helper.  It is NOT yet wired into
`_d_sw_native` — that's a separate iter-870+ step (the FB chain is
unstable for independent reasons per iter-868; wiring this single
fix won't change the stability picture).  The helper exists so a
future Check 4 architectural port can use it.

Tests cover:
- The pure formula on synthetic inputs (sign + magnitude exact).
- The bounded_domain gate (return unchanged when True).
- Locality: only the 4 cube-vertex cells of `ke` change.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.core.fv3_sw_core import _apply_legacy_d_sw4_corner_ke_fix


def _synthetic_inputs(n, seed):
    rng = np.random.default_rng(seed)
    ke = jnp.asarray(rng.normal(size=(6, n + 1, n + 1)))
    ut = jnp.asarray(rng.normal(size=(6, n + 1, n)))
    vt = jnp.asarray(rng.normal(size=(6, n, n + 1)))
    u_d = jnp.asarray(rng.normal(size=(6, n, n + 1)))
    v_d = jnp.asarray(rng.normal(size=(6, n + 1, n)))
    return ke, ut, vt, u_d, v_d


def test_bounded_domain_returns_unchanged():
    """Fortran skips d_sw4 corner-KE fix in duogrid mode (which maps
    to bounded_domain=True in legoESM).  The helper must return
    `ke` unchanged in that case."""
    n = 6
    ke, ut, vt, u_d, v_d = _synthetic_inputs(n, seed=11)
    out = _apply_legacy_d_sw4_corner_ke_fix(
        ke, ut, vt, u_d, v_d, dt=100.0, bounded_domain=True)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ke))


def test_legacy_only_cube_vertex_corners_change():
    """In legacy (bounded_domain=False) mode, the helper must touch
    ONLY the four cube-vertex corners ([:, 0, 0], [:, -1, 0],
    [:, 0, -1], [:, -1, -1]).  All other ke cells must be unchanged.
    """
    n = 6
    ke, ut, vt, u_d, v_d = _synthetic_inputs(n, seed=23)
    out = _apply_legacy_d_sw4_corner_ke_fix(
        ke, ut, vt, u_d, v_d, dt=100.0, bounded_domain=False)
    delta = np.asarray(out) - np.asarray(ke)
    nonzero_mask = np.abs(delta) > 1e-14
    expected_corners = np.zeros_like(nonzero_mask)
    for f in range(6):
        expected_corners[f, 0, 0] = True
        expected_corners[f, -1, 0] = True
        expected_corners[f, 0, -1] = True
        expected_corners[f, -1, -1] = True
    interior_nonzero = nonzero_mask & ~expected_corners
    assert not interior_nonzero.any(), (
        f"Iter-869 d_sw4 corner-KE fix touched non-corner cells at "
        f"{np.argwhere(interior_nonzero).tolist()}; should localise "
        f"on cube-vertex corners only.")


def test_sw_corner_formula_exact_on_constant_inputs():
    """On constant inputs (ut=vt=u_d=v_d=1.0), the Fortran SW formula
    reduces to:

        ke(1,1) = dt6 * (1 + 1) * 1 + (1 + 1) * 1 + (1 + 1) * 1
                = dt6 * 2 * 3 = dt / 6 * 6 = dt

    Verifies the SW-corner arithmetic exactly without depending on
    halo behaviour (constant inputs make halo cells equal to
    interior).  The helper's output at the SW corner must equal dt.
    """
    n = 4
    dt = 100.0
    ke = jnp.zeros((6, n + 1, n + 1))
    ones_un = jnp.ones((6, n + 1, n))
    ones_vn = jnp.ones((6, n, n + 1))
    ones_un_e = jnp.ones((6, n, n + 1))   # u_d shape (6, n, n+1)
    ones_vn_e = jnp.ones((6, n + 1, n))   # v_d shape (6, n+1, n)
    out = _apply_legacy_d_sw4_corner_ke_fix(
        ke, ones_un, ones_vn, ones_un_e, ones_vn_e,
        dt=dt, bounded_domain=False)
    # SW corner: dt6 * (2*1 + 2*1 + 2*1) = dt6 * 6 = dt.
    expected_sw = dt
    np.testing.assert_allclose(
        float(out[0, 0, 0]), expected_sw, atol=1e-12,
        err_msg=(f"SW corner: expected {expected_sw}, got "
                 f"{float(out[0, 0, 0])} on constant-1 inputs."))
    # NW corner: dt6 * (2*1 + 2*1 + (1-1)*1) = dt6 * 4 = (2*dt)/3.
    expected_nw = (2 * dt) / 3
    np.testing.assert_allclose(
        float(out[0, 0, -1]), expected_nw, atol=1e-12,
        err_msg=(f"NW corner: expected {expected_nw}, got "
                 f"{float(out[0, 0, -1])} on constant-1 inputs.  "
                 f"NW formula has a `(ut - vt)` term that vanishes "
                 f"on constant inputs."))
    # SE corner: dt6 * (2*1 + 2*1 + (1-1)*1) = dt6 * 4 = (2*dt)/3.
    expected_se = (2 * dt) / 3
    np.testing.assert_allclose(
        float(out[0, -1, 0]), expected_se, atol=1e-12,
        err_msg=(f"SE corner: expected {expected_se}, got "
                 f"{float(out[0, -1, 0])} on constant-1 inputs."))
    # NE corner: dt6 * (2*1 + 2*1 + 2*1) = dt6 * 6 = dt.
    expected_ne = dt
    np.testing.assert_allclose(
        float(out[0, -1, -1]), expected_ne, atol=1e-12,
        err_msg=(f"NE corner: expected {expected_ne}, got "
                 f"{float(out[0, -1, -1])} on constant-1 inputs."))


def test_helper_is_not_wired_into_d_sw_native():
    """Iter-869 ports the helper but does NOT wire it into
    `_d_sw_native`.  Per iter-868, the FB chain is unstable for
    independent reasons; wiring this single fix would not stabilise
    it.  The helper exists for a future Check 4 architectural port.

    AST scan: `_d_sw_native` does NOT call
    `_apply_legacy_d_sw4_corner_ke_fix` directly.  When a future iter
    wires it in, this test should be UPDATED (delete or repurpose).
    """
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "src" / "legoesm" / "core" / "fv3_sw_core.py")
    tree = ast.parse(src.read_text())

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "_d_sw_native"),
        None,
    )
    assert fn is not None, "_d_sw_native function not found."

    helper_calls = [
        c for c in ast.walk(fn)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "_apply_legacy_d_sw4_corner_ke_fix"
    ]
    assert not helper_calls, (
        f"Iter-869: `_apply_legacy_d_sw4_corner_ke_fix` is now called "
        f"from `_d_sw_native` at line(s) {[c.lineno for c in helper_calls]}. "
        f"This test was written when the helper was NOT yet wired in; "
        f"a future iter that wires it should DELETE or REPURPOSE this "
        f"test.")
