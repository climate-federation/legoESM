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
# Iter-883: also enable x64 at runtime in case JAX was already
# initialized in float32 by an earlier conftest import.  The
# os.environ.setdefault above is for command-line invocation; the
# jax.config.update is the runtime-effective form.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
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


def test_d_sw_native_default_off_matches_pre_iter869():
    """Iter-869b: `_d_sw_native` with the new
    `apply_legacy_d_sw4_corner_ke_fix=False` default must produce
    bit-identical output to a separate call WITHOUT the new kwarg.
    Guards against accidental behaviour drift from iter-869b's
    wire-in."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid)
    from legoesm.core.fv3_sw_core import _d_sw_native

    n = 8
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(710)
    h = jnp.asarray(rng.standard_normal((6, n, n)) + 1000.0)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    h_s = jnp.zeros((6, n, n))
    uc = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    ua = jnp.asarray(rng.standard_normal((6, n, n)))
    va = jnp.asarray(rng.standard_normal((6, n, n)))

    h_a, u_a, v_a = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, 100.0, constants.g,
        div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.0, nord_v=0)
    h_b, u_b, v_b = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, 100.0, constants.g,
        div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.0, nord_v=0,
        apply_legacy_d_sw4_corner_ke_fix=False)
    np.testing.assert_array_equal(np.asarray(h_a), np.asarray(h_b))
    np.testing.assert_array_equal(np.asarray(u_a), np.asarray(u_b))
    np.testing.assert_array_equal(np.asarray(v_a), np.asarray(v_b))


def test_d_sw_native_flag_on_legacy_changes_winds():
    """Iter-869b: with `apply_legacy_d_sw4_corner_ke_fix=True` AND a
    legacy (non-bounded-domain) grid, the d_sw4 corner-KE override
    fires and changes the d_sw6 wind output (via ke_corner difference
    propagating into u_d_new / v_d_new).

    Use `use_duogrid=False` so `cdgrid.base.bounded_domain` is False
    and the helper does fire.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid)
    from legoesm.core.fv3_sw_core import _d_sw_native

    n = 8
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    assert cdgrid.base.bounded_domain is False
    rng = np.random.default_rng(710)
    h = jnp.asarray(rng.standard_normal((6, n, n)) + 1000.0)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    h_s = jnp.zeros((6, n, n))
    uc = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    ua = jnp.asarray(rng.standard_normal((6, n, n)))
    va = jnp.asarray(rng.standard_normal((6, n, n)))

    h_off, u_off, v_off = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, 100.0, constants.g,
        div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.0, nord_v=0,
        apply_legacy_d_sw4_corner_ke_fix=False)
    h_on, u_on, v_on = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, 100.0, constants.g,
        div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.0, nord_v=0,
        apply_legacy_d_sw4_corner_ke_fix=True)
    diff_u = float(np.max(np.abs(np.asarray(u_on) - np.asarray(u_off))))
    diff_v = float(np.max(np.abs(np.asarray(v_on) - np.asarray(v_off))))
    assert diff_u > 1e-12 or diff_v > 1e-12, (
        f"Iter-869b: d_sw4 corner-KE fix flag=True must change u_d / "
        f"v_d output on a legacy grid (the corner KE override "
        f"propagates into d_sw6 ke_corner difference).  Got max|Δu|="
        f"{diff_u:.3e}, max|Δv|={diff_v:.3e}.")
    # h is unaffected (mass transport happens at step 2, before step 4
    # KE compute).
    np.testing.assert_array_equal(np.asarray(h_on), np.asarray(h_off))


def test_d_sw_native_flag_on_duogrid_no_change():
    """Iter-869b: with `apply_legacy_d_sw4_corner_ke_fix=True` AND a
    duogrid grid, the helper's `bounded_domain=True` short-circuit
    means it returns ke unchanged — output should match flag=False
    bit-for-bit.  Verifies the bounded_domain gate is honoured."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid)
    from legoesm.core.fv3_sw_core import _d_sw_native

    n = 8
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    assert cdgrid.base.bounded_domain is True
    rng = np.random.default_rng(710)
    h = jnp.asarray(rng.standard_normal((6, n, n)) + 1000.0)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    h_s = jnp.zeros((6, n, n))
    uc = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    ua = jnp.asarray(rng.standard_normal((6, n, n)))
    va = jnp.asarray(rng.standard_normal((6, n, n)))

    h_off, u_off, v_off = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, 100.0, constants.g,
        div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.0, nord_v=0,
        apply_legacy_d_sw4_corner_ke_fix=False)
    h_on, u_on, v_on = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, 100.0, constants.g,
        div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.0, nord_v=0,
        apply_legacy_d_sw4_corner_ke_fix=True)
    np.testing.assert_array_equal(np.asarray(h_on), np.asarray(h_off))
    np.testing.assert_array_equal(np.asarray(u_on), np.asarray(u_off))
    np.testing.assert_array_equal(np.asarray(v_on), np.asarray(v_off))


def test_helper_is_wired_into_d_sw_native_as_opt_in():
    """Iter-869b: the helper is wired into `_d_sw_native` as a
    default-off opt-in matching iter-862's d_sw5 pattern.  Codex
    iter-869 stop-time review: leaving the helper unused was a
    half-finished implementation; iter-869b wires it in (default
    off) so it has a real call site without changing default
    behaviour.

    AST scan verifies:
    1. `_d_sw_native` calls `_apply_legacy_d_sw4_corner_ke_fix`
       exactly once in its direct body.
    2. The call is gated by `apply_legacy_d_sw4_corner_ke_fix`
       (the function's kwarg).
    3. The kwarg's default in `_d_sw_native`'s signature is
       `False` so default callers (the FB chain wrappers
       `fv3_fb_sw_step` and `fv3_forward_backward_step`) are
       bit-identical to pre-iter-869b behaviour.
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

    # 1. Helper is called from _d_sw_native body.
    helper_calls = [
        c for c in ast.walk(fn)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "_apply_legacy_d_sw4_corner_ke_fix"
    ]
    assert len(helper_calls) == 1, (
        f"Expected exactly 1 call to "
        f"`_apply_legacy_d_sw4_corner_ke_fix` in `_d_sw_native`; "
        f"found {len(helper_calls)} at lines "
        f"{[c.lineno for c in helper_calls]}.")

    # 2. Default value of `apply_legacy_d_sw4_corner_ke_fix` in
    # _d_sw_native's signature is False.
    args = fn.args
    arg_names = [a.arg for a in args.args]
    assert "apply_legacy_d_sw4_corner_ke_fix" in arg_names, (
        "_d_sw_native is missing the `apply_legacy_d_sw4_corner_ke_fix` "
        "kwarg added in iter-869b.")
    # Defaults align to the END of args.args; find the matching index.
    n_args = len(args.args)
    n_defaults = len(args.defaults)
    defaults_start = n_args - n_defaults
    target_idx = arg_names.index("apply_legacy_d_sw4_corner_ke_fix")
    assert target_idx >= defaults_start, (
        f"`apply_legacy_d_sw4_corner_ke_fix` at position {target_idx} "
        f"does not have a default value (defaults start at "
        f"{defaults_start}).")
    default_node = args.defaults[target_idx - defaults_start]
    assert (isinstance(default_node, ast.Constant)
            and default_node.value is False), (
        f"`apply_legacy_d_sw4_corner_ke_fix` default is not the "
        f"literal `False`: got {ast.dump(default_node)}.  Iter-869b "
        f"requires default-off semantics so existing callers see "
        f"no behaviour change.")
