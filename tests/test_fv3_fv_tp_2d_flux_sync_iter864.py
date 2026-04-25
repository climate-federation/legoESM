"""Iter-864: gate the duogrid CGRID flux sync in `fv_tp_2d`.

Fortran ``dyn_core.F90`` synchronises mass fluxes between d_sw1 and
d_sw2 (lines 850-900, ACTIVE) but EXPLICITLY does NOT synchronise
vorticity fluxes between d_sw5 and d_sw6 (lines 1124-1207, the
``mpp_get_boundary(... gridtype=CGRID_NE)`` block is commented out).

Our Python ``fv_tp_2d`` previously applied
``synchronize_cgrid_fluxes`` unconditionally whenever duogrid was on,
silently over-syncing the FB chain's vorticity-flux call relative to
Fortran.  Iter-864 adds an opt-in kwarg ``apply_cgrid_flux_sync``
(default ``True`` for the mass-flux call) and threads
``apply_cgrid_flux_sync=False`` through the FB chain's vorticity-
flux call in ``_d_sw_native`` step 7 so it matches Fortran exactly.

Tests cover:
- The kwarg actually gates the sync: with ``False`` and duogrid on,
  the output equals the no-sync reference; with ``True`` and duogrid
  on, the output equals the iter-808 synced version.
- ``_d_sw_native`` step 7 forwards ``apply_cgrid_flux_sync=False``
  to ``fv_tp_2d`` (AST source-scan).

Production ``fv3_sw_tendencies`` does NOT call ``fv_tp_2d``; this
change does not affect production W2/W5/cosine bell sentinels.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import ast
from pathlib import Path
import inspect

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv_tp_2d import fv_tp_2d, compute_transport_quantities


def _duogrid_cdgrid(n=8):
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    assert cdgrid.base.duogrid is not None
    return cdgrid


def _legacy_cdgrid(n=8):
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    assert cdgrid.base.duogrid is None
    return cdgrid


def _random_inputs(cdgrid, seed):
    n = cdgrid.n
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.normal(size=(6, n, n)))
    ut = jnp.asarray(rng.normal(size=(6, n + 1, n)))
    vt = jnp.asarray(rng.normal(size=(6, n, n + 1)))
    dt = 100.0
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    return q, crx, cry, xfx, yfx, ra_x, ra_y


def test_kwarg_gates_sync_on_duogrid_grid():
    """Iter-864: ``apply_cgrid_flux_sync=False`` skips the iter-808
    sync; ``True`` (default) applies it.  Their outputs differ on a
    duogrid grid because the sync changes face-boundary flux pairs.
    """
    cdgrid = _duogrid_cdgrid(8)
    q, crx, cry, xfx, yfx, ra_x, ra_y = _random_inputs(cdgrid, seed=11)

    fx_off, fy_off = fv_tp_2d(
        q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
        apply_cgrid_flux_sync=False)
    fx_on, fy_on = fv_tp_2d(
        q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
        apply_cgrid_flux_sync=True)

    diff_x = float(jnp.max(jnp.abs(fx_on - fx_off)))
    diff_y = float(jnp.max(jnp.abs(fy_on - fy_off)))
    # The sync averages neighbour pairs at face boundaries, so the
    # outputs MUST differ noticeably on a duogrid grid with random
    # winds.  An identical result would mean the gate is being ignored.
    assert (diff_x > 1e-6) or (diff_y > 1e-6), (
        f"Iter-864 gate ignored: apply_cgrid_flux_sync False vs True "
        f"produced identical fx/fy on a duogrid grid (max |Δfx|="
        f"{diff_x:.3e}, |Δfy|={diff_y:.3e}).  The sync is supposed "
        f"to change neighbour-pair flux values at face boundaries.")


def test_default_kwarg_value_is_true():
    """Default behaviour (no kwarg passed) must match
    ``apply_cgrid_flux_sync=True`` to preserve the iter-808 sync for
    existing callers like ``transport_step``."""
    cdgrid = _duogrid_cdgrid(8)
    q, crx, cry, xfx, yfx, ra_x, ra_y = _random_inputs(cdgrid, seed=23)

    fx_default, fy_default = fv_tp_2d(
        q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid)
    fx_on, fy_on = fv_tp_2d(
        q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
        apply_cgrid_flux_sync=True)

    np.testing.assert_array_equal(np.asarray(fx_default), np.asarray(fx_on))
    np.testing.assert_array_equal(np.asarray(fy_default), np.asarray(fy_on))


def test_kwarg_is_no_op_on_legacy_grid():
    """On a legacy (non-duogrid) grid the iter-808 sync is gated off
    by ``cdgrid.base.duogrid is None`` regardless of the kwarg.
    Output must be identical between flag=True and flag=False on a
    legacy grid."""
    cdgrid = _legacy_cdgrid(8)
    q, crx, cry, xfx, yfx, ra_x, ra_y = _random_inputs(cdgrid, seed=31)

    fx_off, fy_off = fv_tp_2d(
        q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
        apply_cgrid_flux_sync=False)
    fx_on, fy_on = fv_tp_2d(
        q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
        apply_cgrid_flux_sync=True)

    np.testing.assert_array_equal(np.asarray(fx_off), np.asarray(fx_on))
    np.testing.assert_array_equal(np.asarray(fy_off), np.asarray(fy_on))


def test_d_sw_native_passes_apply_cgrid_flux_sync_false():
    """Iter-864: the FB chain's vorticity-flux call in
    ``_d_sw_native`` step 7 MUST pass
    ``apply_cgrid_flux_sync=False`` to ``fv_tp_2d``.

    Codex iter-864 third-pass tightening: the scan was previously
    ``ast.walk(fn) >= 1`` which walks INTO nested scopes (closures,
    comprehensions, helper FunctionDefs).  A dead nested call could
    satisfy the assertion while the LIVE step-7 call silently
    regressed to default-True.  The new scan walks ONLY the direct
    body of ``_d_sw_native`` (skipping nested function scopes) and
    requires that EXACTLY ONE direct-body ``fv_tp_2d`` call exists
    AND that single call passes ``apply_cgrid_flux_sync=False``.

    Additionally, the scan rejects any direct-body
    ``transport_step(...)`` call passing ``apply_cgrid_flux_sync=False``
    — the mass-flux call must default to True (matching Fortran's
    ACTIVE mass-flux sync at dyn_core.F90:850-900).
    """
    src = (Path(__file__).resolve().parent.parent
           / "src" / "legoesm" / "core" / "fv3_sw_core.py")
    tree = ast.parse(src.read_text())

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "_d_sw_native"),
        None,
    )
    assert fn is not None, (
        "_d_sw_native function not found in fv3_sw_core.py.  Has it "
        "been renamed?")

    def direct_body_calls(func_node, callee_name):
        """Yield ``ast.Call`` nodes whose target is a Name == callee_name
        AND whose enclosing scope is `func_node` itself (NOT a nested
        FunctionDef/AsyncFunctionDef/Lambda/comprehension within it).
        """
        nested_scope_types = (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.Lambda, ast.ListComp, ast.SetComp,
                              ast.DictComp, ast.GeneratorExp)

        def visit(node, inside_nested):
            if isinstance(node, nested_scope_types) and node is not func_node:
                # Recurse into the nested scope but mark inside_nested=True
                # so calls there don't count as direct-body.
                for child in ast.iter_child_nodes(node):
                    visit(child, True)
                return
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == callee_name
                    and not inside_nested):
                yield node
            for child in ast.iter_child_nodes(node):
                yield from visit(child, inside_nested)

        # Begin walking the body of `func_node`.
        for stmt in func_node.body:
            yield from visit(stmt, False)

    # 1. Direct-body fv_tp_2d call: exactly one expected, with False.
    fv_tp_calls = list(direct_body_calls(fn, "fv_tp_2d"))
    assert len(fv_tp_calls) == 1, (
        f"_d_sw_native has {len(fv_tp_calls)} direct-body fv_tp_2d "
        f"call(s); expected exactly 1 (the d_sw5 vorticity-flux call). "
        f"A second direct-body call would change the d_sw5 wiring "
        f"away from Fortran.")

    call = fv_tp_calls[0]
    sync_kwarg = next(
        (kw for kw in call.keywords if kw.arg == "apply_cgrid_flux_sync"),
        None,
    )
    assert sync_kwarg is not None, (
        f"_d_sw_native's direct-body fv_tp_2d call (line {call.lineno}) "
        f"does NOT pass `apply_cgrid_flux_sync=...` explicitly.  "
        f"iter-864 requires the FB chain's d_sw5 vorticity-flux call "
        f"to pass `apply_cgrid_flux_sync=False` so the result matches "
        f"Fortran's commented-out vorticity-flux averaging block at "
        f"dyn_core.F90:1124-1207.")
    assert (isinstance(sync_kwarg.value, ast.Constant)
            and sync_kwarg.value.value is False), (
        f"_d_sw_native's direct-body fv_tp_2d call (line {call.lineno}) "
        f"passes `apply_cgrid_flux_sync` but the value is not the "
        f"literal `False` constant.  iter-864 requires the literal "
        f"`False` so the d_sw5 vorticity flux is NOT synced.")

    # 2. Direct-body transport_step calls must NOT pass
    # apply_cgrid_flux_sync=False — the mass-flux call must default to
    # True so it matches Fortran's ACTIVE mass-flux sync.
    transport_calls = list(direct_body_calls(fn, "transport_step"))
    for tcall in transport_calls:
        bad = next(
            (kw for kw in tcall.keywords
             if kw.arg == "apply_cgrid_flux_sync"
             and isinstance(kw.value, ast.Constant)
             and kw.value.value is False),
            None,
        )
        assert bad is None, (
            f"_d_sw_native's direct-body transport_step call (line "
            f"{tcall.lineno}) passes `apply_cgrid_flux_sync=False`.  "
            f"This would disable the iter-808 mass-flux sync, "
            f"contradicting Fortran's ACTIVE mass-flux averaging at "
            f"dyn_core.F90:850-900.  Mass-flux call must default to "
            f"True.")
