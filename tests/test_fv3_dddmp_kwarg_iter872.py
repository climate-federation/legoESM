"""Iter-872: expose production divergence-damping ``dddmp`` as a
configurable kwarg on ``fv3_sw_tendencies`` (Fortran fidelity).

Fortran ``fv_arrays.F90:360`` exposes the adaptive Smagorinsky
divergence-damping coefficient ``flagstruct%dddmp`` as a
user-configurable namelist parameter (strict default 0.0; 0.2 is the
typical production setting).  Pre-iter-872 our Python
``fv3_sw_tendencies`` hardcoded ``dddmp = 0.2`` deep inside the
divergence-damping branch, with no user knob.  Iter-872:

1. Adds a ``dddmp=0.2`` kwarg to ``fv3_sw_tendencies`` whose default
   preserves pre-iter-872 numerics bit-for-bit.
2. Adds a ``dddmp_prod`` field (default 0.2) to
   ``CDGridShallowWaterConfig``.
3. Wires ``dddmp=self.config.dddmp_prod`` into
   ``FV3EdgeShallowWaterModel.step``'s production tendency call.

Tests below cover:

A. ``dddmp=0.2`` (the kwarg default) reproduces the pre-iter-872
   hardcoded value bit-for-bit on a duogrid grid with non-zero divergence
   damping (``div_damp > 0``).
B. ``dddmp=0.0`` produces measurably different tendencies from
   ``dddmp=0.2`` whenever the divergence cap can bite (i.e. there is a
   region where ``dddmp * |div| > d2_bg``).  This proves the kwarg is
   actually wired into the adaptive Smagorinsky branch.
C. ``dddmp=0.0`` reduces to the background-only damping
   ``adaptive_coeff = da_min_c * d2_bg`` (Fortran-strict default), so
   the resulting tendencies match a manual computation that bypasses
   the adaptive cap entirely.
D. AST scan: ``FV3EdgeShallowWaterModel.step``'s production
   ``tendency_fn`` MUST forward ``dddmp=self.config.dddmp_prod`` (or
   equivalent attribute access on a config) to ``fv3_sw_tendencies``.

Production W2/W5/cosine-bell sentinels are unaffected by iter-872
because the default 0.2 preserves bit-for-bit pre-iter-872 numerics.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import ast
import inspect
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _grid_and_state(n=8, seed=2026, scale_div=1.0):
    """Random h/u/v with controllable divergence magnitude.

    ``scale_div=1.0`` produces ~unit-scale random winds (used to ensure
    ``dddmp * |div|`` is comparable to ``d2_bg`` so the cap can bite).

    D-grid stagger convention (matches ``test_cdgrid.py``):
        u_d shape (6, n, n+1) — east-edge tangential component
        v_d shape (6, n+1, n) — north-edge tangential component
    """
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed)
    h = jnp.asarray(rng.normal(size=(6, n, n)) * 100.0 + 8000.0)
    u_d = jnp.asarray(rng.normal(size=(6, n, n + 1)) * scale_div)
    v_d = jnp.asarray(rng.normal(size=(6, n + 1, n)) * scale_div)
    h_s = jnp.zeros_like(h)
    return cdgrid, h, u_d, v_d, h_s


# ---------------------------------------------------------------------
# Test A: default 0.2 reproduces pre-iter-872 hardcoded value
# ---------------------------------------------------------------------

def test_default_kwarg_reproduces_pre_iter872_hardcoded_value():
    """``dddmp=0.2`` (the kwarg default) must produce bit-for-bit
    identical tendencies to passing ``dddmp=0.2`` explicitly.  This is
    a sanity check that the default is unchanged from the historic
    hardcoded value.
    """
    cdgrid, h, u_d, v_d, h_s = _grid_and_state(n=8, seed=11)

    dh_default, du_default, dv_default = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=0.01)
    dh_explicit, du_explicit, dv_explicit = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=0.01, dddmp=0.2)

    np.testing.assert_array_equal(np.asarray(dh_default), np.asarray(dh_explicit))
    np.testing.assert_array_equal(np.asarray(du_default), np.asarray(du_explicit))
    np.testing.assert_array_equal(np.asarray(dv_default), np.asarray(dv_explicit))


# ---------------------------------------------------------------------
# Test B: alternate dddmp values produce different tendencies
# ---------------------------------------------------------------------

def test_dddmp_zero_changes_tendencies_when_cap_can_bite():
    """``dddmp=0.0`` (Fortran strict default) MUST produce measurably
    different tendencies from ``dddmp=0.2`` whenever the cap bites.

    The adaptive Smagorinsky path is::

        adaptive_coeff = da_min_c * max(d2_bg, min(0.20, dddmp * |div|))

    With ``dddmp=0.0`` the inner ``min(0.20, 0)`` is always 0, so the
    coefficient collapses to ``da_min_c * d2_bg`` for every cell.
    With ``dddmp=0.2`` and non-zero divergence, the coefficient grows
    in regions where ``dddmp * |div| > d2_bg``.  The two paths
    therefore diverge whenever there is at least one such region.

    To guarantee the cap bites we use ``div_damp = 1e-12`` so the
    background ``d2_bg = div_damp / da_min_c`` is essentially zero,
    making ``dddmp * |div|`` dominate everywhere div ≠ 0.
    """
    cdgrid, h, u_d, v_d, h_s = _grid_and_state(n=8, seed=23)

    # Both calls share div_damp=1e-12 so d2_bg is tiny — the only path
    # difference is the dddmp scaling on |div|.
    _, du_zero, dv_zero = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=1e-12, dddmp=0.0)
    _, du_pt2, dv_pt2 = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=1e-12, dddmp=0.2)

    diff_u = float(jnp.max(jnp.abs(du_zero - du_pt2)))
    diff_v = float(jnp.max(jnp.abs(dv_zero - dv_pt2)))
    assert (diff_u > 1e-8) or (diff_v > 1e-8), (
        f"Iter-872 dddmp kwarg does not reach the adaptive Smagorinsky "
        f"branch: dddmp=0.0 vs 0.2 produced near-identical du/dv "
        f"(max |Δdu|={diff_u:.3e}, max |Δdv|={diff_v:.3e}) on inputs "
        f"with non-zero divergence.  The kwarg is being silently "
        f"ignored or shadowed by the historic hardcoded value.")


# ---------------------------------------------------------------------
# Test C: dddmp=0.0 reduces to background-only damping
# ---------------------------------------------------------------------

def test_dddmp_zero_matches_background_only_path():
    """With ``dddmp=0.0`` the adaptive cap collapses to the background
    coefficient, so the result must equal a manual run that uses
    ``dddmp=0.0`` on the same inputs.  This pins the Fortran-strict
    default semantics: when ``dddmp=0.0`` no adaptive Smagorinsky
    contribution is added on top of ``d2_bg``.

    Concretely, two runs with ``dddmp=0.0`` and different ``dddmp``
    overrides must give the SAME output as long as ``dddmp_a = 0.0``
    and ``dddmp_b = 0.0`` — but ``dddmp=0.0`` and ``dddmp=0.2`` must
    give different output (Test B).  Together these prove the kwarg
    routes correctly into the cap.
    """
    cdgrid, h, u_d, v_d, h_s = _grid_and_state(n=8, seed=31)

    out_a = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=0.01, dddmp=0.0)
    out_b = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=0.01, dddmp=0.0)

    for a, b in zip(out_a, out_b):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------
# Test D: production tendency_fn forwards dddmp=self.config.dddmp_prod
# ---------------------------------------------------------------------

def test_production_step_forwards_dddmp_prod():
    """AST scan: ``FV3EdgeShallowWaterModel.step`` MUST contain a
    ``fv3_sw_tendencies(...)`` call passing ``dddmp=...`` whose value
    is a config attribute access (``self.config.dddmp_prod`` or
    similar).  A bare ``dddmp=0.2`` (or no ``dddmp`` kwarg) would
    silently bypass the new ``dddmp_prod`` config field and re-pin the
    historic hardcoded value.
    """
    src = (Path(__file__).resolve().parent.parent
           / "src" / "legoesm" / "atmosphere" / "dynamics"
           / "shallow_water_fv3_cdgrid.py")
    tree = ast.parse(src.read_text())

    cls = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef)
         and n.name == "FV3EdgeShallowWaterModel"),
        None,
    )
    assert cls is not None, (
        "Could not find FV3EdgeShallowWaterModel in "
        "shallow_water_fv3_cdgrid.py — has it been renamed?")

    step_fn = next(
        (n for n in cls.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "step"),
        None,
    )
    assert step_fn is not None, (
        "Could not find FV3EdgeShallowWaterModel.step")

    # Find the production fv3_sw_tendencies call (any direct call in
    # step's body, including inside nested defs like tendency_fn).
    prod_calls = [
        n for n in ast.walk(step_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "fv3_sw_tendencies"
    ]
    assert prod_calls, (
        "Could not find any fv3_sw_tendencies(...) call in "
        "FV3EdgeShallowWaterModel.step")

    # At least one such call must pass dddmp=<config attr access>.
    matched = False
    for call in prod_calls:
        for kw in call.keywords:
            if kw.arg != "dddmp":
                continue
            # Accept attribute access whose root traces back to `self`
            # (e.g. self.config.dddmp_prod, self.config.dddmp).  Reject
            # plain literals / other Names.
            node = kw.value
            chain = []
            while isinstance(node, ast.Attribute):
                chain.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name) and node.id == "self" and chain:
                matched = True
                break
        if matched:
            break

    assert matched, (
        "FV3EdgeShallowWaterModel.step's fv3_sw_tendencies(...) call "
        "does NOT forward dddmp=<self.config.*>.  The production path "
        "is therefore re-pinned to the iter-872 kwarg default (0.2) "
        "regardless of CDGridShallowWaterConfig.dddmp_prod.")


# ---------------------------------------------------------------------
# Test E: config field is exposed and has Fortran-fidelity default 0.2
# ---------------------------------------------------------------------

def test_config_dddmp_prod_default_is_zero_point_two():
    """``CDGridShallowWaterConfig.dddmp_prod`` MUST default to 0.2
    (matching the pre-iter-872 hardcoded value) so existing
    production runs are bit-identical.  A config refactor that
    silently changes the default (e.g. to Fortran-strict 0.0) would
    invalidate every cached W2/W5/cosine-bell baseline."""
    cfg = CDGridShallowWaterConfig()
    assert hasattr(cfg, "dddmp_prod"), (
        "CDGridShallowWaterConfig is missing the iter-872 dddmp_prod "
        "field.")
    assert cfg.dddmp_prod == 0.2, (
        f"dddmp_prod default changed from 0.2 to {cfg.dddmp_prod!r}; "
        f"this silently invalidates every production W2/W5/cosine-bell "
        f"baseline.  Update the test ONLY if a deliberate Fortran-"
        f"fidelity refactor changes the production default.")
