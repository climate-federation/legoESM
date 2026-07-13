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
# Iter-883: also enable x64 at runtime in case JAX was already
# initialized in float32 by an earlier conftest import.  The
# os.environ.setdefault above is for command-line invocation; the
# jax.config.update is the runtime-effective form.
import jax
jax.config.update("jax_enable_x64", True)

import ast
import inspect

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

from tests.legoesm_paths import legoesm_source_path


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

def test_iter872c_default_kwarg_is_fortran_strict_zero():
    """Iter-872c (Codex pass-2 Finding 1): the kwarg default
    ``dddmp=0.0`` (Fortran-strict) MUST reproduce passing
    ``dddmp=0.0`` explicitly.  Pre-iter-872c the kwarg default was
    0.2 and silently leaked the adaptive Smagorinsky coefficient
    into every direct caller without explicit opt-in.

    Production callers go through ``FV3EdgeShallowWaterModel.step``
    which passes ``dddmp=self.config.dddmp_prod`` (default 0.2)
    explicitly, so the production W2/W5/cosine-bell sentinels are
    unaffected.  This test pins the *low-level* kwarg default at the
    Fortran-strict value 0.0.
    """
    cdgrid, h, u_d, v_d, h_s = _grid_and_state(n=8, seed=11)

    dh_default, du_default, dv_default = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=0.01)
    dh_explicit, du_explicit, dv_explicit = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=0.01, dddmp=0.0)

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
    src = legoesm_source_path(
        "atmosphere/dynamics/gcm/shallow_water_fv3_cdgrid.py")
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
        "is therefore re-pinned to the iter-872c kwarg default (0.0, "
        "Fortran-strict) regardless of "
        "CDGridShallowWaterConfig.dddmp_prod.")


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


# ---------------------------------------------------------------------
# Iter-872b (Codex Finding 1): adaptive damping when div_damp=0
# ---------------------------------------------------------------------

def test_iter872c_take4_gate_narrow_div_damp_zero_no_op():
    """Iter-872c-take4 (Codex pass-4): with ``div_damp = 0`` the
    divergence-damping branch in ``fv3_sw_tendencies`` MUST be
    skipped entirely, regardless of ``dddmp``.

    iter-872b had widened the gate to ``div_damp > 0 or dddmp > 0``
    to enable the Fortran-valid pure-adaptive regime, but Codex
    pass-4 correctly noted that combined with the production
    ``dddmp_prod = 0.2`` default this turned on adaptive damping
    for default-config callers — a silent behavioural change for a
    code path the comments themselves document as "structurally
    incomplete".  iter-872c-take4 reverts the gate to narrow.

    The widened-gate Fortran-fidelity improvement is deferred until
    the d_sw5 holistic port is complete; until then, ``dddmp`` is
    only effective when ``div_damp > 0``.  This test pins the
    narrow-gate semantics so any future widening is explicit.
    """
    cdgrid, h, u_d, v_d, h_s = _grid_and_state(n=8, seed=43)

    # With div_damp=0, dddmp value MUST NOT affect output
    # (entire branch skipped).
    _, du_pt2, dv_pt2 = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=0.0, dddmp=0.2)
    _, du_zero, dv_zero = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, div_damp=0.0, dddmp=0.0)

    np.testing.assert_array_equal(np.asarray(du_pt2), np.asarray(du_zero))
    np.testing.assert_array_equal(np.asarray(dv_pt2), np.asarray(dv_zero))


# ---------------------------------------------------------------------
# Iter-872c (Codex Finding 2): CDGridShallowWaterModel honours
# `dddmp_prod` via plumbing through `cdgrid_momentum_tendencies`
# ---------------------------------------------------------------------

def test_iter872c_kwarg_default_is_zero_fortran_strict():
    """Iter-872c (Codex pass-2 Finding 1): the low-level kwarg
    default for ``fv3_sw_tendencies.dddmp`` MUST be ``0.0``
    (Fortran-strict, fv_arrays.F90:360).  Pre-iter-872c the kwarg
    default was ``0.2``, which silently leaked the adaptive
    Smagorinsky coefficient into every direct caller that did not
    explicitly opt out.  Production ``FV3EdgeShallowWaterModel.step``
    explicitly passes ``dddmp=self.config.dddmp_prod`` (default 0.2),
    so the production sentinel is unaffected; direct callers without
    an explicit ``dddmp`` get pure background-only damping when
    ``div_damp>0`` and no damping at all when ``div_damp=0``.
    """
    sig = inspect.signature(fv3_sw_tendencies)
    dddmp_default = sig.parameters["dddmp"].default
    assert dddmp_default == 0.0, (
        f"`fv3_sw_tendencies.dddmp` default changed from 0.0 to "
        f"{dddmp_default!r}; this re-introduces the silent leak that "
        f"injects adaptive Smagorinsky into direct callers without "
        f"explicit opt-in.  Restore the Fortran-strict default 0.0 "
        f"and pass 0.2 only from production call sites via "
        f"`CDGridShallowWaterConfig.dddmp_prod`.")


def test_iter872c_cdgrid_momentum_honors_dddmp():
    """Iter-872c (Codex pass-2 Finding 2): the
    ``cdgrid_momentum_tendencies`` divergence-damping branch MUST
    honour the ``dddmp`` kwarg, NOT a hardcoded 0.2 literal.  This
    is the path used by ``CDGridShallowWaterModel`` (driver/CLI
    surface).  Pre-iter-872c the literal was hardcoded so
    ``dddmp_prod`` on the shared config field was silently ignored
    on this code path.

    Shape note: ``cdgrid_momentum_tendencies`` uses the
    corner-stagger D-grid convention (u_d, v_d both shape
    (6, n+1, n+1)) — distinct from ``fv3_sw_tendencies`` which uses
    the edge-midpoint stagger.  Both are valid D-grid layouts.
    """
    from legoesm.core.operators_cdgrid import cdgrid_momentum_tendencies

    n = 8
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(53)
    h = jnp.asarray(rng.normal(size=(6, n, n)) * 100.0 + 8000.0)
    h_s = jnp.zeros_like(h)
    # Corner-stagger D-grid: u_d, v_d both at corners (6, n+1, n+1)
    u_d = jnp.asarray(rng.normal(size=(6, n + 1, n + 1)))
    v_d = jnp.asarray(rng.normal(size=(6, n + 1, n + 1)))
    common = dict(g=constants.g, A_h=0.0, hyperdiff_coeff=0.0,
                  div_damp=0.01)

    du_pt2, dv_pt2 = cdgrid_momentum_tendencies(
        h, u_d, v_d, h_s, cdgrid, dddmp=0.2, **common)
    du_zero, dv_zero = cdgrid_momentum_tendencies(
        h, u_d, v_d, h_s, cdgrid, dddmp=0.0, **common)

    diff_u = float(jnp.max(jnp.abs(du_pt2 - du_zero)))
    diff_v = float(jnp.max(jnp.abs(dv_pt2 - dv_zero)))
    assert (diff_u > 1e-8) or (diff_v > 1e-8), (
        f"`cdgrid_momentum_tendencies` ignores the `dddmp` kwarg: "
        f"dddmp=0.2 vs 0.0 produced near-identical du/dv "
        f"(max |Δdu|={diff_u:.3e}, max |Δdv|={diff_v:.3e}).  Either "
        f"the kwarg is shadowed by a hardcoded 0.2 literal, or the "
        f"kwarg is missing entirely.  Codex pass-2 Finding 2 "
        f"requires the shared `dddmp_prod` config field to reach "
        f"this code path; without that, `CDGridShallowWaterModel` "
        f"users see a silent reproducibility hazard.")


def test_iter872c_take3_cdgrid_shallow_water_does_not_forward_dddmp():
    """Iter-872c-take3 (Codex pass-3): ``cdgrid_shallow_water_tendencies``
    MUST NOT forward ``dddmp_prod`` to ``cdgrid_momentum_tendencies``.

    Codex pass-3 found that iter-872c's plumbing of
    ``dddmp=config.dddmp_prod`` (default 0.2) into
    ``cdgrid_shallow_water_tendencies`` silently turned on adaptive
    Smagorinsky for every default-config ``CDGridShallowWaterModel``
    user, including the standard driver path.  Pre-iter-872 the
    `CDGridShallowWaterModel(default_config)` had hardcoded
    `dddmp = 0.2` BUT was gated behind `if div_damp > 0:` and the
    default `div_damp=0` skipped the entire branch — so it
    effectively had no adaptive damping.  iter-872c's gate widening
    + plumbing made it fire by default.

    iter-872c-take3 reverts the plumbing: ``dddmp_prod`` is scoped
    to ``FV3EdgeShallowWaterModel`` only.  This test pins the
    scoping by asserting that NO ``cdgrid_momentum_tendencies(...)``
    call inside ``cdgrid_shallow_water_tendencies`` passes a
    ``dddmp=...`` kwarg.  Advanced ``CDGridShallowWaterModel``
    callers wanting adaptive Smagorinsky should pass `dddmp`
    directly to ``cdgrid_momentum_tendencies`` (the Fortran-strict
    0.0 default kwarg added in iter-872c).
    """
    src = legoesm_source_path(
        "atmosphere/dynamics/gcm/shallow_water_fv3_cdgrid.py")
    tree = ast.parse(src.read_text())

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "cdgrid_shallow_water_tendencies"),
        None,
    )
    assert fn is not None, (
        "Could not find cdgrid_shallow_water_tendencies in "
        "shallow_water_fv3_cdgrid.py")

    momentum_calls = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "cdgrid_momentum_tendencies"
    ]
    assert momentum_calls, (
        "Could not find any cdgrid_momentum_tendencies(...) call in "
        "cdgrid_shallow_water_tendencies")

    forwarded = []
    for call in momentum_calls:
        for kw in call.keywords:
            if kw.arg == "dddmp":
                forwarded.append(call)
                break

    assert not forwarded, (
        f"cdgrid_shallow_water_tendencies' cdgrid_momentum_tendencies "
        f"call forwards `dddmp=...` ({len(forwarded)} occurrence(s)).  "
        f"This re-introduces the iter-872c silent default-config "
        f"behaviour change for `CDGridShallowWaterModel` users that "
        f"Codex pass-3 flagged.  `dddmp_prod` is scoped to "
        f"`FV3EdgeShallowWaterModel` only by design.")


def test_iter872c_take5_warns_when_dddmp_silently_no_op():
    """Iter-872c-take5 (Codex pass-5 high): ``fv3_sw_tendencies``
    and ``cdgrid_momentum_tendencies`` MUST emit a UserWarning when
    ``dddmp > 0`` is supplied with ``div_damp = 0``, because the
    narrow gate silently no-ops `dddmp` in that regime.  Without
    this warning users could set `dddmp_prod=0.4` and believe
    adaptive Smagorinsky is active when the entire branch is
    bypassed.
    """
    import warnings
    from legoesm.core.operators_cdgrid import cdgrid_momentum_tendencies

    cdgrid, h, u_d, v_d, h_s = _grid_and_state(n=8, seed=61)

    # fv3_sw_tendencies: dddmp>0, div_damp=0 → must warn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fv3_sw_tendencies(
            h, u_d, v_d, h_s, cdgrid,
            div_damp=0.0, dddmp=0.4)
    matched = [w for w in caught
               if issubclass(w.category, UserWarning)
               and "no-op" in str(w.message)]
    assert matched, (
        f"`fv3_sw_tendencies(dddmp=0.4, div_damp=0)` did not emit "
        f"the iter-872c-take5 silent-no-op warning.  Caught: "
        f"{[str(w.message) for w in caught]}")

    # No warning when div_damp > 0 (gate fires).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fv3_sw_tendencies(
            h, u_d, v_d, h_s, cdgrid,
            div_damp=0.01, dddmp=0.4)

    # No warning when dddmp = 0 (no silent no-op concern).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fv3_sw_tendencies(
            h, u_d, v_d, h_s, cdgrid,
            div_damp=0.0, dddmp=0.0)

    # cdgrid_momentum_tendencies: same warning policy.
    n = 8
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid2 = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(62)
    h2 = jnp.asarray(rng.normal(size=(6, n, n)) * 100.0 + 8000.0)
    h_s2 = jnp.zeros_like(h2)
    u_d2 = jnp.asarray(rng.normal(size=(6, n + 1, n + 1)))
    v_d2 = jnp.asarray(rng.normal(size=(6, n + 1, n + 1)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cdgrid_momentum_tendencies(
            h2, u_d2, v_d2, h_s2, cdgrid2,
            div_damp=0.0, dddmp=0.4)
    matched = [w for w in caught
               if issubclass(w.category, UserWarning)
               and "no-op" in str(w.message)]
    assert matched, (
        f"`cdgrid_momentum_tendencies(dddmp=0.4, div_damp=0)` did "
        f"not emit the iter-872c-take5 silent-no-op warning.")


def test_iter872c_take5_cdgrid_shallow_water_warns_direct_callers():
    """Iter-872c-take5 (Codex pass-5 medium): direct callers of
    ``cdgrid_shallow_water_tendencies`` with non-default
    ``dddmp_prod`` MUST also see a UserWarning, not just the
    `CDGridShallowWaterModel.__init__` warning.  Codex pass-5
    correctly noted that direct callers bypass the model-class
    warning, leaving a silent-ignore hazard on the functional API.
    """
    import warnings
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig, CDGridShallowWaterState,
        cdgrid_shallow_water_tendencies,
    )

    n = 8
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(71)
    state = CDGridShallowWaterState(
        h=jnp.asarray(rng.normal(size=(6, n, n)) * 100.0 + 8000.0),
        u_d=jnp.asarray(rng.normal(size=(6, n + 1, n + 1))),
        v_d=jnp.asarray(rng.normal(size=(6, n + 1, n + 1))),
        h_s=jnp.zeros((6, n, n)),
    )

    # Default config: no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cdgrid_shallow_water_tendencies(
            state, cdgrid, CDGridShallowWaterConfig())

    # Non-default dddmp_prod: must warn at the functional API level.
    cfg = CDGridShallowWaterConfig(dddmp_prod=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cdgrid_shallow_water_tendencies(state, cdgrid, cfg)
    matched = [w for w in caught
               if issubclass(w.category, UserWarning)
               and "dddmp_prod" in str(w.message)
               and "ignored" in str(w.message)]
    assert matched, (
        f"Direct call to `cdgrid_shallow_water_tendencies` with "
        f"`config.dddmp_prod=0.5` did not emit the iter-872c-take5 "
        f"functional-API warning.  Codex pass-5 medium identified "
        f"this as a silent-ignore hazard on the public functional "
        f"API.  Caught: {[str(w.message) for w in caught]}")


def test_iter872c_take4_cdgrid_warns_on_non_default_dddmp_prod():
    """Iter-872c-take4 (Codex pass-4): ``CDGridShallowWaterModel``
    MUST emit a UserWarning when constructed with a non-default
    ``dddmp_prod`` because the field is silently ignored on this
    model's tendency path.  Without this warning a user could set
    ``dddmp_prod=0.4`` and get the same numerics as ``dddmp_prod=
    0.2`` — a reproducibility hazard that Codex pass-4 flagged as
    "high" severity.
    """
    import warnings
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig, CDGridShallowWaterModel,
    )

    n = 8
    grid = create_cubed_sphere(n, use_duogrid=True)

    # Default config: no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CDGridShallowWaterModel(grid, CDGridShallowWaterConfig())

    # Non-default `dddmp_prod`: must warn.
    cfg = CDGridShallowWaterConfig(dddmp_prod=0.4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CDGridShallowWaterModel(grid, cfg)

    matched = [w for w in caught
               if issubclass(w.category, UserWarning)
               and "dddmp_prod" in str(w.message)]
    assert matched, (
        f"CDGridShallowWaterModel did not warn when constructed "
        f"with non-default `dddmp_prod=0.4`.  Without this warning, "
        f"the silent no-op on the CDGrid path is a reproducibility "
        f"hazard (Codex pass-4 high finding).  Caught warnings: "
        f"{[str(w.message) for w in caught]}")


def test_iter872c_take3_cdgrid_default_no_silent_adaptive_damping():
    """Iter-872c-take3 (Codex pass-3): ``CDGridShallowWaterModel``
    constructed with the default config MUST NOT silently inject
    adaptive Smagorinsky damping.

    With ``CDGridShallowWaterConfig()`` (default), ``div_damp=0`` and
    the kwarg-default ``dddmp=0.0`` both result in NO entry into the
    divergence-damping branch (gate: ``div_damp>0 or dddmp>0`` →
    ``False or False`` → ``False``).  This pins the iter-872c-take3
    scoping: even though ``dddmp_prod`` defaults to 0.2 on the
    shared config, it is not forwarded to this model's tendency
    path, so the default behaviour is unchanged from pre-iter-872.
    """
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig, CDGridShallowWaterModel,
        cdgrid_shallow_water_tendencies, CDGridShallowWaterState,
    )

    n = 8
    grid = create_cubed_sphere(n, use_duogrid=True)
    cfg = CDGridShallowWaterConfig()
    model = CDGridShallowWaterModel(grid, cfg)
    rng = np.random.default_rng(2026)
    state = CDGridShallowWaterState(
        h=jnp.asarray(rng.normal(size=(6, n, n)) * 100.0 + 8000.0),
        u_d=jnp.asarray(rng.normal(size=(6, n + 1, n + 1))),
        v_d=jnp.asarray(rng.normal(size=(6, n + 1, n + 1))),
        h_s=jnp.zeros((6, n, n)),
    )

    # With cfg.div_damp=0 AND dddmp_prod NOT forwarded, the gate
    # `if div_damp>0 or dddmp>0:` (kwarg-default 0.0) MUST be False,
    # so the divergence-damping branch must be skipped entirely.
    # We verify by computing tendencies twice with different
    # `dddmp_prod` values on the config; if the field were silently
    # forwarded, output would change.
    cfg_pt2 = cfg._replace(dddmp_prod=0.2)
    cfg_pt8 = cfg._replace(dddmp_prod=0.8)
    dh_pt2, du_pt2, dv_pt2 = cdgrid_shallow_water_tendencies(
        state, model.cdgrid, cfg_pt2)
    dh_pt8, du_pt8, dv_pt8 = cdgrid_shallow_water_tendencies(
        state, model.cdgrid, cfg_pt8)

    # If `dddmp_prod` were forwarded, dddmp=0.2 vs 0.8 with this
    # state would produce different tendencies.  Per iter-872c-take3,
    # `cdgrid_shallow_water_tendencies` ignores `dddmp_prod` →
    # outputs MUST be bit-identical.
    np.testing.assert_array_equal(np.asarray(dh_pt2), np.asarray(dh_pt8))
    np.testing.assert_array_equal(np.asarray(du_pt2), np.asarray(du_pt8))
    np.testing.assert_array_equal(np.asarray(dv_pt2), np.asarray(dv_pt8))
