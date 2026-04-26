"""Iter-873: regression sentinel pinning the default-OFF state of all
Fortran-fidelity opt-in flags on `CDGridShallowWaterConfig`.

Several iters added opt-in Fortran-fidelity flags to the production
shallow-water config:

- ``fortran_dir_aware_corners`` (iter-765) — dir=1/2 inner corner fill
  (sw_core.F90:3856-3915).
- ``fortran_a2b_corner_avg`` (iter-766) — a2b_ord4 3-pt scalar corner
  average (a2b_edge.F90:385-388).
- ``fortran_vector_corner_fill`` (iter-767) — fill_corners_agrid_r8
  vector swap+sign (fv_mp_mod.F90:1433-1457).
- ``apply_legacy_d_sw4_corner_ke_fix`` (iter-869b) — d_sw4 corner KE
  override (sw_core.F90:1438-1466).
- ``apply_legacy_d_sw5_corner_corrections`` (iter-871b) — d_sw5
  cube-vertex corner corrections.
- ``boundary_fix_skip_corners`` (iter-769) — skip the 4 cube-corner
  cells in boundary_fix smoothing.
- ``use_experimental_csw`` (pre-iter-862) — experimental CSW path.

Each was added default-OFF for explicit reasons (see per-iter doc
entries).  This regression sentinel ensures the default state is
preserved as the codebase evolves.  A test failure here means
someone flipped a default to True without reading the per-flag
deferral rationale.

Iter-873 also asserts the production matrix runner
(`scripts/run_atmosphere_test_matrix.py`) constructs
`CDGridShallowWaterConfig(...)` WITHOUT setting any of these flags
explicitly to True (i.e., relies on the defaults).  This catches a
future change to the matrix runner that silently activates a
fidelity flag and shifts the W2 sentinel baseline.

This is a Fortran-fidelity safety-net iter — no source change, no
behavioural impact, but locks the documented default-OFF state.
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
from pathlib import Path

import pytest

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
)


# Each entry: (field_name, expected_default, iter_added, doc_anchor).
# Note: `fortran_dir_aware_corners` is a function-level kwarg on
# `_arakawa_lamb_gradient` / `fv3_sw_tendencies`, NOT a config field
# (iter-765 marked it FALSIFIED — never promoted to config).  This
# inventory covers only `CDGridShallowWaterConfig` fields.
_FORTRAN_FIDELITY_OPT_IN_FLAGS = [
    ("fortran_a2b_corner_avg", False, "iter-766", "a2b_edge.F90:385-388"),
    ("fortran_vector_corner_fill", False, "iter-767",
     "fv_mp_mod.F90:1433-1457"),
    ("apply_legacy_d_sw4_corner_ke_fix", False, "iter-869b",
     "sw_core.F90:1438-1466"),
    ("apply_legacy_d_sw5_corner_corrections", False, "iter-871b",
     "_d_sw5_corner_divergence inner helper"),
    ("boundary_fix_skip_corners", False, "iter-769",
     "boundary-corner cascaded smoothing"),
    ("use_experimental_csw", False, "pre-iter-862",
     "experimental C-grid path (known unstable)"),
    # Iter-888c: Fortran s11/s14/s15 boundary-formula opt-in surfaced
    # on `CDGridShallowWaterConfig` so the canonical FB MODEL
    # (`FV3FBShallowWaterModel`) can opt in via config (Codex
    # iter-888b stop-time fix).
    ("apply_fortran_xppm_boundary", False, "iter-888c",
     "tp_core.F90:614-628 / :632-647 (s11/s14/s15 boundary formula)"),
]


@pytest.mark.parametrize("field, expected, iter_added, doc_anchor",
                         _FORTRAN_FIDELITY_OPT_IN_FLAGS)
def test_fortran_fidelity_flag_default_is_off(
        field, expected, iter_added, doc_anchor):
    """Each Fortran-fidelity opt-in flag MUST default to its
    documented OFF state.  Flipping a default silently changes the
    production W2/W5/cosine-bell baseline.  Update this test ONLY
    when a deliberate Fortran-fidelity refactor changes the default
    AND the docs/fv3_fortran_fidelity_review.md entry is updated to
    reflect the new behaviour.
    """
    cfg = CDGridShallowWaterConfig()
    assert hasattr(cfg, field), (
        f"CDGridShallowWaterConfig is missing the `{field}` field "
        f"(added in {iter_added} for {doc_anchor}).  Either the "
        f"field was renamed/removed, or the iter-873 sentinel is "
        f"out-of-date.  Update the test if the field was "
        f"deliberately removed.")
    actual = getattr(cfg, field)
    assert actual is expected, (
        f"`{field}` default changed from {expected!r} to {actual!r}.  "
        f"This was added in {iter_added} ({doc_anchor}) as a "
        f"deferred opt-in, default-OFF.  Flipping the default "
        f"silently shifts the production W2/W5/cosine-bell "
        f"baseline.  Either revert the default OR update this "
        f"sentinel + doc entry to record the deliberate change.")


def test_matrix_runner_does_not_activate_fortran_fidelity_flags():
    """AST scan: `scripts/run_atmosphere_test_matrix.py` must NOT
    construct ``CDGridShallowWaterConfig(...)`` with any of the
    iter-765/766/767/769/869b/871b/pre-862 opt-in flags set to True.

    A future change that silently activates one of these flags in
    the matrix runner would shift the W2/W5/cosine-bell baseline
    without an explicit Fortran-fidelity audit.
    """
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "run_atmosphere_test_matrix.py")
    tree = ast.parse(src.read_text())

    flag_names = {f for f, _, _, _ in _FORTRAN_FIDELITY_OPT_IN_FLAGS}

    config_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "CDGridShallowWaterConfig"
    ]
    assert config_calls, (
        "Could not find any CDGridShallowWaterConfig(...) "
        "construction in run_atmosphere_test_matrix.py.")

    for call in config_calls:
        for kw in call.keywords:
            if kw.arg in flag_names:
                # The kwarg is one of our opt-in flags.  Reject if
                # set to True; allow False (explicit opt-out).
                if (isinstance(kw.value, ast.Constant)
                        and kw.value.value is True):
                    raise AssertionError(
                        f"run_atmosphere_test_matrix.py constructs "
                        f"CDGridShallowWaterConfig(..., {kw.arg}="
                        f"True, ...).  This activates a deferred "
                        f"opt-in Fortran-fidelity flag in the "
                        f"production matrix runner, shifting the "
                        f"W2/W5/cosine-bell baseline.  Either "
                        f"revert the matrix runner change OR run "
                        f"the full sentinel suite to rebaseline.  "
                        f"Update this test to allow the new flag "
                        f"if the change is deliberate and audited.")


def test_w2_sentinel_does_not_activate_fortran_fidelity_flags():
    """AST scan: the W2 sentinel test
    (`test_w2_iter761_matrix_v_ll_and_mode4_baseline`) must not
    activate any Fortran-fidelity opt-in flags either.  Same
    rationale as the matrix runner: silent activation would
    invalidate the saved L2/v_ll/mode-4 baselines.
    """
    src = (Path(__file__).resolve().parent.parent
           / "tests" / "unit" / "test_cdgrid_fv3_regression.py")
    tree = ast.parse(src.read_text())

    flag_names = {f for f, _, _, _ in _FORTRAN_FIDELITY_OPT_IN_FLAGS}

    sentinel_fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "test_w2_iter761_matrix_v_ll_and_mode4_baseline"),
        None,
    )
    assert sentinel_fn is not None, (
        "Could not find test_w2_iter761_matrix_v_ll_and_mode4_baseline "
        "in test_cdgrid_fv3_regression.py.  Either the test was "
        "renamed or this sentinel is stale.")

    config_calls = [
        n for n in ast.walk(sentinel_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "CDGridShallowWaterConfig"
    ]
    assert config_calls, (
        "W2 sentinel test does not construct CDGridShallowWaterConfig.")

    for call in config_calls:
        for kw in call.keywords:
            if kw.arg in flag_names:
                if (isinstance(kw.value, ast.Constant)
                        and kw.value.value is True):
                    raise AssertionError(
                        f"W2 sentinel constructs "
                        f"CDGridShallowWaterConfig(..., {kw.arg}="
                        f"True, ...).  This activates an opt-in "
                        f"Fortran-fidelity flag and would shift the "
                        f"saved L2/v_ll/mode-4 baselines.")


def test_iter873_inventory_is_complete():
    """Sanity: the iter-873 inventory of opt-in flags should match
    the actual fields on `CDGridShallowWaterConfig` whose names match
    a known Fortran-fidelity prefix.  If a future iter adds a flag
    matching `fortran_*` or `apply_legacy_*` or `boundary_fix_skip_*`
    or `use_experimental_*` without updating this inventory, the
    sentinel would silently miss the new flag.
    """
    expected_prefixes = (
        "fortran_", "apply_legacy_", "apply_fortran_",
        "boundary_fix_skip_", "use_experimental_",
    )
    cfg = CDGridShallowWaterConfig()
    detected = sorted([
        f for f in cfg._fields
        if any(f.startswith(p) for p in expected_prefixes)
    ])
    inventoried = sorted([f for f, _, _, _ in _FORTRAN_FIDELITY_OPT_IN_FLAGS])

    missing = set(detected) - set(inventoried)
    assert not missing, (
        f"iter-873 inventory is missing Fortran-fidelity opt-in "
        f"flag(s): {sorted(missing)}.  These appear to be "
        f"`fortran_*`/`apply_legacy_*`/`boundary_fix_skip_*`/"
        f"`use_experimental_*` fields on `CDGridShallowWaterConfig` "
        f"but are not covered by the iter-873 sentinel.  Update "
        f"the `_FORTRAN_FIDELITY_OPT_IN_FLAGS` list above with "
        f"each new flag's expected default + iter-tag + doc-anchor.")

    extra = set(inventoried) - set(detected)
    assert not extra, (
        f"iter-873 inventory references field(s) that no longer "
        f"exist on `CDGridShallowWaterConfig`: {sorted(extra)}.  "
        f"These were likely renamed or removed; update the "
        f"inventory accordingly.")
