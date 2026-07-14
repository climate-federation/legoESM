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

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
)


# Each entry: (field_name, expected_default, iter_added, doc_anchor).
# Note: `fortran_dir_aware_corners` is a function-level kwarg on
# `arakawa_lamb_gradient` / `fv3_sw_tendencies`, NOT a config field
# (iter-765 marked it FALSIFIED — never promoted to config).  This
# inventory covers only `CDGridShallowWaterConfig` fields.
_FORTRAN_FIDELITY_OPT_IN_FLAGS = [
    ("fortran_a2b_corner_avg", False, "iter-766", "a2b_edge.F90:385-388"),
    ("fortran_vector_corner_fill", False, "iter-767",
     "fv_mp_mod.F90:1433-1457"),
    ("apply_legacy_d_sw4_corner_ke_fix", False, "iter-869b",
     "sw_core.F90:1438-1466"),
    ("apply_legacy_d_sw5_corner_corrections", False, "iter-871b",
     "d_sw5_corner_divergence inner helper"),
    ("boundary_fix_skip_corners", False, "iter-769",
     "boundary-corner cascaded smoothing"),
    ("use_experimental_csw", False, "pre-iter-862",
     "experimental C-grid path (known unstable)"),
    # Iter-900 added `fortran_faithful_ppm_left`: switches the LEFT-
    # side cube-edge PPM overrides to Fortran's actual al(0)/al(1)/
    # al(2) recipes at the corrected q_face indices (per iter-899
    # investigation of iter-892's 1-cell shift bug).  Iter-900
    # measurement: ON path WORSENS W2 v_ll_Linf by 53.7 % (0.132 ->
    # 0.203 m/s) — iter-892's empirically-good shifted formulas are
    # better than strict Fortran on the smooth W2 cube vertex.
    # Default OFF preserves iter-893 production behavior; this flag
    # remains a Fortran-fidelity opt-in covered by iter-873 inventory.
    ("fortran_faithful_ppm_left", False, "iter-900",
     "tp_core.F90:359-362 (al(0)/al(1)/al(2) at Hypothesis-A q_face indices)"),
    # Iter-903 added `fortran_faithful_ppm_right`: symmetric counter-
    # part to iter-900's LEFT-side flag.  Switches the RIGHT-side
    # cube-edge PPM overrides to Fortran's al(npx-1)/al(npx) recipes
    # at the corrected q_face[n+2,n+3] indices (per iter-899
    # investigation).  al(npx) is PARTIALLY faithful in halo=2
    # (q1(npx+1) is mode='edge' replica); al(npx+1) is not placed
    # (needs halo=3).  Default OFF preserves iter-893 production
    # behavior; iter-904+ measurement determines whether to flip.
    ("fortran_faithful_ppm_right", False, "iter-903",
     "tp_core.F90:365-367 (al(npx-1)/al(npx) at Hypothesis-A q_face indices)"),
    # Iter-904 NOTE: `use_fv3_dsw1_mass_transport` (default OFF, per
    # iter-904) is intentionally NOT listed here — its name doesn't
    # match the iter-873 prefix taxonomy (`fortran_*` /
    # `apply_legacy_*` / `boundary_fix_skip_*` / `use_experimental_*`)
    # and `use_fv3_*` is a separate naming scheme for FV3-vs-non-FV3
    # algorithmic swaps rather than Fortran-formula opt-ins.  The
    # iter-904 default-OFF property is pinned by the dedicated
    # `tests/test_iter904_*.py` sentinels, not by the iter-873
    # inventory check.
    # Iter-888c added `apply_fortran_xppm_boundary` to this inventory
    # as a default-OFF Fortran-fidelity opt-in.  Iter-892 fixed an
    # off-by-one in the iter-889 implementation and discovered the
    # corrected Fortran-faithful path actually IMPROVES W2 v_north
    # Linf at C36 1-day by 19.6% (0.189 → 0.152 m/s).  Iter-893
    # therefore activates the flag on the production matrix runner
    # and W2 sentinel — moving it from "default-OFF opt-in" to
    # "active by default in production".  The CONFIG default on the
    # dataclass remains False (preserves new-user-OFF behaviour);
    # the activation happens at the matrix-runner / W2-sentinel
    # call sites.  Removed from this inventory because the iter-873
    # matrix-runner / W2-sentinel "must NOT activate" check would
    # block the iter-893 production activation.
]

# Iter-893 record: `apply_fortran_xppm_boundary` is INTENTIONALLY
# active in the production matrix runner and W2 sentinel.  See
# iter-893 doc entry for measurement details.
_FORTRAN_FIDELITY_FLAGS_ACTIVE_IN_PRODUCTION = (
    "apply_fortran_xppm_boundary",
)


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
        "fortran_", "apply_legacy_",
        "boundary_fix_skip_", "use_experimental_",
    )
    # Iter-893: `apply_fortran_*` prefix removed because
    # `apply_fortran_xppm_boundary` was promoted from "default-OFF
    # opt-in" to "active by default in production matrix runner +
    # W2 sentinel" (after iter-892 demonstrated 19.6% W2 v_north Linf
    # improvement).  Other `apply_fortran_*` flags (none currently)
    # would not be auto-detected by this completeness check; manual
    # inventory review remains the safety net.
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


# ----------------------------------------------------------------------
# Iter-896: complement to iter-873's "must NOT activate" gates with
# explicit "MUST activate" gates for flags promoted from default-OFF
# opt-in to default-ON in production.  After iter-893 promoted
# `apply_fortran_xppm_boundary=True` on the production matrix runner
# and W2 sentinel, the iter-873 inventory removed that flag from its
# "must NOT activate" list — but no test was added to assert the
# flag IS still active.  iter-896 adds a fast AST-based sentinel
# that catches a hypothetical revert (someone removes
# `apply_fortran_xppm_boundary=True` from the matrix runner / W2
# sentinel without realising the W2 baseline depends on it).  The
# W2 sentinel ceiling (max_v_ll<0.145) catches it at runtime in 20s;
# this AST sentinel catches it in <1s.
# ----------------------------------------------------------------------


def test_iter896_matrix_runner_has_apply_fortran_xppm_boundary_active():
    """AST scan: `scripts/run_atmosphere_test_matrix.py` MUST construct
    its W2/W5 LEGACY `CDGridShallowWaterConfig(...)` with
    ``apply_fortran_xppm_boundary=True``.  Catches a regression that
    silently reverts iter-893's production activation.

    Pre-iter-893 the matrix runner had no `apply_fortran_xppm_boundary`
    kwarg (default-OFF).  Iter-893 added `=True` after iter-892
    demonstrated the Fortran iord<7 boundary formula reduces W2
    v_north Linf by 19.6% (per-face cell-centre) / 16.8% (lat-lon
    regrid).  iter-873 was updated to drop the flag from its
    "must NOT activate" inventory; this iter-896 sentinel is the
    complementary "MUST activate" check.

    Targeting: the W2/W5 LEGACY config is identifiable by the
    `8.0 * _div_damp_cube(...)` div_damp expression.  The cosine
    bell config uses `_div_damp_cube(...)` (no multiplier) and
    deliberately does NOT pass `apply_fortran_xppm_boundary=True`
    because it uses `transport_step` directly without forwarding
    the kwarg from config (the iter-888 path on `_ppm_1d` is a
    different code path).
    """
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "run_atmosphere_test_matrix.py")
    tree = ast.parse(src.read_text())

    config_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "CDGridShallowWaterConfig"
    ]

    # Find the W2/W5 LEGACY construction.  Identifying signature:
    # `div_damp = 8.0 * _div_damp_cube(...)` — the iter-761 canonical
    # 8x multiplier that cosine bell does not use.
    def _is_w2_w5_legacy_div_damp(value_node):
        """Return True iff value_node is `8.0 * _div_damp_cube(...)`."""
        if not (isinstance(value_node, ast.BinOp)
                and isinstance(value_node.op, ast.Mult)):
            return False
        if not (isinstance(value_node.left, ast.Constant)
                and value_node.left.value == 8.0):
            return False
        if not (isinstance(value_node.right, ast.Call)
                and isinstance(value_node.right.func, ast.Name)
                and value_node.right.func.id == "_div_damp_cube"):
            return False
        return True

    w2_w5_legacy_calls = []
    for call in config_calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        div_damp_node = kwargs.get("div_damp")
        if div_damp_node is not None and _is_w2_w5_legacy_div_damp(
                div_damp_node):
            w2_w5_legacy_calls.append(call)

    assert w2_w5_legacy_calls, (
        "Could not find any CDGridShallowWaterConfig(...) with "
        "the W2/W5 LEGACY signature `div_damp=8.0*_div_damp_cube(...)` "
        "in run_atmosphere_test_matrix.py.  Either iter-761's 8x "
        "div_damp multiplier was refactored or this iter-896 sentinel "
        "is stale.")

    for call in w2_w5_legacy_calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        flag_node = kwargs.get("apply_fortran_xppm_boundary")
        assert flag_node is not None, (
            f"run_atmosphere_test_matrix.py W2/W5 LEGACY config "
            f"(line ~{call.lineno}) does NOT pass "
            f"`apply_fortran_xppm_boundary=`.  iter-893 added this "
            f"to activate Fortran's iord<7 cube-edge boundary "
            f"formulas (tp_core.F90:357-369), reducing W2 v_north "
            f"Linf by 16.8% (lat-lon regrid).  Reverting it would "
            f"shift the W2 sentinel baseline 0.132 → 0.158 m/s, "
            f"which the iter-895 max_v_ll<0.145 ceiling would catch "
            f"at runtime — but iter-896 catches it AST-fast.")
        assert (isinstance(flag_node, ast.Constant)
                and flag_node.value is True), (
            f"run_atmosphere_test_matrix.py W2/W5 LEGACY config "
            f"(line ~{call.lineno}) has "
            f"apply_fortran_xppm_boundary={ast.unparse(flag_node)!r}, "
            f"expected True.  Iter-893 set this to True; reverting "
            f"to False reverts the W2 v_north Linf improvement "
            f"(0.132 → 0.158 m/s).")


def test_iter896_w2_sentinel_has_apply_fortran_xppm_boundary_active():
    """AST scan: the W2 sentinel test
    (`test_w2_iter761_matrix_v_ll_and_mode4_baseline`) MUST construct
    its `CDGridShallowWaterConfig(...)` with
    `apply_fortran_xppm_boundary=True`.  Catches the same revert
    class as the matrix-runner sentinel above.
    """
    src = (Path(__file__).resolve().parent.parent
           / "tests" / "unit" / "test_cdgrid_fv3_regression.py")
    tree = ast.parse(src.read_text())

    sentinel_fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "test_w2_iter761_matrix_v_ll_and_mode4_baseline"),
        None,
    )
    assert sentinel_fn is not None

    config_calls = [
        n for n in ast.walk(sentinel_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "CDGridShallowWaterConfig"
    ]
    assert config_calls

    for call in config_calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        flag_node = kwargs.get("apply_fortran_xppm_boundary")
        assert flag_node is not None, (
            f"W2 sentinel test config (line ~{call.lineno}) does "
            f"NOT pass `apply_fortran_xppm_boundary=`.  iter-893 "
            f"set this to True for matrix consistency; reverting "
            f"would desynchronize the sentinel from the matrix "
            f"runner W2 LEGACY config.")
        assert (isinstance(flag_node, ast.Constant)
                and flag_node.value is True), (
            f"W2 sentinel test config (line ~{call.lineno}) has "
            f"apply_fortran_xppm_boundary="
            f"{ast.unparse(flag_node)!r}, expected True.  iter-893 "
            f"sync requires this matches the matrix runner.")


def test_iter896_active_in_production_inventory_reachable():
    """The `_FORTRAN_FIDELITY_FLAGS_ACTIVE_IN_PRODUCTION` constant
    added in iter-893 records the flags that have been promoted
    from default-OFF opt-in to default-ON in production.  iter-896
    asserts the constant exists, contains the iter-893 flag, and
    every entry corresponds to an actual `CDGridShallowWaterConfig`
    field.

    This complements `_FORTRAN_FIDELITY_OPT_IN_FLAGS` (the still-OFF
    inventory).  Together the two cover the full set of
    Fortran-fidelity flags + their production state.
    """
    assert "_FORTRAN_FIDELITY_FLAGS_ACTIVE_IN_PRODUCTION" in globals(), (
        "iter-893 added `_FORTRAN_FIDELITY_FLAGS_ACTIVE_IN_PRODUCTION` "
        "as a record of flags promoted to default-ON in production. "
        "Was it removed?  iter-896 sentinel needs this constant.")

    active = _FORTRAN_FIDELITY_FLAGS_ACTIVE_IN_PRODUCTION
    assert "apply_fortran_xppm_boundary" in active, (
        f"`_FORTRAN_FIDELITY_FLAGS_ACTIVE_IN_PRODUCTION` is missing "
        f"`apply_fortran_xppm_boundary`.  iter-893 added this entry; "
        f"current value: {active!r}.")

    cfg = CDGridShallowWaterConfig()
    for f in active:
        assert hasattr(cfg, f), (
            f"`_FORTRAN_FIDELITY_FLAGS_ACTIVE_IN_PRODUCTION` cites "
            f"`{f}` but `CDGridShallowWaterConfig` has no such "
            f"attribute.  Inventory is stale.")
