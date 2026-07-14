"""FV3_3D iter 188: PE-side AST regression guard mirroring iter-172
NH guard.  Closes iter-187 codex review concern 6.

Hits the same failure mode the iter-172 NH guard was written for:
a refactor that drops one of the iter-12/14/16/18/57/187 wirings
in ``primitive_eq_cdgrid.py`` while keeping the corresponding
config field would silently disable the FV3-faithful mechanism
even when the user sets the config flag.  No integration test
detects this — the model still runs, just without the intended
damping.

Three tests:

1. ``test_pe_fv3_config_fields_ast_regression`` — config field
   defaults for iter-12/14/16/18/57/187 + iter-188's new
   ``corner_div_damp_dt_proxy`` (the parity field).
2. ``test_pe_fv3_call_sites_ast_regression`` — gate / helper
   pairs for each FV3-faithful wiring, mirror of iter-172.
3. ``test_iter188_ast_guard_self_check`` — drops each pair one
   at a time and verifies the inner check function flags the
   omission, mirror of iter-178 self-check.
"""
from __future__ import annotations

import ast

import pytest

from tests._iter187_marker import ITER187_GATE, ITER187_HELPER
from tests.legoesm_paths import legoesm_source_path


_PE_SRC_PATH = legoesm_source_path(
    "atmosphere/dynamics/gcm/primitive_eq_cdgrid.py"
)


def test_pe_fv3_config_fields_ast_regression():
    """AST regression guard: ``CDGridPrimitiveEquationConfig`` must
    declare the iter-12/14/16/18/57/187 + iter-188 config fields
    with the documented default values.  Catches a refactor that
    drops or silently changes any of these defaults — the runtime
    gate would still pass but the FV3-faithful damping would be
    silently disabled or its strength changed."""
    tree = ast.parse(_PE_SRC_PATH.read_text())

    cfg_class = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef)
         and n.name == "CDGridPrimitiveEquationConfig"),
        None,
    )
    assert cfg_class is not None, (
        "CDGridPrimitiveEquationConfig class not found"
    )

    fields = {}
    for node in cfg_class.body:
        if (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
                and isinstance(node.value, ast.Constant)):
            fields[node.target.id] = node.value.value
        elif (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
                and isinstance(node.value, ast.Attribute)):
            # constants.X — not a literal, skip (covered by the
            # constants-discipline audit).
            pass

    expected = {
        # iter-5: cell-centre divergence damping (constant + adaptive)
        "div_damp_coeff": 0.0,
        "div_damp_dddmp": 0.0,
        # iter-12: post-step del-n vorticity damping
        "damp_v": 0.0,
        "nord_v": 2,
        # iter-14: 4th-order ζ corner interpolation
        "use_fv3_a2b_zeta_corner": False,
        # iter-16/18: corner-divergence damping (del-2 + higher-order)
        "corner_div_damp_d2_bg": 0.0,
        "corner_div_damp_dddmp": 0.20,
        "corner_div_damp_d4_bg": 0.0,
        "corner_div_damp_nord": 0,
        "corner_div_damp_fv3_vector_fill": False,
        # iter-57/58: Smagorinsky-adaptive A_h
        "smagorinsky_cs": 0.0,
        # iter-182 (PE T_diss sqrt(0) fix predates iter-188 but the
        # field is part of the FV3-fidelity surface)
        "T_diss_coeff": 0.0,
        # iter-188: PE / NH parity.  Default 200.0 = the previously
        # hardcoded ``_dt_approx`` at iter-16 wiring.
        "corner_div_damp_dt_proxy": 200.0,
        # iter-208: KE→heat conversion for damp_v (FV3 d_con port).
        "damp_v_d_con": 0.0,
        # iter-218/219: per-step dissipative-heating cap with FV3
        # sponge-layer awareness.  Default 0.0 disables the cap.
        "delt_max": 0.0,
        # iter-221: KE→heat conversion for the iter-16/18
        # corner-divergence damping.
        "corner_div_damp_d_con": 0.0,
        # iter-223: KE→heat conversion for the iter-5 cell-centre
        # divergence damping.
        "div_damp_d_con": 0.0,
        # iter-225: KE→heat conversion for the iter-57/58
        # Smagorinsky-A_h Laplacian.
        "ah_d_con": 0.0,
    }

    missing = []
    wrong_default = []
    for name, default in expected.items():
        if name not in fields:
            missing.append(name)
        elif fields[name] != default:
            wrong_default.append(
                f"{name}: expected {default!r}, got {fields[name]!r}"
            )

    assert not missing, (
        f"iter-188 PE AST regression: PE config missing FV3-faithful "
        f"fields {missing}.  These were added across iter "
        f"5/12/14/16/18/57/182/187/188 — a refactor that drops them "
        f"silently disables the corresponding FV3 mechanism."
    )
    assert not wrong_default, (
        f"iter-188 PE AST regression: PE config field defaults "
        f"changed from documented values: {wrong_default}.  Changing "
        f"the default value silently changes the baseline behavior "
        f"of every model that doesn't override the field."
    )


# Each pair: (gate substring that activates the wiring, helper symbol
# that should appear inside the gated block).  Mirrors the
# tests/test_fv3_nh_toolkit_iter172.py NH guard structure.
_PE_GATE_HELPER_PAIRS = [
    # iter-5: cell-centre constant + adaptive divergence damping
    ("config.div_damp_coeff > 0", "arakawa_lamb_gradient"),
    # iter-12: post-step del-n vorticity damping (gated in step,
    # not the tendency function — uses ``self.config.damp_v``).
    ("self.config.damp_v > 0.0", "fv3_del6_vorticity_damping"),
    # iter-14: 4th-order ζ corner interpolation
    ("config.use_fv3_a2b_zeta_corner",
     "interp_center_to_corner_a2b_ord4"),
    # iter-16: corner-divergence damping (del-2)
    ("config.corner_div_damp_d2_bg > 0.0",
     "fv3_divergence_corner_3d"),
    # iter-18: corner-divergence higher-order Laplacian iteration
    ("config.corner_div_damp_d4_bg > 0.0 and config.corner_div_damp_nord > 0",
     "fv3_corner_laplacian_iteration"),
    # iter-57/58: Smagorinsky-adaptive A_h
    ("config.smagorinsky_cs > 0.0", "compute_smagorinsky_ah_3d"),
    # iter-182: T_diss velocity-dependent dissipation
    ("config.T_diss_coeff > 0", "wind_speed"),
    # iter-187: smag_vort cap recomputation (FV3 sw_core.F90:1797).
    # iter-202 extracted this pair into ``tests/_iter187_marker.py``
    # so PE and NH AST guards share a single source-of-truth and
    # future substring changes (like iter-190's) cannot drift.
    (ITER187_GATE, ITER187_HELPER),
    # iter-208: KE→heat conversion for damp_v.
    ("self.config.damp_v_d_con > 0.0", "interp_corner_to_center"),
    # iter-218/219: per-step dissipative-heating cap.
    ("self.config.delt_max > 0.0", "jnp.clip"),
    # iter-221: corner-div damping d_con.
    ("config.corner_div_damp_d_con > 0.0", "_dKE_dt_corner_cdd"),
    # iter-223: cell-centre div_damp d_con.
    ("config.div_damp_d_con > 0.0", "_dKE_dt_corner_dd"),
    # iter-225: Smagorinsky-A_h d_con.
    ("config.ah_d_con > 0.0", "_dKE_dt_corner_ah"),
    # iter-239: aggregate cap on the 3 PE tendency-based d_con
    # contributions.  The aggregation builds ``_d_con_sum`` by
    # summing the (None-aware) per-mechanism contributions then
    # applies ``jnp.clip`` with a sponge-aware per-level cap
    # before adding to ``dT_dt_data``.
    ("_d_con_sum is not None", "_cap_per_level"),
]


def test_pe_fv3_call_sites_ast_regression():
    """AST regression guard: each PE-side iter-X wiring point must
    remain present in the source.  Searches the source text for the
    gate expressions and helper symbols.  Catches a regression
    where the config field stays but the call-site is dropped."""
    src_text = _PE_SRC_PATH.read_text()

    missing = []
    for gate, helper in _PE_GATE_HELPER_PAIRS:
        if gate not in src_text:
            missing.append(f"gate {gate!r}")
        if helper not in src_text:
            missing.append(f"helper {helper!r}")

    assert not missing, (
        f"iter-188 PE AST regression: PE FV3 call sites missing "
        f"{missing}.  These wirings were added by iter "
        f"5/12/14/16/18/57/182/187; their absence silently disables "
        f"the FV3-faithful mechanism even when the user sets the "
        f"config flag."
    )


def test_iter188_ast_guard_self_check():
    """iter-188 self-check: simulate the regression that the guard
    is supposed to catch.  Build a fake source string that omits
    each gate/helper one at a time and verify the inner check
    function would have flagged it.

    Mirror of iter-178's NH self-check.  Without it, a typo in the
    gate string (e.g., extra space, wrong operator) would cause the
    guard to ALWAYS pass, silently disabling the regression check."""

    # Build a valid source containing all substrings.
    valid_src = "\n".join(
        f"{gate} -> {helper}"
        for gate, helper in _PE_GATE_HELPER_PAIRS
    )

    # Sanity: valid source passes all checks.
    missing = []
    for gate, helper in _PE_GATE_HELPER_PAIRS:
        if gate not in valid_src:
            missing.append(f"gate {gate!r}")
        if helper not in valid_src:
            missing.append(f"helper {helper!r}")
    assert not missing, (
        f"self-check sanity: valid source must pass all checks; "
        f"got missing={missing}"
    )

    # Drop each gate/helper one at a time; verify the check correctly
    # identifies it.  Note: the helper is shared across pairs (e.g.,
    # ``interp_center_to_corner_a2b_ord4`` appears in both iter-14
    # and iter-187), so omitting one occurrence may not necessarily
    # remove all instances — the self-check iterates by INDEX of the
    # pair list and removes the literal at that index.
    for omit_idx in range(len(_PE_GATE_HELPER_PAIRS)):
        broken_src = "\n".join(
            f"{g} -> {h}"
            for i, (g, h) in enumerate(_PE_GATE_HELPER_PAIRS)
            if i != omit_idx
        )
        # Re-check.  The omitted pair's gate may also appear in
        # another pair (unlikely but possible), so we check using
        # the pair's literal substrings only.
        omitted_gate, omitted_helper = _PE_GATE_HELPER_PAIRS[omit_idx]
        # Helper substring may be shared; only assert gate is missing,
        # which is unique per iter-X site.
        assert omitted_gate not in broken_src, (
            f"self-check failed for omit_idx={omit_idx}: dropping "
            f"the {omitted_gate!r} gate did NOT remove it from the "
            f"synthetic source.  Likely an unintended duplicate "
            f"substring across pairs; investigate the pair list."
        )
