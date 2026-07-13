"""FV3_3D iter 172: composition + AST regression guard for the four
NH FV3-faithful damping mechanisms ported in iter 168/169/170/171.

Two complementary safety tests:

1. **Composition test**: a single integration test that turns ON
   ALL FOUR new NH knobs (corner-div, damp_v, a2b_zeta_corner,
   cell-centre div_damp + dddmp) at once on a perturbed NH state
   and verifies they compose without instability or AD breakage.
   Catches the most common composition failure: a downstream knob
   silently overwriting an upstream knob's contribution, or a
   shared intermediate (``div_v``) being computed inconsistently
   across consumers.

2. **AST regression guard**: walks the source of
   ``compressible_euler_cdgrid.py`` and asserts that the four new
   config fields exist with the documented default values, AND
   that the four call-sites (``corner_div_damp_d2_bg`` Python
   gate, ``damp_v`` Python gate, ``use_fv3_a2b_zeta_corner``
   Python gate, ``div_damp_coeff`` Python gate) appear in the
   tendency function or the step function.  Catches a regression
   that drops any of the four wirings while keeping the config
   field (which would silently disable the feature).

Both tests are bit-for-bit free at the AST level (test 2) and
purely structural at the integration level (test 1, which uses
``np.isfinite`` + ``rho_drift < 1.0`` rather than tight numerical
asserts).
"""
from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tests._iter187_marker import ITER187_GATE, ITER187_HELPER
from tests.legoesm_paths import legoesm_source_path

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def small_nh_state():
    """Same fixture as the iter-168/169/170/171 NH tests."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, grid.n, grid.n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def test_nh_full_fv3_toolkit_composes(small_nh_state):
    """All four iter-168/169/170/171 NH knobs ON at once: 20 steps
    from a perturbed state stay finite and produce bounded winds."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=172)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14,
        n_acoustic_substeps=4,
        # iter-168: corner-div damping
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        # iter-169: post-step vorticity damping
        damp_v=0.030,
        nord_v=2,
        # iter-170: 4th-order A→B ζ corner
        use_fv3_a2b_zeta_corner=True,
        # iter-171: cell-centre div damping (constant + adaptive)
        div_damp_coeff=1e6,
        div_damp_dddmp=0.20,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    for _ in range(20):
        s = model.step(s, 10.0)

    assert jnp.all(jnp.isfinite(s.u.data)), "u NaN under full toolkit"
    assert jnp.all(jnp.isfinite(s.v.data)), "v NaN under full toolkit"
    assert jnp.all(jnp.isfinite(s.theta_prime.data))
    assert jnp.all(jnp.isfinite(s.rho_prime.data))

    max_u = float(jnp.max(jnp.abs(s.u.data)))
    max_v = float(jnp.max(jnp.abs(s.v.data)))
    # All four damping mechanisms should KEEP wind magnitude bounded
    # — for a small perturbation in a rest-state NH atmosphere over
    # 20 steps × 10s = 200s of physical time, max winds should
    # stabilise well below O(100 m/s).
    assert max_u < 100.0, f"max|u|={max_u:.1f} m/s — toolkit unstable"
    assert max_v < 100.0, f"max|v|={max_v:.1f} m/s — toolkit unstable"


def test_nh_full_fv3_toolkit_differentiable(small_nh_state):
    """``jax.grad`` flows through 5 steps with all four knobs ON.
    Catches AD breakage in any of the four wirings (most likely
    failure mode is a stray ``np.`` op or ``jax.lax.stop_gradient``
    accidentally introduced in one of the new helpers)."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14,
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        damp_v=0.030,
        nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6,
        div_damp_dddmp=0.20,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(theta_p_data):
        s = state._replace(
            theta_prime=state.theta_prime.replace(data=theta_p_data),
        )
        for _ in range(5):
            s = model.step(s, 10.0)
        return jnp.mean(s.u.data ** 2 + s.v.data ** 2)

    grad = jax.grad(loss_fn)(state.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad)), (
        "AD broke under full FV3 toolkit — check for non-AD-safe "
        "ops in iter-168/169/170/171 wirings"
    )


def test_nh_corner_div_damp_nord2_runs_and_differs_from_nord1(
    small_nh_state,
):
    """FV3_3D iter 887: NH-path mirror of iter-886 PE nord=2 regression
    guard.

    Validates open follow-up "Substantive nord>=2 fidelity restructure"
    for the NH compressible-Euler path (ported in iter-168).  Two checks:

    1. ``corner_div_damp_nord=2`` runs end-to-end one step without
       NaN.  Sanity guard — the higher-order Laplacian iteration
       loop in compressible_euler_cdgrid.py:631-632 must be
       JAX-traceable and produce finite output through the NH
       acoustic-substep loop.

    2. ``corner_div_damp_nord=2 + corner_div_damp_fv3_vector_fill=True``
       produces output that differs from ``corner_div_damp_nord=1``.
       Confirms the higher-order ∇⁴ Laplacian contribution to
       ke_correction is functional, not silently a no-op.

    Faithful-implementation goal: NH nord=2 should be a strict
    extension of nord=1, matching the PE path behavior verified
    in iter-886 (test_corner_div_damp_nord2_pe_runs_and_differs_from_nord1).
    """
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=887)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg_nord1 = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14,
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        corner_div_damp_fv3_vector_fill=True,
    )
    cfg_nord2 = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14,
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=2,
        corner_div_damp_fv3_vector_fill=True,
    )

    m_nord1 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_nord1,
    )
    m_nord2 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_nord2,
    )

    s_nord1 = m_nord1.step(s, 10.0)
    s_nord2 = m_nord2.step(s, 10.0)

    # (1) Both paths finite (no NaN from inner Laplacian iteration
    # composed with NH acoustic-substep loop).
    assert jnp.all(jnp.isfinite(s_nord2.u.data)), \
        "NH u NaN under nord=2"
    assert jnp.all(jnp.isfinite(s_nord2.v.data))
    assert jnp.all(jnp.isfinite(s_nord2.theta_prime.data))
    assert jnp.all(jnp.isfinite(s_nord2.rho_prime.data))

    # (2) nord=2 output differs from nord=1.
    max_diff_u = float(jnp.max(jnp.abs(s_nord2.u.data - s_nord1.u.data)))
    assert max_diff_u > 1e-12, (
        "NH nord=2 path produced bit-for-bit identical output to "
        "nord=1; higher-order Laplacian iteration is silently a no-op."
    )


def test_nh_corner_div_damp_nord3_loop_scales(small_nh_state):
    """FV3_3D iter 889: NH-path mirror of iter-888 PE nord=3 stress.

    Validates the NH compressible-Euler ``for _ in range(nord)``
    Laplacian iteration loop in
    ``compressible_euler_cdgrid.py:631-632`` scales beyond nord=2
    (iter-887) to the full FV3 namelist range nord ∈ {1, 2, 3}.

    Two checks:

    1. ``corner_div_damp_nord=3`` runs end-to-end through the NH
       acoustic-substep loop without NaN.
    2. nord=3 NH output differs from nord=2 NH output — rules out
       silent cap due to numerical fixed-point convergence or
       loop truncation.

    Combined with iter-886/887/888 this completes regression-guard
    coverage for both 3D paths across the full FV3 nord-range.
    """
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=889)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg_nord2 = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14,
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=2,
        corner_div_damp_fv3_vector_fill=True,
    )
    cfg_nord3 = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14,
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=3,
        corner_div_damp_fv3_vector_fill=True,
    )

    m_nord2 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_nord2,
    )
    m_nord3 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_nord3,
    )

    s_nord2 = m_nord2.step(s, 10.0)
    s_nord3 = m_nord3.step(s, 10.0)

    # (1) nord=3 NH runs without NaN.
    assert jnp.all(jnp.isfinite(s_nord3.u.data)), \
        "NH u NaN under nord=3"
    assert jnp.all(jnp.isfinite(s_nord3.v.data))
    assert jnp.all(jnp.isfinite(s_nord3.theta_prime.data))
    assert jnp.all(jnp.isfinite(s_nord3.rho_prime.data))

    # (2) nord=3 NH output differs from nord=2 NH output.
    max_diff_u = float(jnp.max(jnp.abs(s_nord3.u.data - s_nord2.u.data)))
    assert max_diff_u > 1e-12, (
        "NH nord=3 produced bit-for-bit identical output to nord=2; "
        "Laplacian iteration loop may be silently capped at 2."
    )


def test_nh_fv3_config_fields_ast_regression():
    """AST regression guard: ``CDGridCompressibleEulerConfig`` must
    declare the four new iter-168/169/170/171 fields with the
    documented default values.  Catches a refactor that drops or
    silently changes any of these defaults — the runtime gate would
    still pass but the FV3-faithful damping would be silently
    disabled or its strength changed."""
    src_path = legoesm_source_path(
        "atmosphere/dynamics/gcm/compressible_euler_cdgrid.py"
    )
    tree = ast.parse(src_path.read_text())

    cfg_class = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef)
         and n.name == "CDGridCompressibleEulerConfig"),
        None,
    )
    assert cfg_class is not None, (
        "CDGridCompressibleEulerConfig class not found"
    )

    # Collect AnnAssign nodes (NamedTuple field declarations) into
    # {field_name: literal_default_value}.
    fields = {}
    for node in cfg_class.body:
        if (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
                and isinstance(node.value, ast.Constant)):
            fields[node.target.id] = node.value.value

    expected = {
        # iter-168 corner-divergence damping
        "corner_div_damp_d2_bg": 0.0,
        "corner_div_damp_dddmp": 0.20,
        "corner_div_damp_d4_bg": 0.0,
        "corner_div_damp_nord": 0,
        "corner_div_damp_fv3_vector_fill": False,
        "corner_div_damp_dt_proxy": 10.0,
        # iter-169 post-step vorticity damping
        "damp_v": 0.0,
        "nord_v": 2,
        # iter-170 a2b zeta corner
        "use_fv3_a2b_zeta_corner": False,
        # iter-171 cell-centre div damping
        "div_damp_coeff": 0.0,
        "div_damp_dddmp": 0.0,
        # iter-173 async-halo overlap (PE parity, MPI optimisation).
        # Added to the iter-172 guard by iter-178 to close the
        # gap that the field landed AFTER iter-172 was written.
        "use_async_halo": False,
        # iter-180 Smagorinsky-adaptive A_h (PE iter 57-58 parity).
        # Added by iter-186 to extend the guard for the latest
        # config addition.
        "smagorinsky_cs": 0.0,
        # iter-193 post-step damp_w + nord_w (FV3 d_sw1 port).
        "damp_w": 0.0,
        "nord_w": 2,
        # iter-203 KE→heat conversion for damp_w (FV3 d_sw1 d_con).
        "damp_w_d_con": 0.0,
        # iter-209 KE→heat conversion for damp_v (NH mirror of PE
        # iter-208).
        "damp_v_d_con": 0.0,
        # iter-218/219: per-step dissipative-heating cap with FV3
        # sponge-layer awareness.  Default 0.0 disables the cap.
        "delt_max": 0.0,
        # iter-222: NH mirror of PE iter-221 corner-div d_con.
        "corner_div_damp_d_con": 0.0,
        # iter-224: NH mirror of PE iter-223 cell-centre div_damp
        # d_con.
        "div_damp_d_con": 0.0,
        # iter-226: NH mirror of PE iter-225 Smagorinsky-A_h
        # d_con.
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
        f"iter-172 AST regression: NH config missing FV3-faithful "
        f"fields {missing}.  These were added by iter "
        f"168/169/170/171 — a refactor that drops them silently "
        f"disables the corresponding FV3 mechanism."
    )
    assert not wrong_default, (
        f"iter-172 AST regression: NH config field defaults changed "
        f"from documented values: {wrong_default}.  Changing the "
        f"default value silently changes the baseline behavior of "
        f"every model that doesn't override the field."
    )


# FV3_3D iter 196: factor out as a module-level constant so the
# outer ``test_nh_fv3_call_sites_ast_regression`` and the inner
# ``test_iter178_ast_guard_self_check`` consume the SAME source of
# truth.  Previously each test maintained its own local copy and the
# two could drift apart silently — iter 193 added a pair to the outer
# test only, leaving the self-check stale.  iter 196 closes that
# maintenance hazard with a shared constant.
NH_GATE_HELPER_PAIRS = [
    # iter-168: corner-divergence damping
    ("config.corner_div_damp_d2_bg > 0.0", "fv3_divergence_corner_3d"),
    # iter-169: post-step vorticity damping
    ("self.config.damp_v > 0.0", "fv3_del6_vorticity_damping"),
    # iter-170: a2b zeta corner
    ("config.use_fv3_a2b_zeta_corner",
     "interp_center_to_corner_a2b_ord4"),
    # iter-171: cell-centre div damping
    ("config.div_damp_coeff > 0.0", "arakawa_lamb_gradient"),
    # iter-173: async-halo overlap dispatch (added by iter-178
    # to close the gap that the iter-172 guard predates iter-173).
    ("config.use_async_halo and _hb_div ==",
     "overlapped_arakawa_lamb_gradient"),
    # iter-180: Smagorinsky-adaptive A_h dispatch (added by
    # iter-186 to extend the guard).  When ``smagorinsky_cs > 0``
    # AND ``A_h > 0``, the adaptive coefficient is computed via
    # the existing helper.  Both gates required since
    # Smagorinsky is gated INSIDE the ``A_h > 0`` block.
    ("config.smagorinsky_cs > 0.0", "compute_smagorinsky_ah_3d"),
    # iter-187: smag_vort cap recomputation inside the
    # ``corner_div_damp_d4_bg > 0 AND nord > 0`` branch.  iter-202
    # extracted this pair into ``tests/_iter187_marker.py`` so
    # PE and NH AST guards share a single source-of-truth and
    # future substring changes (like iter-190's) cannot drift.
    (ITER187_GATE, ITER187_HELPER),
    # iter-193: post-step damp_w + nord_w (FV3 d_sw1 port,
    # sw_core.F90:1080-1086).  Reuses the SW backbone
    # ``_del6_vt_flux``.
    ("self.config.damp_w > 0.0", "_del6_vt_flux"),
    # iter-203: damp_w KE→heat (d_con).
    ("self.config.damp_w_d_con > 0.0", "heat_half"),
    # iter-209: damp_v KE→heat (d_con NH mirror of PE iter-208).
    # iter-895: substring updated from _exner_ref_broadcast (stale)
    # to _exner_ref_b1 matching the actual variable name at line 1268.
    ("self.config.damp_v_d_con > 0.0", "_exner_ref_b1"),
    # iter-218/219: per-step dissipative-heating cap.
    ("self.config.delt_max > 0.0", "jnp.clip"),
    # iter-222: corner-div damping d_con.
    ("config.corner_div_damp_d_con > 0.0", "_dKE_dt_corner_cdd"),
    # iter-224: cell-centre div_damp d_con.
    ("config.div_damp_d_con > 0.0", "_dKE_dt_corner_dd"),
    # iter-226: Smagorinsky-A_h d_con.
    ("config.ah_d_con > 0.0", "_dKE_dt_corner_ah"),
    # iter-240: NH mirror of iter-239 aggregate cap.  Sums the 3
    # NH tendency-based d_con contributions and applies the
    # iter-219 sponge-aware cap (k=0 → 0.1×, k=1 → 0.5×, k≥2 → 1×)
    # before adding to ``dtheta_p_dt``.
    ("_d_con_sum is not None", "_sponge_factor"),
]


def test_nh_fv3_call_sites_ast_regression():
    """AST regression guard: the FV3-faithful wirings in iter
    168/169/170/171/173/180/187/193 must remain present in the
    source.  Searches the source text for the Python-static gate
    expressions that activate each mechanism.  Catches a regression
    where the config field stays but the call-site is dropped."""
    src_path = legoesm_source_path(
        "atmosphere/dynamics/gcm/compressible_euler_cdgrid.py"
    )
    src_text = src_path.read_text()

    missing = []
    for gate, helper in NH_GATE_HELPER_PAIRS:
        if gate not in src_text:
            missing.append(f"gate {gate!r}")
        if helper not in src_text:
            missing.append(f"helper {helper!r}")

    assert not missing, (
        f"iter-172/178 AST regression: NH FV3 call sites missing "
        f"{missing}.  These wirings were added by iter "
        f"168/169/170/171/173; their absence silently disables the "
        f"FV3-faithful mechanism even when the user sets the config "
        f"flag."
    )


def test_iter178_ast_guard_self_check():
    """iter-178 self-check: simulate the regression that the guard
    is supposed to catch.  Build a fake source string that omits
    each gate/helper one at a time and verify the inner check
    function would have flagged it.

    This is a "test the test" sanity check — without it, a typo
    in the gate string (e.g., extra space, wrong operator) would
    cause the guard to ALWAYS pass, silently disabling the
    regression check.  iter-178 noticed this exact failure mode
    when extending the guard to iter-173: the dispatch substring
    used a different ``and _hb_div ==`` form than I'd initially
    typed."""
    # FV3_3D iter 196: use the shared ``NH_GATE_HELPER_PAIRS``
    # constant so the self-check and the outer test cannot drift
    # apart.  Prior versions kept a local copy here that became stale
    # when iter-190 changed the iter-187 substring and iter-193 added
    # the damp_w pair.
    gate_helper_pairs = NH_GATE_HELPER_PAIRS

    # Build a valid source containing all substrings.  Verify the
    # check passes.
    valid_src = "\n".join(
        f"{gate} -> {helper}" for gate, helper in gate_helper_pairs
    )
    missing = []
    for gate, helper in gate_helper_pairs:
        if gate not in valid_src:
            missing.append(f"gate {gate!r}")
        if helper not in valid_src:
            missing.append(f"helper {helper!r}")
    assert not missing, (
        f"self-check sanity: valid source must pass all checks; "
        f"got missing={missing}"
    )

    # Now drop one gate; verify the check correctly identifies it.
    for omit_idx in range(len(gate_helper_pairs)):
        broken_src = "\n".join(
            f"{g} -> {h}" for i, (g, h) in enumerate(gate_helper_pairs)
            if i != omit_idx
        )
        missing = []
        for gate, helper in gate_helper_pairs:
            if gate not in broken_src:
                missing.append(f"gate {gate!r}")
            if helper not in broken_src:
                missing.append(f"helper {helper!r}")
        assert missing, (
            f"self-check failed for omit_idx={omit_idx}: dropping "
            f"the {gate_helper_pairs[omit_idx][0]!r} gate did NOT "
            f"trip the guard.  This means the guard is silently "
            f"passing — likely a typo in the gate substring."
        )
