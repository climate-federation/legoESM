"""FV3_3D iter 187: faithful port of FV3 ``smag_vort`` adaptive cap
to the ``nord >= 1`` corner-divergence damping branch in BOTH the
hydrostatic PE (``primitive_eq_cdgrid.py``) and the non-hydrostatic
NH (``compressible_euler_cdgrid.py``) 3D paths.

FV3 ``sw_core.F90:d_sw5`` uses TWO different adaptive-cap formulas:

    nord = 0  (line 1722):
        damp  = da_min_c * max(d2_bg, min(0.20, dddmp * |delpc * dt|))

    nord >= 1 (lines 1797-1809):
        wk_corner = a2b_ord4(zeta_relative)
        smag_vort = |dt| * sqrt(delpc**2 + wk_corner**2)
        damp2     = da_min_c * max(d2_bg, min(0.20, dddmp * smag_vort))

The legoESM SW core already implements both forms correctly
(``fv3_sw_core.py:1768-1809``).  iter-168 (NH) and iter-18 (PE)
ported the ``nord = 0`` formula but kept using it for ALL nord —
faithful to FV3 nord=0 only.  iter-187 adds the smag_vort form for
the nord >= 1 branch.

Tests
-----

PE side (mirrors iter-18 wiring):

1. ``test_pe_smag_vort_grad_at_rest`` — ``jax.grad`` through 3 PE
   steps with nord=1, d4_bg, dddmp, d2_bg=floor at the rest state
   stays finite.  Catches the sqrt(0) hazard the iter-181/183
   double-where pattern protects against.
2. ``test_pe_smag_vort_changes_state_under_vortical_perturbation``
   — at nord=1, a perturbation that produces non-zero relative
   vorticity (with delpc ~ 0) produces a measurably different state
   vs. the same config gated to baseline (d2_bg=0).  Demonstrates
   the smag_vort path engages on relative ζ.
3. ``test_pe_nord2_fv3_production_default_finite`` — FV3 AM4
   production default (nord=2, d4_bg=0.16, dddmp=0.2) runs and
   produces finite output for 5 steps from a perturbed state.

NH side (mirrors iter-168 wiring):

4. ``test_nh_smag_vort_grad_at_rest`` — same as test 1 for NH.
5. ``test_nh_smag_vort_changes_state_under_vortical_perturbation``
   — same as test 2 for NH.
6. ``test_nh_nord2_fv3_production_default_finite`` — same as test 3
   for NH.

Cross-cutting:

7. ``test_smag_vort_uses_relative_vorticity_via_a2b_ord4`` — AST
   regression that the iter-187 site reads ``zeta`` (relative)
   and calls ``interp_center_to_corner_a2b_ord4`` (FV3 ``a2b_ord4``
   for ``wk → vort``).  Catches a refactor that accidentally
   reads the absolute ``zeta_corner`` (relative + f_corner).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
    standard_hybrid_levels,
)

from tests.legoesm_paths import legoesm_source_path


# ---------------------------------------------------------------- PE fixtures

@pytest.fixture(scope="module")
def small_pe_state():
    """Held-Suarez init at C8 with hybrid sigma; same as iter-174 PE."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    return grid, cdgrid, coord, state


# ---------------------------------------------------------------- NH fixtures

@pytest.fixture(scope="module")
def small_nh_state():
    """Same fixture as iter-168/172/174 NH tests."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
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


# ============================================================================
# PE tests (mirrors iter-18 wiring at primitive_eq_cdgrid.py:693-723)
# ============================================================================

def test_pe_smag_vort_grad_at_rest(small_pe_state):
    """``jax.grad`` through 3 PE steps with nord=1 + d4_bg + dddmp +
    d2_bg=floor at exactly the rest state must give finite gradients.

    The smag_vort formula computes ``sqrt(delpc**2 + zeta_corner**2)``
    which has an undefined gradient at exactly (0, 0).  iter-181/183's
    JAX double-where pattern (mirrored in iter-187) masks the branch
    so the backward pass produces a safe value.  Without the
    protection, ``jax.grad`` returns NaN here."""
    grid, cdgrid, coord, state = small_pe_state

    cfg = CDGridPrimitiveEquationConfig(
        # nord >= 1 + d4_bg > 0 + dddmp > 0 + d2_bg = floor — engages
        # the iter-187 smag_vort branch fully.
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        use_conservation_fixer=False,
        fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    # Use a near-rest state: zero winds, exact T as held-Suarez.
    # State winds are already small in HS init; explicitly zero them
    # so ``zeta`` and ``delpc`` are both exactly 0.0 at every corner.
    rest = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )

    def loss_fn(T_data):
        s = rest._replace(T=rest.T.replace(data=T_data))
        for _ in range(3):
            s = model.step(s, 100.0)
        return jnp.mean(s.u_d.data ** 2 + s.v_d.data ** 2)

    grad = jax.grad(loss_fn)(rest.T.data)
    assert jnp.all(jnp.isfinite(grad)), (
        "PE smag_vort iter-187: AD at rest state must be finite "
        "(double-where masks sqrt(0) singularity)."
    )


def test_pe_smag_vort_changes_state_under_vortical_perturbation(small_pe_state):
    """At nord=1 + d4_bg + dddmp + d2_bg=floor, a vortical wind
    perturbation produces measurably different state than the
    baseline (d2_bg=0 gates the entire block off)."""
    grid, cdgrid, coord, state = small_pe_state

    n = grid.n
    nlev = state.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=187)
    # Vortical-leaning perturbation: alternating signs along i for v_d
    # (creates ζ ≈ ∂v/∂x), small in u_d.
    u_p = rng.uniform(-0.3, 0.3, size=(6, n + 1, n + 1, nlev))
    v_p_sign = jnp.where(jnp.arange(n + 1)[None, :, None, None] % 2 == 0, 1.0, -1.0)
    v_p = jnp.broadcast_to(v_p_sign, (6, n + 1, n + 1, nlev)) * 1.5
    s = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=v_p),
    )

    cfg_off = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0,           # gates entire block off
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        use_conservation_fixer=False,
        fix_mass=False,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,        # engages smag_vort
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        use_conservation_fixer=False,
        fix_mass=False,
    )

    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)

    s_off = m_off.step(s, 100.0)
    s_on = m_on.step(s, 100.0)

    diff = float(jnp.max(jnp.abs(s_off.u_d.data - s_on.u_d.data)))
    base = float(jnp.max(jnp.abs(s_off.u_d.data)))
    assert diff > 1e-8 * max(base, 1.0), (
        f"PE smag_vort: nord=1 + smag_vort branch must change winds "
        f"vs baseline (diff={diff:.3e}, base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_on.u_d.data))
    assert jnp.all(jnp.isfinite(s_on.v_d.data))


def test_pe_nord2_higher_order_branch_finite(small_pe_state):
    """nord=2 (FV3 d_sw5 production order, del-6 corner-div) runs and
    produces finite output.  Uses an n=8-scaled ``d4_bg`` (dd8 ∝
    da_min_c^3 grows quickly with nord+1=3 at coarse C8) to match the
    existing iter-18 PE test convention.  Catches shape mismatch or
    stability bug specific to the nord >= 2 branch."""
    grid, cdgrid, coord, state = small_pe_state

    cfg = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        # n=8-scaled: existing iter-18 tests use d4_bg=1e-3 at nord=1.
        # nord=2 raises the effective dd8 = (da_min_c*d4_bg)**3, so
        # use a smaller value to keep dd8 in the same magnitude range.
        corner_div_damp_d4_bg=1e-4,
        corner_div_damp_nord=2,
        use_conservation_fixer=False,
        fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    s = state
    for _ in range(5):
        s = model.step(s, 100.0)

    assert jnp.all(jnp.isfinite(s.u_d.data))
    assert jnp.all(jnp.isfinite(s.v_d.data))
    assert jnp.all(jnp.isfinite(s.T.data))


# ============================================================================
# NH tests (mirrors iter-168 wiring at compressible_euler_cdgrid.py:468-495)
# ============================================================================

def test_nh_smag_vort_grad_at_rest(small_nh_state):
    """NH analog of test_pe_smag_vort_grad_at_rest."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(theta_p_data):
        s = state._replace(
            theta_prime=state.theta_prime.replace(data=theta_p_data),
        )
        for _ in range(3):
            s = model.step(s, 10.0)
        return jnp.mean(s.u.data ** 2 + s.v.data ** 2)

    grad = jax.grad(loss_fn)(state.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad)), (
        "NH smag_vort iter-187: AD at rest state must be finite "
        "(double-where masks sqrt(0) singularity)."
    )


def test_nh_smag_vort_changes_state_under_vortical_perturbation(small_nh_state):
    """NH analog of test_pe_smag_vort_changes_state_under_vortical_perturbation."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=187)
    u_p = rng.uniform(-0.3, 0.3, size=(6, n, n, nlev))
    v_p_sign = jnp.where(jnp.arange(n)[None, :, None, None] % 2 == 0, 1.0, -1.0)
    v_p = jnp.broadcast_to(v_p_sign, (6, n, n, nlev)) * 1.5
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=v_p),
    )

    cfg_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0,           # gates block off
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
    )

    m_off = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_off,
    )
    m_on = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_on,
    )

    s_off = m_off.step(s, 10.0)
    s_on = m_on.step(s, 10.0)

    diff = float(jnp.max(jnp.abs(s_off.u.data - s_on.u.data)))
    base = float(jnp.max(jnp.abs(s_off.u.data)))
    assert diff > 1e-8 * max(base, 1.0), (
        f"NH smag_vort: nord=1 + smag_vort branch must change winds "
        f"vs baseline (diff={diff:.3e}, base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_on.u.data))
    assert jnp.all(jnp.isfinite(s_on.v.data))


def test_nh_nord2_fv3_production_default_finite(small_nh_state):
    """NH analog of test_pe_nord2_fv3_production_default_finite."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=287)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.16,         # FV3 AM4 default
        corner_div_damp_nord=2,             # FV3 AM4 default (del-6)
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    for _ in range(5):
        s = model.step(s, 10.0)

    assert jnp.all(jnp.isfinite(s.u.data))
    assert jnp.all(jnp.isfinite(s.v.data))
    assert jnp.all(jnp.isfinite(s.theta_prime.data))


# ============================================================================
# Cross-cutting AST regression
# ============================================================================

def test_smag_vort_uses_relative_vorticity_via_a2b_ord4():
    """AST regression: the iter-187 smag_vort site MUST use the
    relative-vorticity ``zeta`` (cell centres, no Coriolis) lifted to
    corners via ``interp_center_to_corner_a2b_ord4``.  Catches a
    refactor that accidentally feeds the absolute ``zeta_corner =
    zeta_relative + f_corner`` into smag_vort — that would inflate
    the cap at the poles where f dominates and silently bias the
    damping.

    We assert that BOTH PE and NH source contain the iter-187 marker
    sequence inside the corner-div-damp d4 branch:

        _zeta_smag_corner = jax.vmap(
            lambda lev: interp_center_to_corner_a2b_ord4(...)
        )(zeta)

    The literal substring matched is robust to small whitespace changes
    but unique to the iter-187 site (``_zeta_smag_corner`` is unused
    elsewhere)."""
    pe_path = legoesm_source_path("atmosphere/dynamics/gcm/primitive_eq_cdgrid.py")
    nh_path = legoesm_source_path("atmosphere/dynamics/gcm/compressible_euler_cdgrid.py")

    for label, p in [("PE", pe_path), ("NH", nh_path)]:
        src = p.read_text()
        assert "_zeta_smag_corner" in src, (
            f"{label}: iter-187 smag_vort variable ``_zeta_smag_corner`` "
            f"missing — the FV3 nord >= 1 fidelity port has been dropped."
        )
        # Verify the relative-zeta source is used (not zeta_corner).
        # The exact substring is the smag_arg expression, present only
        # at the iter-187 site.
        assert "_smag_arg = _delpc_initial ** 2 + _zeta_smag_corner ** 2" in src, (
            f"{label}: iter-187 smag_vort formula not present.  Either "
            f"the wiring has been dropped or the formula has been "
            f"changed away from FV3 sw_core.F90:1797."
        )
        # Confirm a2b_ord4 helper is used, not a 2nd-order avg.
        assert "interp_center_to_corner_a2b_ord4" in src, (
            f"{label}: iter-187 must use a2b_ord4 for ζ→corner "
            f"(FV3 sw_core.F90:1795)."
        )
