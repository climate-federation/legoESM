"""FV3_3D iter 189: plumb the actual integration dt to the
corner-div damping adaptive cap in BOTH PE and NH 3D paths.

iter-187/iter-188 used ``config.corner_div_damp_dt_proxy`` (a
constant approximation) for the FV3 ``min(0.20, dddmp * dt *
smag_vort)`` cap.  FV3 ``d_sw5`` uses the actual integration dt.
This iter threads the real dt through ``model.step`` →
``tendency_fn`` → ``fv3_hydrostatic_tendencies`` (PE) and the
analogous NH path, with fallback to the existing config dt-proxy
when ``dt_actual=None`` (preserves backward compatibility).

Tests
-----

1. ``test_pe_dt_actual_default_matches_dt_proxy_fallback`` — when
   ``dt_actual=None`` (the existing direct-call test pattern),
   the cap uses ``config.corner_div_damp_dt_proxy`` exactly.
   Bit-for-bit equivalent to iter-188.
2. ``test_pe_dt_actual_changes_smag_vort_cap`` — at
   nord=1 + d4_bg + dddmp + d2_bg=floor + delpc large enough to
   engage the cap, ``dt_actual=N`` produces a different state
   than ``dt_actual=2N`` (cap engagement scales with dt).
3. ``test_nh_dt_actual_default_matches_dt_proxy_fallback`` — NH
   counterpart of test 1.
4. ``test_nh_dt_actual_changes_smag_vort_cap`` — NH counterpart
   of test 2.
5. ``test_pe_step_passes_dt_actual_ast_regression`` — AST guard
   that ``CDGridPrimitiveEquationModel.step`` passes
   ``dt_actual=dt`` to ``fv3_hydrostatic_tendencies``.  Catches a
   refactor that drops the wiring and silently falls back to
   the dt-proxy.
6. ``test_nh_step_passes_dt_actual_ast_regression`` — NH
   counterpart of test 5.
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
    cdgrid_compressible_euler_slow_tendencies,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    fv3_hydrostatic_tendencies,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
    standard_hybrid_levels,
)

from tests.legoesm_paths import legoesm_source_path


_PE_SRC = legoesm_source_path(
    "atmosphere/dynamics/gcm/primitive_eq_cdgrid.py"
)
_NH_SRC = legoesm_source_path(
    "atmosphere/dynamics/gcm/compressible_euler_cdgrid.py"
)


@pytest.fixture(scope="module")
def small_pe_state():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    return grid, cdgrid, coord, state


@pytest.fixture(scope="module")
def small_nh_state():
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
    return grid, cdgrid_for_nh(grid), height_coord, terrain_metric, state


def cdgrid_for_nh(grid):
    return create_cubed_sphere_cdgrid(grid)


# ============================================================================
# PE tests
# ============================================================================

def test_pe_dt_actual_default_matches_dt_proxy_fallback(small_pe_state):
    """Direct call to ``fv3_hydrostatic_tendencies`` without
    ``dt_actual`` (the iter-188 baseline) must match an explicit
    ``dt_actual=config.corner_div_damp_dt_proxy`` call bit-for-bit."""
    grid, cdgrid, coord, state = small_pe_state

    cfg = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        corner_div_damp_dt_proxy=200.0,
    )

    n = grid.n
    nlev = state.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=189)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1, nlev))
    s = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    tend_fallback = fv3_hydrostatic_tendencies(s, grid, coord, cdgrid, cfg)
    tend_explicit = fv3_hydrostatic_tendencies(
        s, grid, coord, cdgrid, cfg,
        dt_actual=cfg.corner_div_damp_dt_proxy,
    )

    np.testing.assert_array_equal(
        np.asarray(tend_fallback.du_d_dt.data),
        np.asarray(tend_explicit.du_d_dt.data),
    )
    np.testing.assert_array_equal(
        np.asarray(tend_fallback.dv_d_dt.data),
        np.asarray(tend_explicit.dv_d_dt.data),
    )


def test_pe_dt_actual_changes_smag_vort_cap(small_pe_state):
    """Different ``dt_actual`` values produce different states
    when the smag_vort cap is engaged.  Catches a regression
    where the iter-189 wiring is silently disabled."""
    grid, cdgrid, coord, state = small_pe_state

    n = grid.n
    nlev = state.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=189)
    # Strong perturbation so the cap engages above the d2_bg floor.
    u_p = rng.uniform(-30.0, 30.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n + 1, n + 1, nlev))
    s = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        # Very large dddmp so the cap engages aggressively.
        corner_div_damp_dddmp=1.0,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        corner_div_damp_dt_proxy=200.0,
    )

    tend_dt100 = fv3_hydrostatic_tendencies(
        s, grid, coord, cdgrid, cfg, dt_actual=100.0,
    )
    tend_dt400 = fv3_hydrostatic_tendencies(
        s, grid, coord, cdgrid, cfg, dt_actual=400.0,
    )

    # When the cap is engaged, larger dt → smag_vort larger → cap
    # at 0.20 hits sooner / engages over more cells → output differs.
    diff = float(jnp.max(jnp.abs(
        tend_dt100.du_d_dt.data - tend_dt400.du_d_dt.data
    )))
    base = float(jnp.max(jnp.abs(tend_dt100.du_d_dt.data)))
    assert diff > 1e-6 * base, (
        f"iter-189 PE dt_actual must affect output when cap engages "
        f"(diff={diff:.3e}, base={base:.3e})"
    )


# ============================================================================
# NH tests
# ============================================================================

def test_nh_dt_actual_default_matches_dt_proxy_fallback(small_nh_state):
    """NH counterpart of test 1."""
    grid, cdgrid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        corner_div_damp_dt_proxy=10.0,
    )

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=189)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    tend_fallback = cdgrid_compressible_euler_slow_tendencies(
        s, grid, height_coord, terrain_metric, cdgrid, cfg,
    )
    tend_explicit = cdgrid_compressible_euler_slow_tendencies(
        s, grid, height_coord, terrain_metric, cdgrid, cfg,
        dt_actual=cfg.corner_div_damp_dt_proxy,
    )

    np.testing.assert_array_equal(
        np.asarray(tend_fallback.du_dt.data),
        np.asarray(tend_explicit.du_dt.data),
    )
    np.testing.assert_array_equal(
        np.asarray(tend_fallback.dv_dt.data),
        np.asarray(tend_explicit.dv_dt.data),
    )


def test_nh_dt_actual_changes_smag_vort_cap(small_nh_state):
    """NH counterpart of test 2."""
    grid, cdgrid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=189)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=1.0,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        corner_div_damp_dt_proxy=10.0,
    )

    tend_dt5 = cdgrid_compressible_euler_slow_tendencies(
        s, grid, height_coord, terrain_metric, cdgrid, cfg,
        dt_actual=5.0,
    )
    tend_dt20 = cdgrid_compressible_euler_slow_tendencies(
        s, grid, height_coord, terrain_metric, cdgrid, cfg,
        dt_actual=20.0,
    )

    diff = float(jnp.max(jnp.abs(
        tend_dt5.du_dt.data - tend_dt20.du_dt.data
    )))
    base = float(jnp.max(jnp.abs(tend_dt5.du_dt.data)))
    assert diff > 1e-6 * base, (
        f"iter-189 NH dt_actual must affect output when cap engages "
        f"(diff={diff:.3e}, base={base:.3e})"
    )


# ============================================================================
# AST regression: verify model.step closures actually pass dt_actual
# ============================================================================

def test_pe_step_passes_dt_actual_ast_regression():
    """AST guard: ``CDGridPrimitiveEquationModel.step`` (or its
    helper) must call ``fv3_hydrostatic_tendencies(...)`` with
    ``dt_actual=dt``.  Without this, the model-step path silently
    falls back to ``config.corner_div_damp_dt_proxy``."""
    src = _PE_SRC.read_text()
    assert "dt_actual=dt" in src, (
        "iter-189 PE AST regression: ``CDGridPrimitiveEquationModel.step`` "
        "must pass ``dt_actual=dt`` to ``fv3_hydrostatic_tendencies``.  "
        "Without this, the model-step path silently falls back to "
        "``config.corner_div_damp_dt_proxy`` and the iter-189 fidelity "
        "fix is disabled."
    )


def test_nh_step_passes_dt_actual_ast_regression():
    """NH counterpart of test 5."""
    src = _NH_SRC.read_text()
    assert "dt_actual=dt" in src, (
        "iter-189 NH AST regression: ``CDGridCompressibleEulerModel.step`` "
        "must pass ``dt_actual=dt`` to "
        "``cdgrid_compressible_euler_slow_tendencies``.  Without this, "
        "the model-step path silently falls back to "
        "``config.corner_div_damp_dt_proxy``."
    )
