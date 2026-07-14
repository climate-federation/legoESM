"""FV3_3D iter 191: coverage tests for the iter-190 a2b_ord4(zeta)
dedup when BOTH iter-170 (``use_fv3_a2b_zeta_corner=True``) and
iter-187 (``corner_div_damp_d4_bg > 0`` + ``corner_div_damp_nord > 0``)
are active simultaneously.  This is the case where iter-190 actually
deduplicates: the SAME ``a2b_ord4(zeta)`` is consumed at the iter-170
``zeta_corner`` site AND at the iter-187 ``_zeta_smag_corner`` site,
so iter-190 introduces a single ``_zeta_a2b_ord4`` local computed
once and reused at both.

Prior to iter-191, no test exercises this combination:

* ``test_div_damp_adaptive.py`` (PE iter-18): nord >= 1 + d4_bg > 0
  but ``use_fv3_a2b_zeta_corner=False`` (default).
* ``test_corner_div_damp_smag_vort_iter187.py`` (PE+NH): nord >= 1
  + d4_bg > 0 but ``use_fv3_a2b_zeta_corner`` not set.
* ``test_fv3_full_toolkit_ad_at_rest_iter184/185.py``:
  ``use_fv3_a2b_zeta_corner=True`` but ``corner_div_damp_nord=0``
  (default), so iter-187 smag_vort branch dormant.

This iter closes the gap with focused tests that:

1. Verify both wirings are exercised: output differs from
   iter-170-only AND iter-187-only configs.
2. Verify finite output (no AD-hazard or numerical issue from
   reusing the same array at two sites).
3. Verify AD-at-rest is still safe with BOTH flags.

Tests are parametrized over PE and NH paths.
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


# ---------------------------------------------------------- shared fixtures

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
    return grid, height_coord, terrain_metric, state


# ============================================================================
# PE path
# ============================================================================

def _pe_perturb(state, grid, seed):
    n = grid.n
    nlev = state.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-0.5, 0.5, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-0.5, 0.5, size=(6, n + 1, n + 1, nlev))
    return state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )


def test_pe_both_flags_finite_and_differs_from_each_alone(small_pe_state):
    """PE: with iter-170 ON + iter-187 ON, output is finite AND
    differs measurably from iter-170-only AND iter-187-only configs.
    Proves iter-190 dedup wires both sites correctly."""
    grid, cdgrid, coord, state = small_pe_state
    s = _pe_perturb(state, grid, 191)

    base_kwargs = dict(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        use_conservation_fixer=False, fix_mass=False,
    )

    cfg_170_only = CDGridPrimitiveEquationConfig(
        **base_kwargs, use_fv3_a2b_zeta_corner=True,
        # disable iter-187 by setting nord=0 (gates off d4 branch)
    )
    # Override nord=0 — the **base_kwargs sets nord=1 so re-construct.
    cfg_170_only = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        # nord=0 + d4_bg=0 disables iter-187 smag_vort branch
        corner_div_damp_d4_bg=0.0,
        corner_div_damp_nord=0,
        use_fv3_a2b_zeta_corner=True,    # iter-170 ON
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_187_only = CDGridPrimitiveEquationConfig(
        **base_kwargs,                   # iter-187 ON via nord=1+d4_bg
        use_fv3_a2b_zeta_corner=False,   # iter-170 OFF
    )
    cfg_both = CDGridPrimitiveEquationConfig(
        **base_kwargs,                   # iter-187 ON
        use_fv3_a2b_zeta_corner=True,    # iter-170 ON
    )

    m_170 = CDGridPrimitiveEquationModel(grid, coord, cfg_170_only)
    m_187 = CDGridPrimitiveEquationModel(grid, coord, cfg_187_only)
    m_both = CDGridPrimitiveEquationModel(grid, coord, cfg_both)

    s_170 = m_170.step(s, 100.0)
    s_187 = m_187.step(s, 100.0)
    s_both = m_both.step(s, 100.0)

    # 1. all finite
    for label, st in [("170", s_170), ("187", s_187), ("both", s_both)]:
        assert jnp.all(jnp.isfinite(st.u_d.data)), f"{label}: u_d NaN"
        assert jnp.all(jnp.isfinite(st.v_d.data)), f"{label}: v_d NaN"

    # 2. both differs from 170-only AND 187-only
    diff_vs_170 = float(jnp.max(jnp.abs(s_both.u_d.data - s_170.u_d.data)))
    diff_vs_187 = float(jnp.max(jnp.abs(s_both.u_d.data - s_187.u_d.data)))
    base = float(jnp.max(jnp.abs(s_both.u_d.data)))
    assert diff_vs_170 > 1e-8 * max(base, 1.0), (
        f"PE iter-190: BOTH config must differ from iter-170-only "
        f"(diff={diff_vs_170:.3e}, base={base:.3e}) — proves "
        f"iter-187 smag_vort branch is exercised when both flags ON."
    )
    assert diff_vs_187 > 1e-8 * max(base, 1.0), (
        f"PE iter-190: BOTH config must differ from iter-187-only "
        f"(diff={diff_vs_187:.3e}, base={base:.3e}) — proves "
        f"iter-170 a2b_ord4 ζ_corner is exercised when both flags ON."
    )


def test_pe_both_flags_grad_at_rest(small_pe_state):
    """PE: ``jax.grad`` with both iter-170 + iter-187 active at rest
    state stays finite.  The dedup'd ``_zeta_a2b_ord4`` is consumed
    at TWO sites; AD must flow correctly through both consumers."""
    grid, cdgrid, coord, state = small_pe_state

    cfg = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        use_fv3_a2b_zeta_corner=True,    # iter-170 ON
        use_conservation_fixer=False, fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

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
        "PE iter-190 dedup: jax.grad with both iter-170 + iter-187 "
        "ON at rest must be finite.  iter-181/183 double-where "
        "pattern guards sqrt(0)."
    )


# ============================================================================
# NH path
# ============================================================================

def _nh_perturb(state, grid, seed):
    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev))
    v_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )


def test_nh_both_flags_finite_and_differs_from_each_alone(small_nh_state):
    """NH counterpart of test_pe_both_flags_finite_and_differs_from_each_alone."""
    grid, height_coord, terrain_metric, state = small_nh_state
    s = _nh_perturb(state, grid, 191)

    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
    )
    cfg_170_only = CDGridCompressibleEulerConfig(
        **common,
        corner_div_damp_d4_bg=0.0,
        corner_div_damp_nord=0,           # iter-187 OFF
        use_fv3_a2b_zeta_corner=True,     # iter-170 ON
    )
    cfg_187_only = CDGridCompressibleEulerConfig(
        **common,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,            # iter-187 ON
        use_fv3_a2b_zeta_corner=False,    # iter-170 OFF
    )
    cfg_both = CDGridCompressibleEulerConfig(
        **common,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,            # iter-187 ON
        use_fv3_a2b_zeta_corner=True,     # iter-170 ON
    )

    m_170 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_170_only,
    )
    m_187 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_187_only,
    )
    m_both = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_both,
    )

    s_170 = m_170.step(s, 10.0)
    s_187 = m_187.step(s, 10.0)
    s_both = m_both.step(s, 10.0)

    for label, st in [("170", s_170), ("187", s_187), ("both", s_both)]:
        assert jnp.all(jnp.isfinite(st.u.data)), f"{label}: u NaN"
        assert jnp.all(jnp.isfinite(st.v.data)), f"{label}: v NaN"

    diff_vs_170 = float(jnp.max(jnp.abs(s_both.u.data - s_170.u.data)))
    diff_vs_187 = float(jnp.max(jnp.abs(s_both.u.data - s_187.u.data)))
    base = float(jnp.max(jnp.abs(s_both.u.data)))
    assert diff_vs_170 > 1e-8 * max(base, 1.0), (
        f"NH iter-190: BOTH must differ from iter-170-only "
        f"(diff={diff_vs_170:.3e}, base={base:.3e})"
    )
    assert diff_vs_187 > 1e-8 * max(base, 1.0), (
        f"NH iter-190: BOTH must differ from iter-187-only "
        f"(diff={diff_vs_187:.3e}, base={base:.3e})"
    )


def test_nh_both_flags_grad_at_rest(small_nh_state):
    """NH counterpart of test_pe_both_flags_grad_at_rest."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,            # iter-187 ON
        use_fv3_a2b_zeta_corner=True,     # iter-170 ON
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
        "NH iter-190 dedup: jax.grad with both iter-170 + iter-187 "
        "ON at rest must be finite."
    )
