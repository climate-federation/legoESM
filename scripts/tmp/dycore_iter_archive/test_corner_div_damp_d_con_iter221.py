"""FV3_3D iter 221: KE→heat ``d_con`` for the iter-16/18 corner-
divergence damping (PE).

iter-208 ported the ``d_con`` energy-conservation block for the
iter-12 ``damp_v`` post-step damping.  iter-221 mirrors the
pattern for the iter-16/18 corner-divergence damping which is the
DOMINANT FV3-faithful KE-removing mechanism on the PE 3D path.

The corner-div mechanism subtracts the gradient of a corner-
staggered KE-correction from the wind tendency:

    du_d_dt -= ∇x(damp * delpc) / 2dx_corner
    dv_d_dt -= ∇y(damp * delpc) / 2dy_corner

The KE removed per second (leading order in dt) is:

    dKE/dt = u_d * du_d_dt_cdd + v_d * dv_d_dt_cdd

The energy-conserving heat addition is (FV3 sw_core.F90 line
1085-1086 + dyn_core.F90 line 1764-1779):

    dT/dt += -corner_div_damp_d_con * (dKE/dt) / c_pd

where the heat tendency is computed at corners then projected to
cell centres via ``interp_corner_to_center`` for the T
tendency.  The 0.5*du² term that FV3 carries in the discrete
form is O(dt) and dropped in this RK3-compatible tendency port.

Tests
-----

1. ``test_corner_div_damp_d_con_off_baseline`` — default 0.0 is
   bit-for-bit baseline (the rest of the corner-div damping
   mechanism is unchanged).
2. ``test_corner_div_damp_d_con_changes_T`` — d_con > 0 changes T
   when winds are non-zero and corner-div damping is active.
3. ``test_corner_div_damp_d_con_no_op_when_corner_div_off`` — gated
   INSIDE the ``corner_div_damp_d2_bg > 0`` block; turning the
   corner-div damp off makes d_con a no-op.
4. ``test_corner_div_damp_d_con_differentiable_at_rest`` — AD-safe
   at rest state (no NaN).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def small_pe_state():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=221)
    n_corners = n + 1
    u_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def _step5(model, state, dt=100.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_corner_div_damp_d_con_off_baseline(small_pe_state):
    """``corner_div_damp_d_con=0.0`` (default) is bit-for-bit equal
    to the no-d_con baseline; the rest of the mechanism unchanged."""
    grid, cdgrid, coord, state = small_pe_state

    common = dict(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_explicit_off = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d_con=0.0,
    )
    cfg_default = CDGridPrimitiveEquationConfig(**common)

    s_explicit = _step5(CDGridPrimitiveEquationModel(
        grid, coord, cfg_explicit_off,
    ), state)
    s_default = _step5(CDGridPrimitiveEquationModel(
        grid, coord, cfg_default,
    ), state)

    np.testing.assert_array_equal(s_explicit.T.data, s_default.T.data)


def test_corner_div_damp_d_con_changes_T(small_pe_state):
    """With d_con > 0 and active corner-div damping + non-zero
    winds, T evolves DIFFERENTLY than the d_con=0 baseline."""
    grid, cdgrid, coord, state = small_pe_state

    common = dict(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_off = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d_con=0.0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d_con=1.0,
    )

    s_off = _step5(CDGridPrimitiveEquationModel(
        grid, coord, cfg_off,
    ), state)
    s_on = _step5(CDGridPrimitiveEquationModel(
        grid, coord, cfg_on,
    ), state)

    diff = float(jnp.max(jnp.abs(s_on.T.data - s_off.T.data)))
    assert diff > 1e-8, (
        f"corner_div_damp_d_con=1.0 must change T relative to "
        f"d_con=0 baseline; got max|ΔT|={diff:.4e}.  Either the "
        f"d_con tendency is not being applied or the corner-div "
        f"damping is not active."
    )


def test_corner_div_damp_d_con_no_op_when_corner_div_off(small_pe_state):
    """d_con is gated INSIDE the corner-div block; if corner-div is
    off, d_con > 0 must be a no-op."""
    grid, cdgrid, coord, state = small_pe_state

    common = dict(
        corner_div_damp_d2_bg=0.0,    # corner-div OFF
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_a = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d_con=0.0,
    )
    cfg_b = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d_con=1.0,
    )
    s_a = CDGridPrimitiveEquationModel(grid, coord, cfg_a).step(state, 100.0)
    s_b = CDGridPrimitiveEquationModel(grid, coord, cfg_b).step(state, 100.0)
    np.testing.assert_array_equal(s_a.T.data, s_b.T.data)


def test_corner_div_damp_d_con_differentiable_at_rest():
    """At rest, ``jax.grad`` w.r.t. wind perturbation amplitude is
    finite when corner_div_damp_d_con > 0."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state_rest = hydrostatic_to_fv3(state_cc, cdgrid)
    state_rest = state_rest._replace(
        u_d=state_rest.u_d.replace(
            data=jnp.zeros_like(state_rest.u_d.data),
        ),
        v_d=state_rest.v_d.replace(
            data=jnp.zeros_like(state_rest.v_d.data),
        ),
    )

    cfg = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss(eps):
        u_d_pert = state_rest.u_d.data + eps * jnp.ones_like(
            state_rest.u_d.data,
        )
        s = state_rest._replace(
            u_d=state_rest.u_d.replace(data=u_d_pert),
        )
        s_new = model.step(s, 100.0)
        return jnp.sum(s_new.T.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        f"PE corner_div_damp_d_con must be AD-safe at rest; got "
        f"grad={g}"
    )
