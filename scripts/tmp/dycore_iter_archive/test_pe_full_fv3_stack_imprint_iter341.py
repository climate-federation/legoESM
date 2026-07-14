"""FV3_3D iter 341: full-stack FV3-faithful integration regression
for the PE 3D path.

PE counterpart of iter-330 NH integration regression.  PE has
fewer FV3-fidelity flags than NH (per iter-331 PE/NH asymmetry):

* ``use_duogrid=True`` (grid construction; iter-333 wires PE
  ke_correction halo)
* ``use_fv3_metric_aware_d_con=True`` (iter-338 metric-aware
  ``damp_v_d_con`` site)
* full FV3 damping toolkit (iter-12/14/16/18/57-59/187/190
  ports)

PE doesn't need cv flag (PE uses cp_air; FV3 cp_air branch
already faithful) or vector halo flag (PE stores winds at
corners; no center-to-corner interp gap) or dynamic Exner (PE
uses actual T; no Π factor).

Tests
-----

1. ``test_pe_full_fv3_stack_finite`` — all fields finite (no NaN).
2. ``test_pe_full_fv3_stack_differentiable_at_rest`` — AD-safe
   under the full PE FV3 stack at rest.
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
def pe_state_duogrid():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=341)
    n_corners = n + 1
    u_p = rng.uniform(-15.0, 15.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-15.0, 15.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def _full_pe_fv3_stack_cfg():
    """All PE FV3-fidelity flags ON + full damping toolkit."""
    return CDGridPrimitiveEquationConfig(
        # iter-16/18 corner-divergence damping
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        # iter-12 damp_v + iter-208 d_con
        damp_v=0.030, nord_v=1,
        damp_v_d_con=1.0,
        # iter-14 a2b_ord4 zeta corner
        use_fv3_a2b_zeta_corner=True,
        # iter-5 cell-centre div_damp + iter-223 d_con
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
        # iter-57/58 Smag-A_h + iter-225 d_con
        A_h=1e6, smagorinsky_cs=0.20,
        ah_d_con=1.0,
        # iter-218/219 sponge-aware delt_max
        delt_max=1.0,
        # iter-338 metric-aware d_con
        use_fv3_metric_aware_d_con=True,
        # PE-specific
        hyperdiff_coeff=0.0,
        hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )


def test_pe_full_fv3_stack_finite(pe_state_duogrid):
    """All PE fields finite under the full FV3 stack."""
    grid, _, coord, state = pe_state_duogrid
    m = CDGridPrimitiveEquationModel(grid, coord, _full_pe_fv3_stack_cfg())
    s = m.step(state, 100.0)
    for fld_name in ("u_d", "v_d", "T", "p_s"):
        fld = getattr(s, fld_name)
        assert np.all(np.isfinite(np.asarray(fld.data))), (
            f"PE full FV3 stack produced non-finite {fld_name}."
        )


def test_pe_full_fv3_stack_differentiable_at_rest(pe_state_duogrid):
    """jax.grad through 2 PE steps with full FV3 stack at rest."""
    grid, _, coord, state = pe_state_duogrid
    rest = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )
    m = CDGridPrimitiveEquationModel(grid, coord, _full_pe_fv3_stack_cfg())

    def loss(amp):
        s = rest._replace(
            u_d=rest.u_d.replace(
                data=amp * jnp.ones_like(rest.u_d.data),
            ),
        )
        for _ in range(2):
            s = m.step(s, 100.0)
        return jnp.mean(s.T.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)
