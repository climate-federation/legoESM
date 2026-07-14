"""FV3_3D iter 216: PE counterpart of iter-175 ``damp_v`` direction
test.  iter-175 verified NH damp_v reduces ``mean(|ζ|)`` on a
vortical IC.  PE iter-12 damp_v has the same backbone
(``fv3_del6_vorticity_damping``) but no analogous direction test.

iter 216 mirrors iter-175 for PE.

Tests
-----

1. ``test_pe_damp_v_reduces_vorticity`` — vortical IC; damp_v=0.030
   (nord_v=0) reduces mean|ζ| more than no-damping baseline.
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
from legoesm.core.operators_cdgrid import dgrid_vorticity
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def vortical_pe_state():
    """PE state with sinusoidal vortical perturbation in v_d (so
    ζ ≈ -∂v/∂x is large).  Mirrors iter-175 NH fixture but for PE
    D-grid corner layout."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    n_corners = n + 1
    i_idx = jnp.arange(n_corners)
    pattern = (
        jnp.sin(2 * jnp.pi * i_idx[None, :, None, None] / n_corners)
        * jnp.ones_like(jnp.arange(n_corners)[None, None, :, None],
                        dtype=jnp.float64)
        * jnp.ones_like(jnp.arange(nlev)[None, None, None, :],
                        dtype=jnp.float64)
    )
    pattern = jnp.broadcast_to(pattern, (6, n_corners, n_corners, nlev))
    V0 = 5.0
    v_perturb = jnp.asarray(V0 * pattern, dtype=jnp.float64)

    state = state._replace(
        u_d=state.u_d.replace(
            data=jnp.zeros_like(state.u_d.data),
        ),
        v_d=state.v_d.replace(data=v_perturb),
    )
    return grid, cdgrid, coord, state


def _mean_abs_vorticity_pe(state, cdgrid):
    """ζ at cell centres for PE state.  PE stores winds at corners
    so use them directly."""
    zeta = dgrid_vorticity(state.u_d.data, state.v_d.data, cdgrid)
    return float(jnp.mean(jnp.abs(zeta)))


def _step5(model, state, dt=100.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_pe_damp_v_reduces_vorticity(vortical_pe_state):
    """PE: damp_v with nord_v=0 (del-2) reduces mean|ζ| vs baseline.
    Uses nord_v=0 for the same reason as iter-175 NH: del-6 produces
    too small a reduction at C8 over 5 steps to discriminate from
    FP noise."""
    grid, cdgrid, coord, state = vortical_pe_state

    cfg_baseline = CDGridPrimitiveEquationConfig(
        # other damping off — isolate damp_v
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_active = CDGridPrimitiveEquationConfig(
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        damp_v=0.030, nord_v=0,    # del-2 for visible C8 effect
        use_conservation_fixer=False, fix_mass=False,
    )

    m_baseline = CDGridPrimitiveEquationModel(grid, coord, cfg_baseline)
    m_active = CDGridPrimitiveEquationModel(grid, coord, cfg_active)

    s_baseline = _step5(m_baseline, state)
    s_active = _step5(m_active, state)

    zeta_baseline = _mean_abs_vorticity_pe(s_baseline, cdgrid)
    zeta_active = _mean_abs_vorticity_pe(s_active, cdgrid)

    assert zeta_active < zeta_baseline, (
        f"PE damp_v=0.030 + nord_v=0 must REDUCE mean|ζ|: "
        f"baseline={zeta_baseline:.4e}, active={zeta_active:.4e} "
        f"(damp_v AMPLIFIED ζ — sign error in iter-12 PE wiring)."
    )
    rel_reduction = (zeta_baseline - zeta_active) / zeta_baseline
    assert rel_reduction > 0.001, (
        f"PE damp_v must reduce mean|ζ| by at least 0.1 % "
        f"(observed {100*rel_reduction:.3f} %).  Direction is the "
        f"key check; magnitude is small at C8 over 5 steps."
    )
