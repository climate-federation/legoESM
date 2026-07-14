"""FV3_3D iter 200: direction-correctness test for the PE
``T_diss_coeff`` velocity-dependent Laplacian dissipation of T.

Existing iter-182 tests verify:

* default ``T_diss_coeff=0.0`` is bit-for-bit baseline,
* ``T_diss_coeff > 0`` with non-zero winds CHANGES T,
* AD safety at the rest state (iter-182 double-where fix).

But NO test verifies the *direction* of the T_diss effect.  T_diss
is a Laplacian DIFFUSION on T: ``dT/dt += T_diss_coeff * |v| * dx
* lap(T)``.  A sign-flipped wiring (``-= T_diss_coeff * ...``) would
still pass "changes-T-when-winds-nonzero" but would AMPLIFY T
variance instead of damping it.

This iter mirrors the iter-174/175/195/198/199 quantitative
direction pattern: initialise T with a sinusoidal perturbation
(small-scale spatial variation that the Laplacian damps strongly),
add nonzero winds (so |v| > 0 engages T_diss), run a few steps,
verify max|T_perturbation| with T_diss > 0 is LESS than baseline.

Tests
-----

1. ``test_T_diss_reduces_T_perturbation_amplitude`` — sinusoidal
   T pattern + uniform wind; T_diss=0.5 reduces ``max|T_pert|``
   after 3 steps vs T_diss=0 baseline.
2. ``test_T_diss_no_amplification_for_typical_range[0.1/0.3/0.5]``
   — T_diss in {0.1, 0.3, 0.5} (FV3_3D iter 182 docstring-cited
   typical range) should never AMPLIFY T variance.
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
def perturbed_T_pe_state():
    """PE state with a sinusoidal T perturbation overlaid on the HS
    init field, plus a uniform 5 m/s wind so |v| > 0 engages T_diss."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    # Add a sinusoidal small-scale T perturbation (small enough not
    # to disturb the HS jet, large enough to be visible).
    i_idx = jnp.arange(n)
    j_idx = jnp.arange(n)
    pattern = (
        jnp.sin(4 * jnp.pi * i_idx[None, :, None, None] / n)
        * jnp.cos(4 * jnp.pi * j_idx[None, None, :, None] / n)
        * jnp.ones((1, 1, 1, nlev), dtype=jnp.float64)
    )
    pattern = jnp.broadcast_to(pattern, (6, n, n, nlev))
    T_amp = 5.0
    T_perturb = T_amp * pattern
    T_data = state.T.data + T_perturb

    # Uniform wind at 5 m/s so |v| > 0 everywhere → T_diss engages.
    nlev_uv = state.u_d.data.shape[-1]
    u_uniform = 5.0 * jnp.ones((6, n + 1, n + 1, nlev_uv))
    v_uniform = jnp.zeros_like(u_uniform)

    state = state._replace(
        T=state.T.replace(data=T_data),
        u_d=state.u_d.replace(data=u_uniform),
        v_d=state.v_d.replace(data=v_uniform),
    )
    return grid, cdgrid, coord, state, T_perturb


def _step_n(model, state, n_steps, dt=200.0):
    s = state
    for _ in range(n_steps):
        s = model.step(s, dt)
    return s


def test_T_diss_reduces_T_perturbation_amplitude(perturbed_T_pe_state):
    """T_diss > 0 reduces the amplitude of a sinusoidal T
    perturbation more than the no-T_diss baseline."""
    grid, cdgrid, coord, state, T_perturb_initial = perturbed_T_pe_state

    cfg_baseline = CDGridPrimitiveEquationConfig(
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_active = CDGridPrimitiveEquationConfig(
        T_diss_coeff=0.5,
        use_conservation_fixer=False, fix_mass=False,
    )

    m_base = CDGridPrimitiveEquationModel(grid, coord, cfg_baseline)
    m_active = CDGridPrimitiveEquationModel(grid, coord, cfg_active)

    s_base = _step_n(m_base, state, n_steps=3)
    s_active = _step_n(m_active, state, n_steps=3)

    # Measure perturbation amplitude as the std around the level-mean
    # T (so the HS background drift doesn't contaminate the metric).
    def _T_pert_std(st):
        T_data = np.asarray(st.T.data, dtype=np.float64)
        T_mean = T_data.mean(axis=(0, 1, 2), keepdims=True)
        return float(np.std(T_data - T_mean))

    std_base = _T_pert_std(s_base)
    std_active = _T_pert_std(s_active)

    assert std_active < std_base, (
        f"T_diss=0.5 must REDUCE T perturbation std relative to "
        f"baseline: baseline_std={std_base:.4e}, "
        f"active_std={std_active:.4e} (T_diss AMPLIFIED variance — "
        f"sign error in the iter-182 wiring or the formula has "
        f"flipped signs)."
    )


@pytest.mark.parametrize("T_diss_coeff", [0.1, 0.3, 0.5])
def test_T_diss_no_amplification_for_typical_range(
    perturbed_T_pe_state, T_diss_coeff,
):
    """Across T_diss_coeff in {0.1, 0.3, 0.5} (iter-182 docstring
    typical range), none should AMPLIFY T perturbation std beyond
    the no-T_diss baseline."""
    grid, cdgrid, coord, state, _ = perturbed_T_pe_state

    cfg_baseline = CDGridPrimitiveEquationConfig(
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_active = CDGridPrimitiveEquationConfig(
        T_diss_coeff=T_diss_coeff,
        use_conservation_fixer=False, fix_mass=False,
    )

    m_base = CDGridPrimitiveEquationModel(grid, coord, cfg_baseline)
    m_active = CDGridPrimitiveEquationModel(grid, coord, cfg_active)

    s_base = _step_n(m_base, state, n_steps=3)
    s_active = _step_n(m_active, state, n_steps=3)

    def _T_pert_std(st):
        T_data = np.asarray(st.T.data, dtype=np.float64)
        T_mean = T_data.mean(axis=(0, 1, 2), keepdims=True)
        return float(np.std(T_data - T_mean))

    std_base = _T_pert_std(s_base)
    std_active = _T_pert_std(s_active)

    assert std_active <= std_base * 1.001, (
        f"T_diss={T_diss_coeff}: must not AMPLIFY T perturbation "
        f"std: baseline={std_base:.4e}, active={std_active:.4e}.  "
        f"Sign error in the T_diss wiring at this coefficient."
    )
