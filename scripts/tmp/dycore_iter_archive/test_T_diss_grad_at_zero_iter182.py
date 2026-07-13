"""FV3_3D iter 182: differentiability of velocity-dependent T
dissipation in the PE 3D path at rest state.

Mirror of the iter-181 fix for the Smagorinsky helper.
``primitive_eq_cdgrid.py`` line 1005 (post-iter-182 edit) had::

    wind_speed = jnp.sqrt(u_cell**2 + v_cell**2)

which has the same singular-gradient-at-zero problem as
iter 181's Smagorinsky sqrt: ``d sqrt(u²+v²) / d u`` at u=v=0
is undefined.  When ``T_diss_coeff > 0``, this propagates into
``dT_dt`` via ``nu_T = T_diss_coeff * wind_speed * dx``, breaking
``jax.grad`` through the PE rest state.

iter 182 applies the same JAX double-where trick to make the
forward pass bit-for-bit unchanged but the backward pass finite.

Tests
-----
1. ``test_pe_T_diss_differentiable_at_rest`` — model-level
   ``jax.grad`` through the PE path with ``T_diss_coeff > 0``,
   starting from EXACTLY the rest state.  Was NaN before iter 182.
2. ``test_pe_T_diss_off_baseline`` — ``T_diss_coeff = 0`` is
   bit-for-bit identical baseline (Python-static gate guard).
3. ``test_pe_T_diss_changes_T_when_winds_nonzero`` — sanity that
   ``T_diss_coeff > 0`` actually has an effect when winds are
   non-zero (the fix didn't accidentally make T_diss inert).
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
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def small_pe_state_rest():
    """Small PE state at the EXACT rest state (zero u_d, v_d).
    ``hydrostatic_to_fv3(held_suarez_init)`` initialises wind to
    ~0 (HS init has tiny perturbations); we then force exactly 0
    so the wind_speed gradient singularity is exposed."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    # Force exact rest state on the D-grid winds.
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )
    return grid, cdgrid, coord, state


def test_pe_T_diss_off_baseline(small_pe_state_rest):
    """T_diss_coeff=0.0 (default) is bit-for-bit identical to the
    field-unset baseline — Python-static gate guard."""
    grid, cdgrid, coord, state = small_pe_state_rest

    cfg_unset = CDGridPrimitiveEquationConfig()
    cfg_zero = CDGridPrimitiveEquationConfig(T_diss_coeff=0.0)

    model_unset = CDGridPrimitiveEquationModel(grid, coord, cfg_unset)
    model_zero = CDGridPrimitiveEquationModel(grid, coord, cfg_zero)

    s_unset = model_unset.step(state, 200.0)
    s_zero = model_zero.step(state, 200.0)

    np.testing.assert_array_equal(
        np.asarray(s_unset.T.data), np.asarray(s_zero.T.data),
    )


def test_pe_T_diss_changes_T_when_winds_nonzero(small_pe_state_rest):
    """T_diss_coeff > 0 actually has an effect on T when winds are
    non-zero.  Sanity: the iter-182 double-where fix didn't
    accidentally make T_diss inert."""
    grid, cdgrid, coord, state = small_pe_state_rest

    n = grid.n
    nlev = state.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=182)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s_perturb = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_off = CDGridPrimitiveEquationConfig(T_diss_coeff=0.0)
    cfg_on = CDGridPrimitiveEquationConfig(T_diss_coeff=0.05)

    model_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    model_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)

    s_off = model_off.step(s_perturb, 200.0)
    s_on = model_on.step(s_perturb, 200.0)

    diff = float(jnp.max(jnp.abs(s_off.T.data - s_on.T.data)))
    base = float(jnp.max(jnp.abs(s_off.T.data)))
    assert diff > 1e-10 * base, (
        f"T_diss_coeff > 0 must change T on perturbed state "
        f"(diff={diff:.3e}, base={base:.3e}).  If 0, the iter-182 "
        f"double-where fix accidentally disabled the T_diss term."
    )


def test_pe_T_diss_differentiable_at_rest(small_pe_state_rest):
    """``jax.grad`` flows through 3 PE steps with ``T_diss_coeff > 0``
    starting from EXACTLY the rest state.  Was NaN before iter 182
    due to the sqrt(0) singularity in
    ``wind_speed = sqrt(u_cell² + v_cell²)``.

    Mirrors the iter-181 ``test_nh_smag_differentiable_at_rest``
    pattern.  Differentiates with respect to T (a scalar that
    doesn't perturb winds away from zero) so the rest-state
    singularity is genuinely exercised by AD."""
    grid, cdgrid, coord, state = small_pe_state_rest

    cfg = CDGridPrimitiveEquationConfig(
        T_diss_coeff=0.05,
        # Disable other knobs that complicate AD; isolate the
        # T_diss path to verify the iter-182 fix specifically.
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss_fn(T_data):
        s = state._replace(T=state.T.replace(data=T_data))
        for _ in range(3):
            s = model.step(s, 200.0)
        return jnp.mean(s.T.data ** 2)

    grad = jax.grad(loss_fn)(state.T.data)
    assert jnp.all(jnp.isfinite(grad)), (
        "AD through PE with T_diss > 0 at rest state must be finite "
        "after iter 182 fix to wind_speed sqrt(0) singularity."
    )
