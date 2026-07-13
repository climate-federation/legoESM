"""FV3_3D iter 185: PE-counterpart umbrella AD-at-rest regression
for the hydrostatic primitive-equation 3D path.

Mirror of iter-184's NH umbrella.  iter 181 fixed the Smagorinsky
helper sqrt(0); iter 182 fixed the PE T_diss wind_speed sqrt(0).
Each had a focused per-helper test, but no umbrella exercises ALL
PE damping knobs simultaneously at rest state.

This iter adds the PE umbrella: ``jax.grad`` through 3 PE steps
with all iter-12/14/16/18/57-58/iter-181-182-fixed mechanisms ON
at rest state.  Catches any future PE-path AD hazard.

Tests
-----
1. ``test_full_pe_toolkit_grad_at_rest`` — every PE damping knob
   ON simultaneously at exactly the rest state, ``jax.grad``
   w.r.t. T gives finite gradients.  Exercises:
   - iter 5/16/18 corner-divergence damping
   - iter 12 post-step vorticity damping (damp_v)
   - iter 14 4th-order ζ corner interp
   - iter 57-58 Smagorinsky-adaptive A_h (relies on iter-181 fix)
   - iter 5 cell-centre adaptive Smag div_damp (dddmp > 0)
   - velocity-dependent T_diss (relies on iter-182 fix)
   - hyperdiff_coeff (4th-order biharmonic)
2. ``test_full_pe_toolkit_grad_at_perturbed`` — sanity that
   perturbed state also works (rest is the challenging case).
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
    """Small PE state at the EXACT rest state.  Same construction
    as iter-182 fixture: HS init then force u_d=v_d=0."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )
    return grid, cdgrid, coord, state


def _full_pe_toolkit_cfg():
    """All PE damping knobs ON at modest values that engage every
    AD-critical path without producing instability over a 3-step
    integration."""
    return CDGridPrimitiveEquationConfig(
        # Static Laplacian + Smagorinsky-adaptive A_h (iter-181 fix)
        A_h=1e6,
        smagorinsky_cs=0.20,
        # Hyperdiffusion (4th-order biharmonic)
        hyperdiff_coeff=1e14,
        # Cell-centre divergence damping (iter 5) with adaptive Smag
        div_damp_coeff=1e6,
        # 4th-order ζ corner (iter 14)
        use_fv3_a2b_zeta_corner=True,
        # Post-step vorticity damping (iter 12)
        damp_v=0.030, nord_v=2,
        # Corner-divergence damping del-2 + del-(2*(nord+1)) (iter 16/18)
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        # iter-187 smag_vort cap + iter-190 dedup with iter-170
        # (extended in iter-192 to engage these in the umbrella).
        # d4_bg + nord > 0 turns on the FV3 ``smag_vort = |dt|*sqrt(
        # delpc² + ζ²)`` cap; combined with use_fv3_a2b_zeta_corner=
        # True (set above), iter-190 dedup is exercised end-to-end
        # at AD-at-rest.
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        # iter-208 KE→heat conversion for damp_v (added to umbrella
        # in iter-210).  FV3 production default ``d_con = 1.0``.
        damp_v_d_con=1.0,
        # iter-218/219 sponge-aware delt_max cap on dissipative
        # heating (added to umbrella in iter-220).  FV3 production
        # default 1.0 K/s; engages PE-specific top-2 sponge skip
        # (k=0,1 uncapped) and interior k≥2 dt*delt_max cap.
        # Verifies AD safety through jnp.clip with the per-level
        # cap-mask.
        delt_max=1.0,
        # iter-221 corner-div damp d_con (added to umbrella in
        # iter-227).  FV3 production default 1.0.  Adds heat
        # tendency to dT_dt at cell centres.
        corner_div_damp_d_con=1.0,
        # iter-223 cell-centre div_damp d_con (added in iter-227).
        div_damp_d_con=1.0,
        # iter-225 Smagorinsky-A_h d_con (added in iter-227).
        # Closes the LAST PE d_con asymmetry — every PE KE-removing
        # mechanism (corner-div, cell-centre div_damp, damp_v, A_h)
        # now contributes to dT_dt simultaneously.
        ah_d_con=1.0,
        # Velocity-dependent T_diss (iter-182 fix)
        T_diss_coeff=0.05,
        # Disable conservation fixers to keep AD focused on the
        # tendency / damping paths (the fixers have their own
        # integration tests).
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )


def test_full_pe_toolkit_grad_at_rest(small_pe_state_rest):
    """``jax.grad`` through 3 PE steps with the FULL toolkit ON
    at rest state.  Differentiating w.r.t. T avoids perturbing
    winds away from zero, exercising the iter-181/182 sqrt-at-zero
    fixes.

    A NaN here would indicate a new AD hazard in the PE path
    beyond what iter-181/182 fixed.
    """
    grid, cdgrid, coord, state = small_pe_state_rest

    cfg = _full_pe_toolkit_cfg()
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss_fn(T_data):
        s = state._replace(T=state.T.replace(data=T_data))
        for _ in range(3):
            s = model.step(s, 200.0)
        return jnp.mean(s.T.data ** 2)

    grad = jax.grad(loss_fn)(state.T.data)
    assert jnp.all(jnp.isfinite(grad)), (
        "AD through full PE FV3 toolkit at rest state must be "
        "finite.  NaN here indicates a new AD hazard has entered "
        "the PE tendency function beyond the iter-181/182 fixes. "
        "Check any new helper added to primitive_eq_cdgrid.py for "
        "unprotected sqrt(0) / log(0) / 1/0 patterns."
    )


def test_full_pe_toolkit_grad_at_perturbed(small_pe_state_rest):
    """Sanity: full toolkit ON at perturbed state also works.
    Verifies the rest-state path is the challenging case."""
    grid, cdgrid, coord, state = small_pe_state_rest

    n = grid.n
    nlev = state.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=185)
    u_p = rng.uniform(-0.5, 0.5, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-0.5, 0.5, size=(6, n + 1, n + 1, nlev))
    state_perturbed = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg = _full_pe_toolkit_cfg()
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss_fn(T_data):
        s = state_perturbed._replace(T=state_perturbed.T.replace(data=T_data))
        for _ in range(3):
            s = model.step(s, 200.0)
        return jnp.mean(s.T.data ** 2)

    grad = jax.grad(loss_fn)(state_perturbed.T.data)
    assert jnp.all(jnp.isfinite(grad))
