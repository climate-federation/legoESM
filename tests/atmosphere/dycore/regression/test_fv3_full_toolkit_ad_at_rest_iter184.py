"""FV3_3D iter 184: comprehensive AD-safety regression for the
full FV3 NH toolkit at rest state.

iter 181/182/183 fixed three sqrt-at-zero gradient hazards
(Smagorinsky helper, PE T_diss wind_speed, SW d_sw5 smag_vort).
Each iter added a focused test for its specific helper.  This
iter adds a comprehensive umbrella test: ``jax.grad`` through 5
NH steps with ALL FV3 toolkit knobs ON simultaneously, starting
from EXACTLY the rest state.

This catches future AD hazards that appear in the NH AD-critical
path.  Any new helper that introduces a sqrt-at-zero, log-of-zero,
divide-by-zero, or similar pattern would be caught when it lands
in the NH path.

The test is structured as ONE comprehensive integration test (not
per-mechanism) so the regression mode is "any combination of NH
toolkit + rest-state AD breaks".

Tests
-----
1. ``test_full_nh_toolkit_grad_at_rest`` — every iter-168/169/170/
   171/180 knob ON simultaneously at rest state, ``jax.grad``
   through 5 steps gives finite gradients.
2. ``test_full_nh_toolkit_grad_at_perturbed`` — same with a small
   perturbation, sanity that the rest-state path is the
   challenging case (perturbed should also work).
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
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def small_nh_state():
    """Standard NH fixture used across iter-168...-181 tests."""
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


def _full_toolkit_cfg():
    """All NH FV3 toolkit knobs ON at PE-tested-or-default values."""
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        # iter-168 corner-divergence damping (del-2)
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        # iter-187 nord >= 1 smag_vort branch + iter-190 dedup
        # (extended in iter-192 to engage these in the umbrella).
        # d4_bg + nord > 0 turns on the smag_vort cap formula, and
        # combined with use_fv3_a2b_zeta_corner=True it exercises
        # the iter-190 dedup'd ``_zeta_a2b_ord4`` shared between
        # the iter-170 ζ_corner site and the iter-187 site.
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        # iter-169 post-step vorticity damping
        damp_v=0.030, nord_v=2,
        # iter-170 4th-order ζ corner
        use_fv3_a2b_zeta_corner=True,
        # iter-171 cell-centre adaptive Smagorinsky div damping
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        # iter-180 Smagorinsky-adaptive A_h (relies on iter-181 fix)
        A_h=1e6, smagorinsky_cs=0.20,
        # iter-193 post-step damp_w + nord_w (FV3 d_sw1 port,
        # added to the umbrella in iter-194).  Reuses the
        # _del6_vt_flux backbone applied per half-level via vmap.
        damp_w=0.030, nord_w=1,
        # iter-203 KE→heat conversion for damp_w (added to umbrella
        # in iter-204).  FV3 production default ``d_con = 1.0``.
        damp_w_d_con=1.0,
        # iter-209 KE→heat conversion for damp_v (NH mirror of PE
        # iter-208).  Added to umbrella in iter-210.
        damp_v_d_con=1.0,
        # iter-218/219 sponge-aware delt_max cap on dissipative
        # heating (added to umbrella in iter-220).  FV3 production
        # default 1.0 K/s; engages the per-level sponge_factor path
        # (k=0 → 0.1×, k=1 → 0.5×, k≥2 → 1×) and verifies AD
        # safety through jnp.clip with sponge masking.
        delt_max=1.0,
        # iter-222 corner-div damp d_con (added to umbrella in
        # iter-227).  FV3 production default 1.0.  Adds heat
        # tendency to dθ_p_dt via the iter-207 Π_ref refinement.
        corner_div_damp_d_con=1.0,
        # iter-224 cell-centre div_damp d_con (added in iter-227).
        div_damp_d_con=1.0,
        # iter-226 Smagorinsky-A_h d_con (added in iter-227).
        # Closes the LAST d_con asymmetry — every NH KE-removing
        # mechanism (corner-div, cell-centre div_damp, damp_v,
        # damp_w, A_h) now contributes to dθ_p_dt simultaneously.
        ah_d_con=1.0,
    )


def test_full_nh_toolkit_grad_at_rest(small_nh_state):
    """``jax.grad`` through 5 NH steps with ALL FV3 toolkit knobs
    ON simultaneously, starting from EXACTLY the rest state.

    This is the umbrella regression: any new AD hazard introduced
    by a future helper would be caught here.  Differentiates
    w.r.t. theta_prime to avoid perturbing winds away from zero
    (where the sqrt-at-zero hazards live)."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = _full_toolkit_cfg()
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
        "AD through full NH FV3 toolkit at rest state must be "
        "finite.  NaN here indicates a new AD hazard has entered "
        "the toolkit beyond the iter-181/182/183 sqrt-at-zero "
        "fixes.  Check any new helpers added to the NH tendency "
        "function for unprotected sqrt(0) / log(0) / 1/0 patterns."
    )


def test_full_nh_toolkit_grad_at_perturbed(small_nh_state):
    """Sanity: same toolkit with a small u perturbation also works.
    Verifies the rest-state path is the challenging case and the
    perturbed path is a regular regression — both should pass."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=184)
    u_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev))
    state_perturbed = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
    )

    cfg = _full_toolkit_cfg()
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(theta_p_data):
        s = state_perturbed._replace(
            theta_prime=state_perturbed.theta_prime.replace(
                data=theta_p_data
            ),
        )
        for _ in range(5):
            s = model.step(s, 10.0)
        return jnp.mean(s.u.data ** 2 + s.v.data ** 2)

    grad = jax.grad(loss_fn)(state_perturbed.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad))
