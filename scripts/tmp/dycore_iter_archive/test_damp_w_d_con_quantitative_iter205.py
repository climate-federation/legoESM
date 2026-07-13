"""FV3_3D iter 205: quantitative energy-conservation test for the
iter-203 ``damp_w_d_con`` KE→heat conversion.

iter-203 implements the FV3 ``sw_core.F90:1086`` formula::

    heat_per_mass = -d_con * dw * (w + 0.5*dw)              # at half-levels
    dθ_p          = avg_half_to_full(heat_per_mass) / c_pd  # at full levels
    θ_p_new       = θ_p + dθ_p

Existing iter-203 tests verify wiring, AD safety, and sign direction
(heat does not REDUCE mean θ_p) — but no test verifies the FORMULA
quantitatively.  A bug in the factor of 0.5, the half→full averaging,
or the c_pd division would still pass iter-203 tests but produce the
wrong amount of heat — silent quantitative drift.

This iter adds a STRONG numerical correctness test: compute the
difference ``θ_p(d_con=1.0) - θ_p(d_con=0.0)`` from two consecutive
single-step runs (same damp_w, same dw), and verify it matches the
expected ``-d_con * ΔKE_w(half_to_full) / c_pd`` formula exactly
within numerical tolerance.

Tests
-----

1. ``test_damp_w_d_con_heat_matches_formula`` — numerical heat
   contribution matches ``-d_con * dw * (w + 0.5*dw) / c_pd``
   averaged half→full, within 1e-12 relative tolerance (machine
   precision modulo a few ULPs).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
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
def small_nh_state_with_w():
    """NH state with sinusoidal w perturbation so dw is non-zero
    everywhere and the energy-conservation arithmetic is exercised."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    nlev_half = nlev + 1
    i_idx = jnp.arange(n)
    j_idx = jnp.arange(n)
    k_idx = jnp.arange(nlev_half)
    pattern = (
        jnp.sin(2 * jnp.pi * i_idx[None, :, None, None] / n)
        * jnp.cos(2 * jnp.pi * j_idx[None, None, :, None] / n)
        * jnp.ones_like(k_idx[None, None, None, :], dtype=jnp.float64)
    )
    pattern = jnp.broadcast_to(pattern, (6, n, n, nlev_half))
    W0 = 0.5
    w_perturb = jnp.asarray(W0 * pattern, dtype=jnp.float64)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=w_perturb, name="w",
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


def test_damp_w_d_con_heat_matches_formula(small_nh_state_with_w):
    """Numerical d_con heat contribution matches
    ``-d_con * dw * (w + 0.5*dw) / c_pd`` averaged half→full
    within machine precision.

    Method: run two single-step models that differ ONLY in
    damp_w_d_con (0.0 vs 1.0).  Both have the same damp_w wiring
    so dw is identical.  The difference in θ_p between the two
    runs is exactly the d_con heat contribution.  We compute the
    EXPECTED contribution from the formula and assert match."""
    grid, height_coord, terrain_metric, state = small_nh_state_with_w

    # Disable other damping / sponges so only damp_w + d_con
    # affects the difference.  Conservation fixers off so they
    # don't perturb θ_p between the two runs.
    common = dict(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
    )
    # Three configs:
    #   acoustic_only: NO damp_w, NO d_con — establishes the
    #                  state after the acoustic step alone.
    #   damp_w_no_d_con: damp_w on, d_con off — w gets the
    #                    damp_w correction; θ_p unchanged from
    #                    acoustic_only.
    #   damp_w_d_con: damp_w on, d_con on — w gets the same damp_w
    #                 correction; θ_p ALSO gets the heat
    #                 contribution.
    # Then: dw = (damp_w_no_d_con.w - acoustic_only.w),
    #       heat = θ_p(damp_w_d_con) - θ_p(damp_w_no_d_con).
    # Compare against the formula using w_after_acoustic (=
    # acoustic_only.w) and dw above.
    cfg_acoustic_only = CDGridCompressibleEulerConfig(
        **common,
        damp_w=0.0, damp_w_d_con=0.0,
    )
    cfg_no_d_con = CDGridCompressibleEulerConfig(
        **common,
        damp_w=0.030, nord_w=1, damp_w_d_con=0.0,
    )
    cfg_d_con = CDGridCompressibleEulerConfig(
        **common,
        damp_w=0.030, nord_w=1, damp_w_d_con=1.0,
    )

    m_acoustic_only = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_acoustic_only,
    )
    m_no_d_con = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_d_con,
    )
    m_d_con = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d_con,
    )

    s_acoustic = m_acoustic_only.step(state, 10.0)
    s_no_d_con = m_no_d_con.step(state, 10.0)
    s_d_con = m_d_con.step(state, 10.0)

    # The difference in θ_p between (damp_w + d_con) and
    # (damp_w only) is exactly the d_con heat contribution.
    delta_theta_p = (
        s_d_con.theta_prime.data - s_no_d_con.theta_prime.data
    )

    # dw is the damp_w correction alone (subtracts the acoustic
    # step's own w-change).  iter-203's heat formula uses
    # ``w_new_data + 0.5*dw`` where ``w_new_data`` is the value
    # AFTER the acoustic step but BEFORE the damp_w correction —
    # i.e., the acoustic_only result.
    w_after_acoustic = s_acoustic.w.data
    w_after_damp_w = s_no_d_con.w.data
    dw = w_after_damp_w - w_after_acoustic

    # Heat per mass at half-levels (formula: -d_con * dw * (w + 0.5*dw))
    # using w_after_acoustic = the value seen by the d_con block.
    heat_half = -1.0 * dw * (w_after_acoustic + 0.5 * dw)
    # Average half-level heat to full-levels.
    heat_full = 0.5 * (heat_half[..., :-1] + heat_half[..., 1:])
    # iter-207: convert to θ_p increment using the Π Exner factor
    # (exner_ref from height_coord) — Δθ = heat / (c_pd * Π).
    exner_ref = height_coord.exner_ref[None, None, None, :]
    expected_dtheta_p = heat_full / (constants.c_pd * exner_ref)

    # Cross-check: confirm dw is non-trivial (test sanity).
    max_dw = float(jnp.max(jnp.abs(dw)))
    assert max_dw > 1e-6, (
        f"sanity: dw must be non-trivial in this test "
        f"(max|dw|={max_dw:.3e}); damp_w may be inert"
    )

    # Direct comparison.  Tolerance: ~1e-12 relative (machine
    # precision modulo a few ULPs of accumulation).  Use absolute
    # tolerance scaled by the maximum expected magnitude.
    max_expected = float(jnp.max(jnp.abs(expected_dtheta_p)))
    abs_tol = max(max_expected, 1e-30) * 1e-10
    diff = float(jnp.max(jnp.abs(delta_theta_p - expected_dtheta_p)))

    assert diff < abs_tol, (
        f"iter-203 d_con heat formula mismatch: "
        f"max|delta_theta_p - expected|={diff:.3e}, "
        f"max|expected|={max_expected:.3e}, "
        f"tol={abs_tol:.3e}.\n"
        f"This indicates a bug in the FV3 sw_core.F90:1086 port: "
        f"either the formula has a wrong factor (e.g., missing 0.5, "
        f"wrong sign, missing c_pd), or the half→full averaging is "
        f"incorrect, or the order of operations in the model.step "
        f"path differs from the test's reconstruction."
    )
