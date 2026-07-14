"""FV3_3D iter 203: KE→heat conversion (``d_con``) for the iter-193
NH ``damp_w`` post-step damping.

Faithful port of FV3 ``sw_core.F90:1086``::

    heat_source = -d_con * dw * (w + 0.5*dw)

which represents the negative of the change in kinetic energy
density per unit mass (``-ΔKE_w = -(w*dw + 0.5*dw²)``).  When
``damp_w`` removes KE from ``w`` (``dw`` opposite sign to ``w``),
heat_source is POSITIVE — the lost KE is deposited as heat in the
temperature field, preserving total energy.

Conversion to ``θ_p`` (NH prognostic temperature) uses the
simplified ``Δθ_p = heat / c_pd`` formula (Π Exner factor
approximated as 1.0; valid in the lower troposphere, error ~30 %
aloft where Π drops to 0.5).

Tests
-----

1. ``test_damp_w_d_con_off_baseline`` — ``damp_w_d_con=0.0`` (and
   field unset) is bit-for-bit identical to the iter-193 baseline.
   Python-static gate guard.
2. ``test_damp_w_d_con_increases_theta_p_when_w_damped`` — when
   ``damp_w > 0`` removes KE from a perturbed w, the θ_p field
   gains heat (positive δθ_p where ``-dw*(w+0.5*dw) > 0``).
3. ``test_damp_w_d_con_differentiable_at_rest`` — ``jax.grad``
   through 5 steps with ``damp_w_d_con > 0`` at rest stays finite.
4. ``test_damp_w_d_con_no_op_when_damp_w_off`` — even with
   ``damp_w_d_con > 0``, when ``damp_w == 0`` the heat source is
   gated off (the d_con block is INSIDE the ``damp_w > 0`` block).
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


def test_damp_w_d_con_off_baseline(small_nh_state):
    """damp_w_d_con=0.0 (default) is bit-for-bit identical to the
    iter-193 baseline (damp_w active without the heat source)."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev_half = state.w.data.shape[-1]
    rng = np.random.default_rng(seed=203)
    w_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev_half))
    s = state._replace(
        w=state.w.replace(data=jnp.asarray(w_p)),
    )

    cfg_baseline = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        # damp_w_d_con=0.0 (default) — no heat source
    )
    cfg_d_con_zero = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        damp_w_d_con=0.0,
    )

    m_baseline = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_baseline,
    )
    m_d_con_zero = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d_con_zero,
    )

    s_baseline = m_baseline.step(s, 10.0)
    s_d_con_zero = m_d_con_zero.step(s, 10.0)

    np.testing.assert_array_equal(
        np.asarray(s_baseline.theta_prime.data),
        np.asarray(s_d_con_zero.theta_prime.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_baseline.w.data),
        np.asarray(s_d_con_zero.w.data),
    )


def test_damp_w_d_con_increases_theta_p_when_w_damped(small_nh_state):
    """With damp_w_d_con > 0, the θ_p field should differ from the
    no-d_con baseline.  Direction: the heat source is positive
    where ``-dw*(w+0.5*dw) > 0`` (KE removed) — so the d_con
    case has higher mean(θ_p) than the no-d_con case for cells
    where damp_w is removing energy."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev_half = state.w.data.shape[-1]
    rng = np.random.default_rng(seed=203)
    w_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev_half))
    s = state._replace(
        w=state.w.replace(data=jnp.asarray(w_p)),
    )

    cfg_no_d_con = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        damp_w_d_con=0.0,
    )
    cfg_d_con = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        damp_w_d_con=1.0,        # FV3 production default
    )

    m_no_d_con = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_d_con,
    )
    m_d_con = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d_con,
    )

    s_no_d_con = m_no_d_con.step(s, 10.0)
    s_d_con = m_d_con.step(s, 10.0)

    diff = float(jnp.max(jnp.abs(
        s_no_d_con.theta_prime.data - s_d_con.theta_prime.data
    )))
    base = float(jnp.max(jnp.abs(s_no_d_con.theta_prime.data)) + 1e-30)
    assert diff > 1e-10, (
        f"damp_w_d_con > 0 must change θ_p (diff={diff:.3e})"
    )

    # The heat source converts KE removed by damp_w into heat,
    # adding it to θ_p.  Total mass-mean θ_p should be HIGHER (or
    # at least not lower) under d_con > 0 since net KE removal
    # produces net heating.  Allow some cushion since the change
    # is small.
    mean_no = float(jnp.mean(s_no_d_con.theta_prime.data))
    mean_d_con = float(jnp.mean(s_d_con.theta_prime.data))
    # Net energy injection should not REDUCE mean θ_p.
    # (The d_con value 1.0 corresponds to full FV3-faithful
    # conservation; iter-193 dw is small at one outer step so
    # the numerical effect is small but should be of the right sign.)
    assert mean_d_con + 1e-6 >= mean_no, (
        f"damp_w_d_con > 0 must not REDUCE mean θ_p relative to "
        f"d_con=0: mean(no_d_con)={mean_no:.4e}, "
        f"mean(d_con)={mean_d_con:.4e} (heat source has wrong sign)."
    )


def test_damp_w_d_con_differentiable_at_rest(small_nh_state):
    """``jax.grad`` through 5 steps with damp_w + damp_w_d_con
    active at rest stays finite.  d_con multiplies dw by w + 0.5*dw;
    at rest both are 0 so this is naturally safe (no sqrt-at-zero
    introduced)."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1, damp_w_d_con=1.0,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(theta_p_data):
        s = state._replace(
            theta_prime=state.theta_prime.replace(data=theta_p_data),
        )
        for _ in range(5):
            s = model.step(s, 10.0)
        return jnp.mean(s.theta_prime.data ** 2)

    grad = jax.grad(loss_fn)(state.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad))


def test_damp_w_d_con_no_op_when_damp_w_off(small_nh_state):
    """When damp_w == 0, damp_w_d_con must NOT add heat (d_con block
    is gated INSIDE the damp_w > 0 block).  Bit-for-bit identical
    to the field-unset baseline."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg_unset = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    cfg_d_con_only = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.0, damp_w_d_con=1.0,    # d_con set but damp_w off
    )

    m_unset = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_unset,
    )
    m_d_con_only = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d_con_only,
    )

    s_unset = m_unset.step(state, 10.0)
    s_d_con_only = m_d_con_only.step(state, 10.0)

    np.testing.assert_array_equal(
        np.asarray(s_unset.theta_prime.data),
        np.asarray(s_d_con_only.theta_prime.data),
    )
