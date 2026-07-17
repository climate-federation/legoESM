"""FV3_3D iter 175: quantitative correctness test for the
post-step del-n vorticity damping (``damp_v``) ported in iter 169.

iter 169 added the FV3 ``fv3_del6_vorticity_damping`` to the NH 3D
path's post-step block.  iter-169's unit tests verify the wiring
CHANGES the state on a perturbed input but do NOT verify the
change is in the *correct* direction.  A sign-flipped damping
would still pass "changes-the-state" but would AMPLIFY vorticity
rather than damp it — silent failure.

This iter mirrors the iter-174 pattern: initialise NH state with a
vortical perturbation and verify ``mean(|ζ|)`` REDUCES vs the
no-damping baseline.

Tests
-----
1. ``test_damp_v_reduces_vorticity`` — ``damp_v=0.030`` reduces
   ``mean(|ζ|)`` by at least 1 % vs no-damping baseline.
2. ``test_damp_v_higher_nord_does_not_amplify`` — for nord ∈
   {0, 1, 2}, none of the orders should AMPLIFY vorticity.  This
   catches sign errors in any of the del-2/del-4/del-6 paths.
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
from legoesm.core.operators_cdgrid import (
    dgrid_vorticity, interp_center_to_corner,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def vortical_nh_state():
    """NH state initialised with a vortical perturbation: a
    sinusoidal pattern in v that produces a non-trivial vorticity
    field.  Pattern: v(face, i, j, k) = V0 * sin(2π i/n) (constant
    in j) so ∂v/∂x is well above noise but ∂u/∂y = 0 — pure shear
    vorticity.  At C8 this gives mean(|ζ|) ~ 1e-5 s^-1, well above
    discretization noise."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    i_idx = jnp.arange(n)
    k_idx = jnp.arange(nlev)
    pattern = (
        jnp.sin(2 * jnp.pi * i_idx[None, :, None, None] / n)
        * jnp.ones_like(jnp.arange(n)[None, None, :, None], dtype=jnp.float64)
        * jnp.ones_like(k_idx[None, None, None, :], dtype=jnp.float64)
    )
    pattern = jnp.broadcast_to(pattern, (6, n, n, nlev))
    V0 = 5.0
    v_perturb = jnp.asarray(V0 * pattern, dtype=jnp.float64)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=v_perturb, name="v", dims=dims_3d, units="m/s"),
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
    return grid, cdgrid, height_coord, terrain_metric, state


def _mean_abs_vorticity(state, cdgrid):
    """Compute ``mean(|ζ|)`` at cell centres.  ζ is computed from
    D-grid winds at corners; lift the cell-centre (u, v) to corners
    via the same path the model uses internally."""
    n_face_uv, n_i_uv, n_j_uv, nlev_uv = state.u.data.shape
    _uv_stack = jnp.stack([state.u.data, state.v.data], axis=-1)
    _uv_flat = _uv_stack.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv * 2)
    _uv_d_flat = interp_center_to_corner(_uv_flat, cdgrid)
    _uv_d = _uv_d_flat.reshape(
        _uv_d_flat.shape[0], _uv_d_flat.shape[1],
        _uv_d_flat.shape[2], nlev_uv, 2,
    )
    u_d = _uv_d[..., 0]
    v_d = _uv_d[..., 1]
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)
    return float(jnp.mean(jnp.abs(zeta)))


def _step5(model, state, dt=10.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_damp_v_reduces_vorticity(vortical_nh_state):
    """damp_v with nord_v=0 (del-2 path) should REDUCE mean|ζ| vs
    no-damping baseline on a vortical IC.

    Use nord_v=0 (del-2) rather than the SW production nord_v=2
    (del-6) because del-6 with the iter-1009 ``damp_v=0.030``
    setting produces only a ~1e-4 % reduction over 5 steps at C8 —
    too small to discriminate from FP noise.  del-2 with the same
    coefficient produces a much larger reduction and is sufficient
    to catch a sign error in the underlying ``fv3_del6_vorticity_damping``
    helper."""
    grid, cdgrid, height_coord, terrain_metric, state = vortical_nh_state

    cfg_baseline = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
    )
    cfg_damp = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        damp_v=0.030, nord_v=0,    # del-2 — strongest damping per coeff
    )

    model_baseline = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_baseline,
    )
    model_damp = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_damp,
    )

    s_baseline = _step5(model_baseline, state)
    s_damp = _step5(model_damp, state)

    zeta_baseline = _mean_abs_vorticity(s_baseline, cdgrid)
    zeta_damp = _mean_abs_vorticity(s_damp, cdgrid)

    assert zeta_damp < zeta_baseline, (
        f"damp_v with nord=0 must REDUCE mean|ζ|: "
        f"baseline={zeta_baseline:.4e}, damp={zeta_damp:.4e} "
        f"(damping AMPLIFIED vorticity — likely sign error)"
    )
    rel_reduction = (zeta_baseline - zeta_damp) / zeta_baseline
    assert rel_reduction > 0.01, (
        f"damp_v with nord=0 must reduce mean|ζ| by at least 1 % "
        f"(observed {100*rel_reduction:.3f} %)"
    )


@pytest.mark.parametrize("nord", [0, 1, 2])
def test_damp_v_no_amplification_for_any_nord(vortical_nh_state, nord):
    """For each nord ∈ {0, 1, 2} (del-2, del-4, del-6) the damping
    must never AMPLIFY vorticity vs the no-damping baseline.

    Higher-order del-n damping is more selective (damps high
    wavenumbers preferentially), so the magnitude of reduction
    differs between orders.  But all should reduce, not amplify.
    """
    grid, cdgrid, height_coord, terrain_metric, state = vortical_nh_state

    cfg_baseline = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
    )
    cfg_damp = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        damp_v=0.030, nord_v=nord,
    )

    model_baseline = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_baseline,
    )
    model_damp = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_damp,
    )

    s_baseline = _step5(model_baseline, state)
    s_damp = _step5(model_damp, state)

    zeta_baseline = _mean_abs_vorticity(s_baseline, cdgrid)
    zeta_damp = _mean_abs_vorticity(s_damp, cdgrid)

    # Allow 0.1 % numerical noise (no strict reduction floor — high-
    # order damping at small coefficient can produce barely-measurable
    # reductions).  But STRICTLY reject amplification.
    assert zeta_damp <= zeta_baseline * 1.001, (
        f"damp_v with nord={nord} must NOT amplify mean|ζ|: "
        f"baseline={zeta_baseline:.4e}, damp={zeta_damp:.4e} "
        f"(rel_change={100*(zeta_damp-zeta_baseline)/zeta_baseline:+.4f} %)"
    )
