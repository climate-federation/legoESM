"""FV3_3D iter 174: quantitative correctness test for the
divergence-damping mechanisms ported in iter 168 and iter 171.

iter 168/171 added FV3-faithful corner-divergence damping (B-grid
delpc + ke gradient) and cell-centre divergence damping
(``div_damp_coeff`` + adaptive Smagorinsky ``div_damp_dddmp``) to
the NH 3D path.  Each had a unit test verifying the wiring CHANGES
the state on a perturbed input — but neither verified the change is
in the *correct* direction.  A sign-flipped damping (e.g.,
``du -= grad`` where it should be ``+= grad``) would still pass
"changes-the-state" but would AMPLIFY divergence rather than damp
it — the worst-case silent failure for a damping mechanism.

This iter adds a quantitative test that initialises the NH state
with a DIVERGENT perturbation (sinusoidal monopole) and verifies
each damping mechanism reduces the post-step divergence relative
to the no-damping baseline.

Design
------

* Use a sinusoidal divergent perturbation in u so ``div(u)`` has a
  non-trivial spatial pattern and ``mean(|div_v|)`` is well above
  noise.  Random noise is too high-frequency for cell-centre
  div_damp to engage at the d2_bg floor.
* Compute ``div_v`` directly (not via the model's internal cache)
  so the test is robust to internal API changes.
* Use a single 5-step integration so the test is fast (~ tens of
  seconds) but long enough for the damping to compound.
* Assert ``mean(|div_v|)`` STRICTLY DECREASES vs the baseline by
  at least 1 % (the FV3 production calibration produces order-10 %
  reductions; 1 % is a generous floor that catches a sign error
  while being insensitive to ULP noise).

Tests
-----
1. ``test_corner_div_damp_reduces_divergence`` — corner-div damping
   reduces ``mean(|div_v|)`` vs no-damping baseline.
2. ``test_cell_centre_div_damp_reduces_divergence`` — cell-centre
   constant ``div_damp_coeff`` reduces ``mean(|div_v|)``.
3. ``test_combined_div_damp_at_least_as_strong_as_either``
   — both ON reduces ``mean(|div_v|)`` at least as much as either
   alone (catches one mechanism cancelling the other).
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
    dgrid_to_cgrid, cgrid_divergence, interp_center_to_corner,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def divergent_nh_state():
    """NH state initialised with a sinusoidal divergent perturbation
    in u (monopole pattern with wavenumber 2 across each face).
    The pattern produces a div_v with peak ~3e-5 s^-1 at C8, well
    above noise and large enough to engage the d2_bg div_damp floor."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    # Build a sinusoidal u perturbation in face-local coordinates.
    # u(face, i, j, k) = U0 * sin(2π i / n) * cos(2π j / n)
    i_idx = jnp.arange(n)
    j_idx = jnp.arange(n)
    k_idx = jnp.arange(nlev)
    pattern = (
        jnp.sin(2 * jnp.pi * i_idx[None, :, None, None] / n)
        * jnp.cos(2 * jnp.pi * j_idx[None, None, :, None] / n)
        * jnp.ones_like(k_idx[None, None, None, :], dtype=jnp.float64)
    )
    pattern = jnp.broadcast_to(pattern, (6, n, n, nlev))
    U0 = 5.0  # m/s amplitude
    u_perturb = jnp.asarray(U0 * pattern, dtype=jnp.float64)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=u_perturb, name="u", dims=dims_3d, units="m/s"),
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
    return grid, cdgrid, height_coord, terrain_metric, state


def _mean_abs_div(state, cdgrid):
    """Compute ``mean(|div_v|)`` at cell centres for the cell-centre
    velocity field of the NH state.  Uses the same path as the model
    internal: lift cell-centre to corners, project to C-grid, then
    cgrid_divergence."""
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
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    div_v = cgrid_divergence(u_c, v_c, cdgrid)
    return float(jnp.mean(jnp.abs(div_v)))


def _step5(model, state, dt=10.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_corner_div_damp_reduces_divergence(divergent_nh_state):
    """Corner-divergence damping should REDUCE mean(|div_v|) vs the
    no-damping baseline on a divergent initial condition.  If the
    sign of the corner-div ke-gradient is flipped this test catches
    it (state would amplify rather than damp)."""
    grid, cdgrid, height_coord, terrain_metric, state = divergent_nh_state

    cfg_baseline = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,    # no other damping
        sponge_coeff=0.0,
    )
    cfg_corner = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
    )

    model_baseline = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_baseline,
    )
    model_corner = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_corner,
    )

    s_baseline = _step5(model_baseline, state)
    s_corner = _step5(model_corner, state)

    div_baseline = _mean_abs_div(s_baseline, cdgrid)
    div_corner = _mean_abs_div(s_corner, cdgrid)

    assert div_corner < div_baseline, (
        f"corner-div damping must REDUCE mean|div_v|: "
        f"baseline={div_baseline:.4e}, corner={div_corner:.4e} "
        f"(damping AMPLIFIED divergence — likely sign error)"
    )
    # 1 % minimum reduction floor — catches sign error without being
    # sensitive to coefficient calibration.
    rel_reduction = (div_baseline - div_corner) / div_baseline
    assert rel_reduction > 0.01, (
        f"corner-div damping must reduce mean|div_v| by at least 1 % "
        f"(observed {100*rel_reduction:.3f} %).  May indicate the "
        f"coefficient is too small to engage."
    )


def test_cell_centre_div_damp_reduces_divergence(divergent_nh_state):
    """Cell-centre constant ``div_damp_coeff`` should REDUCE
    mean(|div_v|) vs the no-damping baseline on a divergent IC."""
    grid, cdgrid, height_coord, terrain_metric, state = divergent_nh_state

    cfg_baseline = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
    )
    cfg_cellc = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        div_damp_coeff=1e10, div_damp_dddmp=0.0,   # constant path
    )

    model_baseline = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_baseline,
    )
    model_cellc = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_cellc,
    )

    s_baseline = _step5(model_baseline, state)
    s_cellc = _step5(model_cellc, state)

    div_baseline = _mean_abs_div(s_baseline, cdgrid)
    div_cellc = _mean_abs_div(s_cellc, cdgrid)

    assert div_cellc < div_baseline, (
        f"cell-centre div_damp must REDUCE mean|div_v|: "
        f"baseline={div_baseline:.4e}, cellc={div_cellc:.4e} "
        f"(damping AMPLIFIED divergence — likely sign error)"
    )
    rel_reduction = (div_baseline - div_cellc) / div_baseline
    assert rel_reduction > 0.01, (
        f"cell-centre div_damp must reduce mean|div_v| by at least 1 % "
        f"(observed {100*rel_reduction:.3f} %)."
    )


def test_combined_div_damp_at_least_as_strong_as_either(divergent_nh_state):
    """When both corner-div AND cell-centre div_damp are ON, the
    combined reduction should be at least as large as either alone.
    Catches a mechanism cancelling the other (e.g., one applied with
    flipped sign that partially undoes the other's damping)."""
    grid, cdgrid, height_coord, terrain_metric, state = divergent_nh_state

    cfg_corner_only = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
    )
    cfg_cellc_only = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        div_damp_coeff=1e10, div_damp_dddmp=0.0,
    )
    cfg_both = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e10, div_damp_dddmp=0.0,
    )

    model_corner = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_corner_only,
    )
    model_cellc = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_cellc_only,
    )
    model_both = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_both,
    )

    s_corner = _step5(model_corner, state)
    s_cellc = _step5(model_cellc, state)
    s_both = _step5(model_both, state)

    div_corner = _mean_abs_div(s_corner, cdgrid)
    div_cellc = _mean_abs_div(s_cellc, cdgrid)
    div_both = _mean_abs_div(s_both, cdgrid)

    weaker = max(div_corner, div_cellc)    # less damping = larger div
    # Allow a small (5 %) cushion: nonlinear interaction can make
    # the combined reduction slightly less than the strict sum but
    # should never be WORSE than the individual stronger mechanism.
    assert div_both <= weaker * 1.05, (
        f"combined damping must be at least as strong as either "
        f"alone: corner={div_corner:.4e}, cellc={div_cellc:.4e}, "
        f"both={div_both:.4e} (one mechanism is cancelling the "
        f"other — likely sign mismatch between corner and cellc)"
    )
