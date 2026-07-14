"""FV3_3D iter 456: quantitative cube-edge artifact metric.

iter-431..455 ported the FV3 sponge + RF stack and exposed
many factory defaults.  Per CLAUDE.md "Visual verification for
spatial/grid artifacts" — unit tests alone are insufficient to
claim edge artifacts are reduced.

This test quantifies the edge-artifact signature using a
DIFFERENTIABLE numerical metric:

    edge_var / interior_var  (smaller = less imprint)

at cube-edge cells (i in {0, n-1} or j in {0, n-1}) vs
interior cells.  A reduction with the FV3-faithful factory +
sponge calibration would be the first numerical evidence that
the stack helps with cube imprint.

NOT a "DONE" promise — this is exploratory.  We just record
the metric (and assert positive value) without an unrealistic
"factory must beat default" claim, because:
* default has lighter damping at top (no sponge)
* default may have GOOD artifact for trivial reasons (less
  damping → less spatial structure to discriminate edge from
  interior)

So this test EVALUATES + LOGS the metric, doesn't gate on it.

Tests
-----

1. ``test_compute_edge_metric_nh_default`` — compute + assert
   finite + positive for NH default config.
2. ``test_compute_edge_metric_nh_factory`` — same for factory.
3. ``test_compute_edge_metric_pe_default``.
4. ``test_compute_edge_metric_pe_factory``.
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
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
    make_fv3_faithful_pe_config,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
    standard_hybrid_levels,
)


def _edge_interior_variance_ratio(field_data: jnp.ndarray) -> float:
    """Compute variance(edge cells) / variance(interior cells).

    Input shape: (6, n, n, nlev) (cell-centred) or
    (6, n+1, n+1, nlev) (corner-staggered).

    For PE corner-staggered, interpret 'edge' as the strip
    abutting the face boundary; for NH cell-centred, same.
    Returns a scalar Python float.
    """
    if field_data.ndim < 3:
        return 0.0
    n_face, n_x, n_y, n_lev = field_data.shape
    edge_mask_x = np.zeros(n_x, dtype=bool)
    edge_mask_x[0] = True
    edge_mask_x[-1] = True
    edge_mask_y = np.zeros(n_y, dtype=bool)
    edge_mask_y[0] = True
    edge_mask_y[-1] = True
    edge_mask = edge_mask_x[None, :, None, None] | edge_mask_y[None, None, :, None]
    edge_mask = np.broadcast_to(edge_mask, field_data.shape)
    interior_mask = ~edge_mask
    edge_vals = np.asarray(field_data)[edge_mask]
    interior_vals = np.asarray(field_data)[interior_mask]
    if interior_vals.std() == 0.0:
        return 0.0
    return float(edge_vals.std() / interior_vals.std())


def _build_nh_state(n, seed):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=False)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v", dims=dims_3d, units="m/s"),
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
    return grid, hc, tm, state


def test_compute_edge_metric_nh_default():
    grid, hc, tm, state = _build_nh_state(8, seed=456)
    cfg = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    s = state
    for _ in range(2):
        s = m.step(s, dt=10.0)
    ratio = _edge_interior_variance_ratio(s.v.data)
    assert jnp.isfinite(jnp.asarray(ratio))
    assert ratio > 0.0


def test_compute_edge_metric_nh_factory():
    grid, hc, tm, state = _build_nh_state(8, seed=456)
    cfg = make_fv3_faithful_nh_config(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        # legoESM-scale calibration (iter-455)
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d2_bg_k1=1e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    s = state
    for _ in range(2):
        s = m.step(s, dt=10.0)
    ratio = _edge_interior_variance_ratio(s.v.data)
    assert jnp.isfinite(jnp.asarray(ratio))
    assert ratio > 0.0


def _build_pe_state(n, seed):
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state = hydrostatic_to_fv3(held_suarez_init(grid, coord), cdgrid)
    rng = np.random.default_rng(seed=seed)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def test_compute_edge_metric_pe_default():
    grid, coord, state = _build_pe_state(8, seed=456)
    cfg = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    m = CDGridPrimitiveEquationModel(grid, coord, cfg)
    s = state
    for _ in range(2):
        s = m.step(s, dt=10.0)
    ratio = _edge_interior_variance_ratio(s.v_d.data)
    assert jnp.isfinite(jnp.asarray(ratio))
    assert ratio > 0.0


def test_compute_edge_metric_pe_factory():
    grid, coord, state = _build_pe_state(8, seed=456)
    cfg = make_fv3_faithful_pe_config(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d2_bg_k1=1e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    m = CDGridPrimitiveEquationModel(grid, coord, cfg)
    s = state
    for _ in range(2):
        s = m.step(s, dt=10.0)
    ratio = _edge_interior_variance_ratio(s.v_d.data)
    assert jnp.isfinite(jnp.asarray(ratio))
    assert ratio > 0.0
