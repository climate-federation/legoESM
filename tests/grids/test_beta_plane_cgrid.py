"""Tests for the Cartesian beta-plane C-grid geometry option.

Covers (1) geometry correctness — uniform Cartesian metrics and the MITgcm
``f = f0 + beta*y`` Coriolis convention at every stagger point — and (2) that the
lat-lon C-grid ocean model runs on it and the Coriolis force acts at the right
magnitude (an f-plane inertial response).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.latlon import (
    LatLonCGridGeometry,
    create_beta_plane_cgrid_geometry,
    ensure_geometry,
)

# MITgcm tutorial_barotropic_gyre parameters.
NY, NX = 8, 10
DX = 20.0e3
F0, BETA, Y0 = 1.0e-4, 1.0e-11, -20.0e3


def _geom(**kw):
    p = dict(dx_m=DX, f0=F0, beta=BETA, y_origin_m=Y0)
    p.update(kw)
    return create_beta_plane_cgrid_geometry(NY, NX, **p)


def test_returns_cgrid_geometry_passthrough():
    g = _geom()
    assert isinstance(g, LatLonCGridGeometry)
    assert ensure_geometry(g) is g  # idempotent through the model's converter


def test_metrics_are_uniform_cartesian():
    g = _geom(dx_m=DX, dy_m=DX)
    for arr in (g.dx_T, g.dy_T, g.dx_u, g.dy_u, g.dx_v, g.dy_v):
        np.testing.assert_allclose(np.asarray(arr), DX)
    np.testing.assert_allclose(np.asarray(g.area_T), DX * DX)
    np.testing.assert_allclose(np.asarray(g.area_q), DX * DX)
    # No cos(lat) convergence, no grid rotation.
    np.testing.assert_allclose(np.asarray(g.cos_lat), 1.0)
    np.testing.assert_allclose(np.asarray(g.cos_alpha_u), 1.0)
    np.testing.assert_allclose(np.asarray(g.sin_alpha_u), 0.0)


def test_pseudo_lat_is_zero_for_metric_self_consistency():
    """The pseudo-``lat`` MUST be 0 so that operators which recompute
    ``cos(grid.lat)`` for a metric (``divergence_cgrid`` v-face length and the
    flux-form momentum advection) agree with the ones that read the ``cos_lat=1``
    metric field (``gradient_x_cgrid``).  A non-zero
    ``y_c/radius`` pseudo-lat would leave a ``1-cos(y_c/radius)`` (~1.8% at
    ``|y_c|/radius=0.19``) grad<->div metric mismatch that makes the implicit
    free surface non-conservative and flips the marginally-resolved gyre
    turbulent (docs/ocean_fidelity/mitgcm_gyre_energy_conservation.md, the
    iteration-7 resolution)."""
    # A tall box (y up to ~1e6 m) is exactly where the old y_c/radius pseudo-lat
    # reached |lat|~0.19 rad and the cos(lat) error bit.
    g = create_beta_plane_cgrid_geometry(60, 10, dx_m=DX, f0=F0, beta=BETA,
                                         y_origin_m=0.0, cartesian_pseudo_lat=True)
    np.testing.assert_array_equal(np.asarray(g.lat), 0.0)
    np.testing.assert_array_equal(np.asarray(g.lat_T), 0.0)
    # cos(grid.lat) (what the operators recompute) == grid.cos_lat (the metric).
    np.testing.assert_allclose(np.cos(np.asarray(g.lat)), np.asarray(g.cos_lat))
    # The v-face zonal length the divergence recomputes, R*cos(lat_v)*dlon, must
    # equal the uniform stored dx_v (== dx_m) the gradient/explicit metrics use.
    lat_v = 0.5 * (np.concatenate([[0.0], np.asarray(g.lat)]) +
                   np.concatenate([np.asarray(g.lat), [0.0]]))
    face_dx_v = g.radius * np.cos(lat_v) * g.dlon
    np.testing.assert_allclose(face_dx_v, DX, rtol=1e-12)


def test_divergence_gradient_metrics_are_mutually_consistent():
    """Energy-conservation invariant the metric fix restores: ``divergence_cgrid``
    (continuity, drives eta) and ``gradient_x_cgrid`` (the PGF in the predictor /
    corrector) must use the SAME zonal metric, else the implicit free-surface
    predictor-corrector is non-conservative (PGF work != continuity).  The bug was
    that the divergence recomputed the v-face length as ``R*cos(grid.lat_v)*dlon``
    (varying) while the gradient used ``dx_u = R*dlon*grid.cos_lat = dx_m``
    (uniform); with the pseudo-lat now 0 they agree exactly.  This compares the
    two operators' effective zonal metrics by feeding a linear field and reading
    back the implied length (non-vacuous: a y/radius pseudo-lat makes the v-face
    metric ~1.8% short at the north edge, breaking this equality)."""
    from legoesm.grids.operators_latlon_cgrid import (
        divergence_cgrid,
        gradient_x_cgrid,
    )
    ny, nx = 60, 10
    g = create_beta_plane_cgrid_geometry(ny, nx, dx_m=DX, f0=F0, beta=BETA,
                                         y_origin_m=0.0, cartesian_pseudo_lat=True)
    # gradient zonal metric: gradient_x of a unit-slope-in-x field == 1/dx_u.
    ramp_x = jnp.asarray(np.broadcast_to(np.arange(nx, dtype=float) * DX, (ny, nx)).copy())
    gx = np.asarray(gradient_x_cgrid(ramp_x, g))[:, 1:nx]      # interior u-faces
    np.testing.assert_allclose(gx, 1.0, rtol=1e-12)            # slope 1 -> dx_u == DX
    # divergence v-face metric: a unit northward v gives div = (dx_v_face)/area per
    # cell row.  Feed v=1 on interior v-faces, u=0; div_merid telescopes to
    # (face_dx[i+1]-face_dx[i])/area = 0 ONLY if face_dx is uniform == DX (the fix).
    v = jnp.asarray(np.ones((ny + 1, nx)))
    v = v.at[0, :].set(0.0).at[-1, :].set(0.0)
    u = jnp.zeros((ny, nx + 1))
    div = np.asarray(divergence_cgrid(u, v, g))
    # Interior rows: incoming face length == outgoing face length == DX -> div 0.
    np.testing.assert_allclose(div[1:-1, :], 0.0, atol=1e-9)


def test_vorticity_operators_correct_scale_on_beta_plane():
    """Root-cause guard (iteration 8): the vorticity-based operators
    (``curl_vertex_cgrid`` -> ``vector_laplacian_cgrid``) must be at the correct
    ``~u/dx²`` scale on the Cartesian beta-plane.  The curl recomputes the vertex
    area as ``R²·dlon·|Δsin(lat)|``, which COLLAPSES to ~0 when the pseudo-lat is
    pinned to 0 (the divergence/gradient metric-consistency fix), so the safe-A
    pole guard divided by 1.0 and blew the vorticity up ~1e8× (NaN within a few
    steps).  Fixed by reading the grid's stored ``area_q = dx·dy``.  Compares the
    vector Laplacian of ``u=sin(2πy/Ly)`` to the analytic ``∇²u = -(2π/Ly)²u`` —
    non-vacuous (before the fix the ratio was ~5e8)."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        flux_divergence_viscosity_cgrid,
        vector_laplacian_cgrid,
    )
    ny, nx = 32, 4
    g = create_beta_plane_cgrid_geometry(ny, nx, dx_m=DX, dy_m=DX, f0=F0, beta=0.0,
                                         y_origin_m=0.0, cartesian_pseudo_lat=True)
    ly = DX * ny
    y_c = (np.arange(ny) + 0.5) * DX
    k = 2 * np.pi / ly
    u = jnp.asarray(np.sin(k * y_c)[:, None] * np.ones((ny, nx + 1)))
    v = jnp.zeros((ny + 1, nx))
    m = jnp.ones((ny, nx))
    um = jnp.ones((ny, nx + 1))
    vm = jnp.ones((ny + 1, nx)).at[0].set(0.0).at[-1].set(0.0)
    vlap_u, _ = vector_laplacian_cgrid(u, v, g, mask=m, u_mask=um, v_mask=vm)
    fdiv_u, _, _ = flux_divergence_viscosity_cgrid(u, v, g, 1.0, mask=m, u_mask=um, v_mask=vm)
    # The vector Laplacian must agree with the (always-correct) component
    # Laplacian to ~discretisation error — NOT be 1e8x larger.
    ratio = float(np.abs(np.asarray(vlap_u)).max() / (np.abs(np.asarray(fdiv_u)).max() + 1e-300))
    assert 0.5 < ratio < 2.0, f"vector_laplacian mis-scaled on beta-plane (ratio {ratio:.3e})"


def test_rectangular_cells_pin_every_stagger_axis():
    """Anisotropic cells (dx != dy) pin the axis of EVERY stagger metric, so a
    u/v-face dx<->dy swap (the metrics the model divides by) cannot slip past."""
    dx, dy = 20e3, 10e3
    g = _geom(dx_m=dx, dy_m=dy)
    # T-points
    np.testing.assert_allclose(np.asarray(g.dx_T), dx)
    np.testing.assert_allclose(np.asarray(g.dy_T), dy)
    np.testing.assert_allclose(np.asarray(g.area_T), dx * dy)
    # u-points: zonal spacing = dx, meridional extent = dy
    np.testing.assert_allclose(np.asarray(g.dx_u), dx)
    np.testing.assert_allclose(np.asarray(g.dy_u), dy)
    # v-points: zonal extent = dx, meridional spacing = dy
    np.testing.assert_allclose(np.asarray(g.dx_v), dx)
    np.testing.assert_allclose(np.asarray(g.dy_v), dy)
    np.testing.assert_allclose(np.asarray(g.area_q), dx * dy)


def test_coriolis_matches_mitgcm_convention():
    """f = f0 + beta*y, evaluated at each stagger point's own y (ini_cori.F)."""
    g = _geom()
    # T / u points sit at cell-centre y: y_c[j] = y0 + (j+1/2) dy.
    y_c = Y0 + (np.arange(NY) + 0.5) * DX
    np.testing.assert_allclose(np.asarray(g.f_T[:, 0]), F0 + BETA * y_c, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(g.f_u[:, 0]), F0 + BETA * y_c, rtol=1e-12)
    # v points sit at the lat interfaces: y_g[j] = y0 + j dy, length NY+1.
    y_g = Y0 + np.arange(NY + 1) * DX
    np.testing.assert_allclose(np.asarray(g.f_v[:, 0]), F0 + BETA * y_g, rtol=1e-12)
    # f is y-only (constant along x).
    assert np.allclose(np.asarray(g.f_T), np.asarray(g.f_T[:, :1]))


def test_beta_zero_is_f_plane():
    g = _geom(beta=0.0)
    np.testing.assert_allclose(np.asarray(g.f_T), F0)
    np.testing.assert_allclose(np.asarray(g.f_v), F0)


def test_stagger_shapes():
    g = _geom()
    assert g.f_T.shape == (NY, NX)
    assert g.f_u.shape == (NY, NX + 1)
    assert g.f_v.shape == (NY + 1, NX)
    assert g.area_q.shape == (NY + 1, NX + 1)


def test_ocean_model_runs_and_coriolis_acts():
    """The lat-lon C-grid ocean model runs on a beta-plane and f deflects flow.

    f-plane (beta=0), uniform initial zonal flow, no forcing: meridional velocity
    must grow ~ f*u0 (inertial response), staying finite and physical — NOT a
    barotropic-CFL blow-up (use a shallow layer + many substeps for stability).
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    f0 = 1.0e-4
    g = create_beta_plane_cgrid_geometry(16, 16, dx_m=50e3, f0=f0, beta=0.0)
    z = create_ocean_z_star(1, H_max=500.0)
    base = rest_state_latlon_cgrid_ocean(g, z, land_lat_threshold=90.0)
    cfg = LatLonCGridOceanConfig(
        use_conservation_fixer=False,
        enable_runtime_checks=False,
        n_barotropic_substeps=40,
        differentiable_barotropic=True,
    )
    model = LatLonCGridOceanModel(g, z, cfg)

    u0, dt = 0.05, 120.0
    s = base._replace(u=base.u.replace(data=jnp.full(base.u.data.shape, u0)))
    v_hist = []
    for _ in range(8):
        s = model.step(s, dt)
        v_hist.append(float(jnp.max(jnp.abs(s.v.data))))

    assert bool(jnp.all(jnp.isfinite(s.u.data)))
    assert bool(jnp.all(jnp.isfinite(s.v.data)))
    # Coriolis deflects: v grows from zero, monotonically, at the f*u0 scale
    # (per-step Δv ~ f*u0*dt = 6e-4) — physical, not a 100 m/s blow-up.
    assert v_hist[-1] > v_hist[0] > 0.0
    assert v_hist[-1] < 1.0  # nowhere near a CFL blow-up
    per_step = v_hist[0]
    assert 1e-5 < per_step < 1e-2  # order f*u0*dt


def test_invalid_unknown_kwarg_rejected():
    with pytest.raises(TypeError):
        create_beta_plane_cgrid_geometry(4, 4, dx_m=1e3, f0=1e-4)  # missing beta
