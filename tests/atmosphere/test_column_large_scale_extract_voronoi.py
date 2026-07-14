"""MPAS/Voronoi column-forcing extractor (iter 73).

Closes the long-deferred third grid family: a worst column flagged on an
unstructured MPAS/Voronoi grid can now have its large-scale forcing extracted for
the LES spin-off (lat-lon + cubed-sphere already worked).  The Voronoi path reuses
the grid-agnostic continuity (``omega_from_divergence``) + advection
(``advective_tendency``) chains; only the horizontal operators are TRiSK
(``gradient_edge_3d`` + Perot ``reconstruct_cell_velocity`` for the cell gradient,
``divergence_cell_3d`` for the cell divergence of the edge-normal velocity).

Single-rank / full-mesh (the local TRiSK operators carry no halo exchange), same
assumption as the lat-lon / cubed extractors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.forcing.idealized.column_forcing import ColumnLargeScaleState
from legoesm.atmosphere.forcing.column_large_scale_extract import (
    extract_column_forcing,
    extract_column_forcing_voronoi,
)
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.grids.voronoi import create_voronoi_mesh, reconstruct_cell_velocity

jax.config.update("jax_enable_x64", True)

_NLEV = 8
_CELL = 40          # an interior cell of the level-2 (162-cell) mesh
_THETA_K = 290.0


def _mesh_sigma():
    return create_voronoi_mesh(2), create_sigma_coordinate(_NLEV)


def _uniform_edge_flow(mesh, u_east, v_north):
    """Edge-normal velocity of a UNIFORM (u_east, v_north) flow: V·n̂ =
    u_east·cos(angleEdge) + v_north·sin(angleEdge), broadcast over levels."""
    ae = jnp.asarray(mesh.angleEdge)
    u_n = u_east * jnp.cos(ae) + v_north * jnp.sin(ae)
    return jnp.broadcast_to(u_n[:, None], (mesh.nEdges, _NLEV))


def _state(mesh, *, theta_const=True):
    p_s = jnp.full((mesh.nCells,), 1.0e5)
    sigma = create_sigma_coordinate(_NLEV)
    p_full = p_s[:, None] * jnp.asarray(sigma.sigma_full)
    if theta_const:
        # θ uniform across CELLS (varies only with level via exner) ⇒ ∇θ = 0.
        T = _THETA_K * exner_function(p_full)
    else:
        T = (_THETA_K + 5.0 * jnp.cos(jnp.asarray(mesh.latCell))[:, None]) \
            * exner_function(p_full)
    # q_v VARIES across cells (a real horizontal gradient ⇒ non-vacuous advection).
    q_v = 0.01 + 4.0e-3 * jnp.cos(jnp.asarray(mesh.latCell))[:, None] \
        * jnp.ones((1, _NLEV))
    return T, q_v, p_s


def test_uniform_flow_reconstruction_is_exact_orientation_regression():
    """Perot reconstruction of a LOCALLY-uniform (u_east, v_north) edge field
    returns that (u, v) at an EQUATORIAL cell (where the local east/north frame is
    well-conditioned — a constant east/north field is singular at the poles, so the
    probe is only meaningful at low latitude) — pins the angleEdge / edge-normal
    SIGN + ORIENTATION convention that BOTH the cell-velocity AND the cell-gradient
    reconstruction in the extractor rely on (Codex orientation check). A swapped
    cos/sin or a sign flip would return (≈−3, ≈7) or negated components, far
    outside this tolerance."""
    mesh, _ = _mesh_sigma()
    u_edge = _uniform_edge_flow(mesh, 7.0, -3.0)
    u_east, v_north = reconstruct_cell_velocity(u_edge[:, 0], mesh)   # 2-D path
    eq = int(np.argmin(np.abs(np.asarray(mesh.latCell))))             # equatorial cell
    assert float(np.asarray(u_east)[eq]) == pytest.approx(7.0, abs=0.15)
    assert float(np.asarray(v_north)[eq]) == pytest.approx(-3.0, abs=0.15)


def test_extract_voronoi_constant_theta_zero_advection_nonvacuous_moisture():
    """Uniform θ ⇒ ∇θ = 0 ⇒ θ-advection ≡ 0 (analytic); a cell-VARYING q_v ⇒
    NON-ZERO moisture advection (the operator genuinely computed a gradient)."""
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh, theta_const=True)
    u_edge = _uniform_edge_flow(mesh, 8.0, 2.0)
    out = extract_column_forcing_voronoi(
        T=T, q_v=q_v, u_edge=u_edge, p_s=p_s, mesh=mesh,
        sigma_coord=sigma, lat_rad=float(mesh.latCell[_CELL]), col_index=(_CELL,))
    assert isinstance(out, ColumnLargeScaleState)
    assert out.T.shape == (_NLEV,) and out.omega.shape == (_NLEV,)
    np.testing.assert_allclose(np.asarray(out.theta_adv), 0.0, atol=1e-10)
    # Non-vacuous: q varies across cells ⇒ a real moisture-advection tendency.
    assert float(np.max(np.abs(np.asarray(out.qv_adv)))) > 1e-12
    # Geostrophic wind is computed (mid-lat cell, outside the cutoff) + finite —
    # NON-zero here because q_v varies with latitude (a virtual-temperature /
    # geopotential gradient → a real thermal wind); the EXACT zero anchor is the
    # truly-uniform-state test below.
    assert out.u_geo is not None and np.all(np.isfinite(np.asarray(out.u_geo)))


def test_extract_voronoi_geostrophic_uniform_state_zero_wind():
    """ANALYTIC ANCHOR (Codex): a TRULY uniform state (uniform T, q_v, p_s across
    cells) has zero geopotential + ln(p_s) gradient ⇒ the geostrophic wind is
    EXACTLY zero (the Perot reconstruction of a zero edge-gradient is zero) at a
    mid-latitude cell — and finite (no NaN/inf)."""
    mesh, sigma = _mesh_sigma()
    p_s = jnp.full((mesh.nCells,), 1.0e5)
    p_full = p_s[:, None] * jnp.asarray(sigma.sigma_full)
    T = 290.0 * exner_function(p_full)           # identical at every cell
    q_v = jnp.full((mesh.nCells, _NLEV), 0.01)   # uniform moisture too
    u_edge = _uniform_edge_flow(mesh, 6.0, 0.0)
    out = extract_column_forcing_voronoi(
        T=T, q_v=q_v, u_edge=u_edge, p_s=p_s, mesh=mesh, sigma_coord=sigma,
        lat_rad=float(mesh.latCell[42]), col_index=(42,))     # ~−45°, outside cutoff
    np.testing.assert_allclose(np.asarray(out.u_geo), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(out.v_geo), 0.0, atol=1e-9)


_MIDLAT_CELL = 42       # ~−45°, well outside the 5° equatorial geostrophic cutoff
_EQ_CELL = 102          # ~0°, INSIDE the cutoff (f → 0 ill-posed)


def test_extract_voronoi_geostrophic_meridional_gradient_zonal_jet():
    """A MERIDIONAL temperature gradient ⇒ a Φ that varies with latitude ⇒ a
    finite, predominantly-ZONAL geostrophic wind (|u_geo| > |v_geo|) at a
    mid-latitude cell. Finite (catches the Perot-weight / degenerate-cell NaN
    hazards) + a real O(1) jet (non-vacuous)."""
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh, theta_const=False)   # T = (290 + 5·cos lat)·exner
    u_edge = _uniform_edge_flow(mesh, 6.0, 0.0)
    out = extract_column_forcing_voronoi(
        T=T, q_v=q_v, u_edge=u_edge, p_s=p_s, mesh=mesh, sigma_coord=sigma,
        lat_rad=float(mesh.latCell[_MIDLAT_CELL]), col_index=(_MIDLAT_CELL,))
    ug, vg = np.asarray(out.u_geo), np.asarray(out.v_geo)
    assert ug.shape == (_NLEV,) and np.all(np.isfinite(ug)) and np.all(np.isfinite(vg))
    assert float(np.max(np.abs(ug))) > 1.0                  # a real jet, not noise
    assert float(np.max(np.abs(ug))) > float(np.max(np.abs(vg)))  # predominantly zonal


def test_extract_voronoi_geostrophic_orographic_phis_runs():
    """The optional orographic ``phis`` threads through the Voronoi Perot
    cell-gradient (a single-LEVEL surface field): terrain gives a finite, NON-zero,
    σ-independent geostrophic wind where the flat uniform state gives 0."""
    from legoesm import constants
    mesh, sigma = _mesh_sigma()
    p_s = jnp.full((mesh.nCells,), 1.0e5)
    T = 290.0 * exner_function(p_s[:, None] * jnp.asarray(sigma.sigma_full))
    q_v = jnp.full((mesh.nCells, _NLEV), 0.01)
    u_edge = _uniform_edge_flow(mesh, 6.0, 0.0)
    kw = dict(T=T, q_v=q_v, u_edge=u_edge, p_s=p_s, mesh=mesh, sigma_coord=sigma,
              lat_rad=float(mesh.latCell[42]), col_index=(42,))
    flat = extract_column_forcing_voronoi(**kw)
    phis = constants.g * (800.0 * jnp.sin(2.0 * jnp.asarray(mesh.latCell)))
    oro = extract_column_forcing_voronoi(**kw, phis=phis)
    assert bool(jnp.all(jnp.isfinite(oro.u_geo))) and bool(jnp.all(jnp.isfinite(oro.v_geo)))
    du = np.asarray(oro.u_geo) - np.asarray(flat.u_geo)
    dv = np.asarray(oro.v_geo) - np.asarray(flat.v_geo)
    assert float(np.max(np.abs(du)) + np.max(np.abs(dv))) > 1e-3
    np.testing.assert_allclose(du, du[0], atol=1e-9)               # σ-independent
    np.testing.assert_allclose(dv, dv[0], atol=1e-9)


def test_extract_voronoi_geostrophic_equatorial_cell_is_none():
    """Within _MIN_GEOSTROPHIC_LAT_DEG of the equator (f → 0) the geostrophic wind
    is None ⇒ the column LES falls back to f×V — NEVER worse than the previous
    unconditional None (enabling geostrophic does not degrade equatorial cols)."""
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh, theta_const=False)
    u_edge = _uniform_edge_flow(mesh, 6.0, 0.0)
    out = extract_column_forcing_voronoi(
        T=T, q_v=q_v, u_edge=u_edge, p_s=p_s, mesh=mesh, sigma_coord=sigma,
        lat_rad=float(mesh.latCell[_EQ_CELL]), col_index=(_EQ_CELL,))
    assert out.u_geo is None and out.v_geo is None


def test_extract_voronoi_shapes_and_finite():
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh, theta_const=False)
    u_edge = _uniform_edge_flow(mesh, 6.0, 0.0)
    out = extract_column_forcing_voronoi(
        T=T, q_v=q_v, u_edge=u_edge, p_s=p_s, mesh=mesh,
        sigma_coord=sigma, lat_rad=float(mesh.latCell[_CELL]), col_index=(_CELL,))
    for name in ("T", "p_full", "q_v", "omega", "theta_adv", "qv_adv"):
        arr = np.asarray(getattr(out, name))
        assert arr.shape == (_NLEV,)
        assert np.all(np.isfinite(arr)), name


def test_extract_voronoi_is_surface_flux_free():
    """The voronoi/MPAS extractor produces NO surface BC (prescribe='none', no
    T_s/w_th_s/w_qv_s) — the cross-dispatch invariant matching iter-148's
    build_column_les_setup guard (a surface flux wired into ANY extractor without the
    LES surface-flux bottom BC would make that guard reject the column)."""
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh, theta_const=False)
    u_edge = _uniform_edge_flow(mesh, 6.0, 0.0)
    out = extract_column_forcing_voronoi(
        T=T, q_v=q_v, u_edge=u_edge, p_s=p_s, mesh=mesh,
        sigma_coord=sigma, lat_rad=float(mesh.latCell[_CELL]), col_index=(_CELL,))
    assert out.prescribe == "none"
    assert out.T_s is None and out.w_th_s is None and out.w_qv_s is None


def test_extract_voronoi_divergent_flow_gives_nonzero_omega():
    """A DIVERGENT edge-normal velocity ⇒ non-zero cell divergence ⇒ non-zero ω
    (the continuity chain is genuinely exercised, not vacuously zero)."""
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh, theta_const=True)
    # Latitude-dependent edge flow ⇒ a spatially varying (divergent) field.
    u_edge = (jnp.asarray(mesh.latEdge)[:, None]
              * jnp.ones((1, _NLEV)) * 5.0)
    out = extract_column_forcing_voronoi(
        T=T, q_v=q_v, u_edge=u_edge, p_s=p_s, mesh=mesh,
        sigma_coord=sigma, lat_rad=float(mesh.latCell[_CELL]), col_index=(_CELL,))
    assert np.all(np.isfinite(np.asarray(out.omega)))
    assert float(np.max(np.abs(np.asarray(out.omega)))) > 0.0


def test_dispatch_routes_voronoi_matches_direct():
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh, theta_const=False)
    u_edge = _uniform_edge_flow(mesh, 5.0, 1.0)
    kw = dict(T=T, q_v=q_v, p_s=p_s, sigma_coord=sigma,
              lat_rad=float(mesh.latCell[_CELL]), col_index=(_CELL,))
    via_dispatch = extract_column_forcing(u=u_edge, v=None, grid=mesh, **kw)
    direct = extract_column_forcing_voronoi(u_edge=u_edge, mesh=mesh, **kw)
    np.testing.assert_array_equal(
        np.asarray(via_dispatch.qv_adv), np.asarray(direct.qv_adv))
    np.testing.assert_array_equal(
        np.asarray(via_dispatch.omega), np.asarray(direct.omega))


def test_dispatch_voronoi_rejects_non_none_v():
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh)
    u_edge = _uniform_edge_flow(mesh, 5.0, 0.0)
    with pytest.raises(ValueError, match="requires `v=None`"):
        extract_column_forcing(
            T=T, q_v=q_v, u=u_edge, v=jnp.zeros((mesh.nCells, _NLEV)),
            p_s=p_s, grid=mesh, sigma_coord=sigma,
            lat_rad=0.1, col_index=(_CELL,))


def test_dispatch_voronoi_wrong_arity_and_shapes_raise():
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh)
    u_edge = _uniform_edge_flow(mesh, 5.0, 0.0)
    base = dict(T=T, q_v=q_v, v=None, p_s=p_s, grid=mesh, sigma_coord=sigma,
                lat_rad=0.1)
    with pytest.raises(ValueError, match="must be .cell"):
        extract_column_forcing(u=u_edge, col_index=(1, 2), **base)
    with pytest.raises(ValueError, match="nEdges"):       # wrong u leading dim
        extract_column_forcing(
            u=jnp.zeros((mesh.nCells, _NLEV)), col_index=(_CELL,), **base)


def test_extract_voronoi_rejects_zeroed_edge_sign():
    """A mesh whose edgeSignOnCell is all zero would silently zero the divergence
    (hence ω) — rejected loudly (Codex #d)."""
    mesh, sigma = _mesh_sigma()
    T, q_v, p_s = _state(mesh)
    u_edge = _uniform_edge_flow(mesh, 5.0, 0.0)
    bad = mesh._replace(edgeSignOnCell=jnp.zeros_like(jnp.asarray(mesh.edgeSignOnCell)))
    with pytest.raises(ValueError, match="edgeSignOnCell is all zero"):
        extract_column_forcing_voronoi(
            T=T, q_v=q_v, u_edge=u_edge, p_s=p_s, mesh=bad,
            sigma_coord=sigma, lat_rad=0.1, col_index=(_CELL,))
