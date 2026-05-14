"""Unit tests for the DINO surface-forcing applicators in
``legoesm.ocean.experiments.dino`` (Phase 4 v1; moved from
``scripts/run_dino.py`` 2026-05-14 per CLAUDE.md "Untested-but-live
debt" + "every dispatch branch must have a direct test" rules).

Covers:
- ``dino_lat_lon_surface_forcing_arrays`` (pre-compute + broadcast)
- ``dino_mpas_surface_forcing_arrays`` (pre-compute + edge projection)
- ``apply_dino_lat_lon_surface_forcing`` (wind + T/S/SW for one step)
- ``apply_dino_mpas_surface_forcing`` (wind + T/S/SW for one step)

Mostly checks shapes, sign conventions, and that forcing actually
moves the state in the right direction over one step. Long-time
stability is exercised by the end-to-end smoke tests.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.voronoi import create_regional_voronoi_mesh
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    apply_dino_lat_lon_surface_forcing,
    apply_dino_mpas_surface_forcing,
    create_dino_z_star,
    dino_lat_lon_grid,
    dino_lat_lon_state,
    dino_lat_lon_surface_forcing_arrays,
    dino_mpas_state,
    dino_mpas_surface_forcing_arrays,
)


# Coarse for fast tests
LATLON_N_LON = 25
MPAS_RES_KM = 220.0


@pytest.fixture(scope="module")
def cfg():
    return DINOConfig()


@pytest.fixture(scope="module")
def z_coord(cfg):
    return create_dino_z_star(cfg)


# ---------- Lat-lon forcing-array builder ---------------------------

@pytest.fixture(scope="module")
def latlon_grid(cfg):
    return dino_lat_lon_grid(cfg, n_lon=LATLON_N_LON)


def test_latlon_forcing_arrays_shapes(latlon_grid, cfg):
    f = dino_lat_lon_surface_forcing_arrays(latlon_grid, cfg)
    assert f["T_star_2d"].shape == (latlon_grid.n_lat, latlon_grid.n_lon)
    assert f["S_star_2d"].shape == (latlon_grid.n_lat, latlon_grid.n_lon)
    assert f["Q_sr_2d"].shape == (latlon_grid.n_lat, latlon_grid.n_lon)
    assert f["tau_u_face"].shape == (latlon_grid.n_lat, latlon_grid.n_lon + 1)


def test_latlon_forcing_T_star_at_equator_is_eq_value(latlon_grid, cfg):
    f = dino_lat_lon_surface_forcing_arrays(latlon_grid, cfg)
    lat_deg = np.degrees(np.asarray(latlon_grid.lat))
    j_eq = int(np.argmin(np.abs(lat_deg)))
    T_star = float(np.asarray(f["T_star_2d"])[j_eq, 0])
    assert T_star == pytest.approx(cfg.T_star_eq, abs=1.0)


def test_latlon_forcing_zonally_uniform(latlon_grid, cfg):
    """T*, S*, Q_sr only depend on latitude — every longitude column
    should be identical."""
    f = dino_lat_lon_surface_forcing_arrays(latlon_grid, cfg)
    T_2d = np.asarray(f["T_star_2d"])
    assert (T_2d == T_2d[:, :1]).all()
    S_2d = np.asarray(f["S_star_2d"])
    assert (S_2d == S_2d[:, :1]).all()


# ---------- Lat-lon applicator --------------------------------------

@pytest.fixture(scope="module")
def latlon_state(latlon_grid, z_coord, cfg):
    return dino_lat_lon_state(latlon_grid, z_coord, cfg)


def test_latlon_apply_forcing_returns_finite(latlon_grid, latlon_state, z_coord, cfg):
    forcing = dino_lat_lon_surface_forcing_arrays(latlon_grid, cfg)
    new = apply_dino_lat_lon_surface_forcing(
        latlon_state, forcing, z_coord, cfg, dt=cfg.dt,
    )
    for fld in (new.T, new.S, new.u):
        assert bool(jnp.all(jnp.isfinite(fld.data)))


def test_latlon_apply_forcing_wind_creates_u(latlon_grid, latlon_state, z_coord, cfg):
    """At rest with wind on, top-layer u should become non-zero."""
    forcing = dino_lat_lon_surface_forcing_arrays(latlon_grid, cfg)
    new = apply_dino_lat_lon_surface_forcing(
        latlon_state, forcing, z_coord, cfg, dt=cfg.dt,
    )
    # state.u shape (n_lat, n_lon+1, nlev); top level = [:, :, 0]
    u_top_before = np.asarray(latlon_state.u.data[:, :, 0])
    u_top_after = np.asarray(new.u.data[:, :, 0])
    assert (u_top_before == 0.0).all()
    assert np.abs(u_top_after).max() > 0.0


def test_latlon_apply_forcing_subsurface_u_unchanged(latlon_grid, latlon_state,
                                                     z_coord, cfg):
    """Wind only affects top u layer."""
    forcing = dino_lat_lon_surface_forcing_arrays(latlon_grid, cfg)
    new = apply_dino_lat_lon_surface_forcing(
        latlon_state, forcing, z_coord, cfg, dt=cfg.dt,
    )
    diff = np.asarray(new.u.data) - np.asarray(latlon_state.u.data)
    assert (diff[:, :, 1:] == 0.0).all()


def test_latlon_apply_forcing_T_evolves_subsurface_via_jerlov(
    latlon_grid, latlon_state, z_coord, cfg,
):
    """Jerlov SW penetration should warm subsurface levels too — not
    just the top layer. Tests that the column-distributed Q_sr is
    actually being applied."""
    forcing = dino_lat_lon_surface_forcing_arrays(latlon_grid, cfg)
    new = apply_dino_lat_lon_surface_forcing(
        latlon_state, forcing, z_coord, cfg, dt=cfg.dt,
    )
    dT = np.asarray(new.T.data) - np.asarray(latlon_state.T.data)
    mask = np.asarray(latlon_state.land_mask.data) > 0.5
    # Subsurface levels (k=1..5) should have nonzero dT in tropics where Q_sr > 0
    dT_sub = dT[..., 1:6]
    assert np.abs(dT_sub[mask, :]).max() > 0.0


# ---------- MPAS forcing-array builder ------------------------------

@pytest.fixture(scope="module")
def mpas_mesh(cfg):
    return create_regional_voronoi_mesh(
        lon_range=(cfg.lon_west_deg, cfg.lon_east_deg),
        lat_range=(-cfg.lat_max_deg, cfg.lat_max_deg),
        resolution_km=MPAS_RES_KM,
        periodic_x=True,
    )


def test_mpas_forcing_arrays_shapes(mpas_mesh, cfg):
    f = dino_mpas_surface_forcing_arrays(mpas_mesh, cfg)
    assert f["T_star_1d"].shape == (mpas_mesh.nCells,)
    assert f["S_star_1d"].shape == (mpas_mesh.nCells,)
    assert f["Q_sr_1d"].shape == (mpas_mesh.nCells,)
    assert f["tau_normal"].shape == (mpas_mesh.nEdges,)


def test_mpas_forcing_tau_normal_projected(mpas_mesh, cfg):
    """tau_normal = tau_u(lat_edge) * cos(angleEdge). Verify the sign
    is consistent: at zonal westerly band (-45° N), tau_u > 0 and
    tau_normal for east-pointing edges (angleEdge ≈ 0) > 0."""
    f = dino_mpas_surface_forcing_arrays(mpas_mesh, cfg)
    lat_e = np.degrees(np.asarray(mpas_mesh.latEdge))
    angle_e = np.asarray(mpas_mesh.angleEdge)
    tau_n = np.asarray(f["tau_normal"])
    # Pick edges in southern westerlies band (lat ≈ -45°) pointing east
    # (angle in [-π/4, π/4])
    mask = (np.abs(lat_e + 45.0) < 5.0) & (np.abs(angle_e) < np.pi / 4)
    if mask.any():
        # In this band the wind is positive (paper: tau_u(-45°) = 0.2 N/m²),
        # so tau_normal projected onto east-pointing edges should be positive
        assert (tau_n[mask] > 0.0).any()


# ---------- MPAS applicator -----------------------------------------

@pytest.fixture(scope="module")
def mpas_state(mpas_mesh, z_coord, cfg):
    return dino_mpas_state(mpas_mesh, z_coord, cfg)


def test_mpas_apply_forcing_returns_finite(mpas_mesh, mpas_state, z_coord, cfg):
    forcing = dino_mpas_surface_forcing_arrays(mpas_mesh, cfg)
    new = apply_dino_mpas_surface_forcing(
        mpas_state, forcing, z_coord, cfg, dt=cfg.dt,
    )
    for fld in (new.T, new.S, new.u):
        assert bool(jnp.all(jnp.isfinite(fld.data)))


def test_mpas_apply_forcing_wind_creates_u(mpas_mesh, mpas_state, z_coord, cfg):
    forcing = dino_mpas_surface_forcing_arrays(mpas_mesh, cfg)
    new = apply_dino_mpas_surface_forcing(
        mpas_state, forcing, z_coord, cfg, dt=cfg.dt,
    )
    u_top_before = np.asarray(mpas_state.u.data[:, 0])
    u_top_after = np.asarray(new.u.data[:, 0])
    assert (u_top_before == 0.0).all()
    assert np.abs(u_top_after).max() > 0.0


def test_mpas_apply_forcing_subsurface_u_unchanged(mpas_mesh, mpas_state, z_coord, cfg):
    forcing = dino_mpas_surface_forcing_arrays(mpas_mesh, cfg)
    new = apply_dino_mpas_surface_forcing(
        mpas_state, forcing, z_coord, cfg, dt=cfg.dt,
    )
    diff = np.asarray(new.u.data) - np.asarray(mpas_state.u.data)
    assert (diff[:, 1:] == 0.0).all()


def test_mpas_apply_forcing_T_evolves_subsurface_via_jerlov(
    mpas_mesh, mpas_state, z_coord, cfg,
):
    forcing = dino_mpas_surface_forcing_arrays(mpas_mesh, cfg)
    new = apply_dino_mpas_surface_forcing(
        mpas_state, forcing, z_coord, cfg, dt=cfg.dt,
    )
    dT = np.asarray(new.T.data) - np.asarray(mpas_state.T.data)
    mask = np.asarray(mpas_state.land_mask.data) > 0.5
    dT_sub = dT[..., 1:6]
    assert np.abs(dT_sub[mask, :]).max() > 0.0
