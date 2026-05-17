"""Bulk-metric comparison: legoESM (lat-lon FV + MPAS) vs Veros.

Drives short, physics-aligned reference runs of the Eady-uniform and DINO
test cases on three model variants:

* Veros (B-grid spherical, baseline reference) -- via the Veros runner
  adapters in ``legoesm.ocean.fidelity.veros_configs``.
* legoESM lat-lon C-grid (Mercator / regional channel).
* legoESM MPAS Voronoi (channel for Eady, regional basin for DINO).

After each run we extract a small set of bulk scalar metrics (basin-wide
T/S ranges, |u|max, surface forcing extrema, etc.) and report a side-by-
side table with the relative delta of each legoESM variant against the
Veros reference. By default the script flags any metric whose relative
delta exceeds ``--tolerance`` (default 5%) but still prints the full
table so callers can audit each one.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python scripts/ocean_fidelity/compare_legoesm_vs_veros.py

Add ``--tolerance 0.10`` to relax the gate to 10%, or ``--write-report
docs/ocean_fidelity/legoesm_vs_veros_<sha>.md`` to dump a Markdown report.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402  -- imported after JAX env setup


# ---------------------------------------------------------------------------
# Veros side -- short reference runs through the in-process runner.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Grid-agnostic bulk metrics.
#
# Each model is reduced to cell-centred 3D arrays (T, S, u-on-cells) plus a
# 3D mass-weight tensor ``w[c, k] = area_c * dz_k * mask[c, k]`` so that the
# T / S / u_mean / ke_mean metrics are computed identically on Veros B-grid,
# legoESM lat-lon C-grid, and legoESM MPAS edges.
# ---------------------------------------------------------------------------

def _bulk_eady_metrics(T_cell, u_sq_cell, u_cell, w_cell, dz_per_col,
                        u_dz_sum_per_col, area_per_col,
                        ) -> dict[str, float]:
    """Common Eady-uniform reductions.

    Parameters
    ----------
    T_cell : (Ncols, Nz) cell-centred T (deg C)
    u_sq_cell : (Ncols, Nz) cell-centred |u|^2 (m^2/s^2)
    u_cell : (Ncols, Nz) cell-centred u (m/s)
    w_cell : (Ncols, Nz) wet-cell mass weights, == area * dz * mask
    dz_per_col, u_dz_sum_per_col, area_per_col : (Ncols,) column quantities.
    """
    total = float(w_cell.sum())
    wet_T = T_cell[w_cell > 0]
    T_min = float(wet_T.min()) if wet_T.size else float("nan")
    T_max = float(wet_T.max()) if wet_T.size else float("nan")
    T_mean = float((T_cell * w_cell).sum() / max(total, 1e-30))
    u_mean = float((u_cell * w_cell).sum() / max(total, 1e-30))
    ke_mean = float(0.5 * (u_sq_cell * w_cell).sum() / max(total, 1e-30))
    # Column-mean |u| -- invariant under vertical redistribution between
    # the explicit-tendency (legoESM) and implicit-friction (Veros) wind
    # forcing pathways. Mask dry columns out.
    wet_col = area_per_col > 0
    col_mean_u = np.zeros_like(area_per_col)
    if wet_col.any():
        col_mean_u[wet_col] = (u_dz_sum_per_col[wet_col]
                               / np.maximum(dz_per_col[wet_col], 1e-30))
    u_abs_max = float(np.abs(col_mean_u).max()) if wet_col.any() else float("nan")
    return {
        "T_min": T_min, "T_max": T_max, "T_mean": T_mean,
        "u_abs_max": u_abs_max, "u_mean": u_mean, "ke_mean": ke_mean,
    }


def _bulk_dino_metrics(T_cell, S_cell, u_cell, w_cell,
                        dz_per_col, u_dz_sum_per_col, area_per_col,
                        taux_zonal) -> dict[str, float]:
    """Common DINO reductions."""
    total = float(w_cell.sum())
    wet_T = T_cell[w_cell > 0]
    wet_S = S_cell[w_cell > 0]
    T_min = float(wet_T.min())
    T_max = float(wet_T.max())
    T_mean = float((T_cell * w_cell).sum() / max(total, 1e-30))
    S_min = float(wet_S.min())
    S_max = float(wet_S.max())
    S_mean = float((S_cell * w_cell).sum() / max(total, 1e-30))
    wet_col = area_per_col > 0
    col_mean_u = np.zeros_like(area_per_col)
    if wet_col.any():
        col_mean_u[wet_col] = (u_dz_sum_per_col[wet_col]
                               / np.maximum(dz_per_col[wet_col], 1e-30))
    u_abs_max = float(np.abs(col_mean_u).max())
    return {
        "T_min": T_min, "T_max": T_max, "T_mean": T_mean,
        "S_min": S_min, "S_max": S_max, "S_mean": S_mean,
        "u_abs_max": u_abs_max,
        "taux_min": float(np.asarray(taux_zonal).min()),
        "taux_max": float(np.asarray(taux_zonal).max()),
    }


# ---------------------------------------------------------------------------
# Veros side: extract cell-centred fields + weights from the runner result.
# ---------------------------------------------------------------------------

def _veros_cell_views(result, with_salt: bool = False, with_taux: bool = False):
    """Return canonical (T, S, u_cell, u²_cell, w, dz_col, u_dz_sum, area_col,
    taux_zonal) arrays on the Veros B-grid cell-centres.
    """
    nx = int(result.grid_metadata["nx"])
    ny = int(result.grid_metadata["ny"])
    nz = int(result.grid_metadata["nz"])
    yt = np.asarray(result.grid_metadata["yt"])[2:-2]   # deg
    maskT = np.asarray(result.grid_metadata["maskT"]).astype(bool)[2:-2, 2:-2, :]
    # Prefer the captured ``dzt`` (cell thicknesses) — it is the only
    # source guaranteed to give per-level dz regardless of whether Veros
    # is configured with uniform or stretched z. Fall back to ``zw``
    # interface differences if the older runner cache did not record dzt.
    dzt_meta = result.grid_metadata.get("dzt")
    if dzt_meta is not None:
        dz = np.asarray(dzt_meta, dtype=np.float64)
    else:
        zw = np.asarray(result.grid_metadata.get("zw"), dtype=np.float64)
        zt = np.asarray(result.grid_metadata.get("zt"), dtype=np.float64)
        if zw.ndim == 1 and zw.size == nz + 1:
            dz = np.abs(np.diff(zw))
        else:
            dz = np.abs(np.diff(np.concatenate([[0.0], zt])))
    dz = dz[:nz]

    # Cos(lat) cell area proxy (uniform-dy => area ∝ cos(lat); the
    # absolute scale cancels in the area-weighted mean).
    cos_lat = np.cos(np.radians(yt))[None, :]              # (1, ny)
    area_2d = np.broadcast_to(cos_lat, (nx, ny)).astype(np.float64)
    mask_col = maskT.any(axis=2)                           # (nx, ny)
    area_col = area_2d * mask_col

    temp = np.asarray(result.variables["temp"])[2:-2, 2:-2, :, -1]
    u = np.asarray(result.variables["u"])[2:-2, 2:-2, :, -1]
    # Veros stores ``u`` on a horizontally-shifted velocity grid (the
    # north-east corner of each T-cell). For bulk volume averages a
    # neighbour-mean on each side is sufficient at our coarse
    # resolutions; we approximate ``u_cell ≈ 0.5 (u[i, j] + u[i-1, j])``
    # and just take ``u`` straight where the shift is irrelevant.
    u_cell = u                                             # B-grid: collocated for our purposes
    u_sq_cell = u_cell ** 2

    # Flatten (nx, ny) -> Ncols, keep z axis.
    Ncols = nx * ny
    T_flat = temp.reshape(Ncols, nz)
    S_flat = (np.asarray(result.variables["salt"])[2:-2, 2:-2, :, -1]
              .reshape(Ncols, nz)) if with_salt else None
    u_flat = u_cell.reshape(Ncols, nz)
    usq_flat = u_sq_cell.reshape(Ncols, nz)
    w = (area_2d[..., None] * dz[None, None, :]
         * maskT.astype(np.float64)).reshape(Ncols, nz)
    dz_col = (dz[None, None, :] * maskT.astype(np.float64)
              ).sum(axis=2).reshape(Ncols)
    u_dz_col = (u_cell * dz[None, None, :] * maskT.astype(np.float64)
                ).sum(axis=2).reshape(Ncols)
    area_col_flat = area_col.reshape(Ncols)

    taux_z = None
    if with_taux:
        # Veros ``surface_taux`` is the zonal stress on the U-grid; for
        # the bulk extrema we just take its interior over the wet mask.
        taux = np.asarray(result.variables["surface_taux"])[2:-2, 2:-2]
        taux_z = taux[mask_col]

    return T_flat, S_flat, u_flat, usq_flat, w, dz_col, u_dz_col, area_col_flat, taux_z


def run_veros_eady_uniform() -> dict[str, float]:
    from legoesm.ocean.fidelity import veros_runner
    result = veros_runner.run_veros(
        "eady_uniform", runlen_s=3600.0, force_recompute=False,
    )
    (T, _S, u, usq, w, dz_col, u_dz_col, area_col, _taux
     ) = _veros_cell_views(result, with_salt=False, with_taux=False)
    return _bulk_eady_metrics(T, usq, u, w, dz_col, u_dz_col, area_col)


def run_veros_dino(nx: int = 20, ny: int = 40, nz: int = 12) -> dict[str, float]:
    from legoesm.ocean.fidelity import veros_runner
    result = veros_runner.run_veros(
        "dino", runlen_s=2 * 2700.0, force_recompute=False,
        factory_kwargs={"nx": nx, "ny": ny, "nz": nz, "uniform_z": True},
    )
    (T, S, u, _usq, w, dz_col, u_dz_col, area_col, taux_z
     ) = _veros_cell_views(result, with_salt=True, with_taux=True)
    return _bulk_dino_metrics(T, S, u, w, dz_col, u_dz_col, area_col, taux_z)


# ---------------------------------------------------------------------------
# legoESM Eady-uniform: lat-lon C-grid + MPAS channel.
# ---------------------------------------------------------------------------

# Veros's linear EOS uses these constants (veros.core.density.linear_eq).
_VEROS_RHO_REF = 1024.0
_VEROS_BETA_T = 1.67e-4
_VEROS_BETA_S = 0.78e-3
# Veros stores ``theta0 = 283.0`` Kelvin and references the linear EOS
# against ``theta0 - T_freeze`` in degrees Celsius.
_VEROS_T_REF_C = 283.0 - constants.T_freeze   # 9.85 deg C
_VEROS_S_REF = 35.0


def _veros_linear_eos_config():
    """Build a legoESM ``LinearEOSConfig`` matching Veros's hard-coded EOS."""
    from legoesm.ocean.eos import LinearEOSConfig
    return LinearEOSConfig(
        rho_ref=_VEROS_RHO_REF, alpha_T=_VEROS_BETA_T, beta_S=_VEROS_BETA_S,
        T_ref=_VEROS_T_REF_C, S_ref=_VEROS_S_REF,
    )


def _eady_legoesm_config():
    """``EadyUniformConfig`` aligned with the Veros adapter parameters."""
    from legoesm.ocean.experiments.eady_uniform import EadyUniformConfig
    return EadyUniformConfig(
        # Geometry / IC parameters identical to Veros eady_uniform adapter.
        H_max=5500.0,
        lat_south=16.0, lat_north=34.0, lat_center=25.0,
        lon_west=0.0, lon_east=10.0,
        N=1.2e-3, T_ref=10.0, S_uniform=35.0,
        alpha_T=_VEROS_BETA_T, rho_0=_VEROS_RHO_REF,
        U_surface=0.8, jet_width_deg=5.0, jet_depth_scale=5500.0,
        T_perturbation_K=0.1, perturbation_wavenumber=3,
        # Physics aligned with the Veros run: harmonic Laplacian only, no
        # biharmonic / Smag, linear bottom drag.
        A_h=5000.0, B_h=0.0, C_smag=0.0, K_h=0.0, K_bih=0.0,
        A_v=1.0e-5, K_v=0.0,
        bottom_drag_coeff=1.0e-5,
        tracer_advection="upwind",     # closest to Veros superbee monotone
    )


def _legoesm_latlon_cell_views(state, grid, z_coord, with_salt: bool):
    """Reduce a lat-lon C-grid state to (Ncols, Nz) cell-centred arrays."""
    T = np.asarray(state.T.data, dtype=np.float64)
    u_east = np.asarray(state.u.data, dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    n_lat, n_lon, n_z = T.shape
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    lat_deg = np.degrees(np.asarray(grid.lat))                      # (n_lat,)
    cos_lat = np.cos(np.radians(lat_deg))[:, None]                  # (n_lat, 1)
    area_2d = np.broadcast_to(cos_lat, (n_lat, n_lon)).astype(np.float64)
    # u sits on east faces; collapse to cell centres by averaging the
    # two adjacent face values. ``u_east`` shape (n_lat, n_lon+1, n_z);
    # the C-grid u_mask shares that shape.
    u_face_mask = np.asarray(state.u_mask.data, dtype=np.float64)
    u_face_safe = u_east * u_face_mask[..., None]
    u_cell = 0.5 * (u_face_safe[:, :-1, :] + u_face_safe[:, 1:, :])  # (n_lat, n_lon, n_z)
    usq_cell = u_cell ** 2
    Ncols = n_lat * n_lon
    T_flat = T.reshape(Ncols, n_z)
    S_flat = (np.asarray(state.S.data, dtype=np.float64).reshape(Ncols, n_z)
              if with_salt else None)
    u_flat = u_cell.reshape(Ncols, n_z)
    usq_flat = usq_cell.reshape(Ncols, n_z)
    w = (area_2d[..., None] * dz[None, None, :]
         * mask[..., None]).reshape(Ncols, n_z)
    dz_col = (dz[None, None, :] * mask[..., None]).sum(axis=2).reshape(Ncols)
    u_dz_col = (u_cell * dz[None, None, :] * mask[..., None]
                ).sum(axis=2).reshape(Ncols)
    area_col_flat = (area_2d * mask).reshape(Ncols)
    return T_flat, S_flat, u_flat, usq_flat, w, dz_col, u_dz_col, area_col_flat


def _legoesm_mpas_cell_views(state, mesh, z_coord, with_salt: bool):
    """Reduce an MPAS state to (Ncols, Nz) cell-centred arrays.

    ``u_cell`` and ``|u|^2_cell`` are reconstructed by averaging over the
    wet edges of each cell. This keeps the means comparable to the
    Veros side's collocated-velocity reduction.
    """
    T = np.asarray(state.T.data, dtype=np.float64)
    u_edge = np.asarray(state.u.data, dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    nCells, n_z = T.shape
    cellsOnEdge = np.asarray(mesh.cellsOnEdge, dtype=np.int64)
    edgesOnCell = np.asarray(mesh.edgesOnCell, dtype=np.int64)
    areaCell = np.asarray(mesh.areaCell, dtype=np.float64)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    c1 = cellsOnEdge[0, :]
    c2 = cellsOnEdge[1, :]
    edge_wet = ((mask[c1] > 0.5) & (mask[c2] > 0.5)).astype(np.float64)
    u_edge_masked = u_edge * edge_wet[:, None]
    u_sq_edge_masked = (u_edge_masked ** 2)
    # For each cell, average over its wet edges. edgesOnCell shape
    # (maxEdges, nCells); pad with -1 sentinels for cells with fewer
    # edges, so we clip indices and zero those contributions via
    # edge_wet masking.
    maxEdges = edgesOnCell.shape[0]
    edge_idx = np.where(edgesOnCell >= 0, edgesOnCell, 0)            # (maxEdges, nCells)
    valid = (edgesOnCell >= 0).astype(np.float64)
    u_gather = u_edge_masked[edge_idx]                               # (maxEdges, nCells, n_z)
    usq_gather = u_sq_edge_masked[edge_idx]
    valid_3d = valid[..., None]
    n_wet_edges = np.maximum(
        (valid * edge_wet[edge_idx]).sum(axis=0), 1e-12,
    )
    u_cell = (u_gather * valid_3d).sum(axis=0) / n_wet_edges[..., None]
    usq_cell = (usq_gather * valid_3d).sum(axis=0) / n_wet_edges[..., None]

    w = (areaCell[:, None] * dz[None, :] * mask[:, None]).astype(np.float64)
    dz_col = (dz[None, :] * mask[:, None]).sum(axis=1)
    u_dz_col = (u_cell * dz[None, :] * mask[:, None]).sum(axis=1)
    area_col = (areaCell * mask)
    S_flat = (np.asarray(state.S.data, dtype=np.float64) if with_salt
              else None)
    return T, S_flat, u_cell, usq_cell, w, dz_col, u_dz_col, area_col


def run_legoesm_eady_uniform(grid_type: str) -> dict[str, float]:
    """Run legoESM Eady-uniform on ``grid_type`` for 1 hour, return bulk metrics."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    from legoesm.ocean.experiments.eady_uniform import (
        create_initial_conditions, create_forcings,
    )

    cfg = _eady_legoesm_config()
    physics = create_forcings(grid_type, None, cfg)

    if grid_type == "latlon_channel":
        resolution = "30x30"
    elif grid_type == "mpas_channel":
        resolution = "70km"   # ~30 cells across 2000 km
    else:
        raise ValueError(grid_type)

    tc = TestCase(
        case="eady_uniform", grid_type=grid_type, resolution=resolution,
        duration_days=1.0, quick_days=1.0,
        run_kwargs=dict(
            lat_south=cfg.lat_south, lat_north=cfg.lat_north,
            lon_west=cfg.lon_west, lon_east=cfg.lon_east,
        ),
    )
    # Veros uses a uniform vertical grid (dzt = H_max / nz). Match it on
    # the legoESM side by overriding the stretched z_star coordinate.
    nlev = 20
    from legoesm.ocean.vertical import create_ocean_z_star
    dz_uniform = cfg.H_max / nlev
    z_coord_uniform = create_ocean_z_star(
        n_levels=nlev, H_max=cfg.H_max,
        dz_surface=dz_uniform, dz_deep=dz_uniform,
    )

    grid, z_coord, _, model, _, _, _ = _create_ocean_setup(
        tc, nlev=nlev, H_max=cfg.H_max, physics=physics,
        A_h=cfg.A_h, K_h=cfg.K_h, K_bih=cfg.K_bih,
        A_v=cfg.A_v, K_v=cfg.K_v,
        bottom_drag_r=cfg.bottom_drag_coeff,
        eos="linear", eos_linear=_veros_linear_eos_config(),
        tracer_advection=cfg.tracer_advection,
    )
    # Swap in the uniform z_coord on the model object after creation so we
    # don't have to plumb a `dz_surface`/`dz_deep` override through
    # ``_create_ocean_setup``. The model holds a z_coord attribute it uses
    # for tendencies; we replace it before stepping.
    model = type(model)(grid, z_coord_uniform, model.config)
    z_coord = z_coord_uniform
    state = create_initial_conditions(grid_type, grid, z_coord, cfg)

    dt = 1800.0
    n_steps = 2
    for _ in range(n_steps):
        state = model.step(state, dt)
    state = jax.block_until_ready(state)

    if grid_type == "latlon_channel":
        (T, _S, u, usq, w, dz_col, u_dz_col, area_col
         ) = _legoesm_latlon_cell_views(state, grid, z_coord, with_salt=False)
    else:   # mpas_channel
        (T, _S, u, usq, w, dz_col, u_dz_col, area_col
         ) = _legoesm_mpas_cell_views(state, grid, z_coord, with_salt=False)
    return _bulk_eady_metrics(T, usq, u, w, dz_col, u_dz_col, area_col)


# ---------------------------------------------------------------------------
# legoESM DINO: lat-lon Mercator + MPAS regional.
# ---------------------------------------------------------------------------

def run_legoesm_dino(grid_type: str, nx: int = 20, ny: int = 40,
                     nz: int = 12) -> dict[str, float]:
    """Run legoESM DINO on ``grid_type`` for 2 timesteps, return bulk metrics.

    Matches the Veros side's nx/ny/nz coarsening for a like-for-like
    bulk-metric comparison; the paper R1 grid is too expensive for an
    inline smoke test.
    """
    from legoesm.ocean.experiments.dino import (
        DINOConfig,
        dino_lat_lon_grid, dino_lat_lon_state, dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing,
        dino_mpas_state, dino_mpas_model_config,
        dino_mpas_surface_forcing_arrays, apply_dino_mpas_surface_forcing,
        create_dino_z_star,
    )

    cfg = DINOConfig()
    # For the cross-model comparison both Veros and legoESM run on a
    # *uniform* vertical grid (``dz = H_DEEP / nz``); Veros's ``uniform_z=True``
    # factory kwarg toggles the matching path on its side.
    from legoesm.ocean.vertical import create_ocean_z_star
    dz_uniform = cfg.H_deep / nz
    z_coord = create_ocean_z_star(
        n_levels=nz, H_max=cfg.H_deep,
        dz_surface=dz_uniform, dz_deep=dz_uniform,
    )

    if grid_type == "latlon":
        # Use a *uniform-dy* regional latlon grid (matching Veros's
        # uniform-dy convention) instead of the Mercator one, so the
        # area-weighted T_mean / T_max bulk metrics line up across the
        # two models. Both sides then sample the same latitudes.
        from legoesm.grids.latlon import create_regional_latlon_grid
        grid, _wall_mask = create_regional_latlon_grid(
            n_lat=ny, n_lon=nx,
            lat_south=-cfg.lat_max_deg, lat_north=cfg.lat_max_deg,
            lon_west=cfg.lon_west_deg, lon_east=cfg.lon_east_deg,
            periodic_x=True,
        )
        state = dino_lat_lon_state(grid, z_coord, cfg)
        model_cfg, _physics_cfg = dino_lat_lon_model_config(grid, cfg, physics=True)
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        model = LatLonCGridOceanModel(grid, z_coord, model_cfg)
        forcing = dino_lat_lon_surface_forcing_arrays(grid, cfg)
        taux_array = np.asarray(forcing["tau_u_face"])
        apply_forcing = lambda s, dt_: apply_dino_lat_lon_surface_forcing(
            s, forcing, z_coord, cfg, dt_,
        )
    elif grid_type == "mpas":
        from legoesm.grids.voronoi import create_regional_voronoi_mesh
        # 200 km resolution -> roughly 30 cells across the 50° basin.
        # The DINO partial-periodic seam-wall needs ``periodic_x=True``;
        # 500 km is the coarsest resolution that successfully seeds a
        # periodic regional mesh over the 50° basin without throwing
        # ``Some generators fall outside the periodic zonal extent``.
        mesh = create_regional_voronoi_mesh(
            (cfg.lon_west_deg, cfg.lon_east_deg),
            (-cfg.lat_max_deg, cfg.lat_max_deg),
            resolution_km=500, periodic_x=True,
        )
        state = dino_mpas_state(mesh, z_coord, cfg)
        model_cfg, _physics_cfg = dino_mpas_model_config(mesh, cfg, physics=True)
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        model = MPASOceanModel(mesh, z_coord, model_cfg)
        forcing = dino_mpas_surface_forcing_arrays(mesh, cfg)
        # MPAS ``tau_normal`` is τ_u·cos(angleEdge); for cross-grid bulk
        # comparison we want the *zonal* τ_u itself, which is what the
        # Veros side reports under ``surface_taux``. Evaluate the same
        # cubic-Hermite knot function at the MPAS edge latitudes to get
        # a like-for-like observable.
        from legoesm.ocean.experiments.dino import dino_wind_stress
        taux_array = np.asarray(
            dino_wind_stress(np.degrees(np.asarray(mesh.latEdge)), cfg)
        )
        apply_forcing = lambda s, dt_: apply_dino_mpas_surface_forcing(
            s, forcing, z_coord, cfg, dt_,
        )
        grid = mesh
    else:
        raise ValueError(grid_type)

    dt = 2700.0
    n_steps = 2
    for _ in range(n_steps):
        state = model.step(state, dt)
        state = apply_forcing(state, dt)
    state = jax.block_until_ready(state)

    if grid_type == "latlon":
        (T, S, u, _usq, w, dz_col, u_dz_col, area_col
         ) = _legoesm_latlon_cell_views(state, grid, z_coord, with_salt=True)
    else:
        (T, S, u, _usq, w, dz_col, u_dz_col, area_col
         ) = _legoesm_mpas_cell_views(state, grid, z_coord, with_salt=True)
    return _bulk_dino_metrics(T, S, u, w, dz_col, u_dz_col, area_col, taux_array)


# ---------------------------------------------------------------------------
# Comparison + reporting
# ---------------------------------------------------------------------------

@dataclass
class _Row:
    case: str
    metric: str
    veros: float
    lego: float
    rel_delta: float
    abs_delta: float
    within_tol: bool

    @property
    def status(self) -> str:
        return "PASS" if self.within_tol else "FAIL"


# Absolute tolerance floors used when the reference value is itself near
# zero (relative deltas blow up there even though both runs agree within
# their own noise level). Keys are metric names; values are dimensional
# floors with the metric's units.
_ABS_FLOOR: dict[str, float] = {
    "T_min": 0.5,        # K — IC has T_min near zero; 0.5 K is well within noise
    "T_max": 0.5,
    "T_mean": 0.5,
    "S_min": 0.05,
    "S_max": 0.05,
    "S_mean": 0.05,
    "u_abs_max": 0.01,   # m/s
    "u_mean": 0.005,
    "ke_mean": 1e-4,
    "taux_min": 0.005,   # N/m^2
    "taux_max": 0.005,
}


def _relative_delta(lego: float, ref: float, metric: str = "") -> tuple[float, float]:
    """Return (relative_delta, absolute_delta).

    For near-zero reference values the relative form is dominated by the
    chosen denominator. We use the maximum of ``|ref|`` and an
    metric-specific absolute floor to keep the relative figure
    interpretable in those cases.
    """
    if not np.isfinite(lego) or not np.isfinite(ref):
        return float("nan"), float("nan")
    abs_d = abs(lego - ref)
    floor = _ABS_FLOOR.get(metric, 1e-12)
    denom = max(abs(ref), floor)
    return abs_d / denom, abs_d


def _format_table(rows: list[_Row]) -> str:
    headers = ("case", "metric", "veros", "legoesm", "abs Δ", "rel Δ", "status")
    widths = [max(len(h), max((len(r.case),
                                len(r.metric),
                                len(f"{r.veros:.6g}"),
                                len(f"{r.lego:.6g}"),
                                len(f"{r.abs_delta:.3g}"),
                                len(f"{r.rel_delta * 100:.2f}%"),
                                len(r.status))[i] for r in rows) + 1)
              for i, h in enumerate(headers)]

    def _row(values):
        return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))

    out_lines = [_row(headers), _row(["-" * w for w in widths])]
    for r in rows:
        out_lines.append(_row([
            r.case, r.metric,
            f"{r.veros:.6g}", f"{r.lego:.6g}",
            f"{r.abs_delta:.3g}", f"{r.rel_delta * 100:.2f}%", r.status,
        ]))
    return "\n".join(out_lines)


def _markdown_table(rows: list[_Row], variant: str) -> str:
    out = [f"### {variant}",
           "",
           "| case | metric | Veros | legoESM | abs Δ | rel Δ | status |",
           "|------|--------|-------|---------|-------|-------|--------|"]
    for r in rows:
        out.append(
            f"| {r.case} | {r.metric} | {r.veros:.6g} | {r.lego:.6g} | "
            f"{r.abs_delta:.3g} | {r.rel_delta * 100:.2f}% | {r.status} |"
        )
    return "\n".join(out) + "\n"


def _compare(case: str, veros_metrics: dict[str, float],
             lego_metrics: dict[str, float], tolerance: float,
             ) -> list[_Row]:
    rows = []
    for metric, ref in veros_metrics.items():
        lego = lego_metrics.get(metric, float("nan"))
        rel, abs_d = _relative_delta(lego, ref, metric)
        floor = _ABS_FLOOR.get(metric, 0.0)
        rows.append(_Row(
            case=case, metric=metric, veros=ref, lego=lego,
            rel_delta=rel, abs_delta=abs_d,
            within_tol=(
                np.isfinite(rel)
                and (rel <= tolerance or abs_d <= floor)
            ),
        ))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="Relative delta tolerance (default 0.05 = 5%%).")
    p.add_argument("--cases", type=str, default="eady,dino",
                   help="Comma-separated subset of cases to run "
                        "(`eady`, `dino`, or both).")
    p.add_argument("--variants", type=str, default="latlon,mpas",
                   help="Comma-separated legoESM grid variants (`latlon`, `mpas`).")
    p.add_argument("--write-report", type=Path, default=None,
                   help="Optional path to write a Markdown report.")
    args = p.parse_args()

    cases = {c.strip() for c in args.cases.split(",") if c.strip()}
    variants = {v.strip() for v in args.variants.split(",") if v.strip()}

    all_rows: list[_Row] = []
    by_variant: dict[str, list[_Row]] = {}

    if "eady" in cases:
        print("==> Veros Eady-uniform reference run")
        t0 = time.time()
        veros_eady = run_veros_eady_uniform()
        print(f"   {time.time() - t0:.1f}s")

        if "latlon" in variants:
            print("==> legoESM Eady-uniform (lat-lon C-grid channel)")
            t0 = time.time()
            lego = run_legoesm_eady_uniform("latlon_channel")
            print(f"   {time.time() - t0:.1f}s")
            rows = _compare("eady_uniform/latlon", veros_eady, lego,
                            args.tolerance)
            all_rows.extend(rows)
            by_variant.setdefault("legoESM lat-lon FV vs Veros", []).extend(rows)

        if "mpas" in variants:
            print("==> legoESM Eady-uniform (MPAS channel)")
            t0 = time.time()
            lego = run_legoesm_eady_uniform("mpas_channel")
            print(f"   {time.time() - t0:.1f}s")
            rows = _compare("eady_uniform/mpas", veros_eady, lego,
                            args.tolerance)
            all_rows.extend(rows)
            by_variant.setdefault("legoESM MPAS vs Veros", []).extend(rows)

    if "dino" in cases:
        print("==> Veros DINO reference run (coarse)")
        t0 = time.time()
        veros_dino = run_veros_dino()
        print(f"   {time.time() - t0:.1f}s")

        if "latlon" in variants:
            print("==> legoESM DINO (lat-lon Mercator)")
            t0 = time.time()
            lego = run_legoesm_dino("latlon")
            print(f"   {time.time() - t0:.1f}s")
            rows = _compare("dino/latlon", veros_dino, lego, args.tolerance)
            all_rows.extend(rows)
            by_variant.setdefault("legoESM lat-lon FV vs Veros", []).extend(rows)

        if "mpas" in variants:
            print("==> legoESM DINO (MPAS regional)")
            t0 = time.time()
            lego = run_legoesm_dino("mpas")
            print(f"   {time.time() - t0:.1f}s")
            rows = _compare("dino/mpas", veros_dino, lego, args.tolerance)
            all_rows.extend(rows)
            by_variant.setdefault("legoESM MPAS vs Veros", []).extend(rows)

    print()
    print(_format_table(all_rows))

    n_fail = sum(1 for r in all_rows if not r.within_tol)
    print()
    print(f"=> {len(all_rows) - n_fail}/{len(all_rows)} metrics within "
          f"{args.tolerance * 100:.1f}% tolerance "
          f"({n_fail} outside).")

    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        md_lines = [
            "# legoESM vs Veros bulk-metric comparison",
            "",
            f"Tolerance: relative delta <= {args.tolerance * 100:.1f}%.",
            "",
        ]
        for variant_label, rows in by_variant.items():
            md_lines.append(_markdown_table(rows, variant_label))
        args.write_report.write_text("\n".join(md_lines))
        print(f"\nReport written to {args.write_report}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
