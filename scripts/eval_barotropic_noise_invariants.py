#!/usr/bin/env python
"""Evaluate Crit 1 invariants for the barotropic-mode-noise issue.

Loads a tendency_3d_means.npz produced by ``run_drake_momentum_budget*.py``
and computes the three invariants from
``docs/issues/barotropic_mode_noise.md``:

  1. var(div(U_baro)) / var(U_baro) on the time-mean field
  2. max |<V_baro>| outside polar caps
  3. sigma(grid-scale V_baro) after a 3-pt meridional Laplacian filter

V_baro is computed as the depth-mean of state_v_mean using dz_ref
weights (z-star-correct using the time-mean h_v would also work but
adds a Reynolds correction which is shown in
docs/research/zstar_vbaro_residual_investigation.md to be small at
this resolution).  Same for U_baro.

Usage:
    JAX_ENABLE_X64=1 python scripts/eval_barotropic_noise_invariants.py [path/to/tendency_3d_means.npz]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig


def _build_geometry():
    cfg = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    return cfg, grid, z_coord


def _depth_mean(field_3d, dz):
    H = float(dz.sum())
    return np.sum(field_3d * dz[None, None, :], axis=-1) / H


def _div_uv(U, V, grid):
    """Compute divergence of (U_baro, V_baro) at cell centres in 1/s.

    U on u-faces (n_lat, n_lon+1), V on v-faces (n_lat+1, n_lon).
    Uses the same FV stencil as ``divergence_cgrid``:
        div = (u_E*dy - u_W*dy + v_N*dx_N - v_S*dx_S) / area
    """
    lat = np.asarray(grid.lat, dtype=np.float64)
    dlat = float(grid.dlat); dlon = float(grid.dlon)
    R = float(grid.radius)
    cos_lat_c = np.cos(np.clip(lat, -np.pi/2 + 1e-9, np.pi/2 - 1e-9))
    lat_v = np.concatenate([
        [lat[0] - 0.5 * dlat],
        0.5 * (lat[:-1] + lat[1:]),
        [lat[-1] + 0.5 * dlat],
    ])
    cos_lat_v = np.cos(np.clip(lat_v, -np.pi/2 + 1e-9, np.pi/2 - 1e-9))
    dx_v = cos_lat_v * R * dlon
    dy = R * dlat
    area = (cos_lat_c * R * dlon * dy)[:, None]
    u_W = U[:, :-1]; u_E = U[:, 1:]
    v_S = V[:-1, :]; v_N = V[1:, :]
    div = (
        (u_E - u_W) * dy
        + v_N * dx_v[1:, None] - v_S * dx_v[:-1, None]
    ) / area
    return div


def _meridional_laplacian(field):
    """3-pt meridional Laplacian filter (high-pass)."""
    out = np.zeros_like(field)
    out[1:-1] = field[2:] - 2.0 * field[1:-1] + field[:-2]
    return out


def _format(name, value, threshold, ok):
    flag = "PASS" if ok else "FAIL"
    return f"  [{flag}] {name:<60} {value:11.4e}  (target {threshold})"


def evaluate(npz_path: Path) -> dict:
    cfg, grid, z_coord = _build_geometry()
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180.0 / np.pi
    lat_v_deg = np.concatenate([
        [lat_deg[0] - 0.5 * float(grid.dlat) * 180.0 / np.pi],
        0.5 * (lat_deg[:-1] + lat_deg[1:]),
        [lat_deg[-1] + 0.5 * float(grid.dlat) * 180.0 / np.pi],
    ])

    d = np.load(npz_path, allow_pickle=False)
    u_mean = d["state_u_mean"]   # (n_lat, n_lon+1, nlev)
    v_mean = d["state_v_mean"]   # (n_lat+1, n_lon, nlev)

    U_baro = _depth_mean(u_mean, dz)        # (n_lat, n_lon+1)
    V_baro = _depth_mean(v_mean, dz)        # (n_lat+1, n_lon)

    # u/v masks: get from the restart used by the run
    restart = np.load(
        Path("results/ocean/global_overturning_50yr_gmredi") /
        "restart_day018250.npz",
        allow_pickle=False,
    )
    u_mask = np.asarray(restart["u_mask"], dtype=np.float64)
    v_mask = np.asarray(restart["v_mask"], dtype=np.float64)
    land_mask = np.asarray(restart["land_mask"], dtype=np.float64)
    U_baro = U_baro * u_mask
    V_baro = V_baro * v_mask

    # ---- Crit 1.1: var(div(U_baro)) / var(U_baro) ----
    div = _div_uv(U_baro, V_baro, grid) * land_mask
    # Restrict variance to wet ocean cells.
    wet_cells = land_mask > 0.5
    wet_u = u_mask > 0.5
    wet_v = v_mask > 0.5
    var_div = float(np.var(div[wet_cells]))
    var_U = float(np.var(U_baro[wet_u]))
    ratio = var_div / max(var_U, 1e-30)

    # ---- Crit 1.2: max |<V_baro>| outside polar caps ----
    # "Polar cap" => |lat_v| > 70 deg.
    polar = np.abs(lat_v_deg) > 70.0
    interior = ~polar
    if np.any(interior):
        V_int = np.where(v_mask > 0.5, np.abs(V_baro), 0.0)
        max_V = float(np.max(V_int[interior, :]))
    else:
        max_V = 0.0

    # ---- Crit 1.3: sigma(grid-scale V_baro) after 3-pt meridional Laplacian ----
    V_lap = _meridional_laplacian(V_baro * v_mask)
    interior_v = (v_mask > 0.5)
    interior_v[0, :] = False
    interior_v[-1, :] = False
    interior_v[polar, :] = False
    V_lap_in = V_lap[interior_v]
    sigma_lap = float(np.std(V_lap_in)) if V_lap_in.size > 0 else 0.0

    metrics = {
        "var_div_U_over_var_U": ratio,
        "max_abs_V_baro_off_polar": max_V,
        "sigma_3pt_laplacian_V_baro": sigma_lap,
    }
    return metrics


def main():
    if len(sys.argv) >= 2:
        npz_path = Path(sys.argv[1])
    else:
        npz_path = Path(
            "results/ocean/momentum_budget_online/tendency_3d_means.npz")
    if not npz_path.is_file():
        print(f"Error: {npz_path} does not exist", file=sys.stderr)
        sys.exit(1)

    metrics = evaluate(npz_path)
    print(f"\n=== Barotropic noise invariants — {npz_path} ===")
    for k, v in metrics.items():
        print(f"  {k:<40} {v:.4e}")

    print("\nCrit 1 acceptance thresholds:")
    print(_format(
        "var(div(U_baro)) / var(U_baro)  < 0.05",
        metrics["var_div_U_over_var_U"], "0.05",
        metrics["var_div_U_over_var_U"] < 0.05,
    ))
    print(_format(
        "max |<V_baro>|  off polar caps  < 5.0e-3 m/s",
        metrics["max_abs_V_baro_off_polar"], "5e-3",
        metrics["max_abs_V_baro_off_polar"] < 5.0e-3,
    ))
    print(_format(
        "sigma(3-pt Laplacian V_baro)    < 1.0e-2 m/s",
        metrics["sigma_3pt_laplacian_V_baro"], "1e-2",
        metrics["sigma_3pt_laplacian_V_baro"] < 1.0e-2,
    ))


if __name__ == "__main__":
    main()
