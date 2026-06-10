#!/usr/bin/env python
"""Side-by-side MOC streamfunction Ψ(φ, z) and zonal-mean w(φ, z) for the
lat-lon vs MPAS year-50 global overturning runs.

Diagnoses whether the apparent T-meridional-structure difference between
the grids is driven by:
  - a weaker meridional cell on MPAS (higher A_h damping Ekman flow), or
  - similar circulation but smoothed T (GM/Redi centered vs triads), or
  - a vertical-transport bug on MPAS (sign flip, pattern mismatch, etc.).

Outputs:
  results/ocean/cross_grid_comparison/moc_yr050.png        (Ψ panels)
  results/ocean/cross_grid_comparison/wbar_yr050.png       (⟨w⟩ panels)
  results/ocean/cross_grid_comparison/moc_w_summary.txt    (max/min table)

Both grids binned to a common 36-point latitude axis (5° bins, matching
lat-lon native resolution) so the panels are directly comparable.

Usage:
    JAX_ENABLE_X64=1 python scripts/run/global_overturning/compare_moc_and_w_mpas_vs_latlon.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm import constants  # noqa: E402
from ocean_test_matrix.regridding import _bin_to_latlon  # noqa: E402


LATLON_RESTART = Path(
    "results/ocean/global_overturning_50yr_implicit/restart_day018250.npz"
)
MPAS_RESTART = Path(
    "results/ocean/global_overturning_mpas_50yr_implicit_dt600/restart_day018250.npz"
)
OUTPUT_DIR = Path("results/ocean/cross_grid_comparison")

# Common 36-bin lat axis (matches lat-lon 36×72 grid: -87.5, -82.5, …, +87.5)
LAT_BIN_CENTRES = np.linspace(-87.5, 87.5, 36)
LAT_BIN_WIDTH_DEG = 5.0
LAT_BIN_HALFWIDTH_DEG = LAT_BIN_WIDTH_DEG / 2.0

R_EARTH = constants.R_earth  # m
N_LEVELS = 20
H_MAX = 4000.0
DZ_SURFACE = 10.0
DZ_DEEP = 500.0


def _zonal_v_transport_latlon(v, v_mask, dz, lat_v_rad):
    """Zonal-line v-transport per layer at each v-face: sum_i v_i * dx_i * mask.

    v       : (nlat_v, nlon, nlev)  v-faces (nlat_v = nlat+1 typical)
    v_mask  : (nlat_v, nlon)        wet-face mask
    dz      : (nlev,)               layer thickness
    lat_v_rad : (nlat_v,)           v-face latitudes (radians)

    Returns
    -------
    V_per_layer : (nlat_v, nlev)    zonal-line transport per layer (m^3/s/m_depth)
                                    actually integrated_over_lon * v = m^2/s
    Note: multiplied by dz gives m^3/s.  We return the per-layer transport
    (sum_i v_i * dx_i * v_mask), with units m^2/s · m = m^3/s once × dz.
    """
    cos_lat = np.cos(lat_v_rad)
    dlon = 2.0 * np.pi / v.shape[1]
    dx = R_EARTH * cos_lat[:, None] * dlon  # (nlat_v, 1) broadcast over lon
    # Apply mask, sum over longitude
    v_masked = v * v_mask[:, :, None]
    V = np.sum(v_masked * dx[..., None], axis=1)   # (nlat_v, nlev)
    return V


def _moc_from_V(V_per_layer, dz):
    """Cumulative depth integral from the bottom up.

    V_per_layer : (nlat, nlev)   zonal-line v-transport per layer (m^3/s)
                                  needs ×dz -> volume transport
    Returns Psi(nlat, nlev_face) where nlev_face = nlev + 1 (top + interfaces + bottom)
    """
    # Volume transport per layer = V_per_layer × dz
    Vol_per_layer = V_per_layer * dz[None, :]   # (nlat, nlev)
    # Cumulate from bottom up — Ψ at top of layer k = sum from k to nlev-1
    # Ψ(z=bottom) = 0, Ψ(z=top of bottom layer) = Vol[..., -1], ...
    Psi = np.zeros((V_per_layer.shape[0], V_per_layer.shape[1] + 1))
    # Ψ at interface above layer k = Ψ at interface below layer k + Vol[k]
    # Iterate from bottom: Psi[k] = Psi[k+1] + Vol[k]  (since indexing top = 0)
    # Convention: Psi has nlev+1 interfaces; Psi[0] = top of column, Psi[nlev] = bottom = 0
    for k in range(V_per_layer.shape[1] - 1, -1, -1):
        Psi[:, k] = Psi[:, k + 1] + Vol_per_layer[:, k]
    return Psi   # m^3/s, divide by 1e6 for Sv


def _wbar_from_psi(Psi, lat_centres_deg):
    return _wbar_from_psi_var_dy(Psi, lat_centres_deg, LAT_BIN_WIDTH_DEG)


def _wbar_from_psi_var_dy(Psi, lat_centres_deg, bin_width_deg):
    """⟨w⟩(φ, z) = (1/Lx) ∂Ψ/∂y, with Lx = 2π R cos φ."""
    lat_rad = np.deg2rad(lat_centres_deg)
    dy = R_EARTH * np.deg2rad(bin_width_deg)
    Lx = 2.0 * np.pi * R_EARTH * np.cos(lat_rad)
    dPsi_dy = np.zeros_like(Psi)
    dPsi_dy[1:-1] = (Psi[2:] - Psi[:-2]) / (2.0 * dy)
    dPsi_dy[0] = (Psi[1] - Psi[0]) / dy
    dPsi_dy[-1] = (Psi[-1] - Psi[-2]) / dy
    wbar = dPsi_dy / np.where(Lx[:, None] > 0, Lx[:, None], np.nan)
    return wbar


def _compute_latlon():
    print(f"\n--- Lat-lon: {LATLON_RESTART} ---")
    d = np.load(LATLON_RESTART, allow_pickle=False)
    eta = d["eta"]; T = d["T"]; v = d["v"]
    v_mask = np.asarray(d["v_mask"], dtype=np.float64)
    print(f"  shapes: T={T.shape}, v={v.shape}, v_mask={v_mask.shape}")

    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.grids.latlon import create_latlon_grid
    z_coord = create_ocean_z_star(
        n_levels=N_LEVELS, H_max=H_MAX,
        dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP)
    grid = create_latlon_grid(36, 72)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    # v lives at v-faces, between cells in latitude.  Lat of v-faces:
    # If lat-cell-centres are at i_centre, v-faces are at midpoints.
    # The v_mask shape tells us how many v-faces there are.
    lat_c_rad = np.asarray(grid.lat)
    if v_mask.shape[0] == lat_c_rad.size + 1:
        # v-faces at edges (one more than cells)
        lat_v_rad = np.concatenate([
            [lat_c_rad[0] - (lat_c_rad[1] - lat_c_rad[0]) / 2],
            0.5 * (lat_c_rad[:-1] + lat_c_rad[1:]),
            [lat_c_rad[-1] + (lat_c_rad[-1] - lat_c_rad[-2]) / 2],
        ])
    else:
        lat_v_rad = lat_c_rad   # fallback

    V_per_layer = _zonal_v_transport_latlon(v, v_mask, dz, lat_v_rad)
    Psi = _moc_from_V(V_per_layer, dz)   # (nlat_v, nlev+1)

    # Bin Ψ to the common 36-bin latitude axis by interpolating each depth interface
    lat_v_deg = lat_v_rad * 180.0 / np.pi
    Psi_binned = np.zeros((len(LAT_BIN_CENTRES), Psi.shape[1]))
    for k in range(Psi.shape[1]):
        Psi_binned[:, k] = np.interp(LAT_BIN_CENTRES, lat_v_deg, Psi[:, k])

    wbar = _wbar_from_psi(Psi_binned, LAT_BIN_CENTRES)
    print(f"  Ψ range: [{Psi_binned.min()/1e6:.1f}, {Psi_binned.max()/1e6:.1f}] Sv")
    print(f"  ⟨w⟩ range: [{wbar.min()*86400:.2e}, {wbar.max()*86400:.2e}] m/day")

    # Depth interfaces: top of layer k=0 is z=0, bottom of layer k=nlev-1 is z_full[-1]
    z_interfaces = np.concatenate([[0.0], z_full + dz / 2.0])  # rough; not critical for plotting
    # Better: build z_interfaces from dz cumulative
    z_top = np.cumsum(np.concatenate([[0.0], -dz]))   # 0, -dz0, -(dz0+dz1), ...
    return {
        "Psi": Psi_binned,
        "wbar": wbar,
        "lat_centres": LAT_BIN_CENTRES,
        "z_interfaces": z_top,    # length nlev+1
        "z_full": z_full,
    }


def _compute_mpas(bin_width_deg=5.0):
    print(f"\n--- MPAS (bin width {bin_width_deg}°): {MPAS_RESTART} ---")
    d = np.load(MPAS_RESTART, allow_pickle=False)
    T = d["T"]; u_edges = d["u"]
    sub_level = int(d.get("mpas_subdivision_level", 4))
    print(f"  shapes: T={T.shape}, u_edges={u_edges.shape}, sub_level={sub_level}")

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.vertical import create_ocean_z_star
    import jax.numpy as jnp
    from legoesm.ocean.init_mpas import reconstruct_cell_velocity
    mesh = create_voronoi_mesh(sub_level)
    z_coord = create_ocean_z_star(
        n_levels=N_LEVELS, H_max=H_MAX,
        dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)

    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edges), mesh)
    u_east = np.asarray(u_east); v_north = np.asarray(v_north)

    lat_deg = np.asarray(mesh.latCell) * 180.0 / np.pi
    cell_area = np.asarray(mesh.areaCell)   # m^2
    is_ocean = np.asarray(d["land_mask"]) > 0.5

    # Bin centres for the chosen width — span same -87.5..+87.5 range
    half = bin_width_deg / 2.0
    bin_centres = np.arange(-90.0 + half, 90.0, bin_width_deg)
    nlat = len(bin_centres)
    nlev = T.shape[1]
    V_per_layer = np.zeros((nlat, nlev))   # m^2/s
    dy_bin = R_EARTH * np.deg2rad(bin_width_deg)
    cells_per_bin = np.zeros(nlat, dtype=int)
    for i, c in enumerate(bin_centres):
        in_bin = is_ocean & (lat_deg >= c - half) & (lat_deg < c + half)
        cells_per_bin[i] = int(in_bin.sum())
        if in_bin.any():
            V_per_layer[i, :] = np.sum(
                v_north[in_bin, :] * cell_area[in_bin, None], axis=0
            ) / dy_bin

    Psi = _moc_from_V(V_per_layer, dz)   # (nlat, nlev+1) m^3/s
    wbar = _wbar_from_psi_var_dy(Psi, bin_centres, bin_width_deg)

    print(f"  Ψ range: [{Psi.min()/1e6:.1f}, {Psi.max()/1e6:.1f}] Sv")
    print(f"  ⟨w⟩ range: [{wbar.min()*86400:.2e}, {wbar.max()*86400:.2e}] m/day")
    print(f"  cells per bin: min={cells_per_bin.min()}, "
          f"mean={cells_per_bin.mean():.1f}, max={cells_per_bin.max()}")

    z_top = np.cumsum(np.concatenate([[0.0], -dz]))
    return {
        "Psi": Psi,
        "wbar": wbar,
        "lat_centres": bin_centres,
        "z_interfaces": z_top,
        "z_full": z_full,
    }


def _plot_panels(panels, field_key, title, units, cmap, levels, outpath,
                  *, sym=True, scale=1.0, level_step_for_label=None):
    """Generic N-panel comparison plot.  panels = list of (label, results dict)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (label, res) in zip(axes, panels):
        F = res[field_key] * scale
        depth = -res["z_interfaces"]
        lat_axis = res["lat_centres"]
        cf = ax.pcolormesh(lat_axis, depth, F.T,
                            vmin=-levels[-1] if sym else levels[0],
                            vmax=levels[-1],
                            cmap=cmap, shading="auto")
        cs = ax.contour(lat_axis, depth, F.T,
                         levels=levels, colors="k", linewidths=0.4,
                         alpha=0.5)
        if sym:
            ax.contour(lat_axis, depth, F.T,
                        levels=[0], colors="k", linewidths=1.0)
        if level_step_for_label is not None:
            cs_label = ax.contour(lat_axis, depth, F.T,
                                    levels=levels[::level_step_for_label],
                                    colors="k", linewidths=0)
            ax.clabel(cs_label, inline=True, fontsize=7, fmt="%g")
        else:
            ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
        ax.invert_yaxis()
        ax.set_xlabel("Latitude (°)")
        ax.set_title(f"{label}: {title}\n[{F.min():+.2f}, {F.max():+.2f}] {units}",
                       fontsize=10)
        ax.grid(alpha=0.2)
        plt.colorbar(cf, ax=ax, fraction=0.045, label=units)
    axes[0].set_ylabel("Depth (m)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LATLON_RESTART.exists():
        print(f"Lat-lon restart not found: {LATLON_RESTART}", file=sys.stderr)
        sys.exit(1)
    if not MPAS_RESTART.exists():
        print(f"MPAS restart not found: {MPAS_RESTART}", file=sys.stderr)
        sys.exit(1)

    res_ll = _compute_latlon()
    res_mp_5 = _compute_mpas(bin_width_deg=5.0)
    res_mp_10 = _compute_mpas(bin_width_deg=10.0)

    # Common Ψ scale: use combined max for direct comparison
    psi_max = max(np.abs(res_ll["Psi"]).max(),
                   np.abs(res_mp_5["Psi"]).max(),
                   np.abs(res_mp_10["Psi"]).max()) / 1e6
    psi_max = float(np.ceil(psi_max / 5.0) * 5.0)
    psi_levels = np.linspace(-psi_max, psi_max, 21)

    wmax_mday = max(np.abs(res_ll["wbar"]).max(),
                     np.abs(res_mp_5["wbar"]).max(),
                     np.abs(res_mp_10["wbar"]).max()) * 86400.0
    wmax_mday = float(np.ceil(wmax_mday * 10) / 10)
    w_levels = np.linspace(-wmax_mday, wmax_mday, 21)

    # ---- MOC: 2-panel (existing layout) ----
    _plot_panels(
        [("Lat-lon", res_ll), ("MPAS (5° bin)", res_mp_5)],
        "Psi", "MOC streamfunction Ψ (Sv)",
        "Sv", "RdBu_r", psi_levels,
        OUTPUT_DIR / "moc_yr050.png",
        sym=True, scale=1.0 / 1e6, level_step_for_label=4)
    _plot_panels(
        [("Lat-lon", res_ll), ("MPAS (5° bin)", res_mp_5)],
        "wbar", "Zonal-mean ⟨w⟩",
        "m/day", "RdBu_r", w_levels,
        OUTPUT_DIR / "wbar_yr050.png",
        sym=True, scale=86400.0, level_step_for_label=4)

    # ---- 3-panel: lat-lon, MPAS 5°, MPAS 10° ----
    _plot_panels(
        [("Lat-lon (5°)", res_ll),
         ("MPAS (5° bin)", res_mp_5),
         ("MPAS (10° bin)", res_mp_10)],
        "Psi", "MOC streamfunction Ψ (Sv)",
        "Sv", "RdBu_r", psi_levels,
        OUTPUT_DIR / "moc_yr050_binwidth_test.png",
        sym=True, scale=1.0 / 1e6, level_step_for_label=4)
    _plot_panels(
        [("Lat-lon (5°)", res_ll),
         ("MPAS (5° bin)", res_mp_5),
         ("MPAS (10° bin)", res_mp_10)],
        "wbar", "Zonal-mean ⟨w⟩",
        "m/day", "RdBu_r", w_levels,
        OUTPUT_DIR / "wbar_yr050_binwidth_test.png",
        sym=True, scale=86400.0, level_step_for_label=4)

    # ---- Summary table ----
    summary_path = OUTPUT_DIR / "moc_w_summary.txt"
    with open(summary_path, "w") as f:
        f.write("MOC and zonal-mean-w comparison, year 50\n")
        f.write("=" * 60 + "\n")
        for label, res in [("Lat-lon (5°)", res_ll),
                            ("MPAS ico4 (5° bin)", res_mp_5),
                            ("MPAS ico4 (10° bin)", res_mp_10)]:
            psi_sv = res["Psi"] / 1e6
            w_mday = res["wbar"] * 86400.0
            f.write(f"\n{label}:\n")
            f.write(f"  Ψ:          [{psi_sv.min():+8.2f}, {psi_sv.max():+8.2f}] Sv\n")
            f.write(f"  Ψ peak abs: {np.abs(psi_sv).max():>8.2f} Sv\n")
            f.write(f"  ⟨w⟩:        [{w_mday.min():+8.3f}, {w_mday.max():+8.3f}] m/day\n")
            f.write(f"  ⟨w⟩ peak abs: {np.abs(w_mday).max():>8.3f} m/day\n")
        # Quantify lumpiness via std of ∂Ψ/∂y (proxied by ⟨w⟩ variability)
        f.write("\n--- Lumpiness diagnostic (std of ⟨w⟩ across all (lat, depth)): ---\n")
        for label, res in [("Lat-lon (5°)", res_ll),
                            ("MPAS ico4 (5° bin)", res_mp_5),
                            ("MPAS ico4 (10° bin)", res_mp_10)]:
            w_mday = res["wbar"] * 86400.0
            f.write(f"  {label:<22s} std=|{np.std(w_mday):.4f}| m/day, "
                    f"peak=|{np.abs(w_mday).max():.4f}| m/day\n")
        f.write("\nIf MPAS 10° has much smaller std than MPAS 5°, the 5° bin "
                "noise is mostly\nbinning artifact (irregular Voronoi sampling).\n"
                "If MPAS 10° still larger than lat-lon, residual TRiSK/Perot "
                "real noise.\n")
    print(f"\nSaved {summary_path}")
    with open(summary_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
