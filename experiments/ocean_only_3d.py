"""Ocean-only 3D legoESM run — dynamics physics diagnostic (NO BGC).

Same grid, land mask, and forcing as ``npzd_3d_ocean.py`` but runs only the
LatLonCGridOceanModel so the temperature/salinity/velocity physics can be
inspected in isolation, without any biogeochemistry coupling.

Physics
-------
- LatLonCGridOceanModel: split-explicit baroclinic/barotropic, C-grid
  advection (DST3), linear EOS, KPP-style vertical mixing, GM/Redi,
  convective adjustment, surface T restoring + wind stress.

Output
------
  experiments/ocean_only_T.png    — temperature maps at 8 depths
  experiments/ocean_only_S.png    — salinity maps at 8 depths
  experiments/ocean_only_speed.png— horizontal current speed at 8 depths
  experiments/ocean_only_sections.png — zonal-mean T and S sections
  experiments/ocean_only_means.npy    — saved annual-mean fields

Usage
-----
    JAX_ENABLE_X64=1 .venv/bin/python experiments/ocean_only_3d.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import (
    rest_state_latlon_cgrid_ocean,
    replace_land_mask,
    idealized_bathymetry_latlon_cgrid,
)
from legoesm.ocean.init_woa import init_ocean_from_woa
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    create_forcings as go_forcings,
    create_eos_config as go_eos,
)
from legoesm.ocean.experiments.global_barotropic_wind import (
    create_simplified_continent_mask,
)

# =============================================================================
# 1. GRID
# =============================================================================
N_LAT, N_LON = 36, 72        # 5-degree global grid
N_LEV = 20
H_MAX = 4000.0
DT    = 1800.0               # 30-min timestep
DAYS  = 365 * 5              # 5-year spin-up

print(f"Grid: {N_LAT}×{N_LON}, {N_LEV} levels, H_max={H_MAX} m")

grid   = create_latlon_grid(N_LAT, N_LON)
z_coord = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX,
                               dz_surface=10.0, dz_deep=400.0)

# =============================================================================
# 2. LAND MASK & BATHYMETRY
# =============================================================================
go_cfg = GlobalOverturningConfig(H_max=H_MAX, n_levels=N_LEV)

# Flat bottom, polar caps closed above 80°
H_bathy, _land_flat = idealized_bathymetry_latlon_cgrid(grid, H_max=H_MAX,
                                                         land_lat_threshold=80.0)

# Simplified continent (Atlantic-like) + Drake Passage open
lon_deg = np.degrees(np.asarray(grid.lon2d))
lat_deg = np.degrees(np.asarray(grid.lat2d))
cont_mask = create_simplified_continent_mask(lon_deg, lat_deg, go_cfg)
land_mask  = jnp.array(cont_mask.astype(np.float64))

# =============================================================================
# 3. INITIAL OCEAN STATE (WOA18 analytical T/S)
# =============================================================================
print("Initialising ocean state (WOA18 analytical T/S)…")
state = rest_state_latlon_cgrid_ocean(
    grid, z_coord,
    T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
)
state = replace_land_mask(state, land_mask)

# Apply WOA18 analytical T/S profiles (analytical fallback, no files needed)
try:
    from legoesm.core.field import Field
    T_woa, S_woa = init_ocean_from_woa(grid, z_coord,
                                        T_path=None, S_path=None)
    state = state._replace(
        T=Field(jnp.array(T_woa), name="T", dims=state.T.dims, units=state.T.units),
        S=Field(jnp.array(S_woa), name="S", dims=state.S.dims, units=state.S.units),
    )
    print("  WOA18 analytical T/S applied.")
except Exception as e:
    print(f"  WOA18 init skipped ({e}); using rest state T/S.")

# =============================================================================
# 4. PHYSICS / DYNAMICS CONFIG
# =============================================================================
import dataclasses as _dc
go_cfg_strong_restore = _dc.replace(go_cfg, tau_T_days=5.0)
physics = go_forcings("latlon", grid, go_cfg_strong_restore)
eos_cfg = go_eos(go_cfg)

model_cfg = LatLonCGridOceanConfig(
    n_barotropic_substeps=30,
    physics=physics,
    A_h=2e5,
    K_h=1e3,
    A_v=1e-3,
    K_v=1e-5,
    bottom_drag_r=1.1e-3,
    eos="linear",
    eos_linear=eos_cfg,
    tracer_advection="dst3",
)
model = LatLonCGridOceanModel(grid, z_coord, model_cfg)
print("Ocean model built.")

# =============================================================================
# 5. TIME LOOP — 5-year spin-up, accumulate running mean in last year
# =============================================================================
land_mask_2d = jnp.array(land_mask)

n_steps     = int(DAYS * 86400.0 / DT)
save_start  = int((DAYS - 365) * 86400.0 / DT)   # last-year start step
save_every  = max(1, int(5 * 86400.0 / DT))       # every 5 days

_shape = (N_LAT, N_LON, N_LEV)
sum_T = np.zeros(_shape)
sum_S = np.zeros(_shape)
sum_U = np.zeros(_shape)
sum_V = np.zeros(_shape)
n_saved = 0

print(f"\nRunning {DAYS // 365}-year ocean-only spin-up: {n_steps} steps (dt={DT}s)…")
t0_wall = time.time()

for k in range(n_steps):
    t_days = k * DT / 86400.0

    # ---- Ocean dynamics only ----
    state = model.step(state, dt=DT)

    # ---- Accumulate last-year running mean ----
    if k >= save_start and k % save_every == 0:
        sum_T += np.asarray(state.T.data)
        sum_S += np.asarray(state.S.data)
        sum_U += np.asarray(state.u.data)
        sum_V += np.asarray(state.v.data)
        n_saved += 1

    if (k + 1) % max(1, n_steps // 50) == 0:
        jax.block_until_ready((state.T.data, state.S.data,
                               state.u.data, state.v.data))
        jax.clear_caches()
        gc.collect()
    if (k + 1) % max(1, n_steps // 20) == 0:
        pct = 100.0 * (k + 1) / n_steps
        elapsed = time.time() - t0_wall
        _T = np.asarray(state.T.data)
        _S = np.asarray(state.S.data)
        print(f"  {pct:5.1f}%  day {t_days:7.1f}  elapsed {elapsed:.0f}s"
              f"  T[min/mean/max]={np.nanmin(_T):.1f}/{np.nanmean(_T):.1f}/{np.nanmax(_T):.1f}"
              f"  S mean={np.nanmean(_S):.2f}")

elapsed = time.time() - t0_wall
print(f"Done in {elapsed:.1f}s  ({elapsed / n_steps * 1000:.1f} ms/step).")
print(f"Accumulated {n_saved} snapshots for annual mean.")

# =============================================================================
# 6. ANNUAL MEAN (last year)
# =============================================================================
T_ann = sum_T / n_saved
S_ann = sum_S / n_saved
U_ann = sum_U / n_saved
V_ann = sum_V / n_saved
speed_ann = np.sqrt(U_ann**2 + V_ann**2)

np.save("experiments/ocean_only_means.npy",
        {"T": T_ann, "S": S_ann, "U": U_ann, "V": V_ann, "speed": speed_ann})
print("Annual means saved to experiments/ocean_only_means.npy")

# =============================================================================
# 7. PLOTTING — 2×4 panel maps at 8 key depths
# =============================================================================
ocean_mask_2d_np = np.asarray(land_mask)          # 1=ocean, 0=land
ocean_mask_3d = ocean_mask_2d_np[:, :, np.newaxis] * np.ones((1, 1, N_LEV))

def _mask_land(arr3d):
    out = arr3d.copy().astype(float)
    out[ocean_mask_3d < 0.5] = np.nan
    return out

T_ann     = _mask_land(T_ann)
S_ann     = _mask_land(S_ann)
speed_ann = _mask_land(speed_ann)

target_depths_m = [5, 25, 50, 100, 200, 500, 1000, 2000]
z_centers = np.abs(np.array(z_coord.z_full_ref))
depth_idx  = [int(np.argmin(np.abs(z_centers - d))) for d in target_depths_m]
depth_labels_sel = [f"{int(z_centers[k])} m" for k in depth_idx]

specs = [
    ("T",     T_ann,     "Temperature (°C)",              "RdYlBu_r", False),
    ("S",     S_ann,     "Salinity (psu)",                "viridis",  False),
    ("speed", speed_ann, "Current speed (m s$^{-1}$)",    "magma",    False),
]

proj     = ccrs.PlateCarree()
data_crs = ccrs.PlateCarree()
ncols, nrows = 4, 2
out_dir = "experiments"
os.makedirs(out_dir, exist_ok=True)

for key, data_3d, title, cmap, diverging in specs:
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.5, nrows * 3.2),
        subplot_kw=dict(projection=proj),
        gridspec_kw=dict(hspace=0.45, wspace=0.08),
    )

    # Roll data by N_LON//2 so lon-0 grid (0→360°) maps to -180→180° extent
    data_3d_rolled = np.roll(data_3d, N_LON // 2, axis=1)

    for panel_i, (k, ax) in enumerate(zip(depth_idx, axes.flatten())):
        data_k = data_3d_rolled[:, :, k]
        valid = data_k[np.isfinite(data_k)]
        if valid.size == 0:
            ax.set_visible(False)
            continue
        p1  = float(np.nanpercentile(data_k, 2))
        p99 = float(np.nanpercentile(data_k, 98))
        if key == "T":
            vmin = max(p1, -2.0)
            vmax = min(p99, 30.0)
        elif key == "S":
            vmin, vmax = p1, p99
        else:
            vmin = 0.0
            vmax = max(p99, 1e-6)

        im = ax.imshow(
            data_k, origin="lower",
            extent=[-180, 180, -90, 90],
            cmap=plt.get_cmap(cmap).with_extremes(bad="#c8c8c8"),
            vmin=vmin, vmax=vmax,
            transform=data_crs, interpolation="nearest",
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.set_extent([-180, 180, -90, 90], crs=data_crs)
        ax.gridlines(linewidth=0.3, color="grey", alpha=0.5,
                     xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30))
        ax.set_title(depth_labels_sel[panel_i], fontsize=10, fontweight="bold", pad=3)
        cb = plt.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.05, shrink=0.9, aspect=25)
        cb.ax.tick_params(labelsize=6)
        cb.ax.xaxis.set_major_formatter(
            plt.matplotlib.ticker.FormatStrFormatter("%.2g"))

    fig.suptitle(
        f"{title}\nlegoESM 3-D ocean-only  —  year-5 annual mean  "
        f"({N_LAT}×{N_LON}, {N_LEV} lev)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out = os.path.join(out_dir, f"ocean_only_{key}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# =============================================================================
# 8. Zonal-mean sections (depth vs latitude) for T and S
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lat_deg_1d = np.degrees(np.asarray(grid.lat))
depths = np.abs(np.array(z_coord.z_full_ref))

for ax, data_3d, label, cmap_s, is_T in [
    (axes[0], T_ann, "Temperature (°C)", "RdYlBu_r", True),
    (axes[1], S_ann, "Salinity (psu)",   "viridis",  False),
]:
    zonal = np.nanmean(data_3d, axis=1)          # (nlat, nlev)
    p1  = float(np.nanpercentile(zonal, 2))
    p99 = float(np.nanpercentile(zonal, 98))
    if is_T:
        vmin2 = max(p1, -2.0)
        vmax2 = min(p99, 30.0)
    else:
        vmin2, vmax2 = p1, p99
    im = ax.contourf(lat_deg_1d, depths, zonal.T,
                     levels=20, cmap=cmap_s, vmin=vmin2, vmax=vmax2)
    ax.set_ylim(depths.max(), depths.min())
    ax.set_xlabel("Latitude (°)", fontsize=10)
    ax.set_ylabel("Depth (m)",    fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.9)

fig.suptitle("Zonal-mean sections — ocean-only year-5 annual mean",
             fontsize=13, fontweight="bold")
out_sec = os.path.join(out_dir, "ocean_only_sections.png")
plt.savefig(out_sec, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_sec}")

print("All done.")
