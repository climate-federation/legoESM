"""
diag_T_MLD.py  –  4-panel ocean diagnostic from ann_means.npy:
  1. Annual-mean SST
  2. SST anomaly from zonal mean  (reveals cold tongue / warm pool)
  3. Mixed Layer Depth  (1 degC criterion applied to annual-mean T profile)
  4. Zonal-mean temperature section (upper 2000 m)
"""
from __future__ import annotations
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, "src")
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import idealized_bathymetry_latlon_cgrid
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig
from legoesm.ocean.experiments.global_barotropic_wind import create_simplified_continent_mask

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# grid
N_LAT, N_LON, N_LEV = 36, 72, 20
H_MAX = 4000.0
grid    = create_latlon_grid(N_LAT, N_LON)
z_coord = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX, dz_surface=10.0, dz_deep=400.0)
lats = np.degrees(np.array(grid.lat))
lons = np.degrees(np.array(grid.lon))
z_c  = np.abs(np.array(z_coord.z_full_ref))
dz   = np.array(z_coord.dz_ref)

# land mask
go_cfg = GlobalOverturningConfig(H_max=H_MAX, n_levels=N_LEV)
_, _ = idealized_bathymetry_latlon_cgrid(grid, H_max=H_MAX, land_lat_threshold=80.0)
lon_deg2d = np.degrees(np.asarray(grid.lon2d))
lat_deg2d = np.degrees(np.asarray(grid.lat2d))
cont_mask = create_simplified_continent_mask(lon_deg2d, lat_deg2d, go_cfg)

# load
ann = np.load("experiments/ann_means.npy", allow_pickle=True).item()
T   = np.array(ann["T"])

# robust mask: cont_mask AND exclude any column with all-zero (boundary mismatch)
all_zero   = np.all(T == 0.0, axis=2)
ocean_mask = cont_mask.astype(bool) & ~all_zero
T_masked   = np.where(ocean_mask[:, :, np.newaxis], T, np.nan)

# roll 0-360 to -180-180
half = N_LON // 2
def roll(a): return np.roll(a, half, axis=1)

sst      = roll(T_masked)[:, :, 0]
sst_zm   = np.nanmean(sst, axis=1, keepdims=True)
sst_anom = sst - sst_zm

# T at ~200m (closest level)
k200  = int(np.argmin(np.abs(z_c - 200.0)))   # index closest to 200 m
T200  = T_masked[:, :, k200]                   # (lat, lon)
T200_r = roll(T200)

# Depth of the 15°C isotherm (D15) — thermocline depth proxy
def d15_isotherm(T3d, z_centers, T_iso=15.0):
    """Depth where T first crosses below T_iso (m). NaN if never."""
    result = np.full(T3d.shape[:2], np.nan)
    for k in range(T3d.shape[2] - 1):
        T_top = T3d[:, :, k]
        T_bot = T3d[:, :, k + 1]
        crossed = np.isfinite(T_top) & np.isfinite(T_bot) & (T_top >= T_iso) & (T_bot < T_iso) & np.isnan(result)
        # linear interpolation for depth
        frac = (T_top - T_iso) / np.where(T_top != T_bot, T_top - T_bot, 1.0)
        z_interp = z_centers[k] + frac * (z_centers[k + 1] - z_centers[k])
        result = np.where(crossed, z_interp, result)
    return result

d15_r = roll(d15_isotherm(T_masked, z_c))
Tzm   = np.nanmean(T_masked, axis=1)

# figure
fig = plt.figure(figsize=(16, 18))
fig.suptitle("legoESM – annual-mean ocean diagnostics", fontsize=14, y=0.99)

def add_map_ax(fig, pos):
    if HAS_CARTOPY:
        ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5, color="k")
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", linestyle="--",
                     x_inline=False, y_inline=False)
    else:
        ax = fig.add_subplot(*pos)
    return ax

def mshow(ax, data, **kw):
    if HAS_CARTOPY:
        return ax.imshow(data, origin="lower", extent=[-180, 180, -90, 90],
                         transform=ccrs.PlateCarree(), interpolation="nearest", aspect="auto", **kw)
    return ax.imshow(data, origin="lower", extent=[-180, 180, -90, 90],
                     interpolation="nearest", aspect="auto", **kw)

# Panel 1: SST
ax1 = add_map_ax(fig, (4, 1, 1))
ocean_sst = sst[np.isfinite(sst)]
vlo, vhi  = np.percentile(ocean_sst, 2), np.percentile(ocean_sst, 98)
im1 = mshow(ax1, sst, cmap=plt.cm.RdYlBu_r.with_extremes(bad="#c8c8c8"), vmin=vlo, vmax=vhi)
plt.colorbar(im1, ax=ax1, pad=0.02, shrink=0.9).set_label("SST (°C)")
ax1.set_title("Annual-mean SST", fontsize=11)

# Panel 2: SST anomaly
ax2 = add_map_ax(fig, (4, 1, 2))
alim = max(3.0, np.percentile(np.abs(sst_anom[np.isfinite(sst_anom)]), 97))
im2 = mshow(ax2, sst_anom, cmap=plt.cm.RdBu_r.with_extremes(bad="#c8c8c8"), vmin=-alim, vmax=alim)
plt.colorbar(im2, ax=ax2, pad=0.02, shrink=0.9).set_label("ΔSST (°C)")
ax2.set_title("SST anomaly from zonal mean  (warm pool / cold tongue / EBC cooling)", fontsize=11)

# Panel 3: T at 200m
ax3 = add_map_ax(fig, (4, 1, 3))
T200_ocean = T200_r[np.isfinite(T200_r)]
im3 = mshow(ax3, T200_r, cmap=plt.cm.RdYlBu_r.with_extremes(bad="#c8c8c8"),
            vmin=np.percentile(T200_ocean, 2), vmax=np.percentile(T200_ocean, 98))
plt.colorbar(im3, ax=ax3, pad=0.02, shrink=0.9).set_label(f"T at {z_c[k200]:.0f} m (°C)")
ax3.set_title(f"Temperature at {z_c[k200]:.0f} m depth  (thermocline structure)", fontsize=11)

# Panel 4: zonal-mean T section
ax4 = fig.add_subplot(4, 1, 4)
z_sel    = z_c <= 2000
z_plot   = z_c[z_sel]
Tzm_plot = Tzm[:, z_sel]
cf = ax4.contourf(lats, z_plot, Tzm_plot.T, levels=20,
                  cmap=plt.cm.RdYlBu_r.with_extremes(bad="#c8c8c8"), extend="both")
cl = ax4.contour(lats, z_plot, Tzm_plot.T, levels=[1,2,5,10,15,20,25],
                 colors="k", linewidths=0.6)
ax4.clabel(cl, fmt="%g°C", fontsize=7)
ax4.invert_yaxis()
ax4.set_xlabel("Latitude")
ax4.set_ylabel("Depth (m)")
ax4.set_title("Zonal-mean temperature (upper 2000 m)", fontsize=11)
ax4.set_xlim(-90, 90)
ax4.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{abs(int(x))}°{'N' if x>=0 else 'S'}")
)
plt.colorbar(cf, ax=ax4, pad=0.02, shrink=0.9).set_label("T (°C)")

plt.tight_layout(rect=[0, 0, 1, 0.98])
out = "experiments/diag_T_MLD.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
print(f"SST range: {np.nanpercentile(sst[np.isfinite(sst)],2):.1f}–{np.nanpercentile(sst[np.isfinite(sst)],98):.1f} °C")
print(f"T at {z_c[k200]:.0f}m range: {np.nanmin(T200_ocean):.1f}–{np.nanmax(T200_ocean):.1f} °C")
