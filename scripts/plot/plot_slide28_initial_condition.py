"""Visualize the geostrophic-adjustment initial condition.

Shows three things explicitly:
  (1) The background vertical T profile (T_surf=20°C → T_deep=2°C, scale=1000m)
  (2) The meridional T-front perturbation: 5°C * cos(lat) * exp(-k/3)
  (3) The final T(lat, z) initial condition (background + perturbation)

Plus a 2D map of the surface T-anomaly to make it obvious that t=0 is NOT
"nothing" — it's an unbalanced thermal-wind state ready to adjust.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.experiments.geostrophic_adjustment import (
    create_initial_conditions as ga_ic,
)
from legoesm.ocean.vertical import create_ocean_z_star

OUT_DIR = Path(
    "results/ocean/slide28_validation/geostrophic_adjustment_logtime"
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = create_latlon_grid(36, 72)
    z = create_ocean_z_star(n_levels=10, H_max=5500.0)
    state = ga_ic("latlon", grid, z)

    T = np.asarray(state.T.data)               # (nlat, nlon, nlev)
    lat = np.rad2deg(np.asarray(grid.lat))     # (nlat,)
    z_full = np.asarray(z.z_full_ref)          # (nlev,) cell centres, negative
    depth = -z_full                            # positive downward, m

    # T-profile vs depth at three latitudes (pick equator, mid, pole-edge)
    j_eq = np.argmin(np.abs(lat))
    j_mid = np.argmin(np.abs(lat - 45))
    j_high = np.argmin(np.abs(lat - 75))
    lon_idx = 0  # any lon

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.6, 1.6]})

    # Left: 1D vertical T profiles at three latitudes
    ax = axes[0]
    ax.plot(T[j_eq, lon_idx, :], depth, "C3-o", label=f"eq ({lat[j_eq]:.0f}°)")
    ax.plot(T[j_mid, lon_idx, :], depth, "C2-s", label=f"mid ({lat[j_mid]:.0f}°)")
    ax.plot(T[j_high, lon_idx, :], depth, "C0-^",
            label=f"high ({lat[j_high]:.0f}°)")
    ax.invert_yaxis()
    ax.set_xlabel("T (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Initial T vertical profile\n(stratification + cos(lat) bump)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Middle: T(lat, depth) — zonal mean cross-section
    ax = axes[1]
    T_zmean = np.nanmean(T, axis=1)            # (nlat, nlev)
    pcm = ax.pcolormesh(lat, depth, T_zmean.T, cmap="RdYlBu_r",
                        shading="auto")
    ax.invert_yaxis()
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Initial T(lat, z) — zonal mean")
    fig.colorbar(pcm, ax=ax, label="T (°C)")

    # Right: surface T anomaly relative to global-mean surface T
    ax = axes[2]
    SST = T[..., 0]                            # surface temperature
    SST_mean = float(np.nanmean(SST))
    SST_anom = SST - SST_mean
    vmax = float(np.nanmax(np.abs(SST_anom)))
    lon = np.rad2deg(np.asarray(grid.lon))
    pcm = ax.pcolormesh(lon, lat, SST_anom, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Initial surface T anomaly\n(T − global mean, mean={SST_mean:.1f}°C)")
    fig.colorbar(pcm, ax=ax, label="ΔT (°C)")

    fig.suptitle(
        "Geostrophic-adjustment initial condition  ·  "
        "rest (η=0, u=0) + meridional T-front  →  thermal-wind imbalance",
        fontsize=11,
    )
    out = OUT_DIR / "slide28_initial_condition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")

    # Quick numerical summary so the user can sanity-check the bump
    print(f"\nSurface T (k=0):")
    print(f"  T at equator = {T[j_eq, lon_idx, 0]:.3f} °C")
    print(f"  T at mid-lat  = {T[j_mid, lon_idx, 0]:.3f} °C")
    print(f"  T at high-lat = {T[j_high, lon_idx, 0]:.3f} °C")
    print(f"  Surface ΔT (eq − pole edge) = "
          f"{T[j_eq, lon_idx, 0] - T[j_high, lon_idx, 0]:.3f} °C")
    print(f"\nDeep T (k=9):")
    print(f"  T at equator = {T[j_eq, lon_idx, 9]:.3f} °C")
    print(f"  T at high-lat = {T[j_high, lon_idx, 9]:.3f} °C")


if __name__ == "__main__":
    main()
