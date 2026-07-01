"""Paper-ready global maps of mean-annual land skin temperature and surface albedo:
ERA5 reference vs the standard CLM5-default parameters vs the ERA5-tuned parameters.

Two rows (skin-T [K], albedo) x three columns (ERA5, CLM default, tuned), Robinson
projection with coastlines, a shared per-row colour scale (fair comparison), area-
weighted RMSE annotated per model panel, panel labels, 300-dpi PNG + vector PDF.

Run: PYTHONPATH=. JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda .venv/bin/python \
        scripts/plot/plot_land_field_maps.py --npz /tmp/era5_hourly.npz \
        --tuned results/land_tuned_fullgrid.json --days 4 --out /tmp/land_field_maps
"""
import argparse
import json

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import scripts.run.train_multilayer_land_era5 as M


def _clm_default_params():
    """STANDARD CLM5 default parameters — nothing tuned to ERA5.  The per-PFT albedo /
    emissivity / z0 / root / water-stress come from the CLM5 table (init_ext_params);
    the snow / glacier / dry-soil scalars are the model's out-of-the-box defaults
    (LandAlbedoConfig defaults + a standard bare-ice albedo), NOT the calibrator's init
    guesses.  This is the honest 'before' baseline for the map comparison."""
    from legoesm.surface_albedo import LandAlbedoConfig
    cp = dict(M.constrain_ext(M.init_ext_params()))
    d = LandAlbedoConfig()
    cp["glac_alb"] = jnp.asarray(0.60)                     # standard bare ice-sheet albedo
    cp["snow_max"] = jnp.asarray(d.alpha_snow_max)         # 0.80
    cp["snow_min"] = jnp.asarray(d.alpha_snow_min)         # 0.50
    cp["snow_dcrit"] = jnp.asarray(d.snow_depth_crit)      # 50 kg/m2
    cp["snow_tau_days"] = jnp.asarray(d.tau_snow_decay / 86400.0)   # 5 days
    cp["soil_dry_boost"] = jnp.asarray(d.soil_dry_albedo_boost)     # 0.11 (CLM)
    return cp


def _annual(cp, data):
    T, A, _ = M.forward_ml(cp, data)
    return np.asarray(T.mean(0)), np.asarray(A.mean(0))   # (ncol,) mean-annual


def _to_grid(vals, place_idx, nlat, nlon):
    """Scatter land-cell values back onto the full lat-lon grid (ocean = NaN).
    ``place_idx`` is the flat grid index for each value (load_training_data shuffles
    the land cells, so this is lidx[sub], NOT the sorted land indices)."""
    g = np.full(nlat * nlon, np.nan)
    g[place_idx] = vals
    return g.reshape(nlat, nlon)


def _wrmse(a, b, w):
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/era5_hourly.npz")
    ap.add_argument("--tuned", default="results/land_tuned_fullgrid.json")
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--out", default="/tmp/land_field_maps")
    args = ap.parse_args()

    M._BULK_SCHEME = "most"
    data = M.load_training_data(args.npz, 100000, 0, args.days)   # all land cells
    w = np.asarray(data["w"])

    cp_def = _clm_default_params()                               # standard CLM5, untuned
    cp_tun = {k: jnp.asarray(v) for k, v in json.load(open(args.tuned)).items()}
    T_def, A_def = _annual(cp_def, data)
    T_tun, A_tun = _annual(cp_tun, data)
    T_era = np.asarray(data["skt"].mean(0)); A_era = np.asarray(data["alb"].mean(0))

    D = np.load(args.npz); lat1, lon1 = D["lat"], D["lon"]; nlat, nlon = lat1.size, lon1.size
    lidx = np.where((D["lsm"].reshape(nlat, nlon) > 0.5).ravel())[0]
    # load_training_data samples land cells in a SHUFFLED order (seed 0); the model
    # outputs follow that order, so each output i lives at grid cell lidx[sub[i]].
    sub = np.random.default_rng(0).choice(lidx.size, size=min(100000, lidx.size),
                                          replace=False)
    place = lidx[sub]
    lon2, lat2 = np.meshgrid(lon1, lat1)
    g = lambda v: _to_grid(v, place, nlat, nlon)

    # rows: (name, unit, cmap, (vmin,vmax), ERA5, default, tuned)
    rows = [
        ("skin temperature", "K", "turbo", (240, 305), T_era, T_def, T_tun),
        ("surface albedo", "", "viridis", (0.05, 0.6), A_era, A_def, A_tun),
    ]
    proj = ccrs.Robinson(central_longitude=0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.6), subplot_kw={"projection": proj})
    labels = "abcdefghi"
    for r, (name, unit, cmap, (vmn, vmx), era, dflt, tun) in enumerate(rows):
        panels = [("ERA5", era, None), ("CLM5 default", dflt, T_era if r == 0 else A_era),
                  ("tuned (this work)", tun, T_era if r == 0 else A_era)]
        for c, (title, field, ref) in enumerate(panels):
            ax = axes[r, c]
            fld = g(field)
            m = ax.pcolormesh(lon2, lat2, np.ma.masked_invalid(fld), cmap=cmap,
                              vmin=vmn, vmax=vmx, transform=ccrs.PlateCarree(),
                              shading="auto", rasterized=True)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.25")
            ax.set_global()
            u = f" [{unit}]" if unit else ""
            ttl = f"({labels[r * 3 + c]}) {title}"
            if ref is not None:
                ttl += f"   RMSE {_wrmse(field, ref, w):.2f}{(' K' if r == 0 else '')}"
            ax.set_title(ttl, fontsize=10)
            if c == 2:
                cb = fig.colorbar(m, ax=axes[r, :], fraction=0.018, pad=0.02,
                                  shrink=0.9)
                cb.set_label(f"{name}{u}", fontsize=9)
    fig.suptitle("Mean-annual land surface: ERA5 vs CLM5-default vs ERA5-tuned parameters",
                 fontsize=12, y=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"# maps -> {args.out}.png / .pdf")
    print(f"# skin-T  RMSE  default {_wrmse(T_def, T_era, w):.3f} -> tuned {_wrmse(T_tun, T_era, w):.3f} K")
    print(f"# albedo  RMSE  default {_wrmse(A_def, A_era, w):.4f} -> tuned {_wrmse(A_tun, A_era, w):.4f}")


if __name__ == "__main__":
    main()
