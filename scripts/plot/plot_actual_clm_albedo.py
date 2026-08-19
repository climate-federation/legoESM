"""The ACTUAL CLM surface albedo field built directly from the CLM surfdata (no legoESM
land model), as the honest 'CLM baseline' for the mean-annual albedo comparison vs ERA5.

CLM mosaic albedo per cell:
    alpha_snowfree = PCT_bare * alpha_soil(SOIL_COLOR)  +  sum_veg PCT_pft * alpha_canopy_pft
    alpha_annual   = (1 - f_snow) * alpha_snowfree       +  f_snow * alpha_snow
where alpha_soil is the CLM soil-color dry/saturated broadband albedo (Oleson et al.
2013, Table 3.3; moisture-blended toward dry for the bare/arid fraction), alpha_canopy
is the CLM5 per-PFT snow-free broadband albedo, and f_snow is a snow-cover climatology
from the ERA5 monthly skin-temperature (months below freezing).

Run: PYTHONPATH=. JAX_ENABLE_X64=1 python scripts/plot/plot_actual_clm_albedo.py \
        --npz /tmp/era5_hourly.npz --tuned /tmp/_v3_best.json
"""
import argparse
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- CLM soil-color broadband albedo (Oleson et al. 2013, CLM Tech Note, Table 3.3) ---
# 20 colour classes; broadband = 0.5*(visible + near-infrared).
_ALBDRY_VIS = np.array([.36,.34,.32,.31,.30,.29,.28,.27,.26,.25,.24,.23,.22,.20,.18,.16,.14,.12,.10,.08])
_ALBDRY_NIR = np.array([.61,.57,.53,.51,.49,.48,.45,.43,.41,.39,.37,.35,.33,.31,.29,.27,.25,.23,.21,.16])
_ALBSAT_VIS = np.array([.25,.23,.21,.20,.19,.18,.17,.16,.15,.14,.13,.12,.11,.10,.09,.08,.07,.06,.05,.04])
_ALBSAT_NIR = np.array([.50,.46,.42,.40,.38,.36,.34,.32,.30,.28,.26,.24,.22,.20,.18,.16,.14,.12,.10,.08])
_SOIL_DRY = 0.5 * (_ALBDRY_VIS + _ALBDRY_NIR)     # broadband dry per class (index 0..19)
_SOIL_SAT = 0.5 * (_ALBSAT_VIS + _ALBSAT_NIR)      # broadband saturated per class

_ALPHA_SNOW = 0.60          # CLM annual-mean broadband snow albedo (fresh~0.8, aged~0.5)


def _nn_regrid(src_lat, src_lon, field, tgt_lat, tgt_lon):
    """Nearest-neighbour regrid a (..., nlat, nlon) field to target 1-D columns [deg]."""
    src_lat = np.asarray(src_lat); src_lon = np.asarray(src_lon) % 360.0
    jlat = np.abs(src_lat[None, :] - np.asarray(tgt_lat)[:, None]).argmin(1)
    jlon = np.abs(src_lon[None, :] - (np.asarray(tgt_lon) % 360.0)[:, None]).argmin(1)
    return np.asarray(field)[..., jlat, jlon]


def actual_clm_albedo(npz):
    from legoesm.land.clm_surface_map import download_clm_surfdata
    from scripts.run.train_land_params_era5 import _TABLE, _PI
    tbl_alb = np.asarray(_TABLE)[:, _PI["albedo_veg"]]   # CLM5 per-PFT canopy albedo (17)

    d = nc.Dataset(download_clm_surfdata())
    slat = np.asarray(d["LATIXY"])[:, 0]; slon = np.asarray(d["LONGXY"])[0, :]
    color = np.asarray(d["SOIL_COLOR"]).astype(int)          # (96,144) 1..20
    pct = np.asarray(d["PCT_NAT_PFT"]) / 100.0               # (15,96,144) fractions

    D = np.load(npz); lat1, lon1 = D["lat"], D["lon"]; nlat, nlon = lat1.size, lon1.size
    latg = np.broadcast_to(lat1[:, None], (nlat, nlon)).ravel()
    long_ = np.broadcast_to(lon1[None, :], (nlat, nlon)).ravel()
    land = (D["lsm"].reshape(nlat, nlon) > 0.5).ravel()

    col = _nn_regrid(slat, slon, color, latg, long_)         # (ncellgrid,)
    frac = _nn_regrid(slat, slon, pct, latg, long_)          # (15, ncellgrid)
    frac = frac / np.clip(frac.sum(0), 1e-6, None)           # renormalise (nat land = 1)

    # soil albedo (bare fraction): blended toward DRY (bare soil is mostly arid/desert).
    soil = 0.7 * _SOIL_DRY[col - 1] + 0.3 * _SOIL_SAT[col - 1]
    # mosaic snow-free albedo: bare (PFT 0) -> soil colour; veg (1..14) -> CLM canopy albedo
    alpha = frac[0] * soil
    for p in range(1, 15):
        alpha = alpha + frac[p] * tbl_alb[p]

    # snow-cover climatology from ERA5 monthly skin-T (fraction of months below freezing)
    from legoesm import constants
    skt = D["skin_temperature"].reshape(12, -1, nlat * nlon).mean(1)   # (12,ncell) monthly
    fsnow = (skt < constants.T_freeze).mean(0)                          # (ncell) 0..1
    alpha_annual = (1 - fsnow) * alpha + fsnow * _ALPHA_SNOW

    era = D["forecast_albedo"].reshape(12, -1, nlat * nlon).mean(1).mean(0)   # (ncell)
    return latg, long_, land, alpha_annual, alpha, era, fsnow


def _to_grid(vals, mask, nlat, nlon):
    g = np.full(nlat * nlon, np.nan); g[mask] = vals[mask]
    return g.reshape(nlat, nlon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/era5_hourly.npz")
    ap.add_argument("--tuned", default="/tmp/_v3_best.json")
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--out", default="/tmp/actual_clm_albedo")
    args = ap.parse_args()

    latg, long_, land, clm_ann, clm_snowfree, era, fsnow = actual_clm_albedo(args.npz)

    # our tuned model albedo (mean-annual) on the same grid
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp, json
    import scripts.run.train_multilayer_land_era5 as M
    M._BULK_SCHEME = "most"
    data = M.load_training_data(args.npz, 100000, 0, args.days)
    cp = {k: jnp.asarray(v) for k, v in json.load(open(args.tuned)).items()}
    _, A, _, _ = M.forward_ml(cp, data)
    D = np.load(args.npz); nlat, nlon = D["lat"].size, D["lon"].size
    lidx = np.where((D["lsm"].reshape(nlat, nlon) > 0.5).ravel())[0]
    sub = np.random.default_rng(0).choice(lidx.size, min(100000, lidx.size), replace=False)
    tuned = np.full(nlat * nlon, np.nan); tuned[lidx[sub]] = np.asarray(A.mean(0))

    w = np.cos(np.deg2rad(latg))
    def bias(x):
        m = land & np.isfinite(x) & np.isfinite(era)
        return np.average(x[m] - era[m], weights=w[m]), np.sqrt(np.average((x[m]-era[m])**2, weights=w[m]))
    for nm, x in [("actual CLM (surfdata)", clm_ann), ("our tuned model", tuned)]:
        b, r = bias(x); print(f"# {nm:26s} albedo bias {b:+.4f}  RMSE {r:.4f}  (vs ERA5)")
    m = land & np.isfinite(clm_ann)
    print(f"# actual-CLM mean-annual albedo (area-wtd land): {np.average(clm_ann[m], weights=w[m]):.3f} "
          f"| ERA5 {np.average(era[m], weights=w[m]):.3f} | snow-free {np.average(clm_snowfree[m], weights=w[m]):.3f}")

    lon2, lat2 = np.meshgrid(D["lon"], D["lat"])
    fields = [("ERA5 forecast albedo", era, "viridis", (0.05, 0.6)),
              ("actual CLM (surfdata)", clm_ann, "viridis", (0.05, 0.6)),
              ("our tuned model", tuned, "viridis", (0.05, 0.6)),
              ("actual CLM - ERA5", clm_ann - era, "RdBu_r", (-0.25, 0.25))]
    proj = ccrs.Robinson(central_longitude=0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.4), subplot_kw={"projection": proj})
    for ax, (title, fld, cmap, (vmn, vmx)) in zip(axes.ravel(), fields):
        mp = ax.pcolormesh(lon2, lat2, np.ma.masked_invalid(_to_grid(fld, land, nlat, nlon)),
                           cmap=cmap, vmin=vmn, vmax=vmx, transform=ccrs.PlateCarree(),
                           shading="auto", rasterized=True)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.25")
        ax.set_global(); ax.set_title(title, fontsize=10)
        plt.colorbar(mp, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Mean-annual land albedo: actual CLM (surfdata soil-colour + PFT) vs ERA5 vs tuned",
                 fontsize=12, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"# map -> {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
