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
    cp["soil_dry_boost"] = jnp.asarray(0.0)     # dry-soil brightening is OUR addition -> off
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
    ap.add_argument("--elev-bands", action="store_true",
                    help="run the tuned column with the sub-grid elevation-band snow "
                         "scheme (gaps 1-6) active")
    args = ap.parse_args()

    M._BULK_SCHEME = "most"
    M._ELEV_BANDS_ON = bool(args.elev_bands)
    data = M.load_training_data(args.npz, 100000, 0, args.days)   # all land cells
    w = np.asarray(data["w"])

    cp_def = _clm_default_params()                               # standard CLM5, untuned
    cp_tun = {k: jnp.asarray(v) for k, v in json.load(open(args.tuned)).items()}

    def _full(cp, bands=False):
        M._ELEV_BANDS_ON = bool(bands)   # forward_ml reads this at call time
        T, A, _ = M.forward_ml(cp, data)
        return np.asarray(T), np.asarray(A)                      # (12, ncol) monthly

    # The untuned-CLM baseline runs the ORIGINAL albedo physics too — the crude LINEAR
    # snow cover min(1, SWE/crit) and no dry-soil brightening (cp_def sets boost=0) — so
    # the middle column is the genuine 'before' (our tanh snow cover + dry soil are part
    # of the contribution, not the baseline).  Monkeypatch the shared snow-cover fn for
    # the default forward, then restore for the tuned (our model) forward.
    import legoesm.surface_albedo as _SA
    _orig_scf = _SA.snow_cover_fraction
    _SA.snow_cover_fraction = lambda sd, cfg: jnp.clip(
        sd / jnp.maximum(cfg.snow_depth_crit, 1e-6), 0.0, 1.0)
    Tf_def, Af_def = _full(cp_def, bands=False)                  # original CLM physics (no bands)
    _SA.snow_cover_fraction = _orig_scf
    Tf_tun, Af_tun = _full(cp_tun, bands=args.elev_bands)        # our model (+ gaps if --elev-bands)
    Tf_era = np.asarray(data["skt"]); Af_era = np.asarray(data["alb"])
    T_def, A_def = Tf_def.mean(0), Af_def.mean(0)                # annual-mean for the maps
    T_tun, A_tun = Tf_tun.mean(0), Af_tun.mean(0)
    T_era, A_era = Tf_era.mean(0), Af_era.mean(0)
    # SPACE-TIME (all 12 months) area-weighted RMSE for the annotation — the standard
    # metric; the annual-mean-field RMSE is smaller because monthly errors partly cancel.
    st = lambda f, ref: float(np.sqrt(np.sum(w[None] * (f - ref) ** 2) / np.sum(w) / 12))
    rmse = {("T", 1): st(Tf_def, Tf_era), ("T", 2): st(Tf_tun, Tf_era),
            ("A", 1): st(Af_def, Af_era), ("A", 2): st(Af_tun, Af_era)}

    D = np.load(args.npz); lat1, lon1 = D["lat"], D["lon"]; nlat, nlon = lat1.size, lon1.size
    lidx = np.where((D["lsm"].reshape(nlat, nlon) > 0.5).ravel())[0]
    # load_training_data samples land cells in a SHUFFLED order (seed 0); the model
    # outputs follow that order, so each output i lives at grid cell lidx[sub[i]].
    sub = np.random.default_rng(0).choice(lidx.size, size=min(100000, lidx.size),
                                          replace=False)
    place = lidx[sub]
    lon2, lat2 = np.meshgrid(lon1, lat1)
    g = lambda v: _to_grid(v, place, nlat, nlon)

    # SWAP the albedo baseline to the ACTUAL CLM albedo (surfdata SOIL_COLOR -> CLM soil
    # table + PCT_NAT_PFT -> CLM5 canopy albedos + snow climatology), NOT the legoESM
    # model run with CLM params.  The actual-CLM field is annual-only, so the albedo row
    # is annotated with the ANNUAL-MEAN spatial RMSE (both baseline and tuned) for a like-
    # for-like comparison; the skin-T row keeps the space-time (monthly) RMSE.
    from scripts.plot.plot_actual_clm_albedo import actual_clm_albedo
    _, _, _, clm_ann, _, _, _ = actual_clm_albedo(args.npz)
    A_def = clm_ann[place]
    ann = lambda f: float(np.sqrt(np.sum(w * (f - A_era) ** 2) / np.sum(w)))
    rmse[("A", 1)] = ann(A_def); rmse[("A", 2)] = ann(A_tun)

    # Panel layout: column 1 = the ERA5 mean-annual FIELD (reference); columns 2-3 = the
    # BIAS (model - ERA5) of the CLM-default baseline and the tuned model, annotated with
    # the area-weighted global-mean bias.  Bias maps are the meaningful spatial metric
    # (a per-cell RMSE map of an annual mean is just |bias|); diverging scale about 0.
    bias_lim = {0: 6.0, 1: 0.2}                     # +-6 K, +-0.2 albedo
    rows = [
        ("skin temperature", "K", "turbo", (240, 305), T_era, T_def, T_tun),
        ("surface albedo", "", "viridis", (0.05, 0.6), A_era, A_def, A_tun),
    ]
    proj = ccrs.Robinson(central_longitude=0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.6), subplot_kw={"projection": proj})
    labels = "abcdefghi"
    mid_title = {0: "CLM5 default params", 1: "actual CLM (surfdata)"}
    for r, (name, unit, cmap, (vmn, vmx), era, dflt, tun) in enumerate(rows):
        u = f" [{unit}]" if unit else ""
        lim = bias_lim[r]
        panels = [("ERA5", era, cmap, (vmn, vmx), False),
                  (f"{mid_title[r]} − ERA5", dflt - era, "RdBu_r", (-lim, lim), True),
                  ("tuned (this work) − ERA5", tun - era, "RdBu_r", (-lim, lim), True)]
        for c, (title, field, cm, (lo, hi), is_bias) in enumerate(panels):
            ax = axes[r, c]
            m = ax.pcolormesh(lon2, lat2, np.ma.masked_invalid(g(field)), cmap=cm,
                              vmin=lo, vmax=hi, transform=ccrs.PlateCarree(),
                              shading="auto", rasterized=True)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.25")
            ax.set_global()
            ttl = f"({labels[r * 3 + c]}) {title}"
            if is_bias:
                b = float(np.sum(w * field) / np.sum(w))     # area-weighted mean bias
                ttl += (f"   bias {b:+.2f} K" if r == 0 else f"   bias {b:+.3f}")
                print(f"# PANEL-BIAS {name} col{c} ({title}): {b:+.4f}", flush=True)
            ax.set_title(ttl, fontsize=10)
            cb = fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02)
            if c == 0:
                cb.set_label(f"{name}{u}", fontsize=8)
            elif c == 2:
                cb.set_label(f"bias{u}", fontsize=8)
    fig.suptitle("Mean-annual land surface: ERA5 reference and model biases "
                 "(CLM default vs ERA5-tuned)", fontsize=12, y=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"# maps -> {args.out}.png / .pdf")
    print(f"# space-time RMSE  skin-T default {rmse[('T',1)]:.3f} -> tuned {rmse[('T',2)]:.3f} K "
          f"| albedo default {rmse[('A',1)]:.4f} -> tuned {rmse[('A',2)]:.4f}")
    print(f"# (annual-mean-field RMSE, smaller: skin-T {_wrmse(T_def,T_era,w):.3f} -> "
          f"{_wrmse(T_tun,T_era,w):.3f} K)")


if __name__ == "__main__":
    main()
