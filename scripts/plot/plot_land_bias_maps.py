"""Global maps of land skin-T and albedo BIAS (model - ERA5), initial vs calibrated.

Runs the multilayer-land forward on ALL land cells with the INITIAL (CLM5 default) and
the CALIBRATED per-PFT + snow parameters, and maps the annual-mean bias for skin
temperature and surface albedo, plus the bias change (|initial| - |calibrated|, positive
= calibration reduced the local bias).

Run: PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python scripts/plot/plot_land_bias_maps.py \
        --npz /tmp/era5_hourly.npz --calibrated results/land_tuned_fullgrid.json --days 4
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

import scripts.run.train_multilayer_land_era5 as M
import scripts.run.train_land_params_era5 as S


def _bias(cp, data, tier):
    """Per-cell annual-mean (model - obs) bias: skin-T [K], albedo, LE [W/m2]."""
    if tier == "slab":
        T, A, L = S.forward(cp, data)                 # (12, ncol)
    else:
        T, A, _, L = M.forward_ml(cp, data)
    tb = np.asarray(T.mean(0) - data["skt"].mean(0))  # (ncol,)
    ab = np.asarray(A.mean(0) - data["alb"].mean(0))
    lb = np.asarray(L.mean(0) - data["le"].mean(0))
    return tb, ab, lb


def _wrms(x, w):
    return float(np.sqrt(np.sum(w * x ** 2) / np.sum(w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/era5_hourly.npz")
    ap.add_argument("--calibrated", required=True)
    ap.add_argument("--bulk", default="most")
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--out", default="/tmp/land_bias_maps.png")
    ap.add_argument("--tier", choices=["multilayer", "slab"], default="multilayer",
                    help="which calibrator's forward/params the JSON belongs to")
    args = ap.parse_args()

    if args.tier == "slab":
        data = S.load_training_data(args.npz, 100000)             # all land cells
        cp_init = dict(S.constrain(S.init_raw_params()))
    else:
        M._BULK_SCHEME = args.bulk
        data = M.load_training_data(args.npz, 100000, 0, args.days)   # all land cells
        # STANDARD CLM5 default (untuned): CLM5-table per-PFT + out-of-the-box snow/
        # glacier/dry-soil scalars (LandAlbedoConfig defaults + standard bare-ice
        # albedo), NOT the calibrator's init guesses.
        from legoesm.surface_albedo import LandAlbedoConfig as _LAC
        _d = _LAC()
        cp_init = dict(M.constrain_ext(M.init_ext_params()))
        cp_init.update(glac_alb=jnp.asarray(0.60), snow_max=jnp.asarray(_d.alpha_snow_max),
                       snow_min=jnp.asarray(_d.alpha_snow_min),
                       snow_dcrit=jnp.asarray(_d.snow_depth_crit),
                       snow_tau_days=jnp.asarray(_d.tau_snow_decay / 86400.0),
                       soil_dry_boost=jnp.asarray(_d.soil_dry_albedo_boost))
    ncol = int(data["lat"].shape[0])
    cal = json.load(open(args.calibrated))
    cp_cal = {k: jnp.asarray(v) for k, v in cal.items()}

    tbi, abi, lbi = _bias(cp_init, data, args.tier)
    tbc, abc, lbc = _bias(cp_cal, data, args.tier)
    w = np.asarray(data["w"])
    print(f"# global land ({ncol} cells, area-weighted, {args.tier}):")
    print(f"#   skin-T  bias {np.average(tbi, weights=w):+.3f} -> {np.average(tbc, weights=w):+.3f} K"
          f"   RMSE {_wrms(tbi, w):.3f} -> {_wrms(tbc, w):.3f} K")
    print(f"#   albedo  bias {np.average(abi, weights=w):+.4f} -> {np.average(abc, weights=w):+.4f}"
          f"   RMSE {_wrms(abi, w):.4f} -> {_wrms(abc, w):.4f}")
    print(f"#   LE      bias {np.average(lbi, weights=w):+.2f} -> {np.average(lbc, weights=w):+.2f} W/m2"
          f"   RMSE {_wrms(lbi, w):.2f} -> {_wrms(lbc, w):.2f} W/m2")

    # reconstruct lon/lat for the land cells (same order as load_training_data)
    D = np.load(args.npz); lat1, lon1 = D["lat"], D["lon"]; nlat, nlon = lat1.size, lon1.size
    lidx = np.where((D["lsm"].reshape(nlat, nlon) > 0.5).ravel())[0]
    latc = np.rad2deg(np.asarray(data["lat"]))
    lonc = lon1[lidx % nlon]; lonc = np.where(lonc > 180, lonc - 360, lonc)

    # 3 rows (skin-T, albedo, LE) x 3 cols (initial bias, calibrated bias, gain)
    rows = [("skin-T bias [K]", tbi, tbc, 8.0, "RdBu_r"),
            ("albedo bias", abi, abc, 0.25, "RdBu_r"),
            ("LE bias [W/m2]", lbi, lbc, 40.0, "RdBu_r")]
    fig, ax = plt.subplots(3, 3, figsize=(16, 10.5))
    for r, (name, bi, bc, lim, cmap) in enumerate(rows):
        gain = np.abs(bi) - np.abs(bc)               # >0 where calibration cut the bias
        glim = np.nanpercentile(np.abs(gain), 98) or lim
        panels = [("initial bias", bi, lim, cmap), ("calibrated bias", bc, lim, cmap),
                  ("bias reduced (|init|-|cal|)", gain, glim, "PiYG")]
        for c, (ttl, v, vl, cm) in enumerate(panels):
            sc = ax[r, c].scatter(lonc, latc, c=v, s=7, cmap=cm, vmin=-vl, vmax=vl)
            ax[r, c].set_title(f"{name} — {ttl}", fontsize=9)
            ax[r, c].set_xlim(-180, 180); ax[r, c].set_ylim(-90, 90)
            ax[r, c].axhline(0, color="k", lw=0.3); plt.colorbar(sc, ax=ax[r, c], fraction=0.03)
    plt.tight_layout(); plt.savefig(args.out, dpi=95)
    print(f"# bias maps -> {args.out}")


if __name__ == "__main__":
    main()
