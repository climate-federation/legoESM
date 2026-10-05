"""Compare land-only surface captures (nh_surface_replay.py --land-only) across what-if arms.

Area-weighted means over a latitude band of land cells (f_land > 0.5) for the
land solve's own outputs: sensible/latent heat, stress, u*, skin temperature,
solver convergence, and an effective heat exchange coefficient
Ch_eff = H / (rho c_p |U| (T_sfc - T_lowest)) over cells with |T_sfc - T_lowest| > dT_min.

Usage: whatif_land_compare.py BASE.npz OTHER.npz [...] [--lat 45 70]
"""
import argparse

import numpy as np

from legoesm import constants


def band_stats(path, lat_lo, lat_hi, dt_min):
    z = np.load(path, allow_pickle=True)
    land = z["f_land"] > 0.0  # the driver packs land columns on f_land > 0, in cell order
    lat, fl, area = z["lat"][land], z["f_land"][land], z["area"][land]
    m = (lat >= lat_lo) & (lat <= lat_hi) & (fl > 0.5)
    if not m.any():
        raise SystemExit(f"{path}: no land cells in {lat_lo}-{lat_hi}")
    rho = z["land_rho_lowest"]
    wind = np.hypot(z["land_u_lowest"], z["land_v_lowest"])
    dT = z["land_T_surface"] - z["land_T_lowest"]
    H = z["land_shflx"]
    fields = {
        "H": H,
        "LE": z["land_lhflx"],
        "tau": z["land_tau_mag"],
        "ustar": np.sqrt(z["land_tau_mag"] / rho),
        "T_sfc": z["land_T_surface"],
        "Tsfc-Tlow": dT,
        "converged": z["land_converged"],
        "n_iters": z["land_n_iters"],
    }
    for k, v in fields.items():
        if not np.isfinite(v[m]).all():
            raise SystemExit(f"{path}: non-finite {k} in band")
    out = {k: float((v[m] * area[m]).sum() / area[m].sum()) for k, v in fields.items()}
    ok = m & (np.abs(dT) > dt_min) & (wind > 0.0)
    ch = H[ok] / (rho[ok] * constants.c_pd * wind[ok] * dT[ok])
    out["Ch_eff_median"] = float(np.median(ch)) if ok.any() else float("nan")
    out["frac_stable"] = float((area[m] * (dT[m] < 0)).sum() / area[m].sum())
    out["n_cells"] = int(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--lat", nargs=2, type=float, default=(45.0, 70.0))
    ap.add_argument("--dt-min", type=float, default=0.5)
    a = ap.parse_args()
    rows = [(f.split("/")[-1], band_stats(f, *a.lat, a.dt_min)) for f in a.files]
    keys = list(rows[0][1])
    print("file\t" + "\t".join(keys))
    for name, r in rows:
        print(name + "\t" + "\t".join(f"{r[k]:.4g}" for k in keys))
    base = rows[0][1]
    for name, r in rows[1:]:
        print("minus base " + name + "\t" + "\t".join(f"{r[k] - base[k]:+.3g}" for k in keys))


if __name__ == "__main__":
    main()
