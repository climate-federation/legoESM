"""Real-data build/validation for the ESA CCI PFT source: 300 m -> 0.25 deg.

Streams the ESA CCI PFT map (~64800x129600 int8) to coarse cover + ESA-native
PFT composition, validates ranges/orientation against known biomes, caches the
result (npz) for the downstream CLM5 crosswalk + assembly, and plots f_land and
the dominant PFT.
"""

from __future__ import annotations

import glob
import os

import numpy as np

import legoesm.driver  # noqa: F401  (break land<->driver import cycle)

from legoesm.land.surface_data.sources.esa_cci_pft import aggregate_esa_pft, LAND_PFT_NAMES
from legoesm.land.surface_data.aggregate import row_cos_weights


def main() -> None:
    f = glob.glob(os.path.expanduser("~/Downloads/ESACCI-LC-L4-PFT*.nc"))[0]
    print("file:", f)
    out = aggregate_esa_pft(f, res_deg=0.25, coarse_rows_per_chunk=1)
    lat, lon = out["lat"], out["lon"]
    fland, flake, fglac = out["f_land"], out["f_lake"], out["f_glacier"]

    w = row_cos_weights(lat)[:, None]
    den = (np.ones_like(fland) * w).sum()
    g = lambda a: float((a * w).sum() / den)
    print(f"grid {fland.shape}  global f_land={g(fland):.3f} f_lake={g(flake):.3f} "
          f"f_glacier={g(fglac):.3f}")

    def at(la, lo):
        i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon - lo)))
        pf = out["pft_frac"][i, j]
        return fland[i, j], LAND_PFT_NAMES[int(np.argmax(pf))], float(pf.max())

    print("%-16s %6s  %-32s %s" % ("location", "f_land", "dominant PFT", "frac"))
    for nm, la, lo in [("Amazon", -3, -60), ("Sahara", 23, 13), ("Congo", 0, 23),
                       ("Boreal-Siberia", 62, 100), ("India-Deccan", 22, 78),
                       ("US-cornbelt", 41, -93), ("Sahel", 15, 10)]:
        fl, dom, fr = at(la, lo)
        print("%-16s %6.2f  %-32s %.2f" % (nm, fl, dom, fr))

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        "data/esa_cover_pft_0p25.npz",
        lat=lat, lon=lon, f_land=fland, f_lake=flake, f_glacier=fglac,
        pft_frac=out["pft_frac"], pft_names=np.array(LAND_PFT_NAMES),
    )
    print("cached -> data/esa_cover_pft_0p25.npz")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dom_idx = np.argmax(out["pft_frac"], axis=-1).astype(float)
    dom_idx[fland < 0.01] = np.nan
    os.makedirs("results/land", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 5))
    im0 = ax[0].pcolormesh(lon, lat, fland, cmap="YlGn", vmin=0, vmax=1, shading="auto")
    ax[0].set_title("ESA f_land @ 0.25 deg"); fig.colorbar(im0, ax=ax[0], shrink=0.8)
    im1 = ax[1].pcolormesh(lon, lat, dom_idx, cmap="tab20", vmin=0, vmax=len(LAND_PFT_NAMES),
                           shading="auto")
    ax[1].set_title("dominant ESA PFT (index)"); fig.colorbar(im1, ax=ax[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig("results/land/esa_pft_0p25.png", dpi=100)
    print("wrote results/land/esa_pft_0p25.png")


if __name__ == "__main__":
    main()
