"""Build the harmonized legoESM surface-data NetCDF from primary sources.

Currently wires the FAO HWSD v2.0 soil source; future sources (GIMMS LAI4g,
ESA-CCI biomass, ...) plug in alongside.  Run once, offline; the runtime loader
``legoesm.land.global_surface_data`` then regrids the output to the model grid.

Example:
    python3 scripts/build_surface_data.py hwsd2 \
        --bil ~/Downloads/HWSD2_RASTER/HWSD2.bil \
        --layers data/hwsd2_cache/HWSD2_LAYERS.csv \
        --res 0.25 --out data/legoesm_surfdata_soil_0p25.nc
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# Break the land<->driver import cycle before importing anything under legoesm.land
# (see tests/land/conftest.py).
import legoesm.driver  # noqa: F401,E402

from legoesm.land.surface_data.sources.hwsd2 import build_hwsd2_surfdata


def _plot_soil(soil: dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat, lon = soil["lat"], soil["lon"]
    panels = [("sand_pct", "sand %", "YlOrBr", 0, 100),
              ("clay_pct", "clay %", "BuPu", 0, 60),
              ("organic", "organic carbon %", "Greens", 0, 10),
              ("bulk_density", "bulk density kg/m3", "cividis", 800, 1800)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for ax, (key, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        field = soil[key][0]                          # top layer (D1)
        im = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"HWSD2 {title} (D1)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    print(f"wrote {out_png}")


def _summarize(soil: dict) -> None:
    for key in ("sand_pct", "clay_pct", "organic", "bulk_density"):
        f = soil[key]
        finite = np.isfinite(f)
        print(f"  {key:14s} layer0 range [{np.nanmin(f[0]):.2f}, {np.nanmax(f[0]):.2f}] "
              f"valid {100 * finite.mean():.1f}%")
    # sand+silt+clay sanity is unavailable (no silt stored), but sand+clay<=100.
    sc = soil["sand_pct"][0] + soil["clay_pct"][0]
    print(f"  sand+clay (D1) max = {np.nanmax(sc):.1f} (<=100 expected)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="source", required=True)
    h = sub.add_parser("hwsd2", help="FAO HWSD v2.0 soil")
    h.add_argument("--bil", default=os.path.expanduser("~/Downloads/HWSD2_RASTER/HWSD2.bil"))
    h.add_argument("--layers", required=True, help="HWSD2.mdb or exported HWSD2_LAYERS.csv")
    h.add_argument("--res", type=float, default=0.25)
    h.add_argument("--out", default="data/legoesm_surfdata_soil_0p25.nc")
    h.add_argument("--chunk", type=int, default=6, help="coarse rows per streaming chunk")
    h.add_argument("--plot", default="results/land/hwsd2_soil_0p25.png")
    args = ap.parse_args()

    if args.source == "hwsd2":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        soil = build_hwsd2_surfdata(
            args.bil, args.layers, args.out, res_deg=args.res,
            coarse_rows_per_chunk=args.chunk,
        )
        print(f"wrote {args.out}  grid {soil['soil_data_frac'].shape}")
        _summarize(soil)
        if args.plot:
            os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
            _plot_soil(soil, args.plot)


if __name__ == "__main__":
    main()
