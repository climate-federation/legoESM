#!/usr/bin/env python
"""Maps of a trained WeatherBench model against ERA5.

Reads the ``.npz`` written by
``scripts/validate/run_weatherbench_eval.py --save-fields`` — the case-mean
forecast and ERA5 verification fields on the WB2 grid, from the SAME forecasts
the scorecard scores — and draws, per field and lead time, three panels:
model, ERA5, and model minus ERA5.

The difference panel is symmetric about zero on a diverging map; the two field
panels share one colour range so they can be compared by eye. Cells that were
below ground (or otherwise unscorable) in any case are blank in all three, for
the same reason the scorecard skips them.

Import-light and JAX-free (login-node runnable).

Usage::

    .venv/bin/python scripts/plot/plot_wb_forecast_maps.py \\
        --fields results/wb_amip/eval/fields_epoch0011.npz \\
        --out-dir results/wb_amip/eval/maps
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Units per headline field key, for the colour-bar label.
_UNITS = {
    "z500": "m^2/s^2", "t850": "K", "q700": "kg/kg", "mslp": "Pa",
    "t2m": "K", "u10": "m/s", "v10": "m/s", "wind_speed_10m": "m/s",
}


def _unit(key: str) -> str:
    for prefix, unit in _UNITS.items():
        if key == prefix:
            return unit
    if key.startswith(("u", "v")):
        return "m/s"
    if key.startswith("t"):
        return "K"
    if key.startswith("q"):
        return "kg/kg"
    return ""


def group_fields(npz) -> dict:
    """``{(key, lead): {"pred", "verif", "count"}}`` from the flat npz names."""
    out: dict = {}
    for name in npz.files:
        if "__" not in name:
            continue
        kind, key, lead = name.split("__")
        out.setdefault((key, int(lead)), {})[kind] = npz[name]
    missing = [k for k, v in out.items()
               if not {"pred", "verif"} <= set(v)]
    if missing:
        raise SystemExit(f"{missing}: the npz is missing a pred/verif pair")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", required=True, type=Path,
                    help="npz from run_weatherbench_eval.py --save-fields")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated field keys (default: all present)")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    npz = np.load(a.fields)
    fields = group_fields(npz)
    lat = npz["wb2_lat_deg"]
    lon = npz["wb2_lon_deg"]
    out_dir = a.out_dir or a.fields.parent / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = (set(a.only.split(",")) if a.only else
              {k for (k, _lead) in fields})

    written = []
    for (key, lead), slot in sorted(fields.items()):
        if key not in wanted:
            continue
        pred = np.asarray(slot["pred"], dtype=float)
        verif = np.asarray(slot["verif"], dtype=float)
        diff = pred - verif
        finite = np.isfinite(pred) & np.isfinite(verif)
        if not finite.any():
            print(f"{key} {lead}h: every cell unscorable, skipped")
            continue
        vmin = float(np.nanmin([pred[finite].min(), verif[finite].min()]))
        vmax = float(np.nanmax([pred[finite].max(), verif[finite].max()]))
        dmax = float(np.nanmax(np.abs(diff[finite]))) or 1.0

        fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.6))
        ext = [float(lon.min()), float(lon.max()),
               float(lat.min()), float(lat.max())]
        for ax, data, title, kw in (
                (axes[0], pred, "model", dict(vmin=vmin, vmax=vmax)),
                (axes[1], verif, "ERA5", dict(vmin=vmin, vmax=vmax)),
                (axes[2], diff, "model - ERA5",
                 dict(vmin=-dmax, vmax=dmax, cmap="RdBu_r"))):
            im = ax.imshow(data, origin="lower", extent=ext,
                           aspect="auto", **kw)
            ax.set_title(title)
            ax.set_xlabel("longitude [deg]")
            fig.colorbar(im, ax=ax, label=_unit(key))
        axes[0].set_ylabel("latitude [deg]")
        fig.suptitle(f"{key}, {lead} h forecast, mean over the eval cases")
        fig.tight_layout()
        png = out_dir / f"map_{key}_{lead}h.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        written.append(png)

    for png in written:
        print(f"wrote {png}")
    if not written:
        raise SystemExit("nothing plotted: check --only against the npz keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
