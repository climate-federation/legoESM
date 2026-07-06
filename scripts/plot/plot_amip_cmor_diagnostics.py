#!/usr/bin/env python
"""Systematic AMIP climate diagnostics from a run's CMOR ``Amon`` output.

Produces one publication-style figure per run: a global map for each standard
field (near-surface air temperature, precipitation, TOA outgoing SW/LW, total
cloud fraction, column water vapour, surface latent/sensible heat flux), the
per-field zonal means, the top-of-atmosphere energy budget, and a table of
area-weighted global means annotated against Earth observational reference
values.  Radiation columns are area-weighted by the model's own sin-latitude
``areacella`` (reproduced here so the plotter needs only the ``Amon`` NetCDF,
not the separately-written ``fx`` table).

Pure ``numpy`` / ``xarray`` / ``matplotlib`` — imports no ``legoesm`` and runs
on the CMOR output of any AMIP/coupled run.

Usage::

    python scripts/plot/plot_amip_cmor_diagnostics.py <run_dir> [--label L] [--out fig.png]

``<run_dir>`` is the run directory that contains ``cmor/Amon/*.nc``.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import xarray as xr

# --- Field table: (CMOR var, unit-scale, unit label, Earth observational
#     reference value, colormap).  The reference values are OBSERVATIONAL
#     comparison targets for the diagnostics table (annual global means from
#     ERA5 / CERES / GPCP), NOT model physical constants — they parameterise
#     the plot annotation only and are deliberately not sourced from
#     ``legoesm.constants`` (which holds model physics constants). ---
FIELD_TABLE = (
    ("tas", 1.0, "K", 288.0, "RdBu_r"),
    ("pr", 86400.0, "mm/day", 2.9, "YlGnBu"),
    ("rsut", 1.0, "W/m2", 100.0, "viridis"),
    ("rlut", 1.0, "W/m2", 239.0, "magma"),
    ("clt", 1.0, "%", 67.0, "Blues"),
    ("prw", 1.0, "mm", 24.5, "GnBu"),
    ("hfls", 1.0, "W/m2", 88.0, "YlOrRd"),
    ("hfss", 1.0, "W/m2", 20.0, "YlOrRd"),
)
_ALBEDO_REF = 0.29   # observational planetary albedo (CERES); annotation only.


def _sinlat_area_weights(nlat: int, nlon: int) -> np.ndarray:
    """Reproduce the model's ``areacella`` weights (exact sin-latitude bands,
    matching ``driver/diagnostics.py::_write_cmip_fixed_files``), normalised to
    sum to 1 — the ``R_earth**2 * dlon`` prefactor cancels in a weighted mean."""
    sin_edges = np.sin(np.radians(np.linspace(-90.0, 90.0, nlat + 1)))
    band = sin_edges[1:] - sin_edges[:-1]
    w = np.broadcast_to(band[:, None], (nlat, nlon)).astype(np.float64)
    return w / w.sum()


def _load_clim(cmor_amon: str, var: str):
    """Time-mean climatology + lat/lon for a CMOR Amon variable, or Nones."""
    fs = sorted(glob.glob(os.path.join(cmor_amon, f"{var}_Amon_*.nc")))
    if not fs:
        return None, None, None
    ds = xr.open_dataset(fs[0])
    da = ds[var]
    tdim = da.dims[0]
    latn = "lat" if "lat" in ds else ("latitude" if "latitude" in ds else None)
    lonn = "lon" if "lon" in ds else ("longitude" if "longitude" in ds else None)
    lat = ds[latn].values if latn else np.arange(da.shape[-2])
    lon = ds[lonn].values if lonn else np.arange(da.shape[-1])
    clim = da.mean(dim=tdim).values
    return clim, lat, lon


def compute_amip_diagnostics(run_dir: str | Path) -> dict:
    """Load a run's CMOR Amon climatology and reduce to diagnostics.

    Returns a dict with per-field ``maps`` (2-D climatology × unit-scale),
    ``zonal`` (zonal means), ``lat``/``lon``, area-weighted ``global_means``
    (var -> (value, earth_ref, unit)), and the TOA ``budget``
    (rsdt/rsut/rlut/R_TOA/albedo) when the SW/LW TOA fields are present.
    Pure + deterministic — the unit-tested core (no matplotlib)."""
    cmor = os.path.join(str(run_dir), "cmor", "Amon")
    tas0, lat, lon = _load_clim(cmor, "tas")
    if tas0 is None:
        raise FileNotFoundError(f"no tas_Amon_*.nc under {cmor}")
    nlat, nlon = tas0.shape
    w = _sinlat_area_weights(nlat, nlon)

    maps, zonal, gmeans = {}, {}, {}
    for var, sc, unit, ref, _cmap in FIELD_TABLE:
        clim, _la, _lo = _load_clim(cmor, var)
        if clim is None:
            continue
        field = clim * sc
        maps[var] = field
        zonal[var] = field.mean(axis=1)
        gmeans[var] = (float(np.sum(field * w)), ref, unit)

    budget = None
    rsdt, _, _ = _load_clim(cmor, "rsdt")
    rsut, _, _ = _load_clim(cmor, "rsut")
    rlut, _, _ = _load_clim(cmor, "rlut")
    if rsdt is not None and rsut is not None and rlut is not None:
        RSDT = float(np.sum(rsdt * w))
        RSUT = float(np.sum(rsut * w))
        RLUT = float(np.sum(rlut * w))
        budget = {
            "rsdt": RSDT, "rsut": RSUT, "rlut": RLUT,
            "R_TOA": RSDT - RSUT - RLUT,
            "albedo": RSUT / max(RSDT, 1e-6),
        }
    return {
        "grid": (nlat, nlon), "lat": lat, "lon": lon,
        "maps": maps, "zonal": zonal, "global_means": gmeans, "budget": budget,
    }


def plot_amip_cmor_diagnostics(run_dir: str | Path, label: str, out_path: str | Path) -> Path:
    """Render the systematic-diagnostics figure to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag = compute_amip_diagnostics(run_dir)
    lat, lon = diag["lat"], diag["lon"]
    nlat, nlon = diag["grid"]

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 5, hspace=0.42, wspace=0.32)
    fig.suptitle(f"AMIP systematic diagnostics — {label}  (grid {nlat}x{nlon}, climatology)",
                 fontsize=15, y=0.98)

    for i, (var, _sc, unit, ref, cmap) in enumerate(FIELD_TABLE):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[r, c])
        if var not in diag["maps"]:
            ax.text(0.5, 0.5, f"{var}\n(missing)", ha="center", va="center")
            ax.axis("off")
            continue
        gm = diag["global_means"][var][0]
        im = ax.pcolormesh(lon, lat, diag["maps"][var], cmap=cmap, shading="auto")
        ax.set_title(f"{var}  gmean={gm:.2f} {unit}  (Earth~{ref})", fontsize=10)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    axz = fig.add_subplot(gs[0:2, 4])
    for var, _sc, unit, _ref, _cmap in FIELD_TABLE:
        if var in diag["zonal"]:
            zm = diag["zonal"][var]
            rng = np.ptp(zm) or 1.0
            axz.plot((zm - zm.min()) / rng, lat, label=f"{var} ({unit})", lw=1.6)
    axz.set_title("zonal means (each min-max normalised)", fontsize=10)
    axz.set_ylabel("lat"); axz.set_xlabel("normalised value")
    axz.legend(fontsize=7, loc="lower right"); axz.grid(alpha=0.3)

    axb = fig.add_subplot(gs[2, 4])
    b = diag["budget"]
    if b is not None:
        keys = ["rsdt", "rsut", "rlut", "R_TOA"]
        axb.bar(range(4), [b[k] for k in keys],
                color=["gold", "tab:blue", "tab:red", "tab:green"])
        axb.set_xticks(range(4)); axb.set_xticklabels(keys, fontsize=8)
        axb.set_title(f"TOA budget  albedo={b['albedo']:.3f} (Earth {_ALBEDO_REF})\n"
                      f"R_TOA={b['R_TOA']:.1f} W/m2", fontsize=9)
        for j, k in enumerate(keys):
            axb.text(j, b[k], f"{b[k]:.0f}", ha="center", va="bottom", fontsize=8)
        axb.axhline(0, color="k", lw=0.5)

    axt = fig.add_subplot(gs[3, 4]); axt.axis("off")
    rows = [f"{v:5s} {gm:8.2f} {u:6s} (Earth~{rf})"
            for v, (gm, rf, u) in diag["global_means"].items()]
    axt.text(0.0, 1.0, "AREA-WEIGHTED GLOBAL MEANS\n" + "\n".join(rows),
             va="top", ha="left", family="monospace", fontsize=9)

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="run directory containing cmor/Amon/*.nc")
    p.add_argument("--label", default=None, help="figure title label")
    p.add_argument("--out", default=None, help="output PNG path")
    args = p.parse_args(argv)
    label = args.label or os.path.basename(str(args.run_dir).rstrip("/"))
    out = args.out or os.path.join(str(args.run_dir), f"amip_diagnostics_{label}.png")
    saved = plot_amip_cmor_diagnostics(args.run_dir, label, out)
    print(f"SAVED {saved}")


if __name__ == "__main__":
    main()
