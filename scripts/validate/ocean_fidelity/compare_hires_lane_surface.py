"""Surface-field agreement between the high-resolution ocean lanes.

FESOM2-JAX (FORCA20 daily store) is the reference; the MPAS and tripole lanes
(``snapshot_final.npz`` from ``run_omip``) are scored against it on one common
lon-lat raster built with the SAME binning for every lane
(``plot_mpas_omip_snapshot.raster_mean``), so the only thing that differs
between columns is the model.  Only raster cells resolved by the reference AND
EVERY lane are scored, with the cos(lat)-weighted statistics of
``compare_omip_nemo`` (``_wstats`` / ``_band_breakdown``), so the numbers are
the same statistic the NEMO scorecard reports.

Sampling caveat, declared in the report: a FESOM daily store is the MEAN over
that calendar day, a lane snapshot is the INSTANTANEOUS state at the end of
its run; the script refuses to score a lane whose end day is not the FESOM
store's day.  Non-finite wet values are fatal (a blown-up lane is not scored
on the cells that survived).

Optionally also compares the global-mean time series: the FESOM daily stores
(area-weighted with the mesh's node cluster area) against each lane's
``mean_timeseries.csv`` -- which ``run_omip`` computes as an UNWEIGHTED mean
over wet cells; the two are the same statistic only on a uniform mesh.

Usage::

    python scripts/validate/ocean_fidelity/compare_hires_lane_surface.py \
        --fesom-daily RUN/daily/day_1958_030 --mesh-dir MESH \
        --lane mpas=RUN/mpas/ico9/snapshot_final.npz \
        --series mpas=RUN/mpas/ico9/mean_timeseries.csv \
        --fesom-daily-root RUN/daily --out-dir OUT
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "scripts" / "plot"))
sys.path.insert(0, str(_ROOT / "scripts" / "validate"))
from plot_mpas_omip_snapshot import raster_mean  # noqa: E402
from compare_omip_nemo import _band_breakdown, _wstats  # noqa: E402

FIELDS = ("sst", "sss", "ssh")


def load_fesom_daily(store, mesh_dir):
    import zarr
    g = zarr.open_group(str(store), mode="r")
    area = np.load(Path(mesh_dir) / "area.npy")[:, 0]
    date = str(dict(g.attrs).get("calendar_date", ""))
    m = re.fullmatch(r"(\d{4})-doy(\d{3})", date)
    if not m:
        raise ValueError(f"{store}: calendar_date {date!r} is not a daily store (YYYY-doyDDD)")
    out = {k: np.asarray(g[k][0], dtype=np.float64) for k in FIELDS}
    out["lon"] = np.asarray(g["lon"][:], dtype=np.float64)
    out["lat"] = np.asarray(g["lat"][:], dtype=np.float64)
    out["area"] = area
    out["day"] = int(m.group(2))
    out["label"] = f"FESOM {date} (daily mean)"
    return out


def load_lane_snapshot(path):
    s = np.load(path, allow_pickle=False)
    lat, lon = s["lat_T"], s["lon_T"]
    # run_omip stores ``land_mask`` with 1 = WET on every grid (its own
    # diagnostics and compare_mld_dbm.py read it as ``> 0.5``).
    wet = s["land_mask"] > 0.5
    if s["T"].ndim == lat.ndim + 1:          # (..., nlev) layout, all lanes
        sst, sss = s["T"][..., 0], s["S"][..., 0]
    else:
        raise ValueError(f"{path}: T has shape {s['T'].shape} for lat_T {lat.shape}")
    if np.nanmax(np.abs(lat)) <= np.pi + 1e-6:
        raise ValueError(f"{path}: lat_T looks like radians; the writer must emit degrees")
    out = {"lon": lon[wet], "lat": lat[wet], "sst": sst[wet], "sss": sss[wet],
           "ssh": s["eta"][wet], "day": float(s["_time_s"]) / 86400.0}
    bad = {k: int((~np.isfinite(out[k])).sum()) for k in FIELDS}
    if any(bad.values()):
        raise ValueError(f"{path}: non-finite wet values {bad}; a blown-up lane is not scored")
    out["label"] = f"{Path(path).parents[1].name} day {out['day']:g} (instantaneous)"
    return out


def raster(d, res_deg):
    grids = {}
    for k in FIELDS:
        lon_e, lat_e, grids[k] = raster_mean(d["lon"], d["lat"], d[k], res_deg)
    return lon_e, lat_e, grids


def common_mask(ref_grids, lane_grids_list):
    m = np.ones_like(ref_grids["sst"], dtype=bool)
    for grids in (ref_grids, *lane_grids_list):
        for k in FIELDS:
            m &= np.isfinite(grids[k])
    return m


def score(ref, lane, lat_c, mask):
    area = np.cos(np.deg2rad(lat_c))[:, None] * mask
    if area.sum() <= 0:
        raise ValueError("no raster cell is resolved by the reference and every lane")
    stats = {}
    for k in FIELDS:
        w = _wstats(lane[k], ref[k], area)
        stats[k] = {"lane_mean": w["lego_mean"], "ref_mean": w["nemo_mean"], "bias": w["bias"],
                    "rms": w["rmse"], "corr": (w["corr"] if np.isfinite(w["corr"]) else None),
                    "n_cells": int(mask.sum()),
                    "band_bias": {b: (None if v is None else v["bias"])
                                  for b, v in _band_breakdown(lane[k], ref[k], area, lat_c).items()}}
    return stats


def fesom_series(root, mesh_dir, n_days):
    import zarr
    area = np.load(Path(mesh_dir) / "area.npy")[:, 0]
    rows = []
    for d in range(1, n_days + 1):
        store = Path(root) / f"day_1958_{d:03d}"
        if not store.exists():
            break
        g = zarr.open_group(str(store), mode="r")
        rows.append([float(d)] + [float((np.asarray(g[k][0], dtype=np.float64) * area).sum()
                                        / area.sum()) for k in ("sst", "sss")])
    return np.asarray(rows)


def lane_series(csv):
    a = np.genfromtxt(csv, delimiter=",", names=True)
    return np.column_stack([a["time_days"], a["mean_SST"], a["mean_SSS"]])


def plot(out_dir, lon_e, lat_e, ref, lanes, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 1 + 2 * len(lanes)
    fig, axes = plt.subplots(len(FIELDS) + (1 if series else 0), ncol,
                             figsize=(5.2 * ncol, 3.2 * (len(FIELDS) + 1)), squeeze=False)
    lim = {"sst": (-2, 32, 3), "sss": (30, 38, 1.5), "ssh": (-2, 2, 0.5)}
    for r, k in enumerate(FIELDS):
        lo, hi, dl = lim[k]
        m = axes[r, 0].pcolormesh(lon_e, lat_e, ref["grids"][k], vmin=lo, vmax=hi, cmap="viridis")
        axes[r, 0].set_title(f"{ref['label']} {k}")
        fig.colorbar(m, ax=axes[r, 0])
        for c, (name, L) in enumerate(lanes.items()):
            m = axes[r, 1 + 2 * c].pcolormesh(lon_e, lat_e, L["grids"][k], vmin=lo, vmax=hi, cmap="viridis")
            axes[r, 1 + 2 * c].set_title(f"{L['label']} {k}")
            fig.colorbar(m, ax=axes[r, 1 + 2 * c])
            d = np.where(ref["mask"], L["grids"][k] - ref["grids"][k], np.nan)
            m = axes[r, 2 + 2 * c].pcolormesh(lon_e, lat_e, d, vmin=-dl, vmax=dl, cmap="RdBu_r")
            s = L["stats"][k]
            axes[r, 2 + 2 * c].set_title(f"{name} - FESOM {k}: bias {s['bias']:+.3f} rms {s['rms']:.3f}")
            fig.colorbar(m, ax=axes[r, 2 + 2 * c])
    if series:
        for j, (k, col) in enumerate((("sst", 1), ("sss", 2))):
            ax = axes[-1, j]
            for name, arr in series.items():
                ax.plot(arr[:, 0], arr[:, col], label=name)
            ax.set_xlabel("day"); ax.set_ylabel(f"global mean {k}"); ax.legend(); ax.grid(alpha=0.3)
            ax.set_title("FESOM area-weighted; lanes unweighted wet-cell mean (run_omip csv)", fontsize=8)
        for j in range(2, ncol):
            axes[-1, j].axis("off")
    fig.tight_layout()
    png = Path(out_dir) / "hires_lane_surface.png"
    fig.savefig(png, dpi=110)
    return png


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fesom-daily", required=True, help="FESOM daily store at the matched day")
    p.add_argument("--mesh-dir", required=True)
    p.add_argument("--lane", action="append", default=[], help="name=snapshot_final.npz (repeatable)")
    p.add_argument("--series", action="append", default=[], help="name=mean_timeseries.csv (repeatable)")
    p.add_argument("--fesom-daily-root", default=None, help="daily/ root for the FESOM global-mean series")
    p.add_argument("--series-days", type=int, default=60)
    p.add_argument("--res-deg", type=float, default=1.0)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ref = load_fesom_daily(a.fesom_daily, a.mesh_dir)
    lon_e, lat_e, ref["grids"] = raster(ref, a.res_deg)
    lat_c = 0.5 * (lat_e[1:] + lat_e[:-1])
    lanes = {}
    for spec in a.lane:
        name, path = spec.split("=", 1)
        L = load_lane_snapshot(path)
        if abs(L["day"] - ref["day"]) > 1e-6:
            raise ValueError(f"lane {name} ends at day {L['day']:g} but the FESOM store is day {ref['day']}")
        _, _, L["grids"] = raster(L, a.res_deg)
        lanes[name] = L
    ref["mask"] = common_mask(ref["grids"], [L["grids"] for L in lanes.values()])
    for L in lanes.values():
        L["stats"] = score(ref["grids"], L["grids"], lat_c, ref["mask"])
    series = {}
    if a.fesom_daily_root:
        series["fesom"] = fesom_series(a.fesom_daily_root, a.mesh_dir, a.series_days)
    for spec in a.series:
        name, path = spec.split("=", 1)
        series[name] = lane_series(path)

    sha = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    report = {"git_sha": sha, "args": vars(a), "ref": ref["label"],
              "sampling": "reference = FESOM daily mean over the calendar day; lanes = instantaneous "
                          "end-of-run state at the same day number",
              "series_note": "fesom = node-area-weighted mean; lanes = run_omip unweighted wet-cell mean",
              "lanes": {n: {"label": L["label"], "stats": L["stats"]} for n, L in lanes.items()},
              "series_end": {n: {"day": float(s[-1, 0]), "sst": float(s[-1, 1]), "sss": float(s[-1, 2]),
                                 "sst_day1": float(s[0, 1]), "sss_day1": float(s[0, 2])}
                             for n, s in series.items() if len(s)}}
    (out_dir / "hires_lane_surface.json").write_text(json.dumps(report, indent=1, allow_nan=False))
    png = plot(out_dir, lon_e, lat_e, ref, lanes, series)
    print(f"reference {ref['label']}; {int(ref['mask'].sum())} raster cells resolved by every lane")
    print(f"{'lane':10s} {'field':4s} {'ref_mean':>9s} {'lane_mean':>9s} {'bias':>8s} {'rms':>7s} {'corr':>6s}")
    for n, L in lanes.items():
        for k in FIELDS:
            s = L["stats"][k]
            print(f"{n:10s} {k:4s} {s['ref_mean']:9.3f} {s['lane_mean']:9.3f} {s['bias']:+8.3f} {s['rms']:7.3f} {(s['corr'] if s['corr'] is not None else float('nan')):6.3f}"
                  + "  band bias: " + " ".join(f"{b}={v:+.3f}" for b, v in s["band_bias"].items() if v is not None))
    for n, v in report["series_end"].items():
        print(f"series {n}: day {v['day']:g} sst {v['sst_day1']:.3f} -> {v['sst']:.3f}  sss {v['sss_day1']:.4f} -> {v['sss']:.4f}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
