#!/usr/bin/env python
"""Full OMIP-vs-NEMO diagnostic battery for a legoESM ocean snapshot.

Complements ``compare_omip_nemo.py`` (surface SST/SSS/MLD scoring) with the
interior + transport diagnostics that surface metrics hide:

  * **Horizontal depth snapshots** of T and S at a set of target depths
    (default 0/100/300/1000 m): model | NEMO | difference.
  * **Zonal-mean depth sections** of T and S (lat vs depth): model | NEMO |
    difference, with the model column linearly interpolated onto the NEMO
    depth axis for the difference panel.
  * **Mixed-layer depth** (reuses ``ocean.diagnostics.mixed_layer_depth``,
    dsigma=0.01 to match NEMO ``mldr10_1``).
  * **NEMO reference transports** (AMOC @ target latitude, ACC @ Drake, global
    meridional heat transport) read via ``nemo_transports`` — printed + saved as
    a reference baseline (the model-side counterparts are archived by the runner
    from the live grid/mesh, which a saved snapshot does not carry).

Grid-agnostic: the legoESM snapshot may be a lat-lon / tripole C-grid 2-D field
or an MPAS / cube flattened cell list — each (lat, lon) sample is regridded to a
common regular lat-lon grid by the SAME inverse-distance kdtree regrid as the
surface scorer (imported, not re-derived).

Usage:
    python scripts/validate/diag_omip_nemo_battery.py \
        --legoesm-snapshot results/omip_nemo/ico7_mld/snapshot_final.npz \
        --nemo-gridt   .../RUN_REF/ORCA1_1m_..._grid_T.nc --nemo-month 3 \
        --nemo-gridv   .../RUN_REF/ORCA1_1y_..._grid_V.nc \
        --nemo-gridu   .../RUN_REF/ORCA1_1y_..._grid_U.nc \
        --domain-cfg   .../INPUTS/.../domain_cfg.nc \
        --grid-label "MPAS ico7" \
        --output-dir results/omip_nemo/battery_ico7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse the surface scorer's curvilinear->latlon IDW regrid + NEMO month
# selection (sibling module in scripts/validate; importing does not run main()).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_omip_nemo import (regrid_curv_to_latlon, _load_nemo, _wstats,  # noqa: E402
                               _band_breakdown)

_TARGET_DEPTHS_M = (0.0, 100.0, 300.0, 1000.0)


def _load_legoesm_3d(path):
    """Load a legoESM snapshot's 3-D T/S + geometry (grid-agnostic).

    Returns flattened-sample arrays: ``T``/``S`` shape ``(ncells, nlev)``,
    ``lat``/``lon``/``mask``/``H_bathy`` shape ``(ncells,)``, ``z`` the
    level-centre depths ``(nlev,)`` (positive down)."""
    s = np.load(path)
    keys = set(getattr(s, "files", []))
    if "z_center_ref" not in keys:
        raise KeyError(f"{path} lacks z_center_ref (needed for interior "
                       "diagnostics); re-run with a current run_omip_core2.")
    T = np.asarray(s["T"], dtype=np.float64)
    S = np.asarray(s["S"], dtype=np.float64)
    # Flatten any leading spatial dims to a single cell axis; keep level last.
    nlev = T.shape[-1]
    T = T.reshape(-1, nlev)
    S = S.reshape(-1, nlev)
    lat = np.asarray(s["lat_T"], dtype=np.float64).ravel()
    lon = np.asarray(s["lon_T"], dtype=np.float64).ravel()
    mask = np.asarray(s["land_mask"], dtype=np.float64).ravel()
    Hb = (np.asarray(s["H_bathy"], dtype=np.float64).ravel()
          if "H_bathy" in keys else np.full(lat.shape, np.inf))
    z = np.abs(np.asarray(s["z_center_ref"], dtype=np.float64).ravel())  # +down
    if not (T.shape[0] == lat.shape[0] == lon.shape[0] == mask.shape[0]):
        raise ValueError(f"snapshot cell-count mismatch: T={T.shape} "
                         f"lat={lat.shape} mask={mask.shape}")
    # _save_snapshot writes lat_T/lon_T in DEGREES; guard against a radian
    # snapshot silently regridding to the wrong geography (codex).
    if np.nanmax(np.abs(lat)) > 90.5:
        raise ValueError(f"snapshot lat_T max |lat|={np.nanmax(np.abs(lat)):.1f} "
                         "deg > 90: expected degrees, got radians? Refusing to "
                         "regrid to a scrambled geography.")
    # Per-level WET mask: the model fills below-seafloor cells with the deepest
    # ACTIVE value (finite, not NaN), so finiteness alone does NOT exclude rock.
    # A level is wet where its TOP interface is above the local sea floor AND in
    # the ocean.  Using the top interface (NOT the cell centre) keeps thin bottom
    # PARTIAL cells whose centre is below H_bathy but whose top is above it --
    # the centre-vs-bathy test silently drops those shelf/slope cells (codex
    # HIGH).  z_half_ref is not stored in the snapshot, so reconstruct the top
    # interface as the midpoint between adjacent centres (surface at 0).
    z_top = np.empty_like(z)
    z_top[0] = 0.0
    z_top[1:] = 0.5 * (z[:-1] + z[1:])
    wet3d = ((z_top[None, :] < Hb[:, None]) & (mask[:, None] > 0.5)).astype(np.float64)
    return {"T": T, "S": S, "lat": lat, "lon": lon, "mask": mask,
            "H_bathy": Hb, "z": z, "nlev": nlev, "wet3d": wet3d}


def _load_nemo_3d(path, month, tidx):
    """NEMO 3-D potential temperature + salinity + level depths.

    Reuses ``_load_nemo`` for the surface fields / mask / month selection and
    adds the 3-D ``thetao``/``so`` (name fallbacks) averaged the same way."""
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)
    # NEMO 5.0.x writes potential temperature as ``to`` and salinity ``so``;
    # older streams use thetao/toce/votemper and soce/vosaline.
    tname = next((n for n in ("thetao", "toce", "votemper", "to") if n in ds), None)
    sname = next((n for n in ("so", "soce", "vosaline") if n in ds), None)
    if tname is None or sname is None:
        raise KeyError(f"{path}: no 3-D T ({tname}) or S ({sname}) variable")
    zname = next((n for n in ("deptht", "olevel", "z", "nav_lev")
                  if n in ds[tname].dims or n in ds), None)
    tdim = "time_counter" if "time_counter" in ds[tname].dims else None
    if month is not None and tdim is not None:
        from compare_omip_nemo import _nemo_record_months
        nt = int(ds.sizes[tdim])
        rec = _nemo_record_months(ds, tdim, nt)
        midx = ([i for i in range(nt) if rec[i] == month] if rec is not None
                else list(range(month - 1, nt, 12)))
        sel = lambda v: ds[v].isel({tdim: midx}).mean(dim=tdim)
    else:
        sel = (lambda v: ds[v].isel({tdim: tidx}) if tdim else ds[v])

    # Horizontal dim names are taken from nav_lat (the 2-D lat coord) so T/S are
    # transposed to (depth, y, x) consistently with lat/lon -- a variable whose
    # source order is (depth, x, y) would otherwise become a transposed map
    # while nav_lat/nav_lon stay (y, x) (codex).
    nav = ds["nav_lat"]
    yx_dims = [d for d in nav.dims]                  # (y, x) for ORCA nav_lat

    def _to_zyx(da):
        """Transpose a DataArray to (depth, y, x) by DIMENSION NAME."""
        dims = list(da.dims)
        zd = zname if (zname in dims) else next(d for d in dims if d not in yx_dims)
        ordered = [zd] + [d for d in yx_dims if d in dims]
        # Append any leftover dims (defensive) so transpose covers all axes.
        ordered += [d for d in dims if d not in ordered]
        return np.asarray(da.transpose(*ordered).values, dtype=np.float64)

    T3 = _to_zyx(sel(tname))
    S3 = _to_zyx(sel(sname))
    z = (np.asarray(ds[zname].values, dtype=np.float64).ravel()
         if zname is not None and zname in ds else None)
    if z is None:                      # depth axis is the leading T3 axis order
        z = np.arange(T3.shape[0], dtype=np.float64)
    lat = np.asarray(ds["nav_lat"].values, dtype=np.float64)
    lon = np.asarray(ds["nav_lon"].values, dtype=np.float64) % 360.0
    # NEMO land/fill is NaN after CF-decode -> finite == ocean.
    return {"T": np.where(np.isfinite(T3), T3, np.nan),
            "S": np.where(np.isfinite(S3), S3, np.nan),
            "z": np.abs(z), "lat": lat, "lon": lon}


def _nearest_level(z_axis, target_m):
    return int(np.argmin(np.abs(np.asarray(z_axis) - target_m)))


def _regrid_level(field2d_or_1d, lat, lon, mask, tgt_lat, tgt_lon):
    """IDW-regrid one level; cells finite & wet are the source ocean cells.

    A level entirely below the sea floor (no finite/wet source cell -- e.g. the
    abyssal NEMO levels deeper than any ocean column) has nothing to regrid;
    return an all-NaN field + zero ocean flag rather than raising, so a full
    depth-column section loop tolerates its empty bottom levels."""
    fin = np.isfinite(field2d_or_1d).ravel() & (np.asarray(mask).ravel() > 0.5)
    if not fin.any():
        shape = (tgt_lat.size, tgt_lon.size)
        return np.full(shape, np.nan), np.zeros(shape)
    src = np.where(np.isfinite(field2d_or_1d), field2d_or_1d, 0.0)
    out, ocean = regrid_curv_to_latlon(src, lat, lon, fin.astype(np.float64),
                                       tgt_lat, tgt_lon)
    return out, ocean


def _zonal_section(T_levels, lat, lon, wet3d, z_axis, tgt_lat, tgt_lon):
    """Zonal-mean (over ocean lon) section: returns array (nlev, n_tgt_lat).

    ``wet3d`` is a per-level wet mask ``(ncells, nlev)`` so below-seafloor
    cells are excluded level-by-level (NOT a single 2-D land mask, which would
    let shelf columns contribute fabricated values below their sea floor)."""
    nlev = T_levels.shape[-1]
    sec = np.full((nlev, tgt_lat.size), np.nan)
    for k in range(nlev):
        g, oc = _regrid_level(T_levels[..., k], lat, lon, wet3d[..., k],
                              tgt_lat, tgt_lon)
        gm = np.where(oc > 0.5, g, np.nan)
        with np.errstate(invalid="ignore"):
            sec[k] = np.nanmean(gm, axis=1)
    return sec


def _interp_to_depths(sec, z_src, z_tgt):
    """Linear-interp a (nlev_src, nlat) section onto z_tgt depths per column."""
    out = np.full((z_tgt.size, sec.shape[1]), np.nan)
    for j in range(sec.shape[1]):
        col = sec[:, j]
        good = np.isfinite(col)
        if good.sum() >= 2:
            out[:, j] = np.interp(z_tgt, z_src[good], col[good],
                                  left=col[good][0], right=np.nan)
    return out


def _plot_depth_snapshots(out_dir, name, units, tgt_lat, tgt_lon,
                          model_slices, nemo_slices, depths, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nd = len(depths)
    fig, ax = plt.subplots(nd, 3, figsize=(15, 3.4 * nd), squeeze=False)
    for i, d in enumerate(depths):
        Lm, ocL = model_slices[i]
        Nm, ocN = nemo_slices[i]
        oc = (ocL > 0.5) & (ocN > 0.5)
        L = np.where(oc, Lm, np.nan); N = np.where(oc, Nm, np.nan)
        vmin = np.nanpercentile([L, N], 1); vmax = np.nanpercentile([L, N], 99)
        for a, dat, ttl in [(ax[i, 0], L, f"{label}"), (ax[i, 1], N, "NEMO")]:
            im = a.pcolormesh(tgt_lon, tgt_lat, dat, vmin=vmin, vmax=vmax,
                              cmap="RdYlBu_r", shading="auto")
            a.set_title(f"{ttl} {name} @ {d:.0f} m"); plt.colorbar(im, ax=a, shrink=0.8)
        dd = L - N
        dm = np.nanpercentile(np.abs(dd), 98) or 1.0
        im = ax[i, 2].pcolormesh(tgt_lon, tgt_lat, dd, vmin=-dm, vmax=dm,
                                 cmap="RdBu_r", shading="auto")
        ax[i, 2].set_title(f"Δ {name} @ {d:.0f} m ({label}-NEMO)")
        plt.colorbar(im, ax=ax[i, 2], shrink=0.8)
    fig.suptitle(f"{label} vs NEMO — {name} [{units}] horizontal snapshots",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_depth_snapshots.png", dpi=85)
    plt.close(fig)


def _plot_section(out_dir, name, units, tgt_lat, zL, secL, zN, secN, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    secL_on_N = _interp_to_depths(secL, zL, zN)
    diff = secL_on_N - secN
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    # Shared colour scale over BOTH sections. secL (model nlev) and secN (NEMO
    # nlev) have DIFFERENT shapes, so stack the FINITE values flat rather than
    # np.nanpercentile([secL, secN]) (which builds an inhomogeneous array).
    both = np.concatenate([secL[np.isfinite(secL)].ravel(),
                           secN[np.isfinite(secN)].ravel()])
    vmin, vmax = ((np.percentile(both, 2), np.percentile(both, 98))
                  if both.size else (0.0, 1.0))
    for a, z, sec, ttl in [(ax[0], zL, secL, label), (ax[1], zN, secN, "NEMO")]:
        im = a.pcolormesh(tgt_lat, z, sec, vmin=vmin, vmax=vmax,
                          cmap="RdYlBu_r", shading="auto")
        a.set_title(f"{ttl} zonal-mean {name}"); a.set_ylabel("depth [m]")
        a.set_xlabel("lat"); a.invert_yaxis(); plt.colorbar(im, ax=a, shrink=0.85)
    dm = np.nanpercentile(np.abs(diff), 98) if np.isfinite(diff).any() else 1.0
    dm = float(dm) if (np.isfinite(dm) and dm > 0) else 1.0
    im = ax[2].pcolormesh(tgt_lat, zN, diff, vmin=-dm, vmax=dm, cmap="RdBu_r",
                          shading="auto")
    ax[2].set_title(f"Δ zonal-mean {name} ({label}-NEMO)")
    ax[2].set_ylabel("depth [m]"); ax[2].set_xlabel("lat"); ax[2].invert_yaxis()
    plt.colorbar(im, ax=ax[2], shrink=0.85)
    fig.suptitle(f"{label} vs NEMO — zonal-mean {name} [{units}] section",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_zonal_section.png", dpi=90)
    plt.close(fig)
    # RMSE on the common (NEMO) depth grid, area-unweighted but lat-cos weighted.
    w = np.cos(np.deg2rad(tgt_lat))[None, :] * np.ones_like(diff)
    fin = np.isfinite(diff)
    rmse = float(np.sqrt(np.nansum((diff[fin]) ** 2 * w[fin]) / np.nansum(w[fin]))) \
        if fin.any() else float("nan")
    return rmse


def _nemo_transports(args):
    """NEMO reference AMOC/ACC/MHT scalars (reuse nemo_transports). Best-effort."""
    rep = {}
    try:
        from nemo_transports import (nemo_amoc_at_latitude, nemo_acc_drake,
                                     nemo_mht)
    except Exception as e:                                   # pragma: no cover
        return {"error": f"nemo_transports import failed: {e}"}
    dc = args.domain_cfg
    if args.nemo_gridv and dc:
        try:
            rep["AMOC"] = nemo_amoc_at_latitude(str(args.nemo_gridv), str(dc),
                                                target_lat=args.amoc_lat)
        except Exception as e:
            rep["AMOC_error"] = str(e)
    if args.nemo_gridu and dc:
        try:
            rep["ACC_Drake"] = nemo_acc_drake(str(args.nemo_gridu), str(dc))
        except Exception as e:
            rep["ACC_error"] = str(e)
    if args.nemo_gridv and args.nemo_gridt and dc:
        try:
            m = nemo_mht(str(args.nemo_gridv), str(args.nemo_gridt), str(dc))
            rep["MHT"] = {k: m[k] for k in ("nh_peak_PW", "nh_peak_lat",
                                            "sh_min_PW", "sh_min_lat")}
        except Exception as e:
            rep["MHT_error"] = str(e)
    return rep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legoesm-snapshot", type=Path, required=True)
    p.add_argument("--nemo-gridt", type=Path, required=True)
    p.add_argument("--nemo-gridv", type=Path, default=None)
    p.add_argument("--nemo-gridu", type=Path, default=None)
    p.add_argument("--domain-cfg", type=Path, default=None)
    p.add_argument("--nemo-month", type=int, default=None)
    p.add_argument("--nemo-time-idx", type=int, default=-1)
    p.add_argument("--amoc-lat", type=float, default=26.5)
    p.add_argument("--res-deg", type=float, default=1.0)
    p.add_argument("--depths-m", type=str, default="0,100,300,1000",
                   help="Comma-separated target depths [m] for horizontal snapshots.")
    p.add_argument("--grid-label", type=str, default="legoESM")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/omip_nemo/battery"))
    args = p.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    depths = [float(x) for x in args.depths_m.split(",") if x.strip()]

    L = _load_legoesm_3d(args.legoesm_snapshot)
    N = _load_nemo_3d(args.nemo_gridt, args.nemo_month, args.nemo_time_idx)
    print(f"[load] legoESM T{L['T'].shape} nlev={L['nlev']} | "
          f"NEMO T{N['T'].shape} nlev={N['z'].size}")

    r = args.res_deg
    tgt_lat = np.arange(-89.5, 90.0, r)
    tgt_lon = np.arange(0.5, 360.0, r)

    report = {"legoesm_snapshot": str(args.legoesm_snapshot),
              "nemo_gridt": str(args.nemo_gridt), "depths_m": depths,
              "section_rmse": {}}

    # --- horizontal depth snapshots (T, S) ---
    for fld, units in (("T", "degC"), ("S", "psu")):
        mod_sl, nem_sl = [], []
        for d in depths:
            kL = _nearest_level(L["z"], d)
            kN = _nearest_level(N["z"], d)
            mod_sl.append(_regrid_level(L[fld][..., kL], L["lat"], L["lon"],
                                        L["wet3d"][..., kL], tgt_lat, tgt_lon))
            nem_sl.append(_regrid_level(N[fld][kN], N["lat"], N["lon"],
                                        np.isfinite(N[fld][kN]).astype(float),
                                        tgt_lat, tgt_lon))
        name = "SST/T" if fld == "T" else "SSS/S"
        _plot_depth_snapshots(out, fld, units, tgt_lat, tgt_lon, mod_sl, nem_sl,
                              depths, args.grid_label)
        print(f"[snapshots] {fld} at depths {depths} -> {fld}_depth_snapshots.png")
        # Per-band bias at each depth: does the surface cold/salty bias reach into
        # the thermocline (advective / large-scale) or stay confined to the mixed
        # layer above it (surface-flux-driven)?  Same latitude bands as the surface
        # scorer, reusing _band_breakdown on the already-common-grid level slices.
        depth_bands = {}
        for i, d in enumerate(depths):
            Lm, ocL = mod_sl[i]
            Nm, ocN = nem_sl[i]
            area = (np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones_like(tgt_lon)[None, :]
                    * ((ocL > 0.5) & (ocN > 0.5)))
            depth_bands[f"{d:.0f}m"] = _band_breakdown(Lm, Nm, area, tgt_lat)
        report[f"{fld}_depth_bands"] = depth_bands
        print(f"[{fld} bands by depth]")
        for dk, bands in depth_bands.items():
            nh = bands.get("NH_midlat_23N_45N")
            if nh is not None:
                print(f"  {dk:>6s}  NH-midlat bias {nh['bias']:+6.2f} {units}  "
                      f"(lego {nh['lego_mean']:.2f} nemo {nh['nemo_mean']:.2f})")

    # --- zonal-mean depth sections (T, S) ---
    for fld, units in (("T", "degC"), ("S", "psu")):
        secL = _zonal_section(L[fld], L["lat"], L["lon"], L["wet3d"], L["z"],
                              tgt_lat, tgt_lon)
        # NEMO levels: regrid each then zonal-mean (build a (lat,lon) per level).
        nlevN = N["z"].size
        secN = np.full((nlevN, tgt_lat.size), np.nan)
        for k in range(nlevN):
            g, oc = _regrid_level(N[fld][k], N["lat"], N["lon"],
                                  np.isfinite(N[fld][k]).astype(float),
                                  tgt_lat, tgt_lon)
            with np.errstate(invalid="ignore"):
                secN[k] = np.nanmean(np.where(oc > 0.5, g, np.nan), axis=1)
        rmse = _plot_section(out, fld, units, tgt_lat, L["z"], secL, N["z"],
                             secN, args.grid_label)
        report["section_rmse"][fld] = rmse
        print(f"[section] {fld} zonal-mean section RMSE={rmse:.3f} {units} "
              f"-> {fld}_zonal_section.png")

    # --- NEMO reference transports ---
    tr = _nemo_transports(args)
    report["nemo_transports"] = tr
    if tr:
        print(f"[transports] NEMO reference: {json.dumps(tr, default=float)}")

    (out / "battery_report.json").write_text(json.dumps(report, indent=2,
                                                        default=float))
    print(f"[done] battery -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
