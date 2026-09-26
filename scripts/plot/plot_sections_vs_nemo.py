"""Depth cross-sections of temperature and salinity: tripole | MPAS | NEMO.

The surface scorecards (``compare_three_way_nemo``) and the four-way map figure
answer "how far is each grid from NEMO AT THE SURFACE".  Neither looks below it,
and the campaign's two open questions -- the equatorial thermocline behind the
cold-tongue bias, and whether the polar salinity agreement is skin-deep or holds
through the water column -- are both questions about the INTERIOR.  This plots
the interior on the same footing the surface figures already use.

SECTIONS PER FIELD
------------------
* GLOBAL ZONAL MEAN, depth vs latitude.  Every cell in a row of the regular
  lat-lon target sits at the same latitude, so a plain row mean is already the
  area-weighted zonal mean -- no cos(lat) factor belongs here (it belongs in a
  GLOBAL mean, which this is not).
* EQUATORIAL PACIFIC, depth vs longitude, averaged over ``--eq-halfwidth``
  degrees either side of the equator.  cos(lat) varies by under 0.1% across a
  2-degree band, so the plain mean is the area-weighted one to well past
  plotting precision.
* Repeatable BASIN sections at a fixed latitude (``--lat-section``) and
  MERIDIONAL sections at a fixed longitude (``--lon-section``).  The
  meridional one is the cold-tongue view: a section along the equator shows
  the thermocline's east-west slope but not how tightly it is confined to the
  equator, and a global zonal mean averages the two together.

WHAT IS HELD FIXED
------------------
All three sources go through the SAME IDW regridder onto the SAME lat-lon
target as the surface scorecard (``compare_omip_nemo.regrid_curv_to_latlon``),
level by level, and only cells resolved and finite on ALL THREE are drawn.  A
difference panel over cells where one source is missing is not a bias, so the
common mask is applied before any mean is taken.

THE VERTICAL AXIS IS ASSERTED, NOT INTERPOLATED.  Both our grids carry
``z_center_ref`` and NEMO carries ``deptht``; on eORCA1 these are the same 75
levels.  Interpolating one onto the other would be a silent choice of vertical
remapping, so instead the two axes are required to agree and the run fails
loudly if they ever stop agreeing.

Cells below the sea floor are excluded by the model's own bathymetry
(``H_bathy``) rather than by a magnitude test on the tracer.  A level counts as
wet where its CENTRE is above the sea floor, which drops the deepest partial
cell of each column -- conservative, and it never admits a dry cell.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "validate"))
sys.path.insert(0, str(_HERE.parents[1] / "validate" / "ocean_fidelity"))
from compare_omip_nemo import _load_legoesm, regrid_curv_to_latlon  # noqa: E402
# Nearest-wet classification, reused from the three-way scorecard rather than
# re-derived. The regridder's own flag is a DISTANCE-TO-DATA flag, not a
# land/sea test: a target cell on land or below the sea floor keeps it as long
# as some wet source lies within 2.5 deg, and then carries a value
# extrapolated from offshore. Reusing the regridder without also reusing this
# exclusion was a real defect in the first version of this plotter -- it let
# extrapolated coastline and bathymetry artifacts into the difference panels.
# Private-name import matches the sibling plotter (plot_fourway_grids_nemo).
from compare_three_way_nemo import _nn_wet_mask as nn_wet_mask  # noqa: E402

# NEMO writes Conservative Temperature under several names depending on
# version and XIOS field definition; likewise practical/absolute salinity.
# Checked in this order and NAMED in the error, because a silent fallback to a
# surface-only field would plot a 2-D variable as if it were a section.
_T3D_CANDS = ("thetao", "bigthetao", "votemper", "toce", "to", "temperature")
_S3D_CANDS = ("so", "soce", "vosaline", "salinity", "sosaline")
_DEPTH_CANDS = ("deptht", "olevel", "depth", "nav_lev")


def _pick(ds, cands, kind, need_dims=3):
    for name in cands:
        if name in ds.variables and ds[name].ndim >= need_dims:
            return name
    raise SystemExit(
        f"FATAL: no {kind} variable among {cands} with >={need_dims} dims.\n"
        f"       present: {sorted(ds.variables)}")


def load_nemo_3d(path, tidx):
    """NEMO 3-D T and S at one time record, plus coordinates.

    Land and below-floor cells are already NaN here (xarray CF-decodes
    ``_FillValue``), so finiteness alone is the wet test -- the same rule the
    surface scorecard uses.
    """
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)
    print(f"[nemo] variables in {Path(path).name}:\n       {sorted(ds.variables)}")
    tname = _pick(ds, _T3D_CANDS, "3-D temperature")
    sname = _pick(ds, _S3D_CANDS, "3-D salinity")
    dname = _pick(ds, _DEPTH_CANDS, "depth coordinate", need_dims=1)
    print(f"[nemo] using T={tname} S={sname} depth={dname} time_idx={tidx}")

    def sel(v):
        da = ds[v]
        tdim = next((d for d in da.dims if d.startswith("time")), None)
        if tdim is not None:
            da = da.isel({tdim: tidx})
        return np.asarray(da, dtype=np.float64)

    T = sel(tname)
    S = sel(sname)
    if T.ndim != 3 or S.ndim != 3:
        raise SystemExit(f"FATAL: expected (depth, y, x) after time select, "
                         f"got T{T.shape} S{S.shape}")
    depth = np.asarray(ds[dname], dtype=np.float64).squeeze()
    mld = sel("mldr10_1") if "mldr10_1" in ds.variables else None
    return {"T3d": T, "S3d": S, "depth": depth, "mld": mld,
            "lat": np.asarray(ds["nav_lat"], dtype=np.float64),
            "lon": np.asarray(ds["nav_lon"], dtype=np.float64) % 360.0}


def lego_wet_3d(src, z):
    """(nlev,) list of wet masks on the source cloud/curvilinear frame.

    ``mask`` is the surface ocean flag the scorecard already uses (>0.5 ocean);
    ``H_bathy`` is the column depth in metres.
    """
    surf = np.asarray(src["mask"]) > 0.5
    H = src.get("H_bathy")
    if H is None:
        raise SystemExit("FATAL: snapshot has no H_bathy; cannot mask below "
                         "the sea floor without inventing a rule")
    H = np.asarray(H, dtype=np.float64)
    return [surf & (H > zk) for zk in z]


def regrid_column(field_lastaxis, src, wet_by_level, tgt_lat, tgt_lon, res_deg):
    """Regrid a (..., nlev) source field level by level onto the lat-lon target.

    Returns (values (nlat, nlon, nlev), resolved flag (nlat, nlon, nlev)).
    """
    nlev = field_lastaxis.shape[-1]
    out = np.full((tgt_lat.size, tgt_lon.size, nlev), np.nan)
    okf = np.zeros_like(out)
    for k in range(nlev):
        wet = wet_by_level[k]
        if not wet.any():
            continue
        v, ok = regrid_curv_to_latlon(field_lastaxis[..., k], src["lat"],
                                      src["lon"], wet.astype(np.float64),
                                      tgt_lat, tgt_lon)
        out[..., k] = v
        okf[..., k] = ok * nn_wet_mask(src["lat"], src["lon"], wet,
                                       tgt_lat, tgt_lon)
    return out, okf


def regrid_nemo(field_levfirst, lat, lon, tgt_lat, tgt_lon):
    nlev = field_levfirst.shape[0]
    out = np.full((tgt_lat.size, tgt_lon.size, nlev), np.nan)
    okf = np.zeros_like(out)
    for k in range(nlev):
        lev = field_levfirst[k]
        wet = np.isfinite(lev)
        if not wet.any():
            continue
        v, ok = regrid_curv_to_latlon(np.nan_to_num(lev), lat, lon,
                                      wet.astype(np.float64), tgt_lat, tgt_lon)
        out[..., k] = v
        okf[..., k] = ok * nn_wet_mask(lat, lon, wet, tgt_lat, tgt_lon)
    return out, okf


def _sec_mean(vals, common, axis):
    """Mean over `axis` using only common cells; NaN where a section cell has
    no support on all three sources."""
    cnt = common.sum(axis=axis)
    tot = np.where(common, vals, 0.0).sum(axis=axis)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _artifact_bound(name, label, unit, secs, z, dz_mis):
    """Bound the depth-offset sampling artifact against the plotted difference.

    Our levels and NEMO's agree to under a metre but not exactly, so an
    index-wise difference carries a |d(field)/dz| * |z_NEMO - z_ours| term. If
    that approaches the tripole-minus-NEMO signal, the panel is showing a
    sampling mismatch rather than a model difference.

    EVERY SECTION GETS ITS OWN BOUND. The bound was originally computed on the
    global zonal mean alone, which left every basin and meridional panel
    unbounded -- and a zonal mean is precisely the average that smooths away
    the sharp vertical gradients a basin or equatorial section exists to show,
    so the one number being computed was the least representative one. A noise
    floor that is pre-registered and then measured on a different field than
    the one being quoted is not a noise floor.

    The comparison is LIKE WITH LIKE: a max artifact against a median signal
    is not a ratio and cannot decide anything. The decisive number is the
    FRACTION of section cells where the artifact exceeds the model difference
    at THAT SAME CELL.
    """
    zN = secs[-1]
    with np.errstate(invalid="ignore"):
        artifact = np.abs(np.gradient(zN, z, axis=-1)) * dz_mis[None, :]
        signal = np.abs(secs[0] - zN)
    both = np.isfinite(artifact) & np.isfinite(signal)
    if not both.any():
        print(f"[{name}] {label}: NO cells support an artifact bound")
        return
    frac = float((artifact[both] > signal[both]).mean())
    kmax = np.unravel_index(np.nanargmax(np.where(both, artifact, np.nan)),
                            artifact.shape)
    print(f"[{name}] {label}: artifact exceeds |tripole-NEMO| on "
          f"{100.0 * frac:.1f}% of section cells; worst cell at "
          f"{z[kmax[-1]]:.0f} m, artifact {artifact[kmax]:.4g} vs signal "
          f"{signal[kmax]:.4g} {unit}; medians {np.nanmedian(artifact[both]):.4g} "
          f"vs {np.nanmedian(signal[both]):.4g} {unit}")


def _plot_sections(out, name, unit, x, xlabel, z, secs, labels, title,
                   mlds=None, max_depth_m=None):
    """``secs`` = [ours_1, ..., ours_n, NEMO] (nx, nlev); ``labels`` the n
    legoESM labels; ``mlds`` optional [mld_1, ..., mld_n, mld_NEMO] (nx,) lines
    drawn on the matching panel (both on each difference panel), None = skip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    *ours, Nm = secs
    if len(ours) != len(labels):
        raise ValueError(f"{len(ours)} sections for {len(labels)} labels")
    fin = np.isfinite(Nm)
    for o in ours:
        fin &= np.isfinite(o)
    if not fin.any():
        raise SystemExit(f"FATAL: {name} {xlabel} section has no common cells")
    stack = np.concatenate([o[fin] for o in ours] + [Nm[fin]])
    vmin, vmax = np.percentile(stack, [1, 99])
    dstack = np.concatenate([(o - Nm)[fin] for o in ours])
    dmax = float(np.percentile(np.abs(dstack), 99)) or 1e-6

    if mlds is None:
        mlds = [None] * len(secs)
    mld_N = mlds[-1]
    panels = [(o, f"{lab} {name}", "RdYlBu_r", vmin, vmax, [(m, lab, "k")])
              for o, lab, m in zip(ours, labels, mlds[:-1])]
    panels.append((Nm, f"NEMO {name}", "RdYlBu_r", vmin, vmax, [(mld_N, "NEMO", "k")]))
    panels += [(o - Nm, f"{lab} - NEMO", "RdBu_r", -dmax, dmax,
                [(m, lab, "k"), (mld_N, "NEMO", "m")])
               for o, lab, m in zip(ours, labels, mlds[:-1])]
    fig, ax = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6))
    for axi, (dat, ttl, cm, lo, hi, lines) in zip(np.atleast_1d(ax), panels):
        im = axi.pcolormesh(x, z, dat.T, vmin=lo, vmax=hi, cmap=cm,
                            shading="auto")
        for m, lab, col in lines:
            if m is not None and np.isfinite(m).any():
                axi.plot(x, m, color=col, lw=1.2, ls="--" if lab == "NEMO" else "-",
                         label=f"MLD {lab}")
        if any(m is not None for m, _, _ in lines):
            axi.legend(fontsize=7, loc="lower right")
        axi.set_title(ttl, fontsize=10)
        axi.set_xlabel(xlabel)
        axi.set_ylabel("depth [m]")
        if max_depth_m is not None:
            axi.set_ylim(0.0, max_depth_m)
        axi.invert_yaxis()
        plt.colorbar(im, ax=axi, shrink=0.85)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    p = out / f"{name}_{xlabel.split()[0].lower()}_section.png"
    fig.savefig(p, dpi=105)
    plt.close(fig)
    print(f"[fig] {p}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tripole", required=True)
    p.add_argument("--mpas", default=None,
                   help="optional second legoESM snapshot; omitted = tripole vs NEMO only")
    p.add_argument("--use-mean-fields", action="store_true",
                   help="read the window-mean state (T_mean_hw/S_mean_hw, mld_mean) "
                        "instead of the instantaneous one; NEMO's file is a window mean")
    p.add_argument("--max-depth-m", type=float, default=None,
                   help="clip the depth axis of every panel (e.g. 300 for the thermocline)")
    p.add_argument("--nemo-gridt", required=True)
    p.add_argument("--nemo-time-idx", type=int, default=-1)
    p.add_argument("--res-deg", type=float, default=1.0)
    p.add_argument("--label-tripole", default="tripole")
    p.add_argument("--label-mpas", default="MPAS")
    p.add_argument("--eq-halfwidth", type=float, default=2.0,
                   help="latitude half-width of the equatorial band [deg]")
    p.add_argument("--eq-lon-range", default="130,290",
                   help="Pacific longitude window for the equatorial section, "
                        "degrees east (0-360)")
    p.add_argument("--lat-section", action="append", default=[],
                   help="repeatable basin-resolved section, "
                        "LAT:LON0,LON1:HALFWIDTH:LABEL (degrees east, 0-360). "
                        "A global zonal mean cancels opposite-signed basin "
                        "anomalies and south of 35S averages across "
                        "continents, so a basin verdict needs its own window. "
                        "There is deliberately NO DEFAULT window: the "
                        "longitude span of a basin changes with latitude, and "
                        "a wrong one silently clips the western boundary "
                        "current -- the repo's existing BASINS['atlantic'] "
                        "starts at 290E, which at 26N excludes the Florida "
                        "Current and the DWBC entirely.")
    p.add_argument("--lon-section", action="append", default=[],
                   help="repeatable meridional section, "
                        "LON:LAT0,LAT1:HALFWIDTH:LABEL. LON may be negative "
                        "(140W is -140 or 220), and the window is selected by "
                        "CIRCULAR longitude distance so a section sitting on "
                        "the prime meridian is not silently emptied. Pass it "
                        "with the equals form -- argparse reads a leading "
                        "minus as a flag. This is the cold-tongue view: a "
                        "zonal-mean section averages the equatorial "
                        "thermocline against the off-equatorial one and hides "
                        "the slope entirely.")
    p.add_argument("--depth-tol-m", type=float, default=1.0,
                   help="max allowed mismatch between our z_center_ref and "
                        "NEMO deptht before the run refuses to compare")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tgt_lat = -90.0 + a.res_deg / 2.0 + a.res_deg * np.arange(int(180.0 / a.res_deg))
    tgt_lon = a.res_deg / 2.0 + a.res_deg * np.arange(int(360.0 / a.res_deg))

    trp = _load_legoesm(a.tripole, use_mean=a.use_mean_fields)
    mps = _load_legoesm(a.mpas, use_mean=a.use_mean_fields) if a.mpas else None
    nem = load_nemo_3d(a.nemo_gridt, a.nemo_time_idx)
    print(f"[fields] {'WINDOW MEAN (T_mean_hw/S_mean_hw)' if a.use_mean_fields else 'INSTANTANEOUS'} "
          f"vs NEMO record {a.nemo_time_idx} (a window mean)")

    z_ours = np.asarray(trp["z_center_ref"], dtype=np.float64)
    z_nemo = np.asarray(nem["depth"], dtype=np.float64)
    checks = [("NEMO", z_nemo)]
    if mps is not None:
        checks.insert(0, ("MPAS", np.asarray(mps["z_center_ref"], dtype=np.float64)))
    for tag, zz in checks:
        if zz.shape != z_ours.shape:
            raise SystemExit(f"FATAL: {tag} has {zz.shape} levels, tripole has "
                             f"{z_ours.shape} -- refusing to remap the vertical")
        d = float(np.max(np.abs(zz - z_ours)))
        print(f"[vertical] max |z_{tag} - z_tripole| = {d:.4f} m")
        if d > a.depth_tol_m:
            raise SystemExit(
                f"FATAL: {tag} vertical axis differs from ours by {d:.3f} m "
                f"(> {a.depth_tol_m} m). Comparing them needs an explicit "
                "vertical remapping choice, which this plotter will not make.")
    # WHERE the mismatch sits decides whether it matters. An index-wise
    # difference of two fields sampled at slightly different depths carries a
    # (vertical gradient x depth offset) term, so a sub-metre offset is
    # harmless in the abyss and is NOT harmless in the thermocline. Print the
    # worst levels with their depths rather than trusting the scalar maximum.
    dz_mis = np.abs(z_nemo - z_ours)
    worst = np.argsort(dz_mis)[::-1][:5]
    print("[vertical] largest per-level mismatches (level, our depth m, "
          "NEMO depth m, |diff| m):")
    for k in worst:
        print(f"           k={int(k):3d}  {z_ours[k]:9.3f}  {z_nemo[k]:9.3f}"
              f"  {dz_mis[k]:7.4f}")
    z = z_ours

    sources = [(a.label_tripole, trp, lego_wet_3d(trp, z))]
    if mps is not None:
        sources.append((a.label_mpas, mps, lego_wet_3d(mps, z)))
    labels = tuple(lab for lab, _, _ in sources)
    print("[mask] wet cells: " + ", ".join(
        f"{lab} {sum(int(w.sum()) for w in wet)}" for lab, _, wet in sources)
        + f", NEMO {int(np.isfinite(nem['T3d']).sum())}")
    who = " / ".join(labels) + " / NEMO"

    # Mixed-layer depth lines: ours from mld_mean (window means only; the
    # instantaneous snapshot carries no MLD), NEMO from mldr10_1. Regridded
    # once as a 2-D field, section-averaged with the SURFACE common mask.
    mld_src = []
    for lab, src, wet in sources:
        m = src.get("mld_mean") if a.use_mean_fields else None
        if m is None:
            print(f"[mld] {lab}: no {'mld_mean' if a.use_mean_fields else 'instantaneous MLD'} -> no line")
            mld_src.append(None)
            continue
        v, ok = regrid_curv_to_latlon(np.nan_to_num(np.asarray(m, dtype=np.float64)),
                                      src["lat"], src["lon"], wet[0].astype(np.float64),
                                      tgt_lat, tgt_lon)
        mld_src.append(np.where(ok > 0.5, v, np.nan))
    if nem.get("mld") is not None:
        v, ok = regrid_curv_to_latlon(np.nan_to_num(nem["mld"]), nem["lat"], nem["lon"],
                                      np.isfinite(nem["mld"]).astype(np.float64), tgt_lat, tgt_lon)
        mld_src.append(np.where(ok > 0.5, v, np.nan))
    else:
        print("[mld] NEMO file has no mldr10_1 -> no line")
        mld_src.append(None)

    def _mld_secs(sl2, common2, axis):
        return [None if m is None else _sec_mean(m[sl2], common2[sl2], axis=axis)
                for m in mld_src]

    lon0, lon1 = (float(v) for v in a.eq_lon_range.split(","))
    eqrow = np.abs(tgt_lat) <= a.eq_halfwidth
    eqcol = (tgt_lon >= lon0) & (tgt_lon <= lon1)
    if not eqrow.any() or not eqcol.any():
        raise SystemExit("FATAL: empty equatorial window")

    for name, unit, key in (("T", "degC", "T3d"), ("S", "psu", "S3d")):
        fields = []
        common = None
        for lab, src, wet in sources:
            G, ok = regrid_column(np.asarray(src[key]), src, wet, tgt_lat,
                                  tgt_lon, a.res_deg)
            fields.append(G)
            c = (ok > 0.5) & np.isfinite(G)
            common = c if common is None else (common & c)
        Ng, okN = regrid_nemo(nem[key], nem["lat"], nem["lon"], tgt_lat, tgt_lon)
        fields.append(Ng)
        common &= (okN > 0.5) & np.isfinite(Ng)
        print(f"[{name}] common cells {int(common.sum())} of {common.size}")
        common0 = common[..., 0]

        zon = [_sec_mean(f, common, axis=1) for f in fields]
        _artifact_bound(name, "global zonal mean", unit, zon, z, dz_mis)
        _plot_sections(out, name, unit, tgt_lat, "latitude", z, zon, labels,
                       f"{name} [{unit}] global zonal-mean section — {who} "
                       f"(common cells only)",
                       mlds=_mld_secs(np.s_[:, :], common0, 1), max_depth_m=a.max_depth_m)

        sl = np.ix_(np.where(eqrow)[0], np.where(eqcol)[0], np.arange(z.size))
        sl2 = np.ix_(np.where(eqrow)[0], np.where(eqcol)[0])
        eq = [_sec_mean(f[sl], common[sl], axis=0) for f in fields]
        _artifact_bound(name, "equatorial Pacific", unit, eq, z, dz_mis)
        _plot_sections(out, name, unit, tgt_lon[eqcol], "longitude degE", z, eq, labels,
                       f"{name} [{unit}] equatorial Pacific section, "
                       f"{a.eq_halfwidth:g}S-{a.eq_halfwidth:g}N mean — {who}",
                       mlds=_mld_secs(sl2, common0, 0), max_depth_m=a.max_depth_m)

        for spec in a.lat_section:
            lat0, lonspec, hw, label = spec.split(":")
            lo, hi = (float(v) for v in lonspec.split(","))
            lat0, hw = float(lat0), float(hw)
            rows = np.abs(tgt_lat - lat0) <= hw
            # lo > hi means the window crosses the prime meridian (the South
            # Atlantic does). Select by wrapping, and plot on a shifted
            # coordinate so the x axis stays monotonic -- an unshifted
            # 312..360,0..17 axis would render the section reversed.
            if lo > hi:
                cols = (tgt_lon >= lo) | (tgt_lon <= hi)
                xcoord = np.where(tgt_lon >= lo, tgt_lon - 360.0, tgt_lon)
            else:
                cols = (tgt_lon >= lo) & (tgt_lon <= hi)
                xcoord = tgt_lon
            order = np.argsort(xcoord[cols])
            if not rows.any() or not cols.any():
                raise SystemExit(f"FATAL: --lat-section {spec!r} selects "
                                 f"{int(rows.sum())} rows, {int(cols.sum())} cols")
            s2 = np.ix_(np.where(rows)[0], np.where(cols)[0], np.arange(z.size))
            s22 = np.ix_(np.where(rows)[0], np.where(cols)[0])
            sec = [_sec_mean(f[s2], common[s2], axis=0) for f in fields]
            nsup = int(np.isfinite(sec[0]).sum())
            print(f"[{name}] section {label}: {int(rows.sum())} rows x "
                  f"{int(cols.sum())} cols, {nsup} supported section cells")
            sec = [s[order] for s in sec]
            ml = [None if m is None else m[order] for m in _mld_secs(s22, common0, 0)]
            _artifact_bound(name, label, unit, sec, z, dz_mis)
            _plot_sections(out, f"{name}_{label}", unit, xcoord[cols][order],
                           "longitude degE", z, sec, labels,
                           f"{name} [{unit}] {label} section at "
                           f"{lat0:g} +/- {hw:g} deg, lon {lo:g}-{hi:g}E — {who}",
                           mlds=ml, max_depth_m=a.max_depth_m)

        for spec in a.lon_section:
            lon0, latspec, hw, label = spec.split(":")
            la0, la1 = (float(v) for v in latspec.split(","))
            lon0, hw = float(lon0), float(hw)
            # CIRCULAR distance, so the window is correct at any longitude
            # rather than only away from the wrap. Latitude does NOT wrap, so
            # the row selection needs none of the lat-section's shifted-axis
            # machinery and the x axis is already monotonic.
            dlon = np.abs((tgt_lon - lon0 + 180.0) % 360.0 - 180.0)
            cols = dlon <= hw
            rows = (tgt_lat >= la0) & (tgt_lat <= la1)
            if not rows.any() or not cols.any():
                raise SystemExit(f"FATAL: --lon-section {spec!r} selects "
                                 f"{int(rows.sum())} rows, {int(cols.sum())} cols")
            s2 = np.ix_(np.where(rows)[0], np.where(cols)[0], np.arange(z.size))
            s22 = np.ix_(np.where(rows)[0], np.where(cols)[0])
            sec = [_sec_mean(f[s2], common[s2], axis=1) for f in fields]
            nsup = int(np.isfinite(sec[0]).sum())
            print(f"[{name}] section {label}: {int(rows.sum())} rows x "
                  f"{int(cols.sum())} cols, {nsup} supported section cells")
            _artifact_bound(name, label, unit, sec, z, dz_mis)
            _plot_sections(out, f"{name}_{label}", unit, tgt_lat[rows],
                           "latitude", z, sec, labels,
                           f"{name} [{unit}] {label} section at "
                           f"{lon0:g} +/- {hw:g} deg, lat {la0:g}-{la1:g} — {who}",
                           mlds=_mld_secs(s22, common0, 1), max_depth_m=a.max_depth_m)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
