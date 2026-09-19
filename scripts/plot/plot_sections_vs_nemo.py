"""Depth cross-sections of temperature and salinity: tripole | MPAS | NEMO.

The surface scorecards (``compare_three_way_nemo``) and the four-way map figure
answer "how far is each grid from NEMO AT THE SURFACE".  Neither looks below it,
and the campaign's two open questions -- the equatorial thermocline behind the
cold-tongue bias, and whether the polar salinity agreement is skin-deep or holds
through the water column -- are both questions about the INTERIOR.  This plots
the interior on the same footing the surface figures already use.

TWO SECTIONS PER FIELD
----------------------
* GLOBAL ZONAL MEAN, depth vs latitude.  Every cell in a row of the regular
  lat-lon target sits at the same latitude, so a plain row mean is already the
  area-weighted zonal mean -- no cos(lat) factor belongs here (it belongs in a
  GLOBAL mean, which this is not).
* EQUATORIAL PACIFIC, depth vs longitude, averaged over ``--eq-halfwidth``
  degrees either side of the equator.  cos(lat) varies by under 0.1% across a
  2-degree band, so the plain mean is the area-weighted one to well past
  plotting precision.

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
    return {"T3d": T, "S3d": S, "depth": depth,
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


def _plot_sections(out, name, unit, x, xlabel, z, secs, labels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Tm, Mm, Nm = secs
    fin = np.isfinite(Tm) & np.isfinite(Mm) & np.isfinite(Nm)
    if not fin.any():
        raise SystemExit(f"FATAL: {name} {xlabel} section has no common cells")
    stack = np.concatenate([Tm[fin], Mm[fin], Nm[fin]])
    vmin, vmax = np.percentile(stack, [1, 99])
    dstack = np.concatenate([(Tm - Nm)[fin], (Mm - Nm)[fin]])
    dmax = float(np.percentile(np.abs(dstack), 99)) or 1e-6

    lab_t, lab_m = labels
    panels = [(Tm, f"{lab_t} {name}", "RdYlBu_r", vmin, vmax),
              (Mm, f"{lab_m} {name}", "RdYlBu_r", vmin, vmax),
              (Nm, f"NEMO {name}", "RdYlBu_r", vmin, vmax),
              (Tm - Nm, f"{lab_t} - NEMO", "RdBu_r", -dmax, dmax),
              (Mm - Nm, f"{lab_m} - NEMO", "RdBu_r", -dmax, dmax)]
    fig, ax = plt.subplots(1, 5, figsize=(28, 4.6))
    for axi, (dat, ttl, cm, lo, hi) in zip(ax, panels):
        im = axi.pcolormesh(x, z, dat.T, vmin=lo, vmax=hi, cmap=cm,
                            shading="auto")
        axi.set_title(ttl, fontsize=10)
        axi.set_xlabel(xlabel)
        axi.set_ylabel("depth [m]")
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
    p.add_argument("--mpas", required=True)
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
    p.add_argument("--depth-tol-m", type=float, default=1.0,
                   help="max allowed mismatch between our z_center_ref and "
                        "NEMO deptht before the run refuses to compare")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tgt_lat = -90.0 + a.res_deg / 2.0 + a.res_deg * np.arange(int(180.0 / a.res_deg))
    tgt_lon = a.res_deg / 2.0 + a.res_deg * np.arange(int(360.0 / a.res_deg))

    trp = _load_legoesm(a.tripole)
    mps = _load_legoesm(a.mpas)
    nem = load_nemo_3d(a.nemo_gridt, a.nemo_time_idx)

    z_ours = np.asarray(trp["z_center_ref"], dtype=np.float64)
    z_mpas = np.asarray(mps["z_center_ref"], dtype=np.float64)
    z_nemo = np.asarray(nem["depth"], dtype=np.float64)
    for tag, zz in (("MPAS", z_mpas), ("NEMO", z_nemo)):
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

    wet_t = lego_wet_3d(trp, z)
    wet_m = lego_wet_3d(mps, z)
    print(f"[mask] wet cells: tripole {sum(int(w.sum()) for w in wet_t)}, "
          f"MPAS {sum(int(w.sum()) for w in wet_m)}, "
          f"NEMO {int(np.isfinite(nem['T3d']).sum())}")

    lon0, lon1 = (float(v) for v in a.eq_lon_range.split(","))
    eqrow = np.abs(tgt_lat) <= a.eq_halfwidth
    eqcol = (tgt_lon >= lon0) & (tgt_lon <= lon1)
    if not eqrow.any() or not eqcol.any():
        raise SystemExit("FATAL: empty equatorial window")

    for name, unit, key in (("T", "degC", "T3d"), ("S", "psu", "S3d")):
        Tg, okT = regrid_column(np.asarray(trp[key]), trp, wet_t, tgt_lat,
                                tgt_lon, a.res_deg)
        Mg, okM = regrid_column(np.asarray(mps[key]), mps, wet_m, tgt_lat,
                                tgt_lon, a.res_deg)
        Ng, okN = regrid_nemo(nem[key], nem["lat"], nem["lon"], tgt_lat, tgt_lon)
        common = ((okT > 0.5) & (okM > 0.5) & (okN > 0.5)
                  & np.isfinite(Tg) & np.isfinite(Mg) & np.isfinite(Ng))
        print(f"[{name}] common cells {int(common.sum())} of {common.size}")

        zon = [_sec_mean(f, common, axis=1) for f in (Tg, Mg, Ng)]

        # BOUND THE DEPTH-OFFSET ARTIFACT against the difference being plotted.
        # The index-wise difference carries |d(field)/dz| * |z_NEMO - z_ours|;
        # if that approaches the plotted tripole-minus-NEMO signal, the panel
        # is showing a sampling mismatch rather than a model difference.
        zN = zon[2]
        with np.errstate(invalid="ignore"):
            dfdz = np.gradient(zN, z, axis=-1)
            artifact = np.abs(dfdz) * dz_mis[None, :]
            signal = np.abs(zon[0] - zN)
        # Compare LIKE WITH LIKE. A max artifact against a median signal is
        # not a ratio and cannot decide anything; the decisive number is the
        # FRACTION of section cells where the sampling artifact exceeds the
        # model difference at THAT SAME CELL.
        both = np.isfinite(artifact) & np.isfinite(signal)
        if both.any():
            frac = float((artifact[both] > signal[both]).mean())
            kmax = np.unravel_index(np.nanargmax(np.where(both, artifact, np.nan)),
                                    artifact.shape)
            print(f"[{name}] depth-offset artifact exceeds |tripole-NEMO| on "
                  f"{100.0 * frac:.1f}% of section cells; at the worst cell "
                  f"(depth {z[kmax[-1]]:.0f} m) artifact {artifact[kmax]:.4g} "
                  f"vs signal {signal[kmax]:.4g} {unit}")
            print(f"[{name}] median artifact {np.nanmedian(artifact[both]):.4g} "
                  f"vs median signal {np.nanmedian(signal[both]):.4g} {unit}")
        _plot_sections(out, name, unit, tgt_lat, "latitude", z, zon,
                       (a.label_tripole, a.label_mpas),
                       f"{name} [{unit}] global zonal-mean section — "
                       f"{a.label_tripole} / {a.label_mpas} / NEMO "
                       f"(common cells only)")

        sl = np.ix_(np.where(eqrow)[0], np.where(eqcol)[0], np.arange(z.size))
        eq = [_sec_mean(f[sl], common[sl], axis=0) for f in (Tg, Mg, Ng)]
        _plot_sections(out, name, unit, tgt_lon[eqcol], "longitude degE", z, eq,
                       (a.label_tripole, a.label_mpas),
                       f"{name} [{unit}] equatorial Pacific section, "
                       f"{a.eq_halfwidth:g}S-{a.eq_halfwidth:g}N mean — "
                       f"{a.label_tripole} / {a.label_mpas} / NEMO")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
