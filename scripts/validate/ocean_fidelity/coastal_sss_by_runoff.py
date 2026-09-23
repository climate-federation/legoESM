#!/usr/bin/env python
"""Split the coastal surface-salinity gap between two lanes by how much river
water each cell receives.

WHY THIS EXISTS. The FESOM lane's coastal salinity error is about twice the
tripole's, the excess sits almost entirely in the ring of cells touching the
edge of the scored domain, and it is already at full strength five days into
the run. Two candidates survive, and both switch on at t=0:

  * the initial condition interpolated onto FESOM's mesh, and
  * where the two lanes PUT river runoff.

They are separable by geography rather than by time. The tripole and MPAS
lanes both run NEMO's ``ln_rnf_depth_ini``, which spreads each river's
freshwater down to a depth set by the local runoff climatology (up to 150 m).
The FESOM lane cannot: ``--runoff-depth-nemo-ini`` is on its allowlist's
rejected list and its ocean core carries no spread-depth map, so every drop of
river water lands in the surface cell. If that is what drives the excess, the
excess must live where the rivers are. If it is the initial condition, the
excess should be there on runoff-free coasts too -- Antarctica, the arid
subtropical shores.

WHAT IT DOES. Reads the scoring card's own per-cell dump (so both lanes and
the oracle come through the identical remap path), regrids the runoff
climatology onto the same one-degree target with the repo's existing
``regrid_curv_to_latlon``, and splits one edge ring into strata by the
runoff's OWN quantiles inside that ring -- never a threshold chosen by hand.
Cells with exactly zero runoff get their own stratum, since "no river at all"
is the stratum the hypothesis actually predicts about.

Shares of the gap are computed on AREA-WEIGHTED SQUARED error, which is the
only form that adds across strata; the per-stratum rmse is reported beside it
for magnitude. A recombination check is printed, and a mismatch is fatal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def wrms(err, area, sel):
    """Area-weighted rms of ``err`` over the selected cells."""
    a = area[sel]
    e = err[sel]
    if a.size == 0 or not np.isfinite(a).any():
        raise SystemExit("wrms: empty selection -- refusing to return a number")
    return float(np.sqrt((a * e * e).sum() / a.sum()))


def wsse(err, area, sel):
    """Area-weighted SUM of squared error -- the form that adds across strata."""
    return float((area[sel] * err[sel] * err[sel]).sum())


def _xyz(lat_rad, lon_rad):
    c = np.cos(lat_rad)
    return np.stack([c * np.cos(lon_rad), c * np.sin(lon_rad),
                     np.sin(lat_rad)], axis=-1)


def runoff_on_target(runoff_file, tgt_lat_1d, tgt_lon_1d, max_deg):
    """Annual-maximum river runoff carried to the scorer's target grid by
    NEAREST SOURCE CELL.

    Deliberately not an inverse-distance average. Runoff is concentrated at a
    handful of river mouths, and averaging it over a multi-degree radius
    leaves almost no coastal cell at exactly zero -- which destroyed the first
    version of this test, where the runoff-free stratum came out at 1.5% of
    the ring's area and could not have carried the gap under any hypothesis.
    Nearest-source keeps "there is no river here" a statement about the
    coastline rather than about the search radius.

    The shared regridder is not reused for this one lookup because it carries
    a known axis-drop defect at k=1; the few lines below are the whole of what
    is needed and their shape is explicit.
    """
    import netCDF4 as nc

    with nc.Dataset(runoff_file) as f:
        if "sorunoff" not in f.variables:
            raise SystemExit(
                f"{runoff_file} carries no 'sorunoff'; variables are "
                f"{sorted(f.variables)}")
        rnf = np.asarray(f.variables["sorunoff"][:], dtype=np.float64)
        lat = np.asarray(f.variables["nav_lat"][:], dtype=np.float64)
        lon = np.asarray(f.variables["nav_lon"][:], dtype=np.float64)
    # Annual MAX, not mean: a river that runs hard for one season is still a
    # river, and the depth-spread lever keys off the climatological maximum
    # exactly as NEMO's rn_rnf_max does.
    rnf_max = np.nan_to_num(rnf, nan=0.0).max(axis=0)
    src_ok = (np.isfinite(lat) & np.isfinite(lon)).ravel()
    if not src_ok.any():
        raise SystemExit("runoff file has no usable source coordinates")

    from scipy.spatial import cKDTree

    src = _xyz(np.deg2rad(lat.ravel()[src_ok]), np.deg2rad(lon.ravel()[src_ok]))
    vals = rnf_max.ravel()[src_ok]
    lon2d, lat2d = np.meshgrid(tgt_lon_1d, tgt_lat_1d)
    tgt = _xyz(np.deg2rad(lat2d.ravel()), np.deg2rad(lon2d.ravel()))
    dist, idx = cKDTree(src).query(tgt, k=1)
    dist = np.asarray(dist).reshape(lat2d.shape)
    idx = np.asarray(idx).reshape(lat2d.shape)
    out = vals[idx]
    # Chord length on the unit sphere for the max_deg cutoff.
    cutoff = 2.0 * np.sin(np.deg2rad(max_deg) / 2.0)
    return np.where(dist <= cutoff, out, 0.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True,
                   help="per-cell npz written by the node-distance scoring card")
    p.add_argument("--runoff-file", type=Path, required=True)
    p.add_argument("--ring", type=int, default=1,
                   help="which edge ring to split (1 = the cells touching the "
                        "edge of the scored domain)")
    p.add_argument("--quantiles", type=str, default="50,75,90",
                   help="percentiles of the ring's own NONZERO runoff that "
                        "separate the strata")
    p.add_argument("--max-deg", type=float, default=2.5,
                   help="regridder's search radius, passed straight through")
    args = p.parse_args()

    d = np.load(args.dump, allow_pickle=True)
    need = ("lat2d", "lon2d", "area", "edge_ring",
            "SSS_a", "SSS_b", "SSS_nemo", "scored")
    missing = [k for k in need if k not in d.files]
    if missing:
        raise SystemExit(f"{args.dump} lacks {missing}; it carries {sorted(d.files)}")

    lat2d, lon2d = d["lat2d"], d["lon2d"]
    area = np.asarray(d["area"], dtype=np.float64)
    ring = np.asarray(d["edge_ring"])
    scored = np.asarray(d["scored"], dtype=bool)
    label_a = str(d["label_a"]) if "label_a" in d.files else "lane_a"
    label_b = str(d["label_b"]) if "label_b" in d.files else "lane_b"

    err_a = np.asarray(d["SSS_a"], dtype=np.float64) - np.asarray(d["SSS_nemo"], dtype=np.float64)
    err_b = np.asarray(d["SSS_b"], dtype=np.float64) - np.asarray(d["SSS_nemo"], dtype=np.float64)

    sel_ring = scored & (ring == args.ring) & np.isfinite(err_a) & np.isfinite(err_b)
    n_ring = int(sel_ring.sum())
    if n_ring == 0:
        raise SystemExit(f"no scored cells in edge ring {args.ring}")

    rnf = runoff_on_target(args.runoff_file, lat2d[:, 0], lon2d[0, :],
                           args.max_deg)
    if rnf.shape != area.shape:
        raise SystemExit(f"regridded runoff {rnf.shape} does not match the "
                         f"dump's grid {area.shape}")

    qs = [float(x) for x in args.quantiles.split(",")]
    nz = rnf[sel_ring & (rnf > 0.0)]
    if nz.size == 0:
        raise SystemExit("no runoff anywhere in this ring -- the split is vacuous")
    cuts = np.percentile(nz, qs)

    strata = [("zero runoff", sel_ring & (rnf <= 0.0))]
    lo = 0.0
    for q, c in zip(qs, cuts):
        strata.append((f"runoff <p{q:g} ({lo:.3g}-{c:.3g})",
                       sel_ring & (rnf > lo) & (rnf <= c)))
        lo = c
    strata.append((f"runoff >p{qs[-1]:g} (>{lo:.3g})", sel_ring & (rnf > lo)))

    tot_a = wsse(err_a, area, sel_ring)
    tot_b = wsse(err_b, area, sel_ring)
    tot_gap = tot_b - tot_a
    area_ring = area[sel_ring].sum()

    print(f"edge ring {args.ring}: {n_ring} scored cells, "
          f"{label_a} rmse {wrms(err_a, area, sel_ring):.4f}  "
          f"{label_b} rmse {wrms(err_b, area, sel_ring):.4f}")
    print(f"runoff strata from the ring's OWN nonzero percentiles {qs} "
          f"= {np.array2string(cuts, precision=4)}")
    print(f"{'stratum':30s} {'cells':>6s} {'area':>7s} "
          f"{label_a[:8]:>9s} {label_b[:8]:>9s} {'excess':>9s} "
          f"{'% of gap':>9s} {'lift':>7s}")
    acc_a = acc_b = 0.0
    for name, sel in strata:
        n = int(sel.sum())
        if n == 0:
            print(f"{name:30s} {0:6d}   (empty)")
            continue
        sa, sb = wsse(err_a, area, sel), wsse(err_b, area, sel)
        acc_a += sa
        acc_b += sb
        share = (sb - sa) / tot_gap if tot_gap != 0.0 else float("nan")
        afrac = area[sel].sum() / area_ring
        # LIFT = share of the gap divided by share of the area. A gap that
        # simply follows the coastline gives 1.0 in every stratum whatever its
        # size; a gap that needs rivers gives ~0 where there are none. This is
        # the statistic the first version of this probe lacked, and its
        # absence is what let a stratum holding 1.5% of the area be read as
        # though it could have refuted anything.
        lift = share / afrac if afrac > 0 else float("nan")
        ra, rb = wrms(err_a, area, sel), wrms(err_b, area, sel)
        print(f"{name:30s} {n:6d} {afrac:7.3f} "
              f"{ra:9.4f} {rb:9.4f} {rb - ra:+9.4f} "
              f"{100.0 * share:8.1f}% {lift:7.2f}")

    # A stratification that loses error is not a stratification.
    for got, want, who in ((acc_a, tot_a, label_a), (acc_b, tot_b, label_b)):
        if not np.isclose(got, want, rtol=1e-9, atol=0.0):
            raise SystemExit(
                f"FATAL: {who} strata do not recombine: {got:.10g} vs {want:.10g}")
    print(f"recombine OK: {label_a} {acc_a:.6g} == {tot_a:.6g}; "
          f"{label_b} {acc_b:.6g} == {tot_b:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
