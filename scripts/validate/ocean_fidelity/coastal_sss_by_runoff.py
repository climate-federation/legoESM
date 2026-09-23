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
import importlib.util
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()


def _load_regridder():
    """Reuse the scorer's own IDW regridder rather than writing a second one."""
    path = _THIS.parent.parent / "compare_omip_nemo.py"
    if not path.is_file():
        raise SystemExit(f"cannot find the shared regridder at {path}")
    spec = importlib.util.spec_from_file_location("_cmp_omip_nemo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cmp_omip_nemo"] = mod
    spec.loader.exec_module(mod)
    return mod.regrid_curv_to_latlon


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


def runoff_on_target(runoff_file, tgt_lat_1d, tgt_lon_1d, regrid, max_deg):
    """Annual-maximum river runoff, regridded onto the scorer's target grid."""
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
    src_ok = np.isfinite(lat) & np.isfinite(lon)
    out, flag = regrid(rnf_max, lat, lon, src_ok,
                       tgt_lat_1d, tgt_lon_1d, k=4, max_deg=max_deg)
    return np.where(flag > 0.5, out, 0.0)


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

    regrid = _load_regridder()
    rnf = runoff_on_target(args.runoff_file, lat2d[:, 0], lon2d[0, :],
                           regrid, args.max_deg)
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
          f"{label_a[:8]:>9s} {label_b[:8]:>9s} {'excess':>9s} {'% of gap':>9s}")
    acc_a = acc_b = 0.0
    for name, sel in strata:
        n = int(sel.sum())
        if n == 0:
            print(f"{name:30s} {0:6d}   (empty)")
            continue
        sa, sb = wsse(err_a, area, sel), wsse(err_b, area, sel)
        acc_a += sa
        acc_b += sb
        share = 100.0 * (sb - sa) / tot_gap if tot_gap != 0.0 else float("nan")
        ra, rb = wrms(err_a, area, sel), wrms(err_b, area, sel)
        print(f"{name:30s} {n:6d} {area[sel].sum()/area_ring:7.3f} "
              f"{ra:9.4f} {rb:9.4f} {rb - ra:+9.4f} {share:8.1f}%")

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
