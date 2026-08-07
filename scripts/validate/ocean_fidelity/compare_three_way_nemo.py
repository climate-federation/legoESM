"""Three-way scorecard: legoESM tripole vs legoESM MPAS vs NEMO ORCA1.

The standing question is not "is one grid close to NEMO" but the pair of
questions the campaign actually has to answer at once:

  1. how far is EACH of our two grids from the NEMO oracle, and
  2. do our two grids agree with EACH OTHER,

on the SAME cells, with the SAME regridder, at the SAME time, so that the two
answers are comparable to each other.  ``compare_omip_nemo.py`` answers (1) for
one arm at a time and ``compare_grids_tripole_mpas.py`` answers (2); neither
puts all three fields on one common footing, and stitching two separately-masked
scorecards together silently compares different cell sets.

WHAT THIS DOES
--------------
All three sources are IDW-regridded onto ONE regular lat-lon target with the
SAME helper the NEMO scorecard uses (``compare_omip_nemo.regrid_curv_to_latlon``)
and only cells resolved on ALL THREE are scored.  A target cell counts as ocean
only if the nearest cell of EVERY source is wet (``--mask-mode nearest``): the
regridder's own coverage flag is a distance-to-wet-data flag that keeps land
cells within 2.5 deg of ocean and fills them by extrapolation, which would
contaminate the near-land sub-domain most of all.  Reported per field
(SST, SSS, MLD):

  * global + per-latitude-band area-weighted bias / RMSE / pattern correlation
    of tripole-vs-NEMO, MPAS-vs-NEMO, and tripole-vs-MPAS;
  * an ARCTIC (>=60N) and a NEAR-LAND (coastal) sub-domain, the two regions the
    campaign keeps failing in, scored as their own masks rather than inferred
    from the 45N band, with near-land further split into its Arctic and
    non-Arctic parts (the Siberian and Canadian shelves are both, so an
    undivided coastal number cannot separate the two failure modes);
  * for MLD, the median difference and the fraction of columns deeper than
    ``--deep-mld-m``, which separate a broad depth shift from a few columns
    convecting to the sea floor;
  * zonal means of all three curves plus the two bias curves.

MLD is the de Boyer Montegut / Treguier density-threshold diagnostic at
delta_sigma = 0.01 to MATCH NEMO ``mldr10_1`` (the same convention
``compare_omip_nemo`` uses -- a larger threshold biases the model deep).
SSH IS DELIBERATELY ABSENT.  An earlier version scored ``zos`` against our
``eta`` after removing each field's area-weighted mean, which reconciles a
spatially UNIFORM offset and nothing else -- not a spatial inverse-barometer
term, not a different free-surface diagnostic, not a datum difference, none of
which have been matched between the two models.  Putting such a field through
the same bias/RMSE/correlation machinery as the fidelity fields presents it as
a verdict it cannot support, so it is removed rather than caveated.  Reinstate
it only once the two diagnostics' conventions are reconciled explicitly.

WHAT IT DOES NOT DO
-------------------
It does not certify that the two arms are a controlled pair -- that is a
property of the RUNS, not of the scorer.  For the xgrid_* pair, both arms
deliberately dropped every flag MPAS hard-errors on (--dm2dc, --isf, --bbl-adv,
--sw-rgb-chl, --gateway-transports, --iwm, the four --tke-* knobs) and both use
the annual-WOA IC, so BOTH are degraded from the NEMO-faithful tripole
configuration, equally.  The tripole-vs-NEMO numbers from such a pair are
therefore a floor, not the tripole's best fidelity; the run_manifest of each
arm is SUMMARISED into the report (command line, git-dirty flag, version)
so the caveat travels with the numbers.

Usage:
    python scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
        --tripole results/omip_nemo/xgrid_trp_d30/snapshot_final.npz \
        --mpas    results/omip_nemo/xgrid_mpas_d30/snapshot_final.npz \
        --nemo-gridt .../ORCA1_1m_20000101_20041231_grid_T.nc --nemo-month 1 \
        --out-dir results/omip_nemo/xgrid3_d30_m01
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # scripts/validate
from compare_omip_nemo import (  # noqa: E402
    _band_breakdown,
    _load_legoesm,
    _load_nemo,
    _wstats,
    regrid_curv_to_latlon,
)

# Sub-domains scored on their own masks (NOT inferred from the latitude bands):
# the two regions this campaign keeps failing in.
_ARCTIC_LAT_N = 60.0

# A zonal-mean row is drawn only where the three sources jointly resolve at
# least this FRACTION of the cells the common ocean mask offers in that row.
# A bare cell count is not comparable across resolutions or latitudes (10 cells
# is 3% of a 1-degree row and 6% of a 2-degree one, and a polar row has far
# fewer ocean cells to begin with); a fraction of the available support is.
_MIN_ZONAL_SUPPORT_FRAC = 0.25


def _manifest_summary(snapshot_path):
    """Provenance of the run a snapshot came from, carried into the report.

    The caveats on these numbers ("this arm dropped --iwm", "this arm ran a
    dirty tree") live in the run manifest, and a scorecard that only records
    the snapshot PATH loses them the moment the directory is renamed.
    """
    mf = Path(snapshot_path).parent / "run_manifest.json"
    if not mf.exists():
        return {"run_manifest": None, "note": f"no run_manifest.json beside {snapshot_path}"}
    try:
        m = json.loads(mf.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"run_manifest": str(mf), "error": f"unreadable: {exc}"}
    repro = m.get("reproducibility", {})
    return {"run_manifest": str(mf),
            "command_line": m.get("run", {}).get("command_line"),
            "git_dirty": repro.get("git_dirty"),
            "legoesm_version": repro.get("legoesm_version"),
            "creation_time": m.get("run", {}).get("creation_time")}


def _nn_wet_mask(src_lat_deg, src_lon_deg, src_wet, tgt_lat_deg, tgt_lon_deg):
    """Classify each TARGET cell wet/dry by its nearest SOURCE cell.

    ``regrid_curv_to_latlon`` returns a coverage flag that is 1 wherever a
    target cell lies within ``max_deg`` (2.5 deg) of any WET source cell.  That
    is a distance-to-data flag, NOT a land/sea classification: a target cell
    sitting on land keeps coverage=1 as long as some ocean cell is within 2.5
    deg, and then carries an IDW value extrapolated from offshore.  Scoring
    such a cell compares two extrapolations, and it does so preferentially at
    coastlines -- i.e. exactly inside the near-land sub-domain this scorecard
    exists to measure.

    Nearest-neighbour over ALL source cells (wet and dry alike) gives a real
    classification: the target is wet iff the source cell physically closest to
    it is wet.
    """
    from scipy.spatial import cKDTree
    lat = np.asarray(src_lat_deg, dtype=np.float64).ravel()
    lon = np.asarray(src_lon_deg, dtype=np.float64).ravel()
    wet = np.asarray(src_wet).ravel() > 0.5
    tree = cKDTree(_xyz_deg(lat, lon))
    tlon2d, tlat2d = np.meshgrid(tgt_lon_deg, tgt_lat_deg)
    _, idx = tree.query(_xyz_deg(tlat2d.ravel(), tlon2d.ravel()), k=1)
    return wet[idx].reshape(tlat2d.shape)


def _xyz_deg(lat_deg, lon_deg):
    """Unit-sphere Cartesian from DEGREES (the regridder's helper takes radians)."""
    la = np.deg2rad(lat_deg)
    lo = np.deg2rad(lon_deg)
    cl = np.cos(la)
    return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], axis=-1)


def build_ocean_mask(coverage, sources, tgt_lat, tgt_lon, mode):
    """The set of target cells that may be scored.

    ``coverage`` is the intersection of the regridder's own validity flags and
    is kept in the conjunction even in 'nearest' mode -- it is NOT redundant:
    nearest-wet classification has no distance limit, while coverage refuses a
    cell whose nearest wet datum is farther than the regridder's 2.5 deg
    validity radius.

    LIMITATION (do not overstate this mask).  Nearest-CENTRE classification has
    no cell polygons and no ocean connectivity, so where a strait or a land
    barrier is narrower than the ~111 km target cell -- the Canadian
    Archipelago, the Danish straits -- a cell can still be misclassified, and
    the value regridder can still interpolate across a barrier because it picks
    its four wet neighbours geometrically.  There is no error bound here.  The
    near-land sub-domain is a screening diagnostic, not a coastline-resolved
    one; that needs a topology-aware remapper.
    """
    if mode == "coverage":
        return coverage
    if mode != "nearest":
        raise ValueError(f"unknown mask mode {mode!r}")
    wet = np.logical_and.reduce([
        _nn_wet_mask(S["lat"], S["lon"], S["mask"], tgt_lat, tgt_lon)
        for S in sources])
    return coverage & wet


def _smooth_common_footprint(field, valid, tgt_lat, tgt_lon, radius_deg):
    """Great-circle top-hat smoothing of a target-grid field, ONE physical
    radius for every source.

    The IDW regridder takes a fixed k=4 source neighbours, whose PHYSICAL
    footprint therefore differs per mesh: a ~60 km MPAS cell keeps short-scale
    structure that a ~111 km ORCA1 cell has already smoothed away.  Because
    NEMO is itself ORCA1, that asymmetry can flatter tripole-vs-NEMO purely as
    a regridding artifact.  Changing the target RESOLUTION does not test this
    -- it moves the sample points but leaves each source's own k=4 footprint
    untouched -- so the control has to impose one common physical low-pass on
    all three fields after regridding.  If the ranking survives this, it is not
    a regrid artifact.
    """
    from scipy.spatial import cKDTree
    lon2d, lat2d = np.meshgrid(tgt_lon, tgt_lat)
    pts = _xyz_deg(lat2d.ravel(), lon2d.ravel())
    v = valid.ravel()
    tree = cKDTree(pts[v])
    vals = np.asarray(field, dtype=np.float64).ravel()[v]
    # Chord length of the great-circle radius on the unit sphere.
    chord = 2.0 * np.sin(np.deg2rad(radius_deg) / 2.0)
    out = np.full(vals.shape, np.nan)
    for i, nb in enumerate(tree.query_ball_point(pts[v], r=chord)):
        if nb:
            out[i] = float(np.mean(vals[nb]))
    full = np.full(field.shape, np.nan).ravel()
    full[v] = out
    return full.reshape(field.shape)


def _coastal_mask(ocean, n_cells):
    """Ocean cells within ``n_cells`` cells (Chebyshev) of a LAND cell.

    ``n_cells`` iterations of an 8-neighbour (3x3) dilation of the land
    complement, so "within 2 cells" means a square halo of half-width 2 and
    includes diagonals.

    Longitude is periodic, so the roll wraps in x.  Latitude is NOT: shifting
    off the top or bottom row contributes NOTHING.  Padding y with land (the
    previous behaviour) marked every pole-adjacent open-ocean cell as coastal,
    which at ``--coast-cells 2`` silently swept the first and last two latitude
    rows into the near-land sub-domain; wrapping y instead would join the two
    poles, which is worse.  A domain edge is not a coastline.
    """
    if n_cells < 1:
        raise ValueError(f"--coast-cells must be >= 1, got {n_cells}")

    def _shift(a, dy, dx):
        out = np.zeros_like(a)
        s = np.roll(a, dx, axis=1)          # x periodic
        if dy == 0:
            return s
        if dy > 0:                          # move content down; top row empty
            out[dy:] = s[:-dy]
        else:                               # move content up; bottom row empty
            out[:dy] = s[-dy:]
        return out

    grown = ~ocean
    for _ in range(int(n_cells)):
        nxt = grown.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nxt |= _shift(grown, dy, dx)
        grown = nxt
    return ocean & grown


def _lego_mld(L):
    """Density-threshold MLD of a legoESM snapshot, or None if the snapshot
    predates the saved MLD geometry.  Same convention as compare_omip_nemo."""
    if L.get("z_center_ref") is None or L.get("H_bathy") is None:
        return None
    from legoesm.ocean.diagnostics import mixed_layer_depth
    z_c = np.asarray(L["z_center_ref"], dtype=np.float64)
    Hb = np.asarray(L["H_bathy"], dtype=np.float64)
    wet = ((z_c[(None,) * Hb.ndim + (slice(None),)] < Hb[..., None])
           & (L["mask"][..., None] > 0.5)).astype(np.float64)
    return np.asarray(mixed_layer_depth(L["T3d"], L["S3d"], z_c,
                                        delta_sigma=0.01, wet_mask=wet,
                                        bottom_depth=Hb))


def _demean(field, area):
    """Remove the area-weighted mean over the finite, positive-area cells.

    ``0.0 * nan`` is ``nan``, so weighting a NaN cell by zero area does NOT
    exclude it: a single NaN anywhere in the array made the mean NaN and
    returned an all-NaN field, silently voiding every statistic downstream.
    Select the contributing cells explicitly instead of relying on the weight.
    """
    w = np.asarray(area, dtype=np.float64)
    sel = (w > 0) & np.isfinite(field)
    if not sel.any():
        raise ValueError("_demean: no finite cells with positive area")
    return field - float((w[sel] * np.asarray(field)[sel]).sum() / w[sel].sum())


def _tail(a, b, m, deep_m):
    """Distribution of the difference, and of the two fields' deep tails.

    A mean bias and an RMSE cannot tell a BROAD shift from a FEW runaway
    columns, and for MLD that distinction is the whole diagnosis: an Arctic
    RMSE of 300 m with a bias of +80 m is either every column 80 m too deep or
    a handful convecting to the sea floor.  The median difference and the
    fraction of each field deeper than ``deep_m`` separate the two.
    """
    if not m.any():
        return None
    d = (a - b)[m]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    av, bv = a[m], b[m]
    return {"n": int(d.size),
            "median_diff": float(np.median(d)),
            "p90_abs_diff": float(np.percentile(np.abs(d), 90)),
            "p99_abs_diff": float(np.percentile(np.abs(d), 99)),
            "frac_a_deeper_than": float(np.mean(av > deep_m)),
            "frac_b_deeper_than": float(np.mean(bv > deep_m)),
            "deep_threshold": deep_m}


def _scored(a, b, area, tgt_lat, sub, deep_m=None):
    """Global + band + sub-domain stats of ``a`` against ``b``."""
    out = {"global": _wstats(a, b, area),
           "bands": _band_breakdown(a, b, area, tgt_lat)}
    for name, m in sub.items():
        am = area * m
        out[name] = _wstats(a, b, am) if am.sum() > 0 else None
    if deep_m is not None:
        out["tails"] = {"global": _tail(a, b, area > 0, deep_m)}
        for name, m in sub.items():
            out["tails"][name] = _tail(a, b, (area > 0) & m, deep_m)
    return out


def _json_safe(obj):
    """Recursively replace non-finite floats with None so the report is valid
    JSON.  An undefined correlation is genuinely absent, not a number."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _fmt(s):
    return (f"bias {s['bias']:+8.3f}  rmse {s['rmse']:7.3f}  corr {s['corr']:6.3f}"
            if s else "  (no cells)")


def _git_sha(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tripole", type=Path, required=True)
    p.add_argument("--mpas", type=Path, required=True)
    p.add_argument("--nemo-gridt", type=Path, required=True)
    p.add_argument("--nemo-month", type=int, default=None,
                   help="Calendar month 1-12 averaged across the file "
                        "(seasonally-matched reference for an instantaneous snapshot).")
    p.add_argument("--nemo-time-idx", type=int, default=-1)
    p.add_argument("--res-deg", type=float, default=1.0)
    p.add_argument("--coast-cells", type=int, default=2,
                   help="Near-land sub-domain width in target cells (default 2 "
                        "= within ~2 deg of the coast of the COMMON mask).")
    p.add_argument("--smooth-radius-deg", type=float, default=None,
                   help="Instrument control: after regridding, replace every "
                        "field by its mean over this great-circle radius, the "
                        "SAME physical radius for all three sources. Use it to "
                        "test whether a ranking is a regrid artifact -- the IDW "
                        "stencil's k=4 neighbours span a different physical "
                        "distance on a ~60 km MPAS cell than on a ~111 km "
                        "ORCA1 cell. Changing --res-deg does NOT test this.")
    p.add_argument("--mask-mode", choices=("nearest", "coverage"), default="nearest",
                   help="How a target cell is classified ocean.  'nearest' "
                        "(default) requires the nearest SOURCE cell of every "
                        "source to be wet -- a real land/sea classification.  "
                        "'coverage' keeps the regridder's distance-to-wet-data "
                        "flag, which admits land cells within 2.5 deg of ocean "
                        "and is retained only to reproduce older scorecards.")
    p.add_argument("--deep-mld-m", type=float, default=500.0,
                   help="MLD depth [m] above which a column counts as 'deep "
                        "convection' in the tail diagnostic (default 500).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--label-tripole", default="tripole")
    p.add_argument("--label-mpas", default="MPAS")
    a = p.parse_args()

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for pth in (a.tripole, a.mpas, a.nemo_gridt):
        if not pth.exists():
            raise SystemExit(f"FATAL: missing input {pth}")

    # Cell CENTRES of a global grid of spacing res_deg.  The previous
    # hard-coded -89.5/0.5 offsets are the centres only at 1 degree; at 2 they
    # gave -89.5..88.5, which is neither centred nor global.
    if not np.isfinite(a.res_deg) or a.res_deg <= 0:
        raise SystemExit(f"--res-deg must be finite and positive, got {a.res_deg}")
    if not (180.0 / a.res_deg).is_integer() or not (360.0 / a.res_deg).is_integer():
        raise SystemExit(f"--res-deg {a.res_deg} does not divide the globe evenly")
    tgt_lat = -90.0 + a.res_deg / 2.0 + a.res_deg * np.arange(int(180.0 / a.res_deg))
    tgt_lon = a.res_deg / 2.0 + a.res_deg * np.arange(int(360.0 / a.res_deg))

    T = _load_legoesm(a.tripole)
    M = _load_legoesm(a.mpas)
    N = _load_nemo(a.nemo_gridt, a.nemo_time_idx, month=a.nemo_month)
    print(f"[load] tripole {T['sst'].shape}  MPAS {M['sst'].shape}  "
          f"NEMO {N['sst'].shape} ({N['n_time']} records)")

    def rg(src, field, mask=None):
        return regrid_curv_to_latlon(field, src["lat"], src["lon"],
                                     src["mask"] if mask is None else mask,
                                     tgt_lat, tgt_lon)

    sstT, ocT = rg(T, T["sst"])
    sstM, ocM = rg(M, M["sst"])
    sstN, ocN = rg(N, N["sst"])
    sssT, _ = rg(T, T["sss"])
    sssM, _ = rg(M, M["sss"])
    sssN, _ = rg(N, N["sss"])

    coverage = (ocT > 0.5) & (ocM > 0.5) & (ocN > 0.5)
    ocean = build_ocean_mask(coverage, (T, M, N), tgt_lat, tgt_lon, a.mask_mode)
    print(f"[mask] mode={a.mask_mode}: coverage-only would keep "
          f"{int(coverage.sum())} cells, scoring {int(ocean.sum())} "
          f"({int(coverage.sum() - ocean.sum())} dropped as land under the "
          "nearest-source classification)")
    n_common = int(ocean.sum())
    if n_common == 0:
        raise SystemExit("FATAL: no cells resolved on all three sources")
    area = (np.cos(np.deg2rad(tgt_lat))[:, None]
            * np.ones_like(tgt_lon)[None, :]) * ocean
    lat2d = tgt_lat[:, None] * np.ones((1, tgt_lon.size))
    arctic = (lat2d >= _ARCTIC_LAT_N) & ocean
    coastal = _coastal_mask(ocean, a.coast_cells)
    # near-land is split by latitude because the two failure modes overlap
    # geographically: the Siberian/Canadian shelves are BOTH Arctic and coastal,
    # so an undifferentiated "near_land" number cannot say whether a coastal
    # error is a coastline problem or the Arctic problem seen through a
    # coastal mask.
    sub = {"arctic_N_of_60N": arctic,
           "near_land": coastal,
           "near_land_arctic": coastal & arctic,
           "near_land_nonarctic": coastal & ~arctic}
    print(f"[common] {n_common} cells on all three; "
          + ", ".join(f"{k} {int(v.sum())}" for k, v in sub.items()))

    report = {
        "generated_by": str(_HERE),
        "git_sha": _git_sha(_HERE.parents[3]),
        "tripole_snapshot": str(a.tripole), "mpas_snapshot": str(a.mpas),
        "nemo_gridt": str(a.nemo_gridt), "nemo_month": a.nemo_month,
        "nemo_time_idx": None if a.nemo_month else a.nemo_time_idx,
        "res_deg": a.res_deg, "coast_cells": a.coast_cells,
        "mask_mode": a.mask_mode,
        "n_coverage_cells": int(coverage.sum()),
        "n_common_cells": n_common,
        "n_cells_per_subdomain": {k: int(v.sum()) for k, v in sub.items()},
        "run_manifests": {
            k: _manifest_summary(p) for k, p in (("tripole", a.tripole),
                                                 ("mpas", a.mpas))},
        "caveat": ("Fidelity numbers inherit the RUN configuration; a matched "
                   "cross-grid pair is degraded from the NEMO-faithful tripole "
                   "config on both arms. See each arm's run_manifest.json."),
        "fields": {},
    }

    plot_fields = {}          # name -> (Tg, Mg, Ng, unit)
    # Each entry carries its OWN scoring area: MLD and SSH are resolved on
    # a smaller domain than SST/SSS, and reusing the SST area would score cells
    # where it does not exist on all three sources.
    fields = [("SST", sstT, sstM, sstN, "degC", area),
              ("SSS", sssT, sssM, sssN, "psu", area)]

    # --- MLD (density threshold, matched to NEMO mldr10_1) -------------------
    mldT_raw, mldM_raw = _lego_mld(T), _lego_mld(M)
    if N.get("mld") is None:
        print("[MLD] SKIPPED: NEMO grid_T has no mldr10_1")
    elif mldT_raw is None or mldM_raw is None:
        which = [n for n, v in (("tripole", mldT_raw), ("MPAS", mldM_raw)) if v is None]
        print(f"[MLD] SKIPPED: snapshot(s) {which} lack z_center_ref/H_bathy")
    else:
        mldT, ocTm = regrid_curv_to_latlon(np.nan_to_num(mldT_raw, nan=0.0),
                                           T["lat"], T["lon"],
                                           np.isfinite(mldT_raw).astype(np.float64),
                                           tgt_lat, tgt_lon)
        mldM, ocMm = regrid_curv_to_latlon(np.nan_to_num(mldM_raw, nan=0.0),
                                           M["lat"], M["lon"],
                                           np.isfinite(mldM_raw).astype(np.float64),
                                           tgt_lat, tgt_lon)
        mldN, ocNm = regrid_curv_to_latlon(np.nan_to_num(N["mld"], nan=0.0),
                                           N["lat"], N["lon"],
                                           np.isfinite(N["mld"]).astype(np.float64),
                                           tgt_lat, tgt_lon)
        # MLD is scored on the intersection of the MLD coverage and the common
        # ocean, which can be smaller than `ocean`; carry its own area.
        mld_ok = ocean & (ocTm > 0.5) & (ocMm > 0.5) & (ocNm > 0.5)
        if not mld_ok.any():
            raise SystemExit("FATAL: MLD has no cells resolved on all three sources")
        report["n_mld_cells"] = int(mld_ok.sum())
        report["mld_note"] = (
            f"MLD scored on its own coverage intersection ({int(mld_ok.sum())} "
            f"cells vs {n_common} for SST/SSS); dsigma=0.01 to match NEMO mldr10_1; "
            "snapshot-state MLD, NOT a seasonal-mean MLD.")
        _mld_area = (np.cos(np.deg2rad(tgt_lat))[:, None]
                     * np.ones_like(tgt_lon)[None, :]) * mld_ok
        fields.append(("MLD", mldT, mldM, mldN, "m", _mld_area))
    if a.smooth_radius_deg:
        print(f"[smooth] common-footprint control ON: top-hat mean over "
              f"{a.smooth_radius_deg} deg great-circle radius, applied "
              "IDENTICALLY to all three sources")
        report["smooth_radius_deg"] = a.smooth_radius_deg
        fields = [(n, *(_smooth_common_footprint(f, ar > 0, tgt_lat, tgt_lon,
                                                 a.smooth_radius_deg)
                        for f in (Tg, Mg, Ng)), u, ar)
                  for n, Tg, Mg, Ng, u, ar in fields]

    for name, Tg, Mg, Ng, unit, ar in fields:
        # PER-FIELD finite intersection.  Geometric coverage is not finiteness:
        # a source can be covered and still deliver a NaN (or, through the NEMO
        # loader's nan_to_num, a fabricated 0.0 that IDW spreads), and without
        # this gate a single NaN turns every statistic for the field into a bare
        # NaN in report.json that reads like a number.
        finite3 = np.isfinite(Tg) & np.isfinite(Mg) & np.isfinite(Ng)
        n_drop = int(((ar > 0) & ~finite3).sum())
        if n_drop:
            print(f"  [{name}] WARNING: dropping {n_drop} covered cells that are "
                  f"non-finite on at least one source")
        ar = np.where(finite3, ar, 0.0)
        if not (ar > 0).any():
            raise SystemExit(f"FATAL: {name} has no finite cells on all three sources")
        report.setdefault("n_finite_cells", {})[name] = int((ar > 0).sum())
        sb = {k: (v & (ar > 0)) for k, v in sub.items()}
        deep = a.deep_mld_m if name == "MLD" else None
        rec = {
            "unit": unit,
            "tripole_vs_nemo": _scored(Tg, Ng, ar, tgt_lat, sb, deep),
            "mpas_vs_nemo": _scored(Mg, Ng, ar, tgt_lat, sb, deep),
            "tripole_vs_mpas": _scored(Tg, Mg, ar, tgt_lat, sb, deep),
        }
        report["fields"][name] = rec
        print(f"\n===== {name} [{unit}] =====")
        for pair in ("tripole_vs_nemo", "mpas_vs_nemo", "tripole_vs_mpas"):
            print(f"  {pair:18s} GLOBAL   {_fmt(rec[pair]['global'])}")
        for dom in sub:
            for pair in ("tripole_vs_nemo", "mpas_vs_nemo", "tripole_vs_mpas"):
                print(f"  {pair:18s} {dom:20s} {_fmt(rec[pair][dom])}")
        if deep is not None:
            print(f"  -- distribution (median / p90|diff| / frac deeper than "
                  f"{deep:.0f} m) --")
            for dom in ["global", *sub]:
                for pair in ("tripole_vs_nemo", "mpas_vs_nemo"):
                    t = rec[pair]["tails"].get(dom)
                    if t:
                        print(f"    {pair:18s} {dom:20s} n={t['n']:6d} "
                              f"med {t['median_diff']:+8.1f}  p90 {t['p90_abs_diff']:7.1f}  "
                              f"model {t['frac_a_deeper_than']*100:5.1f}%  "
                              f"NEMO {t['frac_b_deeper_than']*100:5.1f}%")
        print("  -- latitude bands (bias) --")
        for bn in rec["tripole_vs_nemo"]["bands"]:
            bt = rec["tripole_vs_nemo"]["bands"][bn]
            bm = rec["mpas_vs_nemo"]["bands"][bn]
            if bt and bm:
                print(f"    {bn:22s} trp {bt['bias']:+8.3f}   mpas {bm['bias']:+8.3f}"
                      f"   |  rmse trp {bt['rmse']:6.3f} mpas {bm['rmse']:6.3f}")
        plot_fields[name] = (Tg, Mg, Ng, unit, (ar > 0))

    _plot3(out, tgt_lat, tgt_lon, plot_fields, a.label_tripole, a.label_mpas)

    # allow_nan=False: bare NaN is not valid JSON and reads like a number to a
    # downstream parser.  _wstats returns a NaN correlation for a constant or
    # single-cell domain, so scrub those to null and fail loudly on any other.
    (out / "report.json").write_text(json.dumps(_json_safe(report), indent=2,
                                                allow_nan=False))
    print(f"\n[report] {out / 'report.json'}")
    return 0


def _plot3(out, tgt_lat, tgt_lon, plot_fields, lab_t, lab_m):
    """Per field: 5-panel maps (trp | MPAS | NEMO | trp-NEMO | MPAS-NEMO) and a
    combined 3-way zonal-mean + zonal-bias figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, (Tg, Mg, Ng, unit, ok) in plot_fields.items():
        Tm = np.where(ok, Tg, np.nan)
        Mm = np.where(ok, Mg, np.nan)
        Nm = np.where(ok, Ng, np.nan)
        finite = np.isfinite(Tm) & np.isfinite(Mm) & np.isfinite(Nm)
        if not finite.any():
            raise SystemExit(f"FATAL: {name} has no finite common cells to plot")
        stack = np.concatenate([Tm[finite], Mm[finite], Nm[finite]])
        vmin, vmax = np.percentile(stack, [1, 99])
        dstack = np.concatenate([(Tm - Nm)[finite], (Mm - Nm)[finite]])
        dmax = float(np.percentile(np.abs(dstack), 99))
        fig, ax = plt.subplots(1, 5, figsize=(28, 4))
        panels = [(Tm, f"{lab_t} {name}", "RdYlBu_r", vmin, vmax),
                  (Mm, f"{lab_m} {name}", "RdYlBu_r", vmin, vmax),
                  (Nm, f"NEMO {name}", "RdYlBu_r", vmin, vmax),
                  (Tm - Nm, f"{lab_t} - NEMO", "RdBu_r", -dmax, dmax),
                  (Mm - Nm, f"{lab_m} - NEMO", "RdBu_r", -dmax, dmax)]
        for axi, (dat, ttl, cm, lo, hi) in zip(ax, panels):
            im = axi.pcolormesh(tgt_lon, tgt_lat, dat, vmin=lo, vmax=hi,
                                cmap=cm, shading="auto")
            axi.set_title(ttl, fontsize=10)
            axi.axhline(_ARCTIC_LAT_N, color="k", lw=0.5, ls=":")
            plt.colorbar(im, ax=axi, shrink=0.85)
        fig.suptitle(f"{name} [{unit}] — {lab_t} / {lab_m} / NEMO "
                     f"(colour limits = 1-99th pct of the common cells)",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(out / f"{name}_maps3.png", dpi=95)
        plt.close(fig)
        print(f"[map] {out / f'{name}_maps3.png'}")

    n = len(plot_fields)
    fig, ax = plt.subplots(2, n, figsize=(5.5 * n, 8), squeeze=False)
    for j, (name, (Tg, Mg, Ng, unit, ok)) in enumerate(plot_fields.items()):
        # ONE row mask for all three curves, with a minimum count: a row where
        # only one grid resolves a couple of cells must not be drawn next to a
        # NEMO row mean over 300 cells, and the three curves must be means over
        # the SAME cells or their difference is not a bias.
        rowok = ok & np.isfinite(Tg) & np.isfinite(Mg) & np.isfinite(Ng)
        cnt = rowok.sum(axis=1)
        avail = ok.sum(axis=1)
        good = (avail > 0) & (cnt >= _MIN_ZONAL_SUPPORT_FRAC * avail) & (cnt > 0)

        def zm(f):
            z = np.full(f.shape[0], np.nan)
            z[good] = (np.where(rowok, f, 0.0).sum(axis=1)[good] / cnt[good])
            return z
        zT, zM, zN = zm(Tg), zm(Mg), zm(Ng)
        a0 = ax[0][j]
        a0.plot(zT, tgt_lat, label=lab_t, lw=1.4)
        a0.plot(zM, tgt_lat, label=lab_m, lw=1.4, ls="--")
        a0.plot(zN, tgt_lat, label="NEMO", lw=1.8, color="k")
        a0.set_title(f"zonal-mean {name} [{unit}]")
        a0.set_ylabel("latitude"); a0.legend(fontsize=8); a0.grid(alpha=0.3)
        a1 = ax[1][j]
        a1.plot(zT - zN, tgt_lat, label=f"{lab_t} - NEMO", lw=1.4)
        a1.plot(zM - zN, tgt_lat, label=f"{lab_m} - NEMO", lw=1.4, ls="--")
        a1.plot(zT - zM, tgt_lat, label=f"{lab_t} - {lab_m}", lw=1.0, color="g")
        a1.axvline(0.0, color="k", lw=0.8)
        a1.axhline(_ARCTIC_LAT_N, color="r", lw=0.6, ls=":")
        a1.set_title(f"zonal-mean bias {name} [{unit}]")
        a1.set_ylabel("latitude"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    fig.suptitle("Three-way zonal means and biases (common cells only)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "zonal_3way.png", dpi=110)
    plt.close(fig)
    print(f"[map] {out / 'zonal_3way.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
