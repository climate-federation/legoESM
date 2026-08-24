"""Localise the Arctic columns that mix far deeper than NEMO, on NATIVE cells.

The three-way scorecard established the term: 3.72 % of Arctic columns reach
past 500 m in the virtual-salt-flux control and 6.08 % with the real-freshwater
closure, against NEMO's 0.80 %.  That is a fraction, not a mechanism, and a
regridded field cannot say which columns or why.

Our tripole runs ON the eORCA1 mesh, the same mesh NEMO ORCA1 uses, so this
probe compares CELL TO CELL with no regridding at all -- the interpolation is
the single largest source of doubt in every previous coastal/Arctic number here.
The mapping between our (332, 362) arrays and NEMO's (331, 360) is NOT assumed:
it is searched for and then VERIFIED against the stored coordinates, and the
probe exits if the residual is not near machine precision.

THE FIRST QUESTION IS ABOUT THE INSTRUMENT, NOT THE OCEAN.
``mixed_layer_depth`` returns a depth for every wet column.  When a column has
no density crossing at all, what it returns is governed by the bottom, so a
reported "MLD of 800 m" can mean either "this column convected to 800 m" or
"this column never crossed the threshold and 800 m is where the sea floor is".
Those are different physical claims and the deep FRACTION cannot tell them
apart.  Every deep column is therefore classified first:

  bottom_limited : MLD >= --bottom-frac x H_bathy.  The diagnostic saturated;
                   this is a statement about stratification being unresolvable,
                   not about a mixing depth.
  genuine_deep   : MLD past the threshold with water left beneath it.

Only ``genuine_deep`` columns support a sentence about deep convection.  The
split is printed for both arms and for NEMO before any attribution.

For the genuine ones the probe reports the discriminators that separate the
candidate mechanisms, per column and aggregated: ice concentration (brine
rejection), surface salinity and temperature against NEMO in the SAME cell,
whether the column is statically unstable in the upper ocean (convection
actively running vs a relic deep layer), bathymetry (shelf vs basin), and
distance to the tripole fold (a numerical seam rather than physics).

Usage:
    python scripts/validate/ocean_fidelity/arctic_deep_convection_columns.py \\
        --snapshot results/omip_nemo/nemolev_trp_fwreal_d90/snapshot_day0090.npz \\
        --label real-FW \\
        --nemo-gridt .../ORCA1_1m_20000101_20041231_grid_T.nc --nemo-month 3 \\
        --out-dir results/omip_nemo/arctic_convection_fwreal
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
from compare_omip_nemo import _load_legoesm, _load_nemo  # noqa: E402

_ARCTIC_LAT_N = 60.0
# Coordinate agreement demanded of the native-cell mapping.  Our lat/lon and
# NEMO's nav_lat/nav_lon are the same mesh written by different codes, so the
# residual should be at write precision (float32 ~1e-5 deg), not "close".
_COORD_TOL_DEG = 1e-3


def _find_native_offset(lego_lat, lego_lon, nemo_lat, nemo_lon, valid):
    """Find (j0, i0) such that lego[j0:j0+ny, i0:i0+nx] matches NEMO cell-for-cell.

    Our tripole arrays carry halo rows/columns (the east-west cyclic overlap adds
    a column at each end), so the interior is offset from NEMO's array by an
    amount that depends on run configuration.  ASSUMING the offset is exactly the
    class of error this repo keeps paying for, so search the small candidate set
    and verify, rather than hard-coding [0:331, 1:361].

    ``valid`` restricts the comparison to cells where NEMO's coordinates are
    MEANINGFUL.  This is not a convenience: NEMO's grid_T carries a fill row at
    j=0 with ``nav_lat = -1.0`` everywhere, against a real -84.2 in our array, so
    a max-over-all-cells residual is 83 degrees for the CORRECT offset and the
    search reports failure on a perfect match.  Score on valid ocean cells and
    report the residual there.
    """
    ny, nx = nemo_lat.shape
    Ny, Nx = lego_lat.shape
    best = None
    for j0 in range(0, Ny - ny + 1):
        for i0 in range(0, Nx - nx + 1):
            dlat = np.abs(lego_lat[j0:j0 + ny, i0:i0 + nx] - nemo_lat)[valid]
            # Longitude is periodic: compare on the circle.
            dlon = np.abs((lego_lon[j0:j0 + ny, i0:i0 + nx] - nemo_lon + 180.0)
                          % 360.0 - 180.0)[valid]
            if dlat.size == 0:
                raise SystemExit("FATAL: no valid NEMO cells to match against")
            # Longitude is degenerate near the poles (meridians converge), so the
            # latitude residual carries the decision and longitude is scored at a
            # percentile rather than its max.
            score = float(np.max(dlat)) + float(np.percentile(dlon, 99))
            if best is None or score < best[0]:
                best = (score, j0, i0, float(np.max(dlat)),
                        float(np.percentile(dlon, 99)))
    return best


def _static_instability(rho, wet, top_levels):
    """True where density DECREASES with depth in the upper ocean (unstable)."""
    r = np.where(wet, rho, np.nan)[..., :top_levels]
    d = np.diff(r, axis=-1)
    return np.nanmin(d, axis=-1) < 0.0


def _git_sha(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--label", default="legoESM")
    p.add_argument("--nemo-gridt", type=Path, required=True)
    p.add_argument("--nemo-month", type=int, default=None)
    p.add_argument("--nemo-time-idx", type=int, default=-1)
    p.add_argument("--deep-mld-m", type=float, default=500.0)
    p.add_argument("--bottom-frac", type=float, default=0.9,
                   help="A column whose MLD reaches this fraction of its own "
                        "bathymetry is BOTTOM-LIMITED: the density-threshold "
                        "diagnostic found no crossing and returned a depth set "
                        "by the sea floor, which is not a mixing depth.")
    p.add_argument("--unstable-top-m", type=float, default=200.0)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    L = _load_legoesm(a.snapshot)
    N = _load_nemo(a.nemo_gridt, a.nemo_time_idx, month=a.nemo_month)
    if N.get("mld") is None:
        raise SystemExit("FATAL: NEMO grid_T has no mldr10_1 to compare against")
    if L["lat"].ndim != 2:
        raise SystemExit("FATAL: this probe is tripole-only (needs a 2-D mesh); "
                         f"got lat shape {L['lat'].shape}")

    # --- native-cell mapping, searched and then VERIFIED ---------------------
    nemo_lon = np.asarray(N["lon"], dtype=np.float64)
    # NEMO's coordinates are only meaningful where it has ocean data: grid_T
    # carries a fill row (nav_lat = -1.0) that would otherwise dominate the
    # residual and veto the correct offset.
    nemo_valid = np.asarray(N["mask"]) > 0.5
    print(f"[map] matching on {int(nemo_valid.sum())} NEMO ocean cells "
          f"(of {nemo_valid.size}); fill/land cells excluded")
    score, j0, i0, dlat_max, dlon_p99 = _find_native_offset(
        np.asarray(L["lat"], np.float64), np.asarray(L["lon"], np.float64) % 360.0,
        np.asarray(N["lat"], np.float64), nemo_lon, nemo_valid)
    ny, nx = N["sst"].shape
    print(f"[map] best native offset j0={j0} i0={i0}: "
          f"max|dlat|={dlat_max:.2e} deg, p99|dlon|={dlon_p99:.2e} deg")
    if dlat_max > _COORD_TOL_DEG or dlon_p99 > _COORD_TOL_DEG:
        raise SystemExit(
            f"FATAL: no cell-for-cell mapping found (max|dlat|={dlat_max:.3e}, "
            f"p99|dlon|={dlon_p99:.3e} > {_COORD_TOL_DEG}). These are not the "
            "same mesh, or the snapshot is not a tripole run -- refusing to "
            "compare native cells.")

    sl = (slice(j0, j0 + ny), slice(i0, i0 + nx))
    lat = np.asarray(L["lat"], np.float64)[sl]
    lon = np.asarray(L["lon"], np.float64)[sl] % 360.0
    wet2d = np.asarray(L["mask"])[sl] > 0.5
    Hb = np.asarray(L["H_bathy"], np.float64)[sl]
    T3 = np.asarray(L["T3d"], np.float64)[sl]
    S3 = np.asarray(L["S3d"], np.float64)[sl]
    z_c = np.asarray(L["z_center_ref"], np.float64)

    from legoesm.ocean.diagnostics import mixed_layer_depth
    from legoesm.ocean.eos import wright_eos
    wet3d = (z_c[None, None, :] < Hb[..., None]) & wet2d[..., None]
    mld = np.asarray(mixed_layer_depth(T3, S3, z_c, delta_sigma=0.01,
                                       wet_mask=wet3d.astype(np.float64),
                                       bottom_depth=Hb))
    mld_n = np.asarray(N["mld"], np.float64)

    arctic = wet2d & (lat >= _ARCTIC_LAT_N) & np.isfinite(mld) & np.isfinite(mld_n)
    n_arc = int(arctic.sum())
    if n_arc == 0:
        raise SystemExit("FATAL: no Arctic wet columns with finite MLD on both sides")

    deep = arctic & (mld > a.deep_mld_m)
    deep_n = arctic & (mld_n > a.deep_mld_m)
    # EXACT, not heuristic.  diagnostics.mixed_layer_depth ends with
    #     mld = jnp.where(has_crossing, mld_cross, bottom)
    #     mld = jnp.minimum(mld, bottom)
    # so a column that never crossed the threshold returns its bottom depth
    # EXACTLY.  Testing mld == H_bathy therefore identifies the saturated
    # columns outright; --bottom-frac only widens it to "near-bottom".
    saturated = mld >= Hb * (1.0 - 1e-9)
    near_bottom = mld >= a.bottom_frac * Hb
    bottom_lim = deep & near_bottom
    genuine = deep & ~bottom_lim
    bottom_lim_n = deep_n & (mld_n >= a.bottom_frac * Hb)
    print(f"  (exact saturation mld == H_bathy: {int((deep & saturated).sum())} "
          f"of {int(deep.sum())} deep columns -- these found NO density crossing "
          "at all)")

    print(f"\n[{a.label}] Arctic (>= {_ARCTIC_LAT_N:.0f}N) wet columns: {n_arc}")
    print(f"  deeper than {a.deep_mld_m:.0f} m : model {int(deep.sum())} "
          f"({100 * deep.sum() / n_arc:.2f} %)   NEMO {int(deep_n.sum())} "
          f"({100 * deep_n.sum() / n_arc:.2f} %)")
    print("  -- INSTRUMENT SPLIT (a saturated diagnostic is not a mixing depth) --")
    print(f"    bottom-limited (MLD >= {a.bottom_frac:.2f} x H_bathy): "
          f"model {int(bottom_lim.sum())} "
          f"({100 * bottom_lim.sum() / max(deep.sum(), 1):.1f} % of the model's deep "
          f"columns)   NEMO {int(bottom_lim_n.sum())}")
    print(f"    GENUINE deep (water left beneath): model {int(genuine.sum())} "
          f"({100 * genuine.sum() / n_arc:.2f} % of Arctic)")

    report = {
        "generated_by": str(_HERE), "git_sha": _git_sha(_HERE.parents[3]),
        "snapshot": str(a.snapshot), "label": a.label,
        "nemo_gridt": str(a.nemo_gridt), "nemo_month": a.nemo_month,
        "native_offset": {"j0": j0, "i0": i0, "max_abs_dlat_deg": dlat_max,
                          "p99_abs_dlon_deg": dlon_p99,
                          "note": "cell-for-cell, NO regridding; matched on NEMO "
                                  "ocean cells only (grid_T has a nav_lat=-1 fill row)"},
        "deep_mld_m": a.deep_mld_m, "bottom_frac": a.bottom_frac,
        "n_arctic_columns": n_arc,
        "n_deep_model": int(deep.sum()), "n_deep_nemo": int(deep_n.sum()),
        "n_bottom_limited_model": int(bottom_lim.sum()),
        "n_exactly_saturated_model": int((deep & saturated).sum()),
        "n_bottom_limited_nemo": int(bottom_lim_n.sum()),
        "n_genuine_deep_model": int(genuine.sum()),
    }

    if genuine.any():
        # POTENTIAL density in the SAME convention as the MLD diagnostic
        # (diagnostics.mixed_layer_depth: sigma = eos_fn(T, S, p_ref_pa) with
        # p_ref_pa=0.0).  Using in-situ density here would make every column
        # look stably stratified by compression alone.
        # wright_eos requires p as an ARRAY (it calls p.astype); a bare float
        # raises AttributeError. Broadcast the reference pressure explicitly.
        rho = np.asarray(wright_eos(T3, S3, np.zeros_like(T3)))
        ntop = max(int(np.searchsorted(z_c, a.unstable_top_m)), 2)
        unstable = _static_instability(rho, wet3d, ntop)
        ice = np.asarray(np.load(a.snapshot)["ice_concentration"])[sl] \
            if "ice_concentration" in np.load(a.snapshot).files else None

        g = genuine
        dS = (S3[..., 0] - np.asarray(N["sss"], np.float64))
        dT = (T3[..., 0] - np.asarray(N["sst"], np.float64))

        def agg(mask, name):
            return {"n": int(mask.sum()),
                    "median_H_bathy_m": float(np.median(Hb[mask])),
                    "median_mld_m": float(np.median(mld[mask])),
                    "median_nemo_mld_m": float(np.median(mld_n[mask])),
                    "median_dSSS_vs_nemo": float(np.median(dS[mask])),
                    "median_dSST_vs_nemo": float(np.median(dT[mask])),
                    "frac_statically_unstable_top": float(np.mean(unstable[mask])),
                    "frac_ice_covered": (float(np.mean(ice[mask] > 0.15))
                                         if ice is not None else None),
                    "lat_range": [float(lat[mask].min()), float(lat[mask].max())],
                    "_domain": name}

        rest = arctic & ~deep
        report["genuine_deep"] = agg(g, "genuine deep columns")
        report["arctic_rest"] = agg(rest, "other Arctic columns (reference)")
        print(f"\n  GENUINE-DEEP columns vs the rest of the Arctic:")
        for k in ("median_H_bathy_m", "median_mld_m", "median_nemo_mld_m",
                  "median_dSSS_vs_nemo", "median_dSST_vs_nemo",
                  "frac_statically_unstable_top", "frac_ice_covered"):
            v1, v2 = report["genuine_deep"][k], report["arctic_rest"][k]
            f = (lambda x: "n/a" if x is None else f"{x:+9.3f}")
            print(f"    {k:34s} deep {f(v1)}   rest {f(v2)}")

        # The individual columns, so a spatial pattern is checkable rather than
        # asserted.  Sorted by how far past NEMO they are.
        jj, ii = np.where(g)
        order = np.argsort(-(mld[g] - mld_n[g]))
        cols = [{"j": int(jj[k]), "i": int(ii[k]),
                 "lat": float(lat[jj[k], ii[k]]), "lon": float(lon[jj[k], ii[k]]),
                 "mld_m": float(mld[jj[k], ii[k]]),
                 "nemo_mld_m": float(mld_n[jj[k], ii[k]]),
                 "H_bathy_m": float(Hb[jj[k], ii[k]]),
                 "dSSS": float(dS[jj[k], ii[k]]), "dSST": float(dT[jj[k], ii[k]]),
                 "unstable_top": bool(unstable[jj[k], ii[k]]),
                 "ice_conc": (float(ice[jj[k], ii[k]]) if ice is not None else None)}
                for k in order]
        report["genuine_deep_columns"] = cols
        print(f"\n  worst 10 genuine-deep columns (model MLD - NEMO MLD):")
        for c in cols[:10]:
            print(f"    lat {c['lat']:6.2f} lon {c['lon']:7.2f}  mld {c['mld_m']:7.1f} "
                  f"(NEMO {c['nemo_mld_m']:6.1f})  H {c['H_bathy_m']:7.1f}  "
                  f"dSSS {c['dSSS']:+6.3f}  dSST {c['dSST']:+6.2f}  "
                  f"unstable={c['unstable_top']}  ice={c['ice_conc']}")
    else:
        print("\n  NO genuine-deep columns: every deep column is bottom-limited, "
              "i.e. the deep FRACTION is a statement about the diagnostic "
              "saturating, not about convection depth.")

    (a.out_dir / "report.json").write_text(json.dumps(report, indent=2,
                                                      allow_nan=False))
    print(f"\n[report] {a.out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
