#!/usr/bin/env python
"""Jablonowski-Williamson baroclinic wave vs the FV3 duo-grid oracle.

The Zenodo 8327578 duo-grid release ships reference solutions for
``nh.case-13`` = "DCMIP 2016 J&W BC Wave, with perturbation"
(``tools/test_cases.F90:77``) at C48/C96/C192/C384/C768, in BOTH the
duo-grid and the plain (non-duo) cubed-sphere configuration, as 216
hourly frames of ``ps`` (Pa) and ``vcomp`` (m/s, one level, 462 hPa) on a
181x360 lat-lon canvas.

Nothing in this repository compared against it, so the 3D cube lane had
no oracle anchor at all.  This script is that anchor.

WHAT IS COMPARED, AND WHY ONLY ``p_s``
--------------------------------------
Only surface pressure.  The matrix's ``snapshots_latlon.npz`` stores ``v``
at the BOTTOM model level (``run_atmosphere_test_matrix.py`` builds it as
``v[..., -1]``), while the oracle's ``vcomp`` is a single level at
462 hPa -- different quantities, so comparing them would be a
same-name-different-thing error.  ``p_s`` is unambiguous: Pa on both
sides, on the same 181x360 canvas.

The headline metric is the area-weighted RMS deviation of ``p_s`` about
its own area-weighted mean, ``rms'``.  It needs no longitude alignment
(the oracle stores lon 0.5..359.5, the matrix stores -179.5..179.5) and
no cross-dataset normalisation, so it cannot be wrong through a roll or
a weight -- see the ``oracle-fidelity`` rule "prefer weighting-free
invariants".  Maps are drawn on each dataset's OWN longitude axis mapped
into [0, 360), never by assuming a roll.

HONEST PROTOCOL NOTE
--------------------
This is an EXTERNAL TARGET, not a controlled comparison.  The oracle is
C48 / nonhydrostatic / 32 levels / hord6 / duo; a matrix cube run is C36
/ hydrostatic / 40 sigma levels / its own calibrated solver.  The script
quantifies DISTANCE to the duo reference; it attributes nothing on its
own.  The useful control is internal: run the SAME matrix case on
lat-lon and on the cube and compare both to the oracle -- the grid is
then the variable that moved.

Usage:
  jw_duo_oracle_compare.py --ours <snapshots_latlon.npz> [--label NAME]
                           [--ours <...> --label <...>]...
                           [--res C48] [--days 1,5,9] [--out fig.png]
                           [--json out.json]

Exit status is 0 whenever the comparison itself ran; this is an
instrument, not a gate.  ``--max-rms-ratio R`` turns it into a gate that
exits 1 if any arm's day-1 ``rms'`` exceeds R times the duo oracle's.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ZENODO_ROOT = os.environ.get(
    "LEGOESM_FV3_ZENODO_ROOT",
    "/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/extracted/"
    "Code and simulations files")

# The oracle ships both configurations of the same case; carrying BOTH is
# the point -- their difference is what the duo grid actually buys, and it
# is the only scale against which "our cube imprint is large" means
# anything.
ORACLE_ARMS = {
    "duo": "{res}.nh.case-13.alpha0.duo.hord6",
    "plain": "{res}.nh.case-13.alpha0.hord6",
}


def area_weights(lat_deg: np.ndarray, nlon: int) -> np.ndarray:
    """cos(lat) weights normalised to sum to 1 over the (nlat, nlon) canvas."""
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)))[:, None]
    w = np.repeat(w, nlon, axis=1)
    return w / w.sum()


def ps_stats(ps_pa: np.ndarray, lat_deg: np.ndarray) -> dict:
    """Alignment-free surface-pressure statistics, all in hPa.

    ``rms_prime`` is the area-weighted RMS about the area-weighted mean --
    invariant under any longitude roll, so a coordinate-convention
    mismatch cannot manufacture or hide a signal.
    """
    ps = np.asarray(ps_pa, dtype=np.float64)
    if ps.ndim != 2:
        raise ValueError(f"expected a 2-D (nlat, nlon) p_s field, got {ps.shape}")
    if not np.all(np.isfinite(ps)):
        raise ValueError("p_s contains non-finite values -- refusing to score")
    w = area_weights(lat_deg, ps.shape[1])
    mean = float((ps * w).sum())
    return {
        "min_hPa": float(ps.min()) / 100.0,
        "max_hPa": float(ps.max()) / 100.0,
        "mean_hPa": mean / 100.0,
        "rms_prime_hPa": float(np.sqrt((w * (ps - mean) ** 2).sum())) / 100.0,
    }


def load_oracle(res: str, arm: str):
    """Return (times_days, ps[nt, nlat, nlon] Pa, lat, lon 0..360)."""
    import netCDF4 as nc

    sub = ORACLE_ARMS[arm].format(res=res)
    path = os.path.join(ZENODO_ROOT, sub, "rundir", "atmos_daily.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"oracle reference not found: {path}\n"
            f"Set LEGOESM_FV3_ZENODO_ROOT, or pick a resolution that exists "
            f"({', '.join(sorted(_available_resolutions()))}).")
    d = nc.Dataset(path)
    if d.variables["ps"].units.strip() != "Pa":
        raise ValueError(
            f"oracle ps units are {d.variables['ps'].units!r}, expected 'Pa'")
    return (np.asarray(d.variables["time"][:], dtype=np.float64),
            np.asarray(d.variables["ps"][:], dtype=np.float64),
            np.asarray(d.variables["lat"][:], dtype=np.float64),
            np.asarray(d.variables["lon"][:], dtype=np.float64) % 360.0)


def _available_resolutions():
    if not os.path.isdir(ZENODO_ROOT):
        return set()
    return {n.split(".")[0] for n in os.listdir(ZENODO_ROOT)
            if ".nh.case-13." in n}


def load_ours(path: str):
    """Return (times_days, ps Pa, lat, lon 0..360) from a matrix npz."""
    z = np.load(path, allow_pickle=True)
    for k in ("times_days", "p_s", "lat", "lon"):
        if k not in z.files:
            raise KeyError(
                f"{path} has no {k!r}; keys are {sorted(z.files)}. This script "
                f"reads a matrix snapshots_latlon.npz from a 3-D case.")
    ps = np.asarray(z["p_s"], dtype=np.float64)
    if ps.ndim != 3:
        raise ValueError(f"p_s must be (nt, nlat, nlon), got {ps.shape}")
    # A surface pressure in Pa is ~1e5.  Catch an hPa-valued file rather
    # than silently reporting a 100x-wrong rms.
    if not (5.0e4 < float(np.nanmedian(ps)) < 1.5e5):
        raise ValueError(
            f"{path}: median p_s = {np.nanmedian(ps):.4g}, which is not Pa. "
            f"Refusing to compare mismatched units.")
    return (np.asarray(z["times_days"], dtype=np.float64), ps,
            np.asarray(z["lat"], dtype=np.float64),
            np.asarray(z["lon"], dtype=np.float64) % 360.0)


def at_day(times: np.ndarray, day: float, tol: float = 0.02):
    """Index of the frame at ``day``.  Fails loudly rather than snapping to
    a far-away frame -- a silent nearest-frame match is how a day-0/day-1
    lens error gets reported as physics."""
    i = int(np.argmin(np.abs(times - day)))
    if abs(float(times[i]) - day) > tol:
        raise ValueError(
            f"no frame within {tol} d of day {day}; nearest is "
            f"{float(times[i]):.4f} d. Available span "
            f"{float(times.min()):.3f}..{float(times.max()):.3f} d.")
    return i


def common_days(spans: dict[str, tuple[float, float]],
                days: list[float]) -> list[float]:
    """Days that every arm actually spans.

    An arm still being written by a running matrix job covers fewer days
    than the rest.  Scoring the arms over DIFFERENT windows would be the
    classic differing-window confound, so the window is intersected --
    and whatever is dropped is printed, never silently truncated.
    """
    lo = max(s[0] for s in spans.values())
    hi = min(s[1] for s in spans.values())
    keep = [d for d in days if lo - 1e-9 <= d <= hi + 1e-9]
    dropped = [d for d in days if d not in keep]
    if dropped:
        limiter = min(spans.items(), key=lambda kv: kv[1][1])
        print(f"NOTE: dropped day(s) {', '.join('%g' % d for d in dropped)} — "
              f"outside the common window [{lo:g}, {hi:g}] d. Shortest arm is "
              f"{limiter[0]!r} ({limiter[1][0]:g}..{limiter[1][1]:g} d). All "
              f"arms are scored on the SAME window.")
    if not keep:
        raise ValueError(
            f"no requested day lies in the common window [{lo:g}, {hi:g}] d; "
            f"spans are " + "; ".join(
                f"{k}: {v[0]:g}..{v[1]:g}" for k, v in spans.items()))
    return keep


def build_curves(res: str, arms: list[tuple[str, str]], days: list[float]):
    """({label: {day: stats}}, days_used) for the oracle arms plus each of ours.

    Every arm is scored on the SAME set of days.
    """
    loaded: dict[str, tuple] = {}
    for arm in ORACLE_ARMS:
        t, ps, lat, _ = load_oracle(res, arm)
        loaded[f"oracle {arm}"] = (t, ps, lat)
    for label, path in arms:
        t, ps, lat, _ = load_ours(path)
        loaded[label] = (t, ps, lat)

    spans = {k: (float(v[0].min()), float(v[0].max()))
             for k, v in loaded.items()}
    used = common_days(spans, days)
    out = {k: {d: ps_stats(ps[at_day(t, d)], lat) for d in used}
           for k, (t, ps, lat) in loaded.items()}
    return out, used, spans


def plot(res, arms, days, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("oracle duo", None), ("oracle plain", None)] + [
        (lab, path) for lab, path in arms]
    fig, axes = plt.subplots(len(rows), len(days),
                             figsize=(4.2 * len(days), 2.9 * len(rows)),
                             squeeze=False)
    for r, (label, path) in enumerate(rows):
        if path is None:
            t, ps, lat, lon = load_oracle(res, label.split()[1])
        else:
            t, ps, lat, lon = load_ours(path)
        order = np.argsort(lon)          # each dataset on its OWN lon axis,
        lon_s, ps = lon[order], ps[:, :, order]   # mapped to [0,360) -- no
        for c, day in enumerate(days):            # roll is ever assumed
            ax = axes[r][c]
            f = ps[at_day(t, day)] / 100.0
            m = f.mean()
            # Per-panel symmetric scale about the panel's own mean: the
            # arms differ by two orders of magnitude, so a shared scale
            # would render the oracle rows as flat grey.
            a = float(np.abs(f - m).max()) or 1.0
            im = ax.pcolormesh(lon_s, lat, f - m, cmap="RdBu_r",
                               vmin=-a, vmax=a, shading="auto")
            ax.set_title(f"{label} · day {day:g} · ±{a:.2f} hPa", fontsize=8)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel("lat", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.025)
    fig.suptitle(
        f"J&W baroclinic wave · p_s anomaly · FV3 duo-grid oracle ({res}, "
        f"nonhydrostatic, hord6) vs legoESM\n"
        f"each panel scaled to its OWN range — read the ± in the title",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)


def plot_curves(curves, days, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for label, per_day in curves.items():
        y = [per_day[d]["rms_prime_hPa"] for d in days]
        style = dict(lw=2.4, marker="o") if label.startswith("oracle") else \
            dict(lw=1.6, marker="s", ls="--")
        ax.semilogy(days, y, label=label, **style)
    ax.set_xlabel("day")
    ax.set_ylabel("area-weighted rms$'$ of $p_s$  (hPa)")
    ax.set_title("J&W wave amplitude vs the FV3 duo-grid oracle\n"
                 "(alignment-free metric; log scale)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print("wrote", out_png)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", action="append", default=[],
                    help="matrix snapshots_latlon.npz (repeatable)")
    ap.add_argument("--label", action="append", default=[],
                    help="label per --ours, in the same order")
    ap.add_argument("--res", default="C48",
                    help="oracle resolution: C48/C96/C192/C384/C768")
    ap.add_argument("--days", default="1,3,5,7,9")
    ap.add_argument("--map-days", default="1,5,9")
    ap.add_argument("--out", default=None, help="map PNG")
    ap.add_argument("--out-curves", default=None, help="growth-curve PNG")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-rms-ratio", type=float, default=None,
                    help="gate: exit 1 if any arm's day-1 rms' exceeds this "
                         "multiple of the duo oracle's")
    args = ap.parse_args(argv)

    if len(args.label) not in (0, len(args.ours)):
        ap.error("--label must be given once per --ours, or not at all")
    labels = args.label or [os.path.basename(os.path.dirname(p)) or p
                            for p in args.ours]
    arms = list(zip(labels, args.ours))
    days = [float(x) for x in args.days.split(",")]

    curves, days, spans = build_curves(args.res, arms, days)
    hdr = (f"{'arm':<28}" + "".join(f"{('d%g' % d):>11}" for d in days)
           + f"{'span (d)':>14}")
    print(hdr)
    print("-" * len(hdr))
    # The reference window is the oracle's own span -- a fixed property of
    # the released dataset, not evidence about any run.  An arm of OURS
    # that stops short of it is usually a run that CRASHED, so flag it and
    # never flag the oracle rows for being 9 d rather than 10 d.
    ref_hi = spans["oracle duo"][1]
    for label, per_day in curves.items():
        lo, hi = spans[label]
        mark = ("  <-- TRUNCATED, did not reach the day-%g reference window"
                % ref_hi) if (not label.startswith("oracle")
                              and hi < ref_hi - 1e-9) else ""
        print(f"{label:<28}" + "".join(
            f"{per_day[d]['rms_prime_hPa']:>11.4f}" for d in days)
            + f"{lo:>6.2f}..{hi:<6.2f}" + mark)
    print("(values are area-weighted rms' of p_s in hPa; TRUNCATED arms "
          "ended early — check that run's log for a blow-up before quoting "
          "its number as an imprint)")
    print()
    for label, per_day in curves.items():
        print(f"{label:<28} " + "  ".join(
            f"d{d:g}:[{per_day[d]['min_hPa']:.2f},{per_day[d]['max_hPa']:.2f}]"
            for d in days))

    ratios = {}
    if days:
        d0 = days[0]
        base = curves["oracle duo"][d0]["rms_prime_hPa"]
        for label, per_day in curves.items():
            ratios[label] = per_day[d0]["rms_prime_hPa"] / base
        print(f"\nday-{d0:g} rms' relative to the duo oracle:")
        for label, r in ratios.items():
            print(f"  {label:<28} {r:>10.1f}x")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"res": args.res, "days": days, "curves": {
                k: {str(d): v for d, v in per.items()}
                for k, per in curves.items()},
                "day0_ratio_vs_duo": ratios}, fh, indent=2)
        print("wrote", args.json)
    if args.out:
        # Only map days every arm actually has, so no panel is drawn from a
        # different window than its neighbours.
        want = [float(x) for x in args.map_days.split(",")]
        md = [d for d in want if any(abs(d - u) < 1e-9 for u in days)]
        skipped = [d for d in want if d not in md]
        if skipped:
            print(f"NOTE: map days {skipped} are outside the common window; "
                  f"drawing {md}.")
        if md:
            plot(args.res, arms, md, args.out)
        else:
            print("NOTE: no map day inside the common window; skipping maps.")
    if args.out_curves:
        plot_curves(curves, days, args.out_curves)

    if args.max_rms_ratio is not None:
        bad = {k: r for k, r in ratios.items()
               if not k.startswith("oracle") and r > args.max_rms_ratio}
        if bad:
            print("\nJW_DUO_GATE: FAIL " + ", ".join(
                f"{k}={r:.1f}x>{args.max_rms_ratio:g}" for k, r in bad.items()))
            return 1
        print(f"\nJW_DUO_GATE: PASS (all arms <= {args.max_rms_ratio:g}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
