#!/usr/bin/env python
"""Does the calibrated land model change what the ATMOSPHERE receives?

The land parameter tables were fitted with a plant (canopy conductance) model
active, and the coupled runs deployed them with that model switched off — so the
fitted conductance was inert.  This scores the controlled pair that turns it back
on: two AMIP runs identical except for the plant model.

WHAT IT REPORTS, and why each is here:
  * land latent heat  — the quantity the plant model directly throttles, and the
    one whose coupled collapse is the open question.
  * land sensible heat and near-surface temperature — where the energy goes
    instead; a latent drop with no sensible rise means energy went missing.
  * land precipitation — the recycling feedback that turns a local evaporation
    change into a circulation change.
  * the same over OCEAN — a control: the plant model must not move it, and a
    large ocean change means the arms differ in something other than the land.

EVERY number is area-weighted by the model's own cell area and split by the
model's own land fraction.  An unweighted mean on this mesh is not a global mean,
and a "land" mean that includes half-ocean cells is not a land mean.

Run: PYTHONPATH=. python scripts/validate/amip_land_calibrated_ab.py \
         --control landcal_off --calibrated landcal_on
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np

_ROOT = "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs"
# A cell counts as land / ocean when the model's own land fraction says so.
# Cells in between are excluded from BOTH so neither mean is contaminated by the
# other surface — a mostly-ocean cell's evaporation is not a land signal.
_LAND_MIN_FRAC = 0.8
_OCEAN_MAX_FRAC = 0.2


def _open(root, arm, var, table="Amon"):
    import xarray as xr
    hits = sorted(glob.glob(f"{root}/{arm}/cmor/{table}/{var}_{table}_*.nc"))
    if not hits:
        return None
    return xr.open_dataset(hits[0])[var]


def _masks(root, arm):
    """(area, land mask, ocean mask) from the run's OWN fx fields."""
    sftlf = _open(root, arm, "sftlf", table="fx")
    area = _open(root, arm, "areacella", table="fx")
    if sftlf is None or area is None:
        raise SystemExit(
            f"{arm}: no sftlf/areacella in cmor/fx — cannot area-weight or split "
            "land from ocean, and an unweighted mean on this mesh is not a mean.")
    f = np.asarray(sftlf).astype(float)
    if np.nanmax(f) > 1.5:            # percent vs fraction
        f = f / 100.0
    return np.asarray(area).astype(float), f >= _LAND_MIN_FRAC, f <= _OCEAN_MAX_FRAC


def _mean(field, area, mask):
    v = np.asarray(field).astype(float)
    m = mask & np.isfinite(v)
    if not m.any():
        return float("nan")
    return float((v[m] * area[m]).sum() / area[m].sum())


def _maps(args, area, land, rows):
    """Maps: each field in both arms and their difference, land only.

    Ocean is blanked rather than plotted, because the plant model acts on land
    and a shared colour scale spanning both surfaces would hide the land signal
    under the much larger ocean values.  The two arms share one colour scale per
    row so the panels are comparable by eye; the difference panel gets its own
    symmetric scale centred on zero, so its colour says sign directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fetch(arm, var, scale):
        d = _open(args.root, arm, var)
        return None if d is None else np.asarray(d.mean("time")) * scale

    live = [(lbl, v, u, sc) for lbl, v, u, sc in rows
            if _fetch(args.control, v, sc) is not None]
    fig, axes = plt.subplots(len(live), 3, figsize=(16, 3.1 * len(live)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    lat = np.asarray(_open(args.root, args.control, "tas").lat)
    lon = np.asarray(_open(args.root, args.control, "tas").lon)

    for r, (label, var, unit, scale) in enumerate(live):
        a = _fetch(args.control, var, scale)
        b = _fetch(args.calibrated, var, scale)
        a = np.where(land, a, np.nan)
        b = np.where(land, b, np.nan)
        both = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
        lo, hi = np.percentile(both, [2, 98])
        d = b - a
        dlim = float(np.nanpercentile(np.abs(d[np.isfinite(d)]), 98)) or 1.0

        for c, (field, title, cmap, vmin, vmax) in enumerate((
                (a, "plants OFF", "YlGnBu", lo, hi),
                (b, "plants ON", "YlGnBu", lo, hi),
                (d, "ON - OFF", "RdBu_r", -dlim, dlim))):
            ax = axes[r, c]
            m = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax,
                              shading="auto")
            ax.set_title(f"{label} — {title}" if r == 0 or c == 2
                         else f"{label} — {title}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(m, ax=ax, shrink=0.85, label=unit if c == 2 else "")

    fig.suptitle(
        f"Land: {args.control} ({args.label_control}) vs "
        f"{args.calibrated} ({args.label_calibrated})"
        f"  —  ocean blanked, {args.window}", fontsize=11)
    fig.savefig(args.maps, dpi=110)
    print(f"\nmaps written to {args.maps}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=_ROOT)
    ap.add_argument("--control", required=True, help="run with the plant model OFF")
    ap.add_argument("--calibrated", required=True, help="run with it ON")
    ap.add_argument("--maps", metavar="PNG",
                    help="also write comparison maps to this path")
    # The figure must say what the two arms ACTUALLY differ in. The defaults
    # describe the pair this was written for (plant model off vs on); any other
    # pair has to relabel, or the figure claims an experiment nobody ran.
    ap.add_argument("--label-control", default="plants off",
                    help="what the control arm is, for the figure title")
    ap.add_argument("--label-calibrated", default="plants on",
                    help="what the other arm is, for the figure title")
    ap.add_argument("--window", default="5-day January mean",
                    help="the averaging window, for the figure title")
    args = ap.parse_args()

    area, land, ocean = _masks(args.root, args.control)
    area_b, land_b, _ = _masks(args.root, args.calibrated)
    if not (np.allclose(area, area_b) and (land == land_b).all()):
        raise SystemExit(
            "the two arms do not share a grid or a land mask — they differ in "
            "more than the plant model, so no difference between them is "
            "attributable to it.")

    rows = [("land latent heat", "hfls", "W/m2", 1.0),
            ("land sensible heat", "hfss", "W/m2", 1.0),
            ("land near-surface T", "tas", "K", 1.0),
            ("land precipitation", "pr", "mm/day", 86400.0)]
    print(f"{'':<24}{'control':>10}{'calibrated':>12}{'change':>10}")
    print("--- LAND (the plant model acts here) ---")
    out = {}
    for label, var, unit, scale in rows:
        a, b = (_open(args.root, args.control, var),
                _open(args.root, args.calibrated, var))
        if a is None or b is None:
            print(f"{label:<24}{'(not written)':>32}")
            continue
        # Time-mean first, then space: both arms cover the same span by
        # construction (same --days), so the windows match.
        av = _mean(a.mean("time") * scale, area, land)
        bv = _mean(b.mean("time") * scale, area, land)
        out[var] = (av, bv)
        print(f"{label:<24}{av:10.2f}{bv:12.2f}{bv - av:+10.2f}  {unit}")

    print("--- OCEAN (control: the plant model must NOT move this) ---")
    for label, var, unit, scale in (("ocean latent heat", "hfls", "W/m2", 1.0),
                                    ("ocean near-surface T", "tas", "K", 1.0)):
        a, b = (_open(args.root, args.control, var),
                _open(args.root, args.calibrated, var))
        if a is None or b is None:
            continue
        av = _mean(a.mean("time") * scale, area, ocean)
        bv = _mean(b.mean("time") * scale, area, ocean)
        print(f"{label:<24}{av:10.2f}{bv:12.2f}{bv - av:+10.2f}  {unit}")

    print("\nREADING THIS:")
    if "hfls" in out:
        a, b = out["hfls"]
        print(f"  Land evaporation moved {b - a:+.1f} W/m2 ({100 * (b - a) / max(a, 1e-9):+.0f} %).")
        print("  Observed land latent heat is roughly 28-32 W/m2 in the annual mean;")
        print("  a SHORT run from a cold soil column is not comparable to that, so")
        print("  read the CHANGE between the arms, not either absolute value.")
    print("  A large OCEAN change means the arms differ in more than the plant")
    print("  model and nothing here is attributable to it.")
    if args.maps:
        _maps(args, area, land, rows)

    print("\nNOT ANSWERED HERE: whether the flux the land solved equals the flux the")
    print("  atmosphere applied.  The published output carries only the")
    print("  atmosphere's, so that closure check needs a diagnostic that exports")
    print("  both — the reviewers' named follow-up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
