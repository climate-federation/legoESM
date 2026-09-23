#!/usr/bin/env python3
"""Area-weighted spatial scores for a pair of AMIP arms, plus a clear-sky split.

Both reviewers refused the signed regional means the surface-flux pair was
first judged on.  A signed mean lets an over-rainy region cancel a dry one, so
it cannot say whether the field's PATTERN improved; and a top-of-atmosphere
number that does not separate clear sky from cloud cannot say whether a
radiative change came from clouds at all.

So this prints, per field: the area-weighted mean bias AND the area-weighted
root-mean-square error against the same reference, over the whole globe and
over the deep tropics.  Radiation additionally gets its clear-sky counterpart
and the cloud radiative effect (clear minus all sky), which is the part a
cloud or moisture change is actually allowed to move.

Usage: pair_scores.py <run> [<run> ...]
"""
import sys

import numpy as np
sys.path.insert(0, "scripts/validate/amip_bias")
import regional_bias as rb

BOXES = {"GLOBAL": (-90, 90, 0, 360), "tropics 20S-20N": (-20, 20, 0, 360)}
# (field, display scale, unit) -- water fluxes are stored in kg m-2 s-1.
FIELDS = [("pr", 86400.0, "mm/d"), ("evspsbl", 86400.0, "mm/d"),
          ("pr_era5", 86400.0, "mm/d"), ("emp", 86400.0, "mm/d"),
          ("prw", 1.0, "kg/m2"), ("clt", 1.0, "%"),
          ("rsut", 1.0, "W/m2"), ("rlut", 1.0, "W/m2"),
          ("rsutcs", 1.0, "W/m2"), ("rlutcs", 1.0, "W/m2"),
          ("cre_sw", 1.0, "W/m2"), ("cre_lw", 1.0, "W/m2")]


def rms(field, lat, lon, box):
    return float(np.sqrt(rb.region_mean(field ** 2, lat, lon, box)))


def load_pair(run, var):
    """Model field and its reference on the model grid, or None if absent.

    ``cre_*`` is not published: it is built here from the all-sky and
    clear-sky pair on BOTH sides with the same sign convention, so the
    difference is a like-for-like cloud radiative effect and not a mix of two
    definitions.
    """
    if var == "pr_era5":                       # same rain, ERA5 instead of GPCP
        d = rb._load_model(run, "pr")
        lat, lon = np.asarray(d.lat), np.asarray(d.lon)
        ref = rb._ref_clim("pr", rb._month_labels(d), lat, lon, src=rb.ERA5)
        return np.asarray(d["pr"]).mean(0), np.asarray(ref), lat, lon
    if var == "emp":
        # Evaporation minus precipitation, ERA5 on BOTH terms.  A model whose
        # water budget closes has E-P near zero in the monthly mean, so an
        # evaporation bias read against ERA5 and a rain bias read against GPCP
        # can disagree by the two references' own mismatch and manufacture a
        # deficit the model does not have.  One reference, one budget.
        e, p_ = load_pair(run, "evspsbl"), load_pair(run, "pr_era5")
        return (e[0] - p_[0], e[1] - p_[1], e[2], e[3])
    if var.startswith("cre_"):
        allsky, clear = ("rsut", "rsutcs") if var == "cre_sw" else ("rlut", "rlutcs")
        a, c = load_pair(run, allsky), load_pair(run, clear)
        if a is None or c is None:
            return None
        return (c[0] - a[0], c[1] - a[1], a[2], a[3])
    d = rb._load_model(run, var)
    if d is None:
        return None
    lat, lon = np.asarray(d.lat), np.asarray(d.lon)
    ref = rb._ref_clim(var, rb._month_labels(d), lat, lon)
    if ref is None:
        return None
    return np.asarray(d[var]).mean(0), np.asarray(ref), lat, lon


def main(runs):
    for box_name, box in BOXES.items():
        print(f"\n== {box_name}    " + "".join(f"{r:>26}" for r in runs))
        print(f"{'field':<18}" + "".join("      bias      rms" for _ in runs))
        for var, scale, unit in FIELDS:
            cells = []
            for run in runs:
                p = load_pair(run, var)
                if p is None:
                    cells.append(None)
                    continue
                m, o, lat, lon = p
                cells.append((rb.region_mean(m - o, lat, lon, box) * scale,
                              rms(m - o, lat, lon, box) * scale))
            if all(c is None for c in cells):
                continue
            print(f"{var + ' [' + unit + ']':<18}"
                  + "".join("        --       --" if c is None
                            else f"  {c[0]:8.2f} {c[1]:8.2f}" for c in cells))
    print("\nbias = area-weighted mean error; rms = area-weighted RMS error, "
          "same reference and months on both sides.")
    print("cre = clear-sky minus all-sky outgoing flux, built identically for "
          "model and reference.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
