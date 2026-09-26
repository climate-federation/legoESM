"""How much of the real world does the hybrid vertical coordinate forbid?

``standard_hybrid_levels`` builds ``B(eta) = eta**transition_exponent``.  Near
the surface ``dB/deta -> transition_exponent``, so a layer carries POSITIVE
mass only while ``p_s`` stays above a threshold set by the coordinate alone.
Below it the near-surface layers invert and ``dp_from_hybrid`` -- which feeds
the dycore, not merely a diagnostic -- returns negative thicknesses.

Found on #1029: two cells in that regime killed a 200-day idealized run inside
200 steps.  The coordinate's own docstring says a 2.5-degree AMIP run reaches
543 hPa over the Tibetan Plateau, and #1029 exists because the AMIP lat-lon
lane dies only when real topography is present -- so the question is whether
production has been running inside the invalid regime all along.

This probe reports two things the choice of fix depends on:

* per candidate coordinate, the minimum valid surface pressure AND the highest
  elevation it admits -- the second is the one a modeller can judge, because
  "663.9 hPa" does not obviously read as "forbids the Tibetan Plateau";
* against a real elevation dataset, how many cells and how much AREA each
  candidate puts in the invalid regime.

Run: scripts/cluster/issue_sweep_0923/negmass_exposure.sbatch
"""
from __future__ import annotations

import argparse

import numpy as np

from legoesm import constants
from legoesm.grids.vertical import (hybrid_min_valid_surface_pressure,
                                    make_hybrid_levels,
                                    standard_hybrid_levels)


def _threshold(n_levels, exponent, stretching, p_top_Pa=200.0):
    # ``standard_hybrid_levels(n)`` only takes (n_levels, p_ref) and picks
    # p_top / stretching / exponent from a table keyed on n_levels; the knobs
    # this probe sweeps live on ``make_hybrid_levels``, which it delegates to.
    s = make_hybrid_levels(
        n_levels, p_top_Pa=p_top_Pa, transition_exponent=exponent,
        stretching=stretching)
    return float(hybrid_min_valid_surface_pressure(
        np.asarray(s.A_half), np.asarray(s.B_half), constants.p_ref))


def _production_threshold(n_levels):
    """The coordinate a run actually gets from ``standard_hybrid_levels``."""
    s = standard_hybrid_levels(n_levels)
    return float(hybrid_min_valid_surface_pressure(
        np.asarray(s.A_half), np.asarray(s.B_half), constants.p_ref))


def _elevation_for(p_s_Pa, T_K):
    """Elevation whose hydrostatic surface pressure is ``p_s_Pa``.

    The isothermal relation the idealized initial conditions themselves use
    (``p_s = p_ref * exp(-g z / (R_d T))``).  A real atmosphere has a lapse
    rate, so this understates the height a little; it is here to make the
    pressure threshold legible, not to be a barometric standard.
    """
    return float(constants.R_d * T_K / constants.g
                 * np.log(constants.p_ref / p_s_Pa))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlev", type=int, default=40)
    p.add_argument("--T-ref-K", type=float, default=288.0,
                   help="isothermal temperature for the elevation conversion")
    p.add_argument("--elevation-file", default="",
                   help="NetCDF with a real elevation field; omit to skip the "
                        "exposure section and report thresholds only")
    p.add_argument("--elev-var", default="")
    a = p.parse_args(argv)

    cands = [
        ("exponent=3 stretching=2.5 (L40 production)", 3, 2.5),
        ("         exponent=3 stretching=0.0", 3, 0.0),
        ("gentler  exponent=2 stretching=2.5", 2, 2.5),
        ("gentler  exponent=2 stretching=0.0", 2, 0.0),
        ("sigma-like exponent=1 stretching=2.5", 1, 2.5),
        ("sigma-like exponent=1 stretching=0.0", 1, 0.0),
    ]
    prod_thr = _production_threshold(a.nlev)
    prod_z = _elevation_for(prod_thr, a.T_ref_K) if prod_thr > 0 else float("inf")
    print(f"n_levels = {a.nlev}, elevation conversion at T = {a.T_ref_K} K")
    print(f"PRODUCTION standard_hybrid_levels({a.nlev}): min valid p_s = "
          f"{prod_thr/100:.1f} hPa, max elevation "
          f"{'unbounded' if not np.isfinite(prod_z) else f'{prod_z:.0f} m'}\n")
    print(f"{'coordinate':40s} {'min valid p_s':>14s} {'max elevation':>15s}")
    rows = []
    for label, expo, stre in cands:
        thr = _threshold(a.nlev, expo, stre)
        z_max = _elevation_for(thr, a.T_ref_K) if thr > 0 else float("inf")
        rows.append((label, expo, stre, thr, z_max))
        z_txt = "unbounded" if not np.isfinite(z_max) else f"{z_max:9.0f} m"
        print(f"{label:40s} {thr/100:11.1f} hPa {z_txt:>15s}")

    if not a.elevation_file:
        print("\n(no --elevation-file given; exposure section skipped)")
        return 0

    import xarray as xr
    ds = xr.open_dataset(a.elevation_file)
    var = a.elev_var or [v for v in ds.data_vars
                         if ds[v].ndim == 2][0]
    z = np.asarray(ds[var].values, dtype=np.float64)
    lat_name = [c for c in ds[var].dims if "lat" in c.lower()]
    lat = (np.asarray(ds[lat_name[0]].values, dtype=np.float64)
           if lat_name else None)
    print(f"\nelevation field: {a.elevation_file}")
    print(f"  variable {var!r}, shape {z.shape}, "
          f"max {np.nanmax(z):.0f} m, min {np.nanmin(z):.0f} m")

    land = np.where(z > 0.0, z, 0.0)          # ocean contributes p_s = p_ref
    p_s = constants.p_ref * np.exp(
        -constants.g * land / (constants.R_d * a.T_ref_K))

    # Area weights: a lat-lon cell's area goes as cos(lat).  An unweighted cell
    # count over a lat-lon grid is not a global fraction (this repo has been
    # bitten by exactly that), so both are reported and the weighted one is
    # the answer.
    if lat is not None and lat.size == z.shape[0]:
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, z.shape[1]))
    else:
        w = np.ones_like(z)
        print("  WARNING: latitude axis not identified; area weights are "
              "UNIFORM and the percentages below are cell counts, not area")

    print(f"\n{'coordinate':40s} {'cells':>10s} {'% cells':>9s} {'% AREA':>9s}"
          f" {'lowest p_s':>12s}")
    for label, _e, _s, thr, _z in rows:
        bad = p_s < thr
        pct_cells = 100.0 * bad.sum() / bad.size
        pct_area = 100.0 * w[bad].sum() / w.sum()
        print(f"{label:40s} {int(bad.sum()):10d} {pct_cells:8.3f}% "
              f"{pct_area:8.3f}% {p_s.min()/100:9.1f} hPa")
    print("\nA coordinate with 0.000% is valid everywhere this dataset has "
          "land; anything above that is running the dycore on negative layer "
          "thicknesses over that fraction of the planet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
