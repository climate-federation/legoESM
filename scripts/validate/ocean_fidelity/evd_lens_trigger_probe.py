"""Would NEMO's EVD trigger fire on OUR cold-tongue lens columns at night?

Offline falsifier for the claim that the equatorial warm lens survives because
our production runs carry no convective adjustment (ORCA1: ``ln_zdfevd=T``,
``rn_evd=100`` on tracers where ``MIN(rn2, rn2b) <= -1e-12``).  On a day-30
snapshot (00Z = mid-afternoon in the cold tongue) the top cell is warmest; a
night of net cooling on the top cell alone makes it colder than the cell below
and the trigger fires -- IF the N2 used is NEMO's ``bn2`` (compressibility-free).
This probe evaluates NEMO's bn2 (Roquet TEOS-10 alpha/beta at the reference
depth ladder) for the snapshot as written and after imposing the night
cooling, and prints the fraction of box columns whose FIRST interior interface
(cells 1-2) is unstable.  (The model's legacy in-situ trigger is not
re-derived here: a lookalike is not the quantity; the runtime firing
diagnostic in the arm measures the real one.)

A firing fraction near 1 after cooling is the prediction; ~0 refutes the
mechanism.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from legoesm import constants
from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2

_C_P = 3991.87    # J/kg/K, NEMO rcp
_RHO_0 = 1026.0


def _box(z, lo, hi, halfwidth):
    lat = np.asarray(z["lat_T"], dtype=np.float64)
    lon = np.asarray(z["lon_T"], dtype=np.float64) % 360.0
    wet = np.asarray(z["land_mask"], dtype=np.float64) > 0.5
    return wet & (np.abs(lat) <= halfwidth) & (lon >= lo) & (lon < hi)


def _n2_top_interfaces(T, S, gdept, gdepw_int, n_if=3):
    """Signed NEMO bn2 at the first ``n_if`` interior interfaces, (ncol, n_if)."""
    T = T[:, :n_if + 1]
    S = S[:, :n_if + 1]
    n2 = compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept[:n_if + 1], gdepw_int[:n_if], eos_form="teos10",
        e3w_source="depth_difference")
    return np.asarray(n2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--lon-lo", type=float, default=220.0)
    ap.add_argument("--lon-hi", type=float, default=240.0)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--night-flux-wm2", type=float, default=100.0,
                    help="net upward (cooling) surface flux applied to the top "
                         "cell only, for --night-hours")
    ap.add_argument("--night-hours", type=float, default=4.0)
    ap.add_argument("--threshold", type=float, default=-1e-12,
                    help="NEMO zdfevd fires where N2 <= this")
    a = ap.parse_args()

    z = np.load(a.snapshot)
    box = _box(z, a.lon_lo, a.lon_hi, a.lat_halfwidth)
    T = np.asarray(z["T"], dtype=np.float64)[box]
    S = np.asarray(z["S"], dtype=np.float64)[box]
    gdept = np.abs(np.asarray(z["z_center_ref"], dtype=np.float64))
    gdepw_int = np.abs(np.asarray(z["z_interface_ref"], dtype=np.float64))[1:-1]
    ok = np.isfinite(T[:, :4]).all(axis=1) & np.isfinite(S[:, :4]).all(axis=1)
    T, S = T[ok], S[ok]
    n = T.shape[0]
    dz0 = float(np.abs(np.asarray(z["z_interface_ref"])[1]
                       - np.asarray(z["z_interface_ref"])[0]))
    cool = a.night_flux_wm2 * a.night_hours * 3600.0 / (_RHO_0 * _C_P * dz0)
    print(f"{a.snapshot}")
    print(f"box {a.lon_lo:.0f}-{a.lon_hi:.0f}E |lat|<={a.lat_halfwidth}: "
          f"{n} columns; top cell {dz0:.2f} m; night cooling of the top cell "
          f"{a.night_flux_wm2:.0f} W/m2 x {a.night_hours:.0f} h = {cool:.2f} K")
    print(f"box-mean T levels 0-3: {np.round(T[:, :4].mean(axis=0), 3)}  "
          f"(depths {np.round(gdept[:4], 2)} m)\n")

    Tn = T.copy()
    Tn[:, 0] -= cool
    print(f"{'state':>14}{'trigger':>10}{'if 1-2':>10}{'if 2-3':>10}"
          f"{'if 3-4':>10}   median N2(if 1-2)")
    for tag, TT in (("as written", T), ("after night", Tn)):
        for name, n2 in (("nemo_bn2", _n2_top_interfaces(TT, S, gdept, gdepw_int)),):
            fire = (n2 <= a.threshold).mean(axis=0)
            print(f"{tag:>14}{name:>10}" + "".join(f"{f:>10.3f}" for f in fire)
                  + f"   {np.median(n2[:, 0]):.3e}")
    print("\nfraction = share of box columns whose interface is unstable "
          "(would carry avt = rn_evd). NEMO's own 5-day-mean occupancy at "
          "1-2 m in this box is ~0.5-1.0 (equatorial_diffusivity_oracle).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
