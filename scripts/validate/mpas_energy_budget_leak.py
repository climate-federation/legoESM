"""Column energy-budget LEAK for an MPAS AMIP run (#1354 / #1515).

THE QUESTION.  The day-130+ thermal runaway warms the interior at roughly
0.17 K/day, which is about 20 W/m^2 of column imbalance.  Two explanations:

  * energy is created INTERNALLY -- heating applied without the matching flux,
    a physics bookkeeping bug -- in which case the budget does not close; or
  * the fluxes themselves are biased, in which case the budget DOES close and
    the search moves to which flux is wrong.

One number decides it, and this computes it.

SIGN CONVENTION, stated because the whole answer is a sign (CLAUDE.md gate).
Everything below is energy INTO THE ATMOSPHERIC COLUMN, positive:

  * ``toa_net``  = sw_down_toa - sw_up_toa - lw_up_toa, positive DOWN
    (``energy_budget.py``), so it enters the column as ``+toa_net``.
  * ``sw_net_sfc`` / ``lw_net_sfc`` are documented ``+into surface``
    (``radiation/integration.py:504``).  Radiation absorbed by the SURFACE
    LEAVES the column, so it enters as ``-(sw_net_sfc + lw_net_sfc)``.
  * ``hfss`` / ``hfls`` are positive UP, from surface into the column, so they
    enter as ``+hfss + hfls``.  ``E`` is column MOIST static energy, so the
    latent flux belongs in it.

  dE/dt = toa_net - (sw_net_sfc + lw_net_sfc) + hfss + hfls + LEAK

The tracker also reports ``residual = toa_net - dE/dt`` directly.  Substituting
dE/dt = toa_net - residual and cancelling ``toa_net``:

  LEAK = (sw_net_sfc + lw_net_sfc) - hfss - hfls - residual

NB the #1354 run card stated this as ``residual - sfc_net_rad_up - SH - LH``,
which has the radiation term the wrong way round; the derivation above is the
one used here and the card is corrected to match.

READING (pre-registered):
  LEAK > 0        => energy created internally (LEAK enters dE/dt as a source).
                     A physics bookkeeping bug, and very likely #1515 too.
  LEAK ~ 0        => the budget CLOSES.  The warming is then either a real
                     boundary-flux error, or MSE COMPENSATION -- the column
                     warming while it dries, so that moist static energy barely
                     moves while T rises.

THE FOLLOW-UP, computed here so one run answers both (GLM review).  If the
budget closes, splitting dE/dt into its THERMAL and LATENT parts separates
those two:

  latent part  ~  L_v * d(CWV)/dt
  thermal part ~  dE/dt - latent part

A latent tendency near -20 W/m^2 while the thermal part is +20 means
compensation; a latent tendency near zero means the energy arrived through a
boundary flux.  #1354 reports CWV flat at 23-26 kg/m^2 while T rises, which
predicts the SECOND -- so this is a real test of that report, not a formality.

A PREDICTION, WRITTEN BEFORE THE INTERVAL-MEAN RUN LANDED (2026-09-05), so it
cannot be fitted afterwards.  The column energy is

    E = int (c_p*T + L_v*q_v - L_f*q_frozen + Phi + KE) dp/g

and the flux list above contains NO precipitation term.  So when frozen
condensate PRECIPITATES out, q_frozen falls, the ``-L_f*q_frozen`` term rises,
and E rises with nothing on the flux side to match it.  That reads as a
spurious energy SOURCE of

    L_f * (frozen precipitation rate)  =  3.87 W/m^2 per mm/day

(L_f from ``legoesm.constants``; 1 mm/day = 1/86400 kg/m^2/s).  Sign: dE/dt too HIGH by X
makes ``residual = toa_net - dE/dt`` too low by X, hence LEAK too HIGH by X --
a POSITIVE apparent leak, which is the sign observed.

The ``L_f*precip`` column below is the UPPER BOUND on that term (every
millimetre frozen); the real value is that times the frozen fraction.  If the
interval-mean leak lands near it, the residual is this definitional gap and not
a physics bookkeeping bug -- the fix is then to add the frozen-precipitation
energy flux to the budget, not to hunt the physics.  If the leak is much LARGER
than the bound, the gap cannot explain it and the physics hunt is back on.

AND WHAT THIS CANNOT SEE (also GLM): moist-static-energy closure is blind to a
precipitation MASS-flux inconsistency that does not also mis-state the latent
heating.  The water budget is a separate question, so the run's own
``moisture_residual`` (E - P - dW/dt) is printed alongside rather than left for
a second probe.

The MPAS lane writes its own ``timeseries.npz`` at the END of the run (it
short-circuits the unified collector), so there are no incremental
``energy_chunk_*.npz`` files on this path -- looking for them and reporting
their absence as "the tracker did not fire" is wrong, and this script reads the
right artifact.

DO NOT QUOTE A NUMBER FROM THIS WITHOUT RUNNING THE CROSS-CHECK.
``scripts/validate/mpas_energy_sampling_crosscheck.py`` compares these
channels against the run's own CMOR time-accumulated means.  The tracker is
fed ``ModelDriver._mpas_sfc_accum`` interval means (slots 0-7) when the CMOR
feed is on and stamps ``energy_flux_interval_mean``; ``assert_interval_means``
below refuses a snapshot-sourced series.

Prints numbers only.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from legoesm import constants

_NEEDED = ("energy_toa_net", "energy_dE_dt", "energy_residual",
           "sw_net_sfc", "lw_net_sfc", "hfss", "hfls")
# Present on the MPAS lightweight series; used for the thermal/latent split and
# the water budget. Absent -> those columns are simply not printed.
_OPTIONAL = ("CWV", "moisture_residual", "days",
             "energy_flux_interval_mean",
             # `precip` feeds the frozen-precipitation energy bound. Omitting
             # it here silently disabled that whole block -- the same
             # collected-but-not-delivered failure codex found for the
             # interval-mean flag, caught this time by its own test.
             "precip")


def load(path: pathlib.Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    missing = [k for k in _NEEDED if k not in d]
    if missing:
        raise SystemExit(
            f"{path} lacks {missing}; present: {sorted(d.files)[:20]}. The "
            "energy channels are only written by the MPAS/spectral lightweight "
            "series -- check the run actually reached its first diagnostic.")
    out = {k: np.asarray(d[k], dtype=float) for k in _NEEDED}
    for k in _OPTIONAL:
        if k in d:
            out[k] = np.asarray(d[k], dtype=float)

    # ALIGNMENT GUARD.  In the MPAS lane `days` is appended unconditionally at
    # each diagnostic step, while the energy channels are appended inside a
    # further `if` (the energy-budget inputs must all be present).  If that
    # condition is ever false on a diagnostic step, `days` grows and the energy
    # channels do not -- and every subsequent sample is paired with the WRONG
    # day, silently.  A leak computed across a shift is meaningless, so refuse
    # rather than report.
    lengths = {k: len(v) for k, v in out.items()}
    n = lengths[_NEEDED[0]]
    ragged = {k: L for k, L in lengths.items() if L != n}
    if ragged:
        raise SystemExit(
            f"channel lengths disagree in {path}: {lengths}. The energy "
            "channels are appended under a condition that `days` is not, so a "
            "mismatch means the samples are SHIFTED relative to each other and "
            "any leak computed from them pairs the wrong day's fluxes. Refusing "
            "to report a number.")
    return out


def assert_interval_means(t, *, allow_snapshots: bool = False) -> str:
    """Refuse a leak computed from end-of-interval SNAPSHOTS.

    The driver stamps ``energy_flux_interval_mean`` = 1 when the seven energy
    channels are diagnostic-interval means and 0 when they are snapshots.
    Snapshots alias the diurnal cycle of the land-dominated turbulent fluxes:
    measured, the sampled sensible heat flux was 8.0 W/m^2 against an
    accumulated 20.5, and the apparent leak was +34.7 W/m^2 where accumulated
    channels gave ~11.  That is a plausible wrong answer, so it is REFUSED
    rather than annotated -- an annotation gets dropped when the number is
    quoted onward.

    Returns a one-line provenance string for the report header.
    """
    flag = t.get("energy_flux_interval_mean")
    if flag is None:
        msg = ("this series predates the interval-mean flag, so its flux "
               "timing is UNKNOWN (almost certainly snapshots)")
    elif float(np.min(flag)) >= 1.0:
        return "flux timing: diagnostic-INTERVAL MEANS on every sample"
    else:
        n_snap = int((np.asarray(flag) < 1.0).sum())
        msg = (f"{n_snap} of {flag.size} samples are end-of-interval "
               "SNAPSHOTS, not interval means")
    if allow_snapshots:
        return f"flux timing: CONTAMINATED -- {msg} (--allow-snapshots)"
    raise SystemExit(
        f"REFUSING to report a leak: {msg}. Snapshots alias the diurnal cycle "
        "of the land-dominated turbulent fluxes and produce a plausible wrong "
        "answer (#1354/#1353). Re-run with the CMOR feed on so the flux "
        "accumulator runs, or pass --allow-snapshots to see the contaminated "
        "number knowing that is what it is.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("timeseries", type=pathlib.Path,
                   help="timeseries.npz from the run")
    p.add_argument("--skip-days", type=float, default=1.0,
                   help="drop samples at or before this day. The FIRST sample "
                        "is dropped unconditionally regardless: the tracker "
                        "sets its dE_dt (and hence residual) to zero there by "
                        "construction, so its 'leak' is just "
                        "sfc_rad - hfss - hfls and is meaningless.")
    p.add_argument("--allow-snapshots", action="store_true",
                   help="report even when the channels are snapshots rather "
                        "than interval means; the number is then contaminated "
                        "by diurnal aliasing and is labelled as such.")
    args = p.parse_args(argv)

    t = load(args.timeseries)
    provenance = assert_interval_means(
        t, allow_snapshots=args.allow_snapshots)
    days = t.get("days", np.arange(len(t["hfss"]), dtype=float))
    # STRICTLY greater, and index 0 dropped unconditionally: the tracker's
    # first sample has dE_dt == 0 by construction (energy_budget.py: there is
    # no previous energy to difference against), so its residual equals
    # toa_net and the "leak" computed from it is a pure artefact.  Using
    # ">= skip_days" kept exactly that sample when the diagnostic interval is
    # one day and skip_days is 1 (codex review, HIGH).
    keep = days > args.skip_days
    keep[0] = False
    if not keep.any():
        raise SystemExit(
            f"no samples after day {args.skip_days} (excluding the first, "
            "whose dE_dt is zero by construction)")

    sfc_rad = t["sw_net_sfc"] + t["lw_net_sfc"]
    leak = sfc_rad - t["hfss"] - t["hfls"] - t["energy_residual"]

    print(provenance)
    print(f"{args.timeseries}  ({int(keep.sum())} samples after day "
          f"{args.skip_days:g} of {len(days)}; first sample always dropped, "
          "dE_dt is zero there by construction)")
    hdr = (f"{'day':>7s} {'toa_net':>9s} {'dE/dt':>9s} {'residual':>9s} "
           f"{'sfc_rad':>9s} {'hfss':>8s} {'hfls':>8s} {'LEAK':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for i in np.flatnonzero(keep):
        print(f"{days[i]:7.2f} {t['energy_toa_net'][i]:9.3f} "
              f"{t['energy_dE_dt'][i]:9.3f} {t['energy_residual'][i]:9.3f} "
              f"{sfc_rad[i]:9.3f} {t['hfss'][i]:8.3f} {t['hfls'][i]:8.3f} "
              f"{leak[i]:9.3f}")
    # Thermal / latent split of dE/dt (GLM): distinguishes "energy arrived
    # through a boundary flux" from "the column warmed while it dried".
    # Frozen-precipitation energy bound (see the prediction in the docstring).
    if "precip" in t and len(t["precip"]) == len(days):
        lf_precip = constants.L_f * t["precip"] / 86400.0   # mm/day -> W/m^2
        print()
        print("frozen-precipitation energy gap, UPPER BOUND (all precip "
              "frozen): L_f * precip")
        print(f"{'day':>7s} {'precip[mm/d]':>13s} {'L_f*precip':>11s} "
              f"{'LEAK':>9s} {'LEAK - bound':>13s}")
        for i in np.flatnonzero(keep):
            print(f"{days[i]:7.2f} {t['precip'][i]:13.3f} {lf_precip[i]:11.3f} "
                  f"{leak[i]:9.3f} {leak[i] - lf_precip[i]:13.3f}")
        _b, _l = lf_precip[keep], leak[keep]
        _f = np.isfinite(_b) & np.isfinite(_l)
        if _f.any():
            print(f"  means: bound {_b[_f].mean():.3f}  leak {_l[_f].mean():.3f}"
                  f"  leak-bound {(_l[_f] - _b[_f]).mean():.3f} W/m^2")

    if "CWV" in t and len(t["CWV"]) == len(days):
        dt_s = np.gradient(days) * 86400.0
        d_cwv = np.gradient(t["CWV"])
        latent = constants.L_v * d_cwv / np.maximum(dt_s, 1e-30)
        thermal = t["energy_dE_dt"] - latent
        print()
        print("thermal / latent split of dE/dt [W/m^2] "
              "(latent = L_v * d(CWV)/dt):")
        print(f"{'day':>7s} {'dE/dt':>9s} {'latent':>9s} {'thermal':>9s} "
              f"{'CWV':>8s} {'moist_res':>10s}")
        for i in np.flatnonzero(keep):
            mr = (t["moisture_residual"][i]
                  if "moisture_residual" in t else float("nan"))
            print(f"{days[i]:7.2f} {t['energy_dE_dt'][i]:9.3f} "
                  f"{latent[i]:9.3f} {thermal[i]:9.3f} "
                  f"{t['CWV'][i]:8.3f} {mr:10.3f}")

    L = leak[keep]
    finite = np.isfinite(L)
    print()
    print(f"LEAK over the kept window [W/m^2]: mean={L[finite].mean():.3f} "
          f"median={np.median(L[finite]):.3f} sd={L[finite].std():.3f} "
          f"min={L[finite].min():.3f} max={L[finite].max():.3f} "
          f"(non-finite {int((~finite).sum())}/{finite.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
