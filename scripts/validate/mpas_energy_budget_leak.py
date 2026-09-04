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

Prints numbers only.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

_NEEDED = ("energy_toa_net", "energy_dE_dt", "energy_residual",
           "sw_net_sfc", "lw_net_sfc", "hfss", "hfls")
# Present on the MPAS lightweight series; used for the thermal/latent split and
# the water budget. Absent -> those columns are simply not printed.
_OPTIONAL = ("CWV", "moisture_residual", "T_atm", "days")


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
    return out


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
    args = p.parse_args(argv)

    t = load(args.timeseries)
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

    print(f"{args.timeseries}  ({int(keep.sum())} samples after day "
          f"{args.skip_days:g} of {len(days)}; first sample always dropped, "
          f"dE_dt is zero there by construction)")
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
    if "CWV" in t and len(t["CWV"]) == len(days):
        from legoesm import constants
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
