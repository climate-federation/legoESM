#!/usr/bin/env python
"""Does the SCM-RCE column close its water budget, and where does the water go?

THE OBSERVATION.  Every convection scheme in the 2026-08-10 campaign reports a
surface precipitation of 1e-18 to 3e-5 mm/day against the CRM reference's
2.395, while the column sits in a steady state.  Those two facts cannot both
be innocent: over a 300 K ocean with a 5 m/s wind and near-surface air at
~88 % RH, the bulk surface flux (``Ch_neutral`` = 1.5e-3, ``bulk_scheme`` =
"constant") should evaporate of order 3 mm/day.  Steady state then requires a
sink of the same size, and the campaign says the sink is ~0.

THE DISCRIMINATOR.  For a column, ``d(CWV + CWC)/dt = E - P``.  Two runs of the
PRODUCTION entry point (``run_scm_rce``) at different lengths give the storage
term by difference, and each run reports its own P.  Then

    E_implied = d(CWV + CWC)/dt + P

* ``E_implied ~ 0``  => the surface moisture flux is not reaching the column,
  i.e. the column is closed and P = 0 is CONSISTENT rather than a diagnostic
  failure.  The defect would be in the SCM's surface coupling.
* ``E_implied ~ 3 mm/day`` => evaporation is live and something removes that
  water WITHOUT being counted as precipitation.  The defect would be in the
  precipitation diagnostic, or in a clamp/limiter silently destroying water.

Nothing here re-derives the column: it calls the campaign's own
``run_scm_rce`` so the answer applies to the configuration the arms ran, not
to a reconstruction of it.

Run (CPU, ~6 min for the default pair)::

    JAX_ENABLE_X64=1 python scripts/validate/scm_rce_water_budget.py \\
        --reference-dir results/rcemip_ref_sam300
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from legoesm import constants

from scripts.run import run_scm_rce_campaign as camp

MM_PER_M = 1_000.0
RHO_W = 1_000.0          # kg/m^3, for kg/m^2 -> mm of liquid water


def column_water(ref, profile) -> float:
    """Column integral of a mixing-ratio profile [kg/m^2 == mm of water].

    ``ref.mass_weights`` are the normalised mass weights the campaign scores
    with; the column mass is ``p_sfc / g`` for a full column, so the integral
    is ``sum(w * q) * p_sfc / g``.  Uses the campaign's OWN weights rather than
    a second vertical discretisation.
    """
    w = np.asarray(ref.mass_weights, dtype=float)
    q = np.asarray(profile, dtype=float)
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import WING_P_SFC
    return float(np.sum(w * q) * WING_P_SFC / constants.g)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", type=Path,
                    default=Path("results/rcemip_ref_sam300"))
    ap.add_argument("--last-reference-files", type=int, default=5)
    ap.add_argument("--convection", default="sbm")
    ap.add_argument("--microphysics", default="morrison")
    ap.add_argument("--days-short", type=float, default=20.0)
    ap.add_argument("--days-long", type=float, default=25.0)
    ap.add_argument("--dt", type=float, default=camp.DEFAULT_DT_S)
    ap.add_argument("--analysis-days", type=float, default=2.0)
    ap.add_argument("--hard-saturation-adjustment", action="store_true",
                    default=True)
    args = ap.parse_args(argv)

    ref = camp.build_reference_profiles(
        args.reference_dir, args.last_reference_files,
        precip_analysis_days=args.analysis_days,
    )
    cfg = camp.make_physics_config(
        radiation="rrtmgp",
        radiation_update_interval_steps=camp.DEFAULT_RRTMGP_UPDATE_INTERVAL_STEPS,
        convection=args.convection,
        microphysics=args.microphysics,
        hard_saturation_adjustment=args.hard_saturation_adjustment,
    )
    cfg, arm = camp.apply_subsidence_solve_override(
        cfg, "implicit_flux", category="convection")

    print("=" * 78)
    print("SCM-RCE water budget:  d(CWV + CWC)/dt = E - P")
    print("=" * 78)
    print(f"  convection={args.convection}  microphysics={args.microphysics}  "
          f"kernel={arm}")
    print(f"  reference={args.reference_dir}  CRM precip="
          f"{ref.precip_ref_mm_day:.4g} mm/day")

    runs = {}
    for label, days in (("short", args.days_short), ("long", args.days_long)):
        run = camp.run_scm_rce(
            cfg, ref, label=f"budget:{label}", days=days, dt=args.dt,
            analysis_days=args.analysis_days,
            require_equilibrium=False, require_realism=False,
            equil_T_tol_K=camp.EQUIL_T_TOL_K, equil_qv_tol=camp.EQUIL_QV_TOL,
            equil_qcond_tol=camp.EQUIL_QCOND_TOL,
        )
        cwv = column_water(ref, run.qv_profile)
        cwc = column_water(ref, run.qcond_profile)
        runs[label] = (days, run, cwv, cwc)
        print(f"  [{label:5s}] days={days:6.1f}  status={run.status:8s}  "
              f"CWV={cwv:8.3f} mm  CWC={cwc:8.4f} mm  "
              f"P={run.precip_mm_day:.6e} mm/day")

    (d0, r0, cwv0, cwc0) = runs["short"]
    (d1, r1, cwv1, cwc1) = runs["long"]
    if d1 <= d0:
        raise SystemExit("--days-long must exceed --days-short")
    dstore = ((cwv1 + cwc1) - (cwv0 + cwc0)) / (d1 - d0)
    p_mean = 0.5 * (r0.precip_mm_day + r1.precip_mm_day)
    e_implied = dstore + p_mean

    print("-" * 78)
    print(f"  storage  d(CWV+CWC)/dt = {dstore:+.4f} mm/day "
          f"(over days {d0:.0f} -> {d1:.0f})")
    print(f"  reported P            = {p_mean:.6e} mm/day")
    print(f"  IMPLIED E = dS/dt + P = {e_implied:+.4f} mm/day")
    print(f"  CRM reference P       = {ref.precip_ref_mm_day:.4f} mm/day")
    print("-" * 78)
    print("  Reading (the two branches this probe exists to separate):")
    print("    |E| << 1 mm/day   -> the column is CLOSED; no surface moisture")
    print("                         flux reaches it, and P = 0 follows.")
    print("    E ~ CRM P         -> evaporation is live and its water leaves")
    print("                         WITHOUT being counted as precipitation.")
    print("  This probe does not choose between them; it prints E.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
