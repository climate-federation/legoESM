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
SECONDS_PER_DAY_LOCAL = 86_400.0
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
    # BooleanOptionalAction, NOT store_true: with store_true and default=True
    # the flag could never be turned OFF, so the guard could not be A/B'd —
    # and the guard is a candidate water sink, which makes an un-disableable
    # flag exactly the wrong shape for this probe.
    ap.add_argument("--hard-saturation-adjustment",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--surface-wind-m-s", type=float,
                    default=camp.DEFAULT_SCM_RCE_SURFACE_WIND_M_S)
    args = ap.parse_args(argv)
    args_wind = args.surface_wind_m_s

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

    # ---- E measured from the MODEL's own bulk formula, not from arithmetic --
    # The storage identity alone cannot separate "no evaporation" from
    # "evaporation whose water leaves uncounted", because the P it uses is the
    # quantity under suspicion.  Evaluating the shipped surface-flux routine on
    # the equilibrium column the model actually reached breaks that circle.
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import WING_P_SFC
    from legoesm.thermo import saturation_mixing_ratio
    import jax.numpy as jnp

    # The surface-layer config hangs off the ACTIVE turbulence SCHEME, not off
    # TurbulenceConfig itself (which carries only the scheme name and the
    # per-scheme sub-configs).  Assuming the latter cost a run: AttributeError,
    # 'TurbulenceConfig' object has no attribute 'surface' (job 9361582).
    _turb_sub = getattr(cfg.turbulence, cfg.turbulence.scheme, None)
    surf_cfg = getattr(_turb_sub, "surface", None)
    # E must be sampled the SAME WAY as storage and P -- as a window mean, not
    # a single snapshot at the end. With a snapshot, a residual can be blamed
    # on the sampling mismatch and never tested. Both endpoints are available
    # (the two runs), so the window mean is a trapezoid over the same interval
    # the storage difference spans.
    if surf_cfg is None:
        raise SystemExit(
            f"turbulence scheme {cfg.turbulence.scheme!r} exposes no surface- "
            "layer config, so E cannot be measured from the shipped bulk "
            "formula for this configuration.")
    def _E_of(run):
        T_a = float(np.asarray(run.T_profile)[-1])
        q_a = float(np.asarray(run.qv_profile)[-1])
        p_a = float(np.asarray(ref.sigma_full)[-1] * WING_P_SFC)
        rho_a = p_a / (constants.R_d * T_a * (1.0 + 0.608 * q_a))
        q_s = float(saturation_mixing_ratio(
            jnp.asarray(camp.FIXED_SST_K), jnp.asarray(WING_P_SFC)))
        one_ = jnp.ones((1,))
        _a, _b, sh, lh_, _c = compute_surface_fluxes(
            u=one_ * args_wind, v=one_ * 0.0, T=one_ * T_a, q_v=one_ * q_a,
            T_sfc=one_ * camp.FIXED_SST_K, q_sfc=one_ * q_s, rho=one_ * rho_a,
            config=surf_cfg)
        lh_v = float(np.asarray(lh_)[0])
        return (T_a, q_a, float(np.asarray(sh)[0]), lh_v,
                lh_v / constants.L_v * (SECONDS_PER_DAY_LOCAL / RHO_W) * MM_PER_M)

    E_short = _E_of(runs["short"][1])
    E_long = _E_of(runs["long"][1])
    r_long = runs["long"][1]
    T_a = float(np.asarray(r_long.T_profile)[-1])
    q_a = float(np.asarray(r_long.qv_profile)[-1])
    p_a = float(np.asarray(ref.sigma_full)[-1] * WING_P_SFC)
    rho_a = p_a / (constants.R_d * T_a * (1.0 + 0.608 * q_a))
    q_sfc = float(saturation_mixing_ratio(
        jnp.asarray(camp.FIXED_SST_K), jnp.asarray(WING_P_SFC)))
    one = jnp.ones((1,))
    _tx, _ty, shflx, lhflx, _ustar = compute_surface_fluxes(
        u=one * args_wind, v=one * 0.0, T=one * T_a, q_v=one * q_a,
        T_sfc=one * camp.FIXED_SST_K, q_sfc=one * q_sfc, rho=one * rho_a,
        config=surf_cfg,
    )
    lh = float(np.asarray(lhflx)[0])
    e_bulk_mm_day = lh / constants.L_v * (SECONDS_PER_DAY_LOCAL / RHO_W) * MM_PER_M
    print(f"  surface layer on the equilibrium column: T_a={T_a:.3f} K, "
          f"q_a={q_a:.6f}, q_sat(SST)={q_sfc:.6f}, rho={rho_a:.4f} kg/m^3")
    print(f"    bulk_scheme={getattr(surf_cfg, 'bulk_scheme', '?')}  "
          f"Ch_neutral={getattr(surf_cfg, 'Ch_neutral', float('nan')):.3e}")
    print(f"    SHF={float(np.asarray(shflx)[0]):8.3f} W/m^2   "
          f"LHF={lh:8.3f} W/m^2   ->  E(end) = {e_bulk_mm_day:.4f} mm/day")
    e_window = 0.5 * (E_short[4] + E_long[4])
    print(f"    E(start)={E_short[4]:.4f}  E(end)={E_long[4]:.4f}  ->  "
          f"E(window mean) = {e_window:.4f} mm/day")
    e_bulk_mm_day = e_window

    (d0, r0, cwv0, cwc0) = runs["short"]
    (d1, r1, cwv1, cwc1) = runs["long"]
    if d1 <= d0:
        raise SystemExit("--days-long must exceed --days-short")
    dstore = ((cwv1 + cwc1) - (cwv0 + cwc0)) / (d1 - d0)
    p_mean = 0.5 * (r0.precip_mm_day + r1.precip_mm_day)
    e_implied = dstore + p_mean

    # CO-SAMPLED budget: E and P both from the applied tendencies, averaged
    # over the same analysis window inside the driver. This is the version
    # neither reviewer would accept a residual from without believing it.
    e_cos = 0.5 * (r0.evap_mm_day + r1.evap_mm_day)
    p_cos = 0.5 * (r0.precip_mm_day + r1.precip_mm_day)
    print("-" * 78)
    print(f"  CO-SAMPLED (both from the applied tendencies, same window):")
    print(f"    E = {e_cos:+.4f}   P = {p_cos:+.4f}   dS/dt = {dstore:+.4f}"
          f"   residual E-P-dS/dt = {e_cos - p_cos - dstore:+.4f} mm/day")
    print("-" * 78)
    print(f"  storage  d(CWV+CWC)/dt = {dstore:+.4f} mm/day "
          f"(over days {d0:.0f} -> {d1:.0f})")
    print(f"  reported P            = {p_mean:.6e} mm/day")
    print(f"  IMPLIED E = dS/dt + P = {e_implied:+.4f} mm/day  "
          f"(uses the REPORTED P, so it under-reads by any uncounted sink)")
    print(f"  MEASURED E (bulk)     = {e_bulk_mm_day:+.4f} mm/day")
    print(f"  => uncounted sink     = {e_bulk_mm_day - e_implied:+.4f} mm/day")
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
