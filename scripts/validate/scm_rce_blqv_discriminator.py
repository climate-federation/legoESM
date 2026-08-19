#!/usr/bin/env python
"""One-variable arms that discriminate the boundary-layer humidity bias cause.

MEASURED: +5.1 g/kg surface q_v bias common to all ten convection schemes,
identical before and after tuning — the cause is in the SHARED path.  The code
review localised three candidates, and each arm here flips exactly ONE of them
against an identical control, on one representative mass-flux scheme:

* ``control``      — the campaign configuration exactly (louis, bulk "constant",
                     cloudtop_entrainment_efficiency 0.0, wind 5 m/s).
* ``entrainment``  — ``cloudtop_entrainment_efficiency`` 0.0 -> 0.2: the one
                     ventilation term Louis has, which ships disabled.
* ``coare3``       — ``bulk_scheme`` "constant" -> "coare3": stability-dependent
                     exchange plus the native convective-gustiness w* term that
                     is structurally inert under "constant".
* ``hb``           — turbulence scheme louis -> holtslag_boville: the NONLOCAL
                     w*-scaled K-profile with countergradient transport, i.e.
                     the second reviewer's predicted fix.  It tests the
                     shear-only-K hypothesis directly: if the subcloud bias is
                     flux-through-collapsed-K, a K-profile scheme removes it
                     without touching the flux law.
                     MEASURED 2026-08-17: +6.27 vs control +6.18 — NO EFFECT.
                     Self-consistent: w* transport is driven by surface
                     buoyancy flux, which the locked warm-moist state has
                     destroyed; a buoyancy-driven closure cannot break a
                     lock-in that suppresses buoyancy.
* ``downdraft``    — emanuel with its PORTED unsaturated downdraft ENABLED
                     (enable_unsaturated_downdraft, ships OFF): the one
                     mechanism that imports low-theta_e air to the surface
                     layer INDEPENDENT of local surface buoyancy, driven by
                     rain evaporation aloft — i.e. the cold-pool surrogate.
                     This is the surviving candidate after the first four arms
                     all failed to move the surface state, and it is the only
                     one that explains the adjustment-vs-mass-flux bimodality.
                     Necessarily run on emanuel (its scheme owns the port), so
                     its control is the emanuel arm, not mass_flux.

Interpretation, pre-registered:
* entrainment fixes the 0-2 km humidity while E moves little  -> missing
  VENTILATION is dominant;
* coare3 fixes E and the surface state while the humidity aloft persists ->
  the FLUX LAW was dominant;
* neither moves it -> the shear-only K_h collapse under a uniform wind is the
  remaining candidate and needs its own arm (a K floor tied to u*).

Uses the campaign's own ``run_scm_rce`` so the protocol is byte-identical to
the arms being explained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax.numpy as jnp  # noqa: E402

from scripts.run import run_scm_rce_campaign as camp  # noqa: E402

REPORT_Z_KM = (0.5, 1.0, 2.0)


def _with_louis_override(cfg, **kw):
    louis = cfg.turbulence.louis
    surface_kw = {k: v for k, v in kw.items() if k in ("bulk_scheme",)}
    louis_kw = {k: v for k, v in kw.items() if k not in surface_kw}
    if surface_kw:
        louis = louis._replace(surface=louis.surface._replace(**surface_kw))
    if louis_kw:
        louis = louis._replace(**louis_kw)
    return cfg._replace(turbulence=cfg.turbulence._replace(louis=louis))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", default="mass_flux",
                        help="representative mass-flux scheme (the biased class)")
    parser.add_argument("--days", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--analysis-days", type=float, default=5.0)
    parser.add_argument(
        "--reference-dir", type=Path,
        default=Path("/burg-archive/glab/users/pg2328/legoESM/results"
                     "/rcemip_ref_sam300"))
    parser.add_argument("--entrainment-efficiency", type=float, default=0.2)
    parser.add_argument("--arms", default="control,entrainment,coare3,hb")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if "rcemip1_n128" in str(args.reference_dir):
        raise SystemExit("REFUSED: that is our own CRM run, not an oracle.")
    ref = camp.build_reference_profiles(args.reference_dir, 5)
    z_km = np.asarray(ref.z_m) / 1000.0
    qv_ref = np.asarray(ref.qv_ref) * 1000.0
    T_ref = np.asarray(ref.T_ref)
    idx = {zk: int(np.argmin(np.abs(z_km - zk))) for zk in REPORT_Z_KM}

    base = camp.make_physics_config(
        radiation="rrtmgp", convection=args.scheme, turbulence="louis",
        microphysics="morrison", hard_saturation_adjustment=True)
    base, _st = camp.apply_subsidence_solve_override(
        base, "implicit_flux", category="convection")

    # Sanity: print what the CONTROL actually resolves, so the "one variable"
    # claim is verifiable from the log rather than asserted.
    louis = base.turbulence.louis
    print(f"control resolves: bulk_scheme={louis.surface.bulk_scheme!r}  "
          f"cloudtop_entrainment_efficiency="
          f"{louis.cloudtop_entrainment_efficiency!r}")

    dd_base = camp.make_physics_config(
        radiation="rrtmgp", convection="emanuel", turbulence="louis",
        microphysics="morrison", hard_saturation_adjustment=True)
    dd_base, _st3 = camp.apply_subsidence_solve_override(
        dd_base, "implicit_flux", category="convection")
    _c, _s2, dd_sub = camp._active_subconfig(dd_base, "convection")
    dd_cfg = camp._set_active_subconfig(
        dd_base, "convection",
        dd_sub._replace(enable_unsaturated_downdraft=True))

    hb_base = camp.make_physics_config(
        radiation="rrtmgp", convection=args.scheme,
        turbulence="holtslag_boville", microphysics="morrison",
        hard_saturation_adjustment=True)
    hb_base, _st2 = camp.apply_subsidence_solve_override(
        hb_base, "implicit_flux", category="convection")
    # Each arm is (config, surface_wind_m_s or None for the default 5.0).
    # The wind arms answer the boundary-condition question directly: SAM's
    # surface fluxes are driven by resolved gusts (no imposed mean wind), and
    # our fixed 5 m/s is the surrogate — wind2/wind8 bracket it so the
    # sensitivity of the equilibrium surface state to that choice is MEASURED.
    arm_cfgs = {
        "control": (base, {}),
        "entrainment": (_with_louis_override(
            base, cloudtop_entrainment_efficiency=args.entrainment_efficiency),
            {}),
        "coare3": (_with_louis_override(base, bulk_scheme="coare3"), {}),
        "hb": (hb_base, {}),
        # The downdraft pair: emanuel without and with the ported shaft.
        "emanuel_ctl": (dd_base, {}),
        "downdraft": (dd_cfg, {}),
        "wind2": (base, {"surface_wind_m_s": 2.0}),
        "wind8": (base, {"surface_wind_m_s": 8.0}),
        # Review-driven arms (codex adversarial pass on the verdict):
        # f0: RCEMIP is NONROTATING with a freely evolving wind; the control
        # pins the column to a 5 m/s geostrophic target through f=2.5e-5.
        # coriolis_s_inv=0 disables the whole Coriolis+geostrophic block
        # (scm_forcing.py guards on f_c != 0), giving the protocol-relevant
        # dynamical BC in one variable.
        "f0": (base, {"coriolis_s_inv": 0.0}),
        # nosatadj: hard saturation adjustment is ON in the control and is a
        # nonlinear regulator of a 93%-RH surface layer that had no arm.
        "nosatadj": (camp.apply_subsidence_solve_override(
            camp.make_physics_config(
                radiation="rrtmgp", convection="mass_flux",
                turbulence="louis", microphysics="morrison",
                hard_saturation_adjustment=False),
            "implicit_flux", category="convection")[0], {}),
        # subs: large-scale subsidence ships OFF ("none"); the CRM-derived
        # clear-sky subsidence is the one-variable alternative.
        "subs": (base, {"large_scale_forcing": "crm_clear_sky_subsidence"}),
        # f0_coare3: the corrected-protocol candidate. f0 alone fixed the
        # surface state (dqv +6.20 -> +1.37) but the wind spins down and E
        # collapses to 0.70 vs the CRM's 2.73 - SAM carries its fluxes on
        # convective gusts. COARE3's w* gustiness supplies exactly that flux
        # wind independent of the mean wind. PRE-REGISTERED: if E recovers
        # substantially while the surface stays near the f0 state, the pair
        # (f=0, coare3 gustiness) is the defensible SCM-RCEMIP configuration
        # and the campaign should be re-run on it; if E stays collapsed, the
        # gust source must come from convection itself (cold pools) and no
        # bulk-scheme fix suffices.
        "f0_coare3": (_with_louis_override(base, bulk_scheme="coare3"),
                      {"coriolis_s_inv": 0.0}),
    }

    results = {}
    cache: dict = {}
    # "+" is accepted as a separator alongside ",": sbatch --export splits its
    # OWN list on commas, so a comma-separated ARMS value silently truncates to
    # its first element (measured: a two-arm submission ran exactly one arm).
    _arm_list = [a.strip() for a in
                 args.arms.replace("+", ",").split(",") if a.strip()]
    unknown = [a for a in _arm_list if a not in arm_cfgs]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known={sorted(arm_cfgs)}")
    for arm in _arm_list:
        cfg, extra_kw = arm_cfgs[arm]
        run = camp.run_cached(
            cache, cfg, ref, label=f"blqv:{arm}", days=args.days, dt=args.dt,
            **extra_kw,
            analysis_days=args.analysis_days, require_equilibrium=False,
            require_realism=False, equil_T_tol_K=camp.EQUIL_T_TOL_K,
            equil_qv_tol=camp.EQUIL_QV_TOL,
            equil_qcond_tol=camp.EQUIL_QCOND_TOL,
            scm_microphysics_substeps=30, scm_convection_substeps=10,
            bl_anchor_top_m=-1.0)
        if not run.T_profile:
            print(f"{arm}: FAILED ({run.reason[:120]})")
            results[arm] = {"status": run.status, "reason": run.reason}
            continue
        qv = np.asarray(run.qv_profile) * 1000.0
        T = np.asarray(run.T_profile)
        rec = {
            "status": run.status,
            "dqv_sfc": float(qv[-1] - qv_ref[-1]),
            **{f"dqv_{zk:g}km": float(qv[idx[zk]] - qv_ref[idx[zk]])
               for zk in REPORT_Z_KM},
            "dT_sfc": float(T[-1] - T_ref[-1]),
            "sfc_rh": run.sfc_relative_humidity,
            "sfc_dT": run.sfc_delta_T_K,
            "evap": run.evap_mm_day,
            "precip": run.precip_mm_day,
            "thermo": run.thermo_score,
            # Equilibrium evidence (review finding: the probe discarded it).
            "drift_T_rmse_K": run.drift_T_rmse_K,
            "drift_qv_rmse": run.drift_qv_rmse,
        }
        results[arm] = rec
        print(f"{arm:12s} dqv sfc {rec['dqv_sfc']:+6.2f}  "
              + " ".join(f"{zk:g}km {rec[f'dqv_{zk:g}km']:+6.2f}"
                         for zk in REPORT_Z_KM)
              + f"  dT_sfc {rec['dT_sfc']:+5.2f}  RH {rec['sfc_rh']:.3f}  "
              f"SST-Ta {rec['sfc_dT']:+5.2f}  E {rec['evap']:.2f}  "
              f"P {rec['precip']:.2f}  thermo {rec['thermo']:.3f}")

    print(f"\nCRM: RH 0.752, SST-Ta 3.06 K, E {ref.evap_ref_mm_day:.2f} "
          f"mm/day ({ref.evap_ref_note}); biases should -> 0.")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"scheme": args.scheme, "days": args.days, "arms": results},
            indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
