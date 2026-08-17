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
    parser.add_argument("--arms", default="control,entrainment,coare3")
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

    arm_cfgs = {
        "control": base,
        "entrainment": _with_louis_override(
            base, cloudtop_entrainment_efficiency=args.entrainment_efficiency),
        "coare3": _with_louis_override(base, bulk_scheme="coare3"),
    }

    results = {}
    cache: dict = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        cfg = arm_cfgs[arm]
        run = camp.run_cached(
            cache, cfg, ref, label=f"blqv:{arm}", days=args.days, dt=args.dt,
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
        }
        results[arm] = rec
        print(f"{arm:12s} dqv sfc {rec['dqv_sfc']:+6.2f}  "
              + " ".join(f"{zk:g}km {rec[f'dqv_{zk:g}km']:+6.2f}"
                         for zk in REPORT_Z_KM)
              + f"  dT_sfc {rec['dT_sfc']:+5.2f}  RH {rec['sfc_rh']:.3f}  "
              f"SST-Ta {rec['sfc_dT']:+5.2f}  E {rec['evap']:.2f}  "
              f"P {rec['precip']:.2f}  thermo {rec['thermo']:.3f}")

    print("\nCRM: RH 0.752, SST-Ta 3.06 K, E 2.73 mm/day; biases should -> 0.")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"scheme": args.scheme, "days": args.days, "arms": results},
            indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
