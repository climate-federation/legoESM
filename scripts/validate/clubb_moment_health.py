"""Is CLUBB's forward solution diverging, or only its gradient?

WHY. Our CLUBB is the WORST of nine closures on Wangara (normalized profile
error 15.2 against 0.501 for the best) and carries a parameter gradient of
1.3e16 there, while being the BEST of the nine on BOMEX (0.075) with a
perfectly ordinary gradient of 0.28. Ruled out by measurement already: rollout
length (the LONGEST case has the SMALLEST gradient), the dry configuration
(equally dry CBL is clean), the score normaliser, and dead parameters.

What correlates is convective vigour, and the dominant coefficients are all
wp2/wp3/skewness terms. Two independent reviews landed on the same reading: a
bound that keeps the wp3-skewness branch finite in real CLUBB is missing or
softened here, the FORWARD solution diverges, and the adjoint merely follows
it. That is a claim about the trajectory, so measure the trajectory.

WHAT IT PRINTS, per step: max|wp3|, min wp2, and max|Skw| where
``Skw = wp3 / wp2^{3/2}`` is the skewness the PDF closure is built on. Real
CLUBB holds |Skw| inside single digits; a departure to tens or beyond, and the
step it happens on, localises the defect in TIME -- for Wangara the suspicion
is the morning transition, when the surface heat flux crosses from the
nocturnal value up through its 13:00 peak.

Usage:
  python scripts/validate/clubb_moment_health.py --case wangara --every 30
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def _load_driver():
    path = _ROOT / "scripts" / "run" / "run_scm_les_turbulence_tuning.py"
    spec = importlib.util.spec_from_file_location("_scm_tuning", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_scm_tuning"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="wangara")
    p.add_argument("--nlev", type=int, default=48)
    p.add_argument("--hours", type=float, default=None,
                   help="default: the case's own length")
    p.add_argument("--every", type=int, default=30,
                   help="print every Nth step")
    args = p.parse_args(argv)

    drv = _load_driver()
    case = drv.load_case(args.case, nlev=args.nlev, dt=None)
    surface = drv.build_surface_config(case, flux_to_closure=True)
    prescribed = case.forcing.prescribe == "fluxes"
    if drv.surface_flux_varies_in_time(case):
        import dataclasses
        case = dataclasses.replace(
            case, forcing=case.forcing._replace(flux_to_closure=True))
    cfg = drv.build_physics_config(
        "clubb", prescribed_fluxes=prescribed, microphysics="none",
        simple_lw=args.case in drv._SIMPLE_LW_CASES,
        bulk_ch=case.spec.bulk_ch, bulk_ce=case.spec.bulk_ce,
        surface=surface, clubb_prognostic=True,
    )
    scm = drv._create_scm(cfg, case, case.dt)
    from legoesm.atmosphere.physics.turbulence.clubb import (
        unpack_clubb_moments,
    )

    hours = args.hours if args.hours is not None else _case_hours(drv, case)
    nsteps = max(1, int(round(hours * 3600.0 / case.dt)))
    print(f"clubb on {args.case}: {nsteps} steps of {case.dt} s "
          f"({hours:.2f} h), nlev={args.nlev}")
    print(f"{'step':>6} {'hour':>6} {'w_T_sfc':>10} {'min wp2':>11} "
          f"{'max|wp3|':>11} {'max|Skw|':>11} {'max|thl|':>9}")

    step_fn, tend_fn = scm._step_fn, scm._tend_fn
    state, phys = scm.state, scm.phys_state
    # NOT jitted and NOT scanned: this is a per-step host-side inspection, and
    # the point is to see the step the trajectory leaves physical bounds. The
    # cost is one dispatch per step, which for a few thousand steps is minutes.
    worst = (0.0, -1)
    for k in range(nsteps):
        t = k * case.dt
        state, phys = step_fn(state, phys, tend_fn, case.dt, t)
        if k % args.every and k != nsteps - 1:
            continue
        m = unpack_clubb_moments(phys.clubb_moments)
        wp2 = np.asarray(m.wp2)
        wp3 = np.asarray(m.wp3)
        # Skw on the zt levels wp3 lives on; wp2 sits on zm, so use the
        # overlapping first nlev entries -- this is a HEALTH indicator, not the
        # scheme's own Skw, which interpolates. An order-of-magnitude readout
        # does not need the interpolation and inventing one would be a second
        # thing to get wrong.
        n = wp3.shape[1]
        skw = wp3 / np.maximum(wp2[:, :n], 1e-12) ** 1.5
        s_max = float(np.nanmax(np.abs(skw)))
        if s_max > worst[0]:
            worst = (s_max, k)
        wth = float(case.forcing.w_th_s(t)) if case.forcing.w_th_s else 0.0
        print(f"{k:6d} {t/3600.0:6.2f} {wth:10.5f} "
              f"{float(np.nanmin(wp2)):11.3e} {float(np.nanmax(np.abs(wp3))):11.3e} "
              f"{s_max:11.3e} "
              f"{float(np.nanmax(np.abs(np.asarray(m.thlm)))):9.2f}")
        if not np.isfinite(s_max):
            print(f"  NON-FINITE at step {k}; stopping")
            break

    print(f"\nworst |Skw| = {worst[0]:.3e} at step {worst[1]} "
          f"({worst[1] * case.dt / 3600.0:.2f} h)")
    print("READ IT AS: real CLUBB holds |Skw| in single digits. Tens or more, "
          "and the step it starts, is a diverging FORWARD solution -- the "
          "gradient is then following the trajectory, not causing it.")
    return 0


def _case_hours(drv, case) -> float:
    spec = getattr(case, "spec", None)
    for attr in ("hours", "duration_h"):
        v = getattr(spec, attr, None)
        if v:
            return float(v)
    return 8.0 if case.name == "wangara" else 6.0


if __name__ == "__main__":
    raise SystemExit(main())
