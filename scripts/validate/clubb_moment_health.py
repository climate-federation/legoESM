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
    p.add_argument("--freeze-flux", action="store_true",
                   help="hold the surface flux at its t=0 value. THE "
                        "discriminator for Wangara: CBL drives the SAME "
                        "prescribed-flux machinery with a constant value and "
                        "is healthy, so freezing Wangara's separates 'the "
                        "time-varying/traced route is broken' from 'something "
                        "else about this case is'.")
    args = p.parse_args(argv)

    drv = _load_driver()
    case = drv.load_case(args.case, nlev=args.nlev, dt=None)
    surface = drv.build_surface_config(case, flux_to_closure=True)
    prescribed = case.forcing.prescribe == "fluxes"
    if drv.surface_flux_varies_in_time(case):
        import dataclasses
        if args.freeze_flux:
            w0 = float(case.forcing.w_th_s(0.0))
            case = dataclasses.replace(case, forcing=case.forcing._replace(
                flux_to_closure=True, w_th_s=lambda _t, _w=w0: _w))
            print(f"FLUX FROZEN at w'T' = {w0:.5f} K m/s (t=0)")
        else:
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
          f"{'max|wp3|':>11} {'max|Skw|':>11} {'thl_bot':>9} {'min thl':>9}")

    step_fn, tend_fn = scm._step_fn, scm._tend_fn
    from legoesm.atmosphere.physics.turbulence.clubb import (
        unpack_clubb_moments as _unpack,
    )

    # ONE compiled scan, diagnostics emitted as scan outputs. The obvious
    # version -- a Python loop calling scm._step_fn -- RE-TRACES the whole
    # CLUBB step on every call, because that closure is not jitted. At 2880
    # steps that never printed a single row in six hours of walltime.
    def _diag_step(carry, k):
        state, phys = carry
        t = k.astype(jnp.float64) * case.dt
        state, phys = step_fn(state, phys, tend_fn, case.dt, t)
        m = _unpack(phys.clubb_moments)
        wp2, wp3 = m.wp2, m.wp3
        n = wp3.shape[1]
        skw = wp3 / jnp.maximum(wp2[:, :n], 1e-12) ** 1.5
        out = (jnp.min(wp2), jnp.max(jnp.abs(wp3)), jnp.max(jnp.abs(skw)),
               m.thlm[0, 0], jnp.min(m.thlm), jnp.max(m.thlm))
        return (state, phys), out

    _final, hist = jax.lax.scan(
        _diag_step, (scm.state, scm.phys_state), jnp.arange(nsteps))
    wp2_min, wp3_max, skw_max, thl_bot, thl_min, thl_max = (
        np.asarray(h) for h in hist)

    for k in range(0, nsteps, max(1, args.every)):
        t = k * case.dt
        wth = float(case.forcing.w_th_s(t)) if case.forcing.w_th_s else 0.0
        print(f"{k:6d} {t/3600.0:6.2f} {wth:10.5f} {wp2_min[k]:11.3e} "
              f"{wp3_max[k]:11.3e} {skw_max[k]:11.3e} {thl_bot[k]:9.2f} "
              f"{thl_min[k]:9.2f}", flush=True)
    j = int(np.nanargmax(np.abs(skw_max)))
    worst = (float(skw_max[j]), j)
    jb = int(np.nanargmin(thl_bot))
    print(f"\ncoldest bottom-level thlm = {thl_bot[jb]:.2f} K at step {jb} "
          f"({jb * case.dt / 3600.0:.2f} h); it starts at {thl_bot[0]:.2f} K")
    bad = np.where(thl_bot < thl_bot[0] - 5.0)[0]
    if bad.size:
        print(f"bottom level first drops >5 K below its start at step "
              f"{int(bad[0])} ({bad[0] * case.dt / 3600.0:.2f} h)")
    else:
        print("bottom level never drops >5 K below its start")
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
