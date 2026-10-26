"""Localize prognostic CLUBB's blow-up when the surface flux reaches it.

Handing a prescribed-flux deck's surface heat flux to the closure (instead of
zeroing the exchange coefficient and injecting the heat afterwards) leaves most
schemes unchanged and improves the flux-driven ones. Prognostic CLUBB instead
goes from a normalized score of 0.0432 to 772.7 on the dry convective case --
FINITE, not NaN, so it is not an overflow: the column is being driven far too
hard, or driven twice.

The prescribed-flux cases (bomex, cbl, gabls1) all degrade; ekman (no surface
heat flux) and rico (interactive bulk fluxes) are bit-identical, which points
at the surface-flux path rather than at anything else the change touched.

This probe steps ONE column with the flux on and off, side by side, and prints
where the two separate and by how much -- the step, the field, and the surface
kinematic flux the closure is actually using. It reports NUMBERS ONLY; the
interpretation belongs in the analysis that reads them, not baked in here.

Usage::

    python scripts/validate/diag_clubb_surface_flux.py --case cbl --steps 40
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

_DRIVER = (Path(__file__).resolve().parents[1] / "run"
           / "run_scm_les_turbulence_tuning.py")


def _load_driver():
    spec = importlib.util.spec_from_file_location("_tuner", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tuner"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="cbl")
    p.add_argument("--scheme", default="clubb")
    p.add_argument("--nlev", type=int, default=48)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--print-every", type=int, default=1)
    args = p.parse_args(argv)

    import jax.numpy as jnp
    d = _load_driver()

    case = d.load_case(args.case, nlev=args.nlev, dt=None)
    dt = float(case.dt)

    arms = {}
    for label, to_closure in (("fluxoff", False), ("fluxon", True)):
        c = case
        surface = d.build_surface_config(c, flux_to_closure=to_closure)
        prescribed = c.forcing.prescribe == "fluxes"
        if to_closure and prescribed:
            import dataclasses
            c = dataclasses.replace(c, forcing=c.forcing._replace(
                prescribe="none", w_th_s=None, w_qv_s=None))
        cfg = d.build_physics_config(
            args.scheme, prescribed_fluxes=prescribed, surface=surface,
            simple_lw=args.case in d._SIMPLE_LW_CASES)
        scm = d._create_scm(cfg, c, dt)
        arms[label] = {
            "scm": scm, "state": scm.state, "phys": scm.phys_state,
            "surface": surface,
        }
        print(f"[{label}] Cd={surface.Cd_neutral:.4e} Ch={surface.Ch_neutral:.4e} "
              f"prescribed_shflx={surface.prescribed_shflx_w_m2} "
              f"prescribed_lhflx={surface.prescribed_lhflx_w_m2} "
              f"forcing.prescribe={c.forcing.prescribe!r}")
    print(f"case={args.case} scheme={args.scheme} dt={dt}s nlev={args.nlev}")
    print()

    hdr = (f"{'step':>5}{'t[s]':>9}"
           f"{'off Tmin':>10}{'off Tmax':>10}"
           f"{'on Tmin':>10}{'on Tmax':>10}"
           f"{'max|dT|':>11}{'on max|wp2|':>13}{'on max|wpthlp|':>15}")
    print(hdr)

    for k in range(args.steps):
        for a in arms.values():
            a["state"], a["phys"] = a["scm"]._step_fn(
                a["state"], a["phys"], a["scm"]._tend_fn, dt,
                jnp.asarray(k * dt, dtype=jnp.float64))
        Toff = np.asarray(arms["fluxoff"]["state"].T.data).ravel()
        Ton = np.asarray(arms["fluxon"]["state"].T.data).ravel()
        dT = np.nanmax(np.abs(Ton - Toff)) if Ton.size == Toff.size else np.nan

        ph = arms["fluxon"]["phys"]
        mom = getattr(ph, "clubb_moments", None)
        wp2 = wpthlp = np.nan
        if mom is not None:
            m = np.asarray(mom)
            # packed (ncol, 15, nlev+1); slot order is CLUBB's own. Report the
            # largest magnitude in the pack rather than guessing a slot index.
            wp2 = float(np.nanmax(np.abs(m)))
            wpthlp = float(np.nanmax(np.abs(m[:, :, 0])))

        if k % args.print_every == 0 or k == args.steps - 1:
            print(f"{k + 1:>5}{(k + 1) * dt:>9.0f}"
                  f"{np.nanmin(Toff):>10.3f}{np.nanmax(Toff):>10.3f}"
                  f"{np.nanmin(Ton):>10.3f}{np.nanmax(Ton):>10.3f}"
                  f"{dT:>11.4g}{wp2:>13.4g}{wpthlp:>15.4g}")
        if not np.all(np.isfinite(Ton)):
            print(f"  ON went NON-FINITE at step {k + 1}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
