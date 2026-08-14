"""Localize WHERE and WHEN one SCM turbulence closure goes non-finite.

``run_scm_les_turbulence_tuning.py`` reports a blown-up arm as a single word
(``nonfinite_rollout``) and a penalty score, which says a scheme failed but
not which prognostic went first, at which step, or at which level. Ranking a
closure out of a campaign on that alone is guessing; this probe turns it into
a step, a field and a level index.

It steps ONE scheme on ONE case with the SAME construction path the tuning
driver uses (``case.create_scm`` -> ``scm._step_fn`` with ``scm._tend_fn``, so
the large-scale forcing and the surface boundary condition are present exactly
as in the campaign) and prints, per stride, the min/max of every prognostic
plus the turbulence carry. It stops at the FIRST non-finite value and names it.

NUMBERS ONLY. It prints no verdict: a probe that decides what its own output
means gets that opinion quoted back as evidence.

``nanmin``/``nanmax`` are deliberately NOT used -- a NaN is the signal here, so
hiding it in the reduction would defeat the instrument.

Usage::

    python scripts/validate/diag_scheme_blowup.py --case cbl --scheme mynn25
    python scripts/validate/diag_scheme_blowup.py --case cbl --scheme mynn25 \\
        --surface-flux-to-closure
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]


def _load_driver():
    """Import the tuning driver as a module.

    Reusing it is the point: the case registry, the physics-config builder and
    the surface-config builder must be the ONES THE CAMPAIGN RUNS, or the probe
    localizes a blow-up in a column the campaign never integrates.
    """
    path = _ROOT / "scripts" / "run" / "run_scm_les_turbulence_tuning.py"
    spec = importlib.util.spec_from_file_location("scm_les_tuning", path)
    if spec is None or spec.loader is None:                # pragma: no cover
        raise SystemExit(f"cannot import the tuning driver at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scm_les_tuning"] = mod
    spec.loader.exec_module(mod)
    return mod


def _profiles(state, phys):
    """Every prognostic this probe watches, as flat float64 arrays."""
    out = {
        "T": np.asarray(state.T.data).ravel(),
        "qv": np.asarray(state.tracers["q_v"].data).ravel(),
        "u": np.asarray(state.u.data).ravel(),
        "v": np.asarray(state.v.data).ravel(),
    }
    # The turbulence carry is scheme-dependent (TKE for tke/edmf/clubb_lite,
    # qke = 2*TKE for mynn25, the packed moment vector for prognostic CLUBB)
    # and absent for the diagnostic closures, so it is watched only when the
    # state actually carries one.
    tke = getattr(phys, "tke", None)
    if tke is not None:
        out["tke_carry"] = np.asarray(tke).ravel()
    mom = getattr(phys, "clubb_moments", None)
    if mom is not None:
        out["clubb_moments"] = np.asarray(mom).ravel()
    return out


def _first_bad(fields):
    """(name, flat index) of the first non-finite entry, or None."""
    for name, arr in fields.items():
        bad = np.flatnonzero(~np.isfinite(arr))
        if bad.size:
            return name, int(bad[0]), int(bad.size)
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--scheme", required=True)
    p.add_argument("--nlev", type=int, default=48)
    p.add_argument("--dt", type=float, default=None,
                   help="override the case's own dt [s]")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--surface-flux-to-closure", action="store_true",
                   help="hand a prescribed-flux deck's surface heat flux to "
                        "the closure instead of injecting it afterwards; "
                        "matches the campaign flag of the same name.")
    args = p.parse_args(argv)

    import jax.numpy as jnp
    d = _load_driver()

    if args.case not in d.ALL_CASES:
        raise SystemExit(f"unknown case {args.case!r}; "
                         f"choose from {list(d.ALL_CASES)}")
    if args.scheme not in d.TURBULENCE_SCHEMES:
        raise SystemExit(f"unknown scheme {args.scheme!r}; "
                         f"choose from {list(d.TURBULENCE_SCHEMES)}")

    case = d.load_case(args.case, nlev=args.nlev, dt=args.dt)
    dt = float(case.dt)
    prescribed = case.forcing.prescribe == "fluxes"
    surface = d.build_surface_config(
        case, flux_to_closure=args.surface_flux_to_closure)
    if args.surface_flux_to_closure and prescribed:
        import dataclasses
        case = dataclasses.replace(case, forcing=case.forcing._replace(
            prescribe="none", w_th_s=None, w_qv_s=None))
    cfg = d.build_physics_config(
        args.scheme, prescribed_fluxes=prescribed, surface=surface,
        simple_lw=args.case in d._SIMPLE_LW_CASES)
    scm = d._create_scm(cfg, case, dt)

    print(f"case={args.case} scheme={args.scheme} nlev={args.nlev} dt={dt:g}s "
          f"steps={args.steps} ({args.steps * dt / 3600.0:.2f} h) "
          f"prescribe={case.forcing.prescribe!r} "
          f"flux_to_closure={args.surface_flux_to_closure}")
    print(f"surface: Cd={surface.Cd_neutral:.4e} Ch={surface.Ch_neutral:.4e} "
          f"shflx={surface.prescribed_shflx_w_m2} "
          f"lhflx={surface.prescribed_lhflx_w_m2}")

    # One compile, then one host sync per step. Calling ``scm._step_fn``
    # directly re-traces the whole column every step, which is affordable for
    # the 30-step CLUBB probe but not for the thousands of steps it takes to
    # reach a blow-up several hours into a case. ``_tend_fn`` and ``dt`` are
    # captured in the closure so they stay compile-time constants.
    import jax
    step = jax.jit(lambda s, ph, t: scm._step_fn(s, ph, scm._tend_fn, dt, t))

    state, phys = scm.state, scm.phys_state
    watched = list(_profiles(state, phys))
    print(f"watching: {', '.join(watched)}")
    hdr = f"{'step':>6}{'t[h]':>8}"
    for name in watched:
        hdr += f"{name + ' min':>14}{name + ' max':>14}"
    print(hdr)

    for k in range(args.steps):
        state, phys = step(state, phys,
                           jnp.asarray(k * dt, dtype=jnp.float64))
        fields = _profiles(state, phys)
        bad = _first_bad(fields)
        if k % args.print_every == 0 or k == args.steps - 1 or bad:
            row = f"{k + 1:>6}{(k + 1) * dt / 3600.0:>8.3f}"
            for name in watched:
                arr = fields[name]
                row += f"{np.min(arr):>14.5g}{np.max(arr):>14.5g}"
            print(row, flush=True)
        if bad:
            name, idx, count = bad
            print(f"NON-FINITE at step {k + 1} (t={(k + 1) * dt:.1f} s): "
                  f"field={name} first_flat_index={idx} n_bad={count} "
                  f"of {fields[name].size}")
            return 2
    print(f"finite through step {args.steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
