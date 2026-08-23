"""Is a turbulence closure's parameter gradient a slope, or a singularity?

WHY. In the LES-vs-SCM tuning campaign, CLUBB and Smagorinsky report a
directional derivative of -1e15 and -1.8e17 against an objective of order 1,
and no line-search step reduces the loss. Two explanations fit every number:

  CHAOS               identical coefficients reaching the physics by two code
                      paths diverge through rounding alone, and the divergence
                      amplifies as exp(lambda*T). Needs a long rollout.
  ILL-CONDITIONING    one parameter-dependent quantity passes near zero
                      (catastrophic cancellation into a denominator). Needs no
                      time at all.

ROLLOUT LENGTH SEPARATES THEM. Chaos must collapse when T does; a bad
denominator is flat in T. Thirty steps cannot amplify 1e-16 to 1e15, so a large
gradient at half an hour REFUTES chaos arithmetically.

The scoring driver cannot answer this: it refuses a ``--hours`` that does not
match the LES reference end, and rightly so -- the two sides must average the
same window. The question is about the ROLLOUT, not about fidelity to a
reference, so this scores a self-contained functional of the final state
instead and needs no reference at all.

Reuses the driver's own ``_rollout_means``, so the graph differentiated here is
the graph the campaign differentiates, not a lookalike.

Usage:
  python scripts/validate/scm_gradient_conditioning.py \
      --scheme clubb --case bomex --hours 0.25 0.5 1 2 4
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

_ROOT = Path(__file__).resolve().parents[2]


def _load_driver():
    """Import the tuning driver by path (it is a script, not a module)."""
    path = _ROOT / "scripts" / "run" / "run_scm_les_turbulence_tuning.py"
    spec = importlib.util.spec_from_file_location("_scm_tuning", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_scm_tuning"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scheme", default="clubb")
    p.add_argument("--case", default="bomex")
    p.add_argument("--hours", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--nlev", type=int, default=48)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--tier", default="extended")
    p.add_argument("--top", type=int, default=6,
                   help="how many largest-|grad| parameters to name per length")
    args = p.parse_args(argv)

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("JAX_ENABLE_X64=1 is required (the campaign is f64).")

    drv = _load_driver()
    case = drv.load_case(args.case, nlev=args.nlev, dt=args.dt)
    surface = drv.build_surface_config(case, flux_to_closure=True)
    prescribed = case.forcing.prescribe == "fluxes"
    if drv.surface_flux_varies_in_time(case):
        import dataclasses
        case = dataclasses.replace(
            case, forcing=case.forcing._replace(flux_to_closure=True))
    base_cfg = drv.build_physics_config(
        args.scheme, prescribed_fluxes=prescribed, microphysics="none",
        simple_lw=args.case in drv._SIMPLE_LW_CASES,
        bulk_ch=case.spec.bulk_ch, bulk_ce=case.spec.bulk_ce,
        surface=surface, clubb_prognostic=True,
    )
    params0 = drv._initial_params(args.scheme, args.tier)
    print(f"{args.scheme} on {args.case}: {len(params0.constraints)} parameters "
          f"at tier {args.tier}, nlev={args.nlev}")
    print(f"{'hours':>7} {'steps':>7} {'objective':>14} {'|grad|max':>12} "
          f"{'|grad|median':>13}   largest contributors")

    for hours in args.hours:
        # A SELF-CONTAINED functional of the analysis-window mean state. Not the
        # LES score: no reference is involved, so nothing here depends on a
        # matched averaging window. Any smooth functional works -- what is under
        # test is the CONDITIONING of d(state)/d(params), not the metric.
        def loss_fn(params, hours=hours):
            means, _ps = drv._rollout_means(
                params, base_cfg=base_cfg, case=case, dt=case.dt,
                hours=hours, analysis_hours=min(0.25, hours * 0.5),
                chunk_steps=drv.DEFAULT_CHUNK_STEPS,
            )
            # Sum of squares over the scored state, normalised by level count so
            # the value is comparable across lengths.
            return sum(jnp.mean(means[k] ** 2) for k in ("T", "qv", "u", "v"))

        import equinox as eqx
        val, grads = eqx.filter_value_and_grad(loss_fn)(params0)
        stats = drv._grad_stats(grads, 0.0)
        ranked = sorted(stats.items(), key=lambda kv: -kv[1]["abs_max"])
        mags = [s["abs_max"] for _n, s in ranked]
        median = mags[len(mags) // 2]
        top = "  ".join(f"{n.rsplit('.', 1)[-1]}={s['abs_max']:.2g}"
                        for n, s in ranked[:args.top])
        nsteps = max(1, int(round(hours * 3600.0 / case.dt)))
        print(f"{hours:7.2f} {nsteps:7d} {float(val):14.6g} {mags[0]:12.3g} "
              f"{median:13.3g}   {top}", flush=True)

    print("\nREAD IT AS: |grad|max roughly FLAT in length -> ill-conditioning, "
          "and the SAME parameters dominate at every length.\n"
          "            |grad|max growing many orders with length -> chaos, and "
          "the dominant set rotates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
