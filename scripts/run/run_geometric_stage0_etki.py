"""GEOMETRIC calibration — Stage 0 ETKI twin recovery.

Recovers the GEOMETRIC closure parameters (Torres et al. 2025: alpha,
c_eps_geometric, kappa_u, kappa_e, rossby_factor) from the truth observables
written by ``run_geometric_stage0.py``, using Ensemble Transform Kalman Inversion
(``legoesm.training.etki``) under the Perezhogin et al. (arXiv:2604.06398)
exact-IC protocol: every ensemble member starts from the truth's spun-up
attractor snapshot (NO re-equilibration) and is integrated with its OWN sampled
parameters over the averaging window; the loss is the misfit of the §3.2
observation vector (equal-weighted, per-field-normalized mean+std maps of
T, S, ψ, depth-integrated EKE). ACC transport is HELD OUT for early stopping.

This is the protocol shakedown (``docs/planning/geometric_calibration_campaign.md``
§4 Stage 0): truth = the model at known ``GeometricConfig`` defaults, so success =
ETKI drives the ensemble-mean parameters back to those defaults and the misfit
decreases to the averaging-window noise floor.

NOTE the gradient-free ETKI path rebuilds the config eagerly per member (Python
floats), so each member triggers one JIT compile of the step (the documented
ETKI cost; the differentiable adjoint path threads traced params instead). Start
small (``--ne 20 --iterations 4 --member-days 300``) and scale up once the misfit
trend is confirmed.

Usage::

    # wiring smoke (tiny, fast):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run/run_geometric_stage0_etki.py \
        --truth results/ocean/geometric_stage0/truth20 \
        --ne 4 --iterations 2 --member-days 60 --rel-spread 0.1 --smoke

    # first real recovery (GPU 0):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run/run_geometric_stage0_etki.py \
        --truth results/ocean/geometric_stage0/truth20 \
        --ne 30 --iterations 4 --member-days 400 --rel-spread 0.1 \
        --out results/ocean/geometric_stage0/etki
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def run_etki_recovery(*, truth_dir: Path, out_dir: Path, ne: int, iterations: int,
                      member_days: float, sample_every_days: float,
                      rel_spread: float, dt: float, etki_dt: float, seed: int):
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    from legoesm.ocean.physics.lateral_mixing.eke import GeometricConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.training import etki
    from legoesm.training.trainable_ocean_params import (
        GEOMETRIC_TRAINABLE, constrain, ensemble_from_priors,
    )
    from legoesm.ocean.fidelity import geometric_stage0 as g0

    out_dir.mkdir(parents=True, exist_ok=True)
    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        # --- Truth: observation target + exact-IC snapshot ---
        truth_obs = np.load(truth_dir / "observables.npz")
        snapshot = np.load(truth_dir / "truth_snapshot.npz")
        true_meta = json.loads((truth_dir / "truth_meta.json").read_text())
        true_geom = GeometricConfig(**true_meta["geometric_true_params"])

        land_mask = snapshot["land_mask"]
        y, r_diag, packer = g0.build_observation_target(truth_obs, land_mask)
        y = jnp.asarray(y, jnp.float64)
        r_diag = jnp.asarray(r_diag, jnp.float64)
        # Saturation cap for member EKE: 1000× the truth's max mean-EKE is a
        # generous "clearly unphysical runaway" bound (truth equilibrates ~1e3
        # m^3/s^2), so a finite-but-blown-up member is dropped, not averaged in
        # (adversarial-review M1).
        eke_cap = 1.0e3 * float(np.nanmax(np.abs(truth_obs["mean_eke"])))
        print(f"== observation vector n_o={y.shape[0]} "
              f"(mean+std of {g0.OBSERVABLE_FIELDS}) | eke blow-up cap "
              f"{eke_cap:.2e} m^3/s^2 ==", flush=True)

        # --- Forward model: one exact-IC member window -> packed observable g ---
        def _member_g(raw_vec: np.ndarray) -> np.ndarray:
            vals = {sp.constraint.name: float(constrain(jnp.asarray(raw_vec[i]), sp))
                    for i, sp in enumerate(GEOMETRIC_TRAINABLE)}
            geom = GeometricConfig(**vals)
            recipe = g0.build_geometric_recipe(geom)
            cfg = recipe.model_config
            model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
            state = g0.restore_state(recipe.initial_state, snapshot,
                                     model=model, cfg=cfg, grid=recipe.grid)
            block_fn, spb = g0.make_block_stepper(model, recipe.wind_forcing, dt)
            acc = g0.Accumulator()
            try:
                g0.advance_and_observe(
                    block_fn, spb, state, member_days, phase="member",
                    accumulator=acc, sample_every_days=sample_every_days,
                    eke_blowup_cap=eke_cap, verbose=False)
            except RuntimeError:
                return np.full(y.shape[0], np.nan)        # blow-up -> ETKI drops it
            return packer(acc.finalize())

        def forward_fn(theta: jnp.ndarray) -> jnp.ndarray:
            theta = np.asarray(theta)
            gs = []
            for m in range(theta.shape[0]):
                t0 = time.time()
                gm = _member_g(theta[m])
                finite = bool(np.all(np.isfinite(gm)))
                print(f"    member {m+1:3d}/{theta.shape[0]} "
                      f"{'ok ' if finite else 'NaN'} {time.time()-t0:5.0f}s",
                      flush=True)
                gs.append(gm)
            return jnp.asarray(np.stack(gs), jnp.float64)

        # --- ETKI ensemble + loop ---
        key = jax.random.PRNGKey(seed)
        theta0 = ensemble_from_priors(key, ne, GEOMETRIC_TRAINABLE,
                                      rel_spread=rel_spread)
        history = {"misfit": [], "n_valid": [], "params": [], "held_transport": []}

        def _callback(it, step):
            errs = g0.relative_param_error(step.theta, true_geom, GEOMETRIC_TRAINABLE)
            history["misfit"].append(step.misfit)
            history["n_valid"].append(step.n_valid)
            history["params"].append({k: v[0] for k, v in errs.items()})
            mean_rel = float(np.mean([v[2] for v in errs.values()]))
            print(f"\n== iter {it}: misfit {step.misfit:.4e} | n_valid "
                  f"{step.n_valid}/{ne} | mean |Δθ|/θ {mean_rel:.3f} ==")
            for k, (val, true, rel) in errs.items():
                print(f"     {k:18s} {val:11.4g}  (true {true:g}, rel {rel:.3f})")

        print(f"== ETKI: ne={ne}, iters={iterations}, member window {member_days:.0f}d, "
              f"rel_spread={rel_spread}, dt={dt:.0f}s fp64 ==", flush=True)
        theta_final, misfits = etki.run_etki(
            forward_fn, theta0, y, n_iterations=iterations, dt=etki_dt,
            r_diag=r_diag, callback=_callback)

        # --- Persist ---
        final_errs = g0.relative_param_error(theta_final, true_geom,
                                             GEOMETRIC_TRAINABLE)
        result = {
            "truth_dir": str(truth_dir),
            "true_params": true_meta["geometric_true_params"],
            "recovered_params": {k: v[0] for k, v in final_errs.items()},
            "relative_error": {k: v[2] for k, v in final_errs.items()},
            "misfits": [float(m) for m in misfits],
            "ne": ne, "iterations": iterations, "member_days": member_days,
            "rel_spread": rel_spread,
        }
        np.savez_compressed(out_dir / "etki_result.npz",
                            theta_final=np.asarray(theta_final),
                            misfits=np.asarray(misfits))
        (out_dir / "etki_result.json").write_text(json.dumps(result, indent=2))
        print(f"\n== ETKI DONE -> {out_dir} ==")
        print(f"   misfit trajectory: "
              f"{' -> '.join(f'{m:.3e}' for m in misfits)}")
        print(f"   final mean |Δθ|/θ: "
              f"{np.mean([v[2] for v in final_errs.values()]):.3f}")
        return result
    finally:
        set_policy(_prev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True,
                    help="Truth dir (observables.npz + truth_snapshot.npz + truth_meta.json).")
    ap.add_argument("--out", type=Path,
                    default=Path("results/ocean/geometric_stage0/etki"))
    ap.add_argument("--ne", type=int, default=30, help="Ensemble size.")
    ap.add_argument("--iterations", type=int, default=4, help="ETKI iterations.")
    ap.add_argument("--member-days", type=float, default=400.0,
                    help="Exact-IC member averaging window [days].")
    ap.add_argument("--sample-every-days", type=float, default=10.0)
    ap.add_argument("--rel-spread", type=float, default=0.1,
                    help="Prior logit-space spread (0.1 ~ measured 2-4x factors).")
    ap.add_argument("--dt", type=float, default=43200.0)
    ap.add_argument("--etki-dt", type=float, default=1.0,
                    help="ETKI scheduler step (Iglesias & Yang 2021).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny wiring smoke: ne=4, 2 iters, 60-day members.")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    if args.smoke:
        ne, iters, mdays, severy = 4, 2, 60.0, 5.0
    else:
        ne, iters, mdays, severy = (args.ne, args.iterations, args.member_days,
                                    args.sample_every_days)

    run_etki_recovery(
        truth_dir=args.truth, out_dir=args.out, ne=ne, iterations=iters,
        member_days=mdays, sample_every_days=severy, rel_spread=args.rel_spread,
        dt=args.dt, etki_dt=args.etki_dt, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
