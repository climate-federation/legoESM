"""GEOMETRIC calibration — Stage 0 identifiability analysis (adjoint).

Quantifies WHICH observables constrain WHICH GEOMETRIC parameters, and exposes
the sloppy/degenerate directions that the ETKI twin recovery hit (the c_eps drift,
the α/c_eps trade-off). This is the Stage-0 deliverable that EXPLAINS the recovery
plateau and tells us whether an observation vector / joint-IC scheme can break the
degeneracy (``docs/planning/geometric_calibration_campaign.md`` §4, Stage 0 gates).

Method: the differentiable ocean (2026-06-11 audit; GM/Redi
``adjoint_stabilization="stop_gradient_slopes"`` gate, commit dc41c0a33) gives the
sensitivity matrix ``G = ∂g/∂θ`` of the (per-field-normalized) observable maps
w.r.t. the UNCONSTRAINED (logit) parameters — the same coordinates ETKI optimizes,
so each param is auto-scaled by its prior range. ``g`` is the END-STATE map of
T, S, ψ, EKE after a short exact-IC window from the truth attractor; ``G`` is
formed by reverse-mode AD (see below). Then:

``G`` is formed by ``jax.jacrev`` (reverse-mode — the rigid-lid CG solve is a
``custom_vjp``, so forward-mode is unavailable) over a COMPACT scalar observable.

  * Fisher F = GᵀG (5×5). Eigenvalues = the identifiability SPECTRUM; the
    smallest-eigenvalue eigenvector = the sloppiest parameter COMBINATION
    (a degeneracy shows as a small eigenvalue whose eigenvector mixes two params).
  * Per-(field, param) sensitivity ‖∂g_field/∂θ_i‖ — the "which observable
    constrains which parameter" table.

Run at 5- and 30-day windows (the plan's horizons). fp64 throughout.

Usage::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
        scripts/run/run_geometric_stage0_identifiability.py \
        --truth results/ocean/geometric_stage0/truth20 \
        --windows 5 30 --out results/ocean/geometric_stage0/identifiability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run_identifiability(*, truth_dir: Path, out_dir: Path, windows_days, dt: float,
                        stabilization: str = "stop_gradient_slopes"):
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    from legoesm.ocean.physics.lateral_mixing.eke import GeometricConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.training.trainable_ocean_params import (
        GEOMETRIC_TRAINABLE, TrainableOceanParams, constrain,
    )
    from legoesm.ocean.fidelity import geometric_stage0 as g0

    out_dir.mkdir(parents=True, exist_ok=True)
    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        snap = np.load(truth_dir / "truth_snapshot.npz")
        truth_obs = np.load(truth_dir / "observables.npz")
        meta = json.loads((truth_dir / "truth_meta.json").read_text())
        true_geom = GeometricConfig(**meta["geometric_true_params"])
        specs = GEOMETRIC_TRAINABLE
        pnames = [s.constraint.name for s in specs]
        raw_true = TrainableOceanParams.from_values(
            meta["geometric_true_params"], specs).raw

        # Build the model ONCE at truth params (concrete → validation runs), with
        # GM/Redi slope-stabilization, then warm the rigid-lid cache.
        recipe = g0.build_geometric_recipe(true_geom)
        cfg = recipe.model_config._replace(
            gm_redi=recipe.model_config.gm_redi._replace(
                adjoint_stabilization=stabilization))
        model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
        state0 = g0.restore_state(recipe.initial_state, snap, model=model,
                                  cfg=cfg, grid=recipe.grid)
        model._ensure_rigid_lid_data(state0)
        sf = recipe.wind_forcing

        # Compact, physically-targeted scalar observables (the rigid-lid CG solve
        # is custom_vjp → reverse-mode only, so jacfwd over full maps is blocked;
        # jacrev over ~9 scalars is cheap and more interpretable). Each probes a
        # distinct GEOMETRIC lever: stratification & isopycnal structure (GM
        # transport ∝ α/c_eps), EKE level (c_eps), ACC transport (kappa_u).
        wet2 = jnp.asarray(np.asarray(snap["land_mask"]) > 0.5)        # (lat,lon)
        lat = jnp.asarray(np.degrees(np.asarray(recipe.grid.lat)))
        band2 = (lat >= g0.ACC_DRAKE_LAT_S) & (lat <= g0.ACC_DRAKE_LAT_N)
        nwet = jnp.sum(wet2)

        def _wmean(x2):                       # masked spatial mean of a 2-D field
            return jnp.sum(jnp.where(wet2, x2, 0.0)) / nwet

        OBS_NAMES = ("T_volmean", "T_strat", "T_merid_rms", "S_volmean",
                     "psi_transport", "psi_rms", "eke_mean", "eke_max", "eke_rms")

        def observables(state):
            T = state.T.data                  # (lat,lon,lev), lev0=surface
            S = state.S.data
            psi = state.psi                   # (lat+1,lon+1)
            eke = state.eke.data              # (lat,lon)
            w3 = wet2[:, :, None]
            T_vol = jnp.sum(jnp.where(w3, T, 0.0)) / (nwet * T.shape[2])
            T_strat = _wmean(T[:, :, 0]) - _wmean(T[:, :, -1])     # surf - bottom
            Tcol = jnp.sum(jnp.where(w3, T, 0.0), axis=2) / T.shape[2]  # col-mean
            T_merid = jnp.sqrt(_wmean((Tcol - _wmean(Tcol)) ** 2))     # horiz structure
            S_vol = jnp.sum(jnp.where(w3, S, 0.0)) / (nwet * S.shape[2])
            psi_band = jnp.where(band2[:, None], psi[:-1, :-1], jnp.nan)
            psi_T = jnp.nanmax(psi_band) - jnp.nanmin(psi_band)         # ACC transport
            psi_rms = jnp.sqrt(_wmean((psi[:-1, :-1] - _wmean(psi[:-1, :-1])) ** 2))
            eke_m = _wmean(eke)
            eke_mx = jnp.max(jnp.where(wet2, eke, -jnp.inf))
            eke_r = jnp.sqrt(_wmean((eke - eke_m) ** 2))
            return jnp.stack([T_vol, T_strat, T_merid, S_vol, psi_T, psi_rms,
                              eke_m, eke_mx, eke_r])

        def packed_raw(raw, n_steps):
            vals = {s.constraint.name: constrain(raw[i], s)
                    for i, s in enumerate(specs)}
            geom = true_geom._replace(**vals)
            m2 = g0.clone_with_geometric(model, cfg, geom)
            s = state0
            for _ in range(n_steps):
                s = m2._step_impl(s, dt, surface_forcing=sf)
            return observables(s)

        results = {}
        for W in windows_days:
            n_steps = int(round(W * 86400.0 / dt))
            print(f"== window {W} d ({n_steps} steps): jacrev ∂g/∂θ ...", flush=True)
            g0_vals = np.asarray(packed_raw(raw_true, n_steps))        # scale per obs
            ag = np.abs(g0_vals)
            # Floor the per-observable scale RELATIVE to the largest |g| so a
            # near-zero observable cannot get ~1/eps weight and hijack the Fisher
            # spectrum (adversarial-review Q2); warn if any trips it.
            floor = 1e-8 * float(ag.max())
            tripped = [OBS_NAMES[j] for j in range(len(ag)) if ag[j] < floor]
            if tripped:
                print(f"  WARNING: near-zero observables at truth {tripped} "
                      f"(|g| < {floor:.2e}) — scale floored, weight capped", flush=True)
            scale = np.maximum(ag, floor)
            print(f"  g(theta_true) = "
                  f"{', '.join(f'{n}={v:.3e}' for n, v in zip(OBS_NAMES, g0_vals))}",
                  flush=True)
            G_raw = np.asarray(jax.jacrev(lambda r: packed_raw(r, n_steps))(raw_true))
            G = G_raw / scale[:, None]            # dimensionless (n_obs, n_param)
            F = G.T @ G                            # normalized Fisher (5×5)
            evals, evecs = np.linalg.eigh(F)
            order = np.argsort(evals)[::-1]
            evals, evecs = evals[order], evecs[:, order]
            diagF = np.sqrt(np.maximum(np.diag(F), 0.0))
            cond = float(evals[0] / max(evals[-1], 1e-300))
            results[str(W)] = {
                "obs_names": list(OBS_NAMES),
                "jacobian_normalized": G.tolist(),
                "fisher_eigvalues": evals.tolist(),
                "fisher_eigvectors": evecs.tolist(),
                "condition_number": cond,
                "single_param_sensitivity": dict(zip(pnames, diagF.tolist())),
            }

            print(f"\n  Fisher spectrum (desc): "
                  f"{', '.join(f'{e:.3e}' for e in evals)}")
            print(f"  condition number (sloppiness): {cond:.2e}")
            print(f"  sloppiest direction (smallest-eig eigenvector):")
            for n, c in sorted(zip(pnames, evecs[:, -1]), key=lambda t: -abs(t[1])):
                print(f"      {n:18s} {c:+.3f}")
            print(f"  best-constrained direction (largest-eig eigenvector):")
            for n, c in sorted(zip(pnames, evecs[:, 0]), key=lambda t: -abs(t[1])):
                print(f"      {n:18s} {c:+.3f}")
            print(f"  single-param sensitivity ‖∂g/∂θ_i‖ (logit coords, desc):")
            for n, d in sorted(zip(pnames, diagF), key=lambda t: -t[1]):
                print(f"      {n:18s} {d:.3e}")
            print(f"  observable × param sensitivity table (|∂g/∂θ|, normalized):")
            print(f"      {'observable':14s} " + " ".join(f"{n[:8]:>9s}" for n in pnames))
            for j, on in enumerate(OBS_NAMES):
                print(f"      {on:14s} " + " ".join(f"{abs(G[j, i]):9.2e}" for i in range(len(specs))))
            print(flush=True)
            # Save incrementally so a later-window timeout never loses results.
            (out_dir / "identifiability.json").write_text(json.dumps(
                {"truth_dir": str(truth_dir), "param_names": pnames,
                 "windows": results}, indent=2))

        (out_dir / "identifiability.json").write_text(json.dumps(
            {"truth_dir": str(truth_dir), "param_names": pnames,
             "windows": results}, indent=2))
        print(f"== identifiability analysis -> {out_dir}/identifiability.json ==")
        return results
    finally:
        set_policy(_prev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("results/ocean/geometric_stage0/identifiability"))
    ap.add_argument("--windows", type=float, nargs="+", default=[5.0, 30.0],
                    help="Window lengths [days] for the sensitivity (default 5 30).")
    ap.add_argument("--dt", type=float, default=43200.0)
    ap.add_argument("--stabilization", default="stop_gradient_slopes",
                    choices=("stop_gradient_slopes", "stop_gradient_taper", "none"),
                    help="GM/Redi adjoint stabilization. 'none' = exact adjoint "
                         "(use at SHORT windows only) for the gate-invariance "
                         "cross-check.")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    run_identifiability(truth_dir=args.truth, out_dir=args.out,
                        windows_days=args.windows, dt=args.dt,
                        stabilization=args.stabilization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
