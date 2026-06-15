"""GEOMETRIC calibration — Stage 0 truth generator (the twin's reference).

Spins up the 2° Veros-ACC channel with the GEOMETRIC mesoscale closure
(``EKEConfig.closure="geometric"``, Torres et al. 2025) at a KNOWN "true"
``GeometricConfig`` and writes the twin-experiment reference:

  * ``truth_snapshot.npz`` — the full ocean state at the end of the averaging
    window (every Field leaf AND the rigid-lid ψ / AB2 carries), for the
    exact-IC member protocol (``docs/planning/geometric_calibration_campaign.md``
    §3.1; consumed by ``run_geometric_stage0_etki.py``).
  * ``observables.npz`` — the ETKI observation vector: time-mean AND temporal-std
    maps of T, S, ψ and the depth-integrated EKE over the averaging window (§3.2;
    ψ not η — the rigid-lid solver leaves η ≡ 0), plus HELD-OUT ACC transport / KE
    time series for the early-stopping monitor.

The shared spin/observe/diagnostic machinery lives in
``legoesm.ocean.fidelity.geometric_stage0`` (so the truth driver and the ETKI
recovery loop reduce identically). Run in float64.

Usage::

    # de-risk wiring + geometric stability (short, GPU 0):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run/run_geometric_stage0.py \
        --smoke --out results/ocean/geometric_stage0/smoke

    # equilibrated truth generation (~12-14 model yr to EKE plateau; GPU 0):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run/run_geometric_stage0.py \
        --spin-years 20 --avg-days 800 --sample-every-days 10 \
        --out results/ocean/geometric_stage0/truth20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_DAYS_PER_YEAR = 365.0


def run_stage0(*, spin_years: float, avg_days: float, sample_every_days: float,
               dt: float, geom_overrides: dict[str, float], out_dir: Path):
    import jax.numpy as jnp
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    from legoesm.ocean.physics.lateral_mixing.eke import GeometricConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity import geometric_stage0 as g0

    out_dir.mkdir(parents=True, exist_ok=True)
    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        geom = GeometricConfig(**geom_overrides) if geom_overrides else GeometricConfig()
        recipe = g0.build_geometric_recipe(geom)
        cfg = recipe.model_config
        model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
        if cfg.coriolis_scheme == "explicit_ab2":
            model.check_coriolis_stability(dt)
        state = g0.seed_freerun_carries(model, recipe.initial_state, cfg, recipe.grid)
        block_fn, spb = g0.make_block_stepper(model, recipe.wind_forcing, dt)

        # --- Spin-up to the attractor ---
        spin_days = spin_years * _DAYS_PER_YEAR
        print(f"== GEOMETRIC ACC truth: spin-up {spin_days:.0f} days "
              f"(α={geom.alpha}, c_eps={geom.c_eps_geometric}, "
              f"kappa_u={geom.kappa_u}, kappa_e={geom.kappa_e}, "
              f"rossby={geom.rossby_factor}), dt={dt:.0f}s fp64 ==", flush=True)
        state = g0.advance_and_observe(block_fn, spb, state, spin_days, phase="spin")

        # --- Averaging window: observable maps + held-out series ---
        print(f"== averaging window {avg_days:.0f} days, "
              f"sample every {sample_every_days:.0f} days ==", flush=True)
        acc = g0.Accumulator()
        held = {"day": [], "acc_transport_Sv": [], "total_KE_J": []}
        state = g0.advance_and_observe(
            block_fn, spb, state, avg_days, phase="avg", accumulator=acc,
            sample_every_days=sample_every_days, held=held,
            z_coord=recipe.z_coord, grid=recipe.grid,
            rho_0=cfg.constants.rho_0, g_val=cfg.constants.g,
            spin_offset_days=spin_days)
        if acc.n < 2:
            raise RuntimeError(
                f"only {acc.n} observable sample(s) — increase --avg-days or "
                "decrease --sample-every-days")

        # --- Persist truth snapshot (Field leaves + plain-array carries) ---
        # The exact-IC member protocol restarts from this state, so the rigid-lid
        # ψ / AB2 carries (plain arrays) ARE part of the IC — save them too.
        snap = {}
        for f in state._fields:
            obj = getattr(state, f)
            if obj is None:
                continue
            if hasattr(obj, "data"):
                snap[f] = np.asarray(obj.data)
            elif isinstance(obj, jnp.ndarray):
                snap[f] = np.asarray(obj)
        np.savez_compressed(out_dir / "truth_snapshot.npz", **snap)

        # --- Persist observables (ETKI vector) + held-out series ---
        obs = acc.finalize()
        obs["held_day"] = np.asarray(held["day"])
        obs["held_acc_transport_Sv"] = np.asarray(held["acc_transport_Sv"])
        obs["held_total_KE_J"] = np.asarray(held["total_KE_J"])
        obs["n_samples"] = np.asarray(acc.n)
        np.savez_compressed(out_dir / "observables.npz", **obs)

        meta = {
            "geometric_true_params": {
                "alpha": geom.alpha, "c_eps_geometric": geom.c_eps_geometric,
                "kappa_u": geom.kappa_u, "kappa_e": geom.kappa_e,
                "rossby_factor": geom.rossby_factor,
            },
            "spin_days": spin_days, "avg_days": avg_days,
            "sample_every_days": sample_every_days, "dt_s": dt,
            "n_samples": acc.n,
            "acc_transport_mean_Sv": float(np.mean(held["acc_transport_Sv"])),
            "total_KE_mean_J": float(np.mean(held["total_KE_J"])),
            "grid": [int(recipe.grid.n_lat), int(recipe.grid.n_lon)],
        }
        (out_dir / "truth_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\n== TRUTH WRITTEN to {out_dir} ==")
        print(f"   observables: {sorted(k for k in obs)}")
        print(f"   ACC transport (held-out mean): "
              f"{meta['acc_transport_mean_Sv']:.1f} Sv")
        print(f"   total KE (held-out mean): {meta['total_KE_mean_J']:.3e} J")
        return meta
    finally:
        set_policy(_prev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spin-years", type=float, default=20.0,
                    help="Spin-up [model years] before averaging (EKE plateaus "
                         "~12-14 yr on the 2° ACC; default 20).")
    ap.add_argument("--avg-days", type=float, default=800.0,
                    help="Averaging-window length [model days] (default 800).")
    ap.add_argument("--sample-every-days", type=float, default=10.0,
                    help="Observable sampling cadence within the window [days].")
    ap.add_argument("--dt", type=float, default=43200.0,
                    help="Tracer timestep [s] (default 43200 = Veros dt_tracer; "
                         "dt_mom = dt/9 via the recipe's dt_mom_ratio).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for truth_snapshot.npz / observables.npz.")
    ap.add_argument("--smoke", action="store_true",
                    help="Wiring + geometric-stability smoke: 5-day spin, 5-day "
                         "window, daily sampling (overrides the duration args).")
    # GeometricConfig 'true' parameter overrides (default = paper-calibrated).
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--c-eps-geometric", type=float, default=None)
    ap.add_argument("--kappa-u", type=float, default=None)
    ap.add_argument("--kappa-e", type=float, default=None)
    ap.add_argument("--rossby-factor", type=float, default=None)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    geom_overrides = {}
    for field in ("alpha", "c_eps_geometric", "kappa_u", "kappa_e", "rossby_factor"):
        v = getattr(args, field)
        if v is not None:
            geom_overrides[field] = v

    if args.smoke:
        spin_years, avg_days, sample_every = 5.0 / _DAYS_PER_YEAR, 5.0, 1.0
    else:
        spin_years, avg_days, sample_every = (args.spin_years, args.avg_days,
                                              args.sample_every_days)

    run_stage0(spin_years=spin_years, avg_days=avg_days,
               sample_every_days=sample_every, dt=args.dt,
               geom_overrides=geom_overrides, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
