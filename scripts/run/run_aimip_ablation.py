#!/usr/bin/env python
"""AIMIP classical-variant scheme-swap ablation.

Greedy stage-by-stage swap.  Each stage tries N alternative schemes
for one parameterization category (gravity wave -> convection ->
turbulence -> microphysics -> cloud), trains the full AIMIP
classical pipeline end-to-end on the multi-year windowed dataset,
evaluates against the held-out 2017 windows, and picks the variant
with the lowest combined (eval loss + |T bias|) as the stage winner.
The winner is locked in and the next stage runs on top of it.

Outputs (under ``results/aimip_001/``):

* ``aimip_ablation_scorecard.json`` — per-attempt metrics + per-stage
  winners + final winning scheme set.
* ``aimip_ablation_summary.png`` — bar chart of T RMSE + |T bias|
  per attempt, grouped by stage.

Usage::

    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_aimip_ablation.py \\
        --suite config/aimip/aimip_suite.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

logger = logging.getLogger("aimip-ablation")


# ----------------------------------------------------------------------
# Ablation table (stage_name, scheme_dim_kwarg, candidate scheme list).
# Order matters: each stage runs on top of the previous stage's winner.
# Candidates that fail to compile / numerically diverge are recorded
# with ``status="failed"`` and skipped from the winner pick.
# ----------------------------------------------------------------------

ABLATIONS: list[tuple[str, str, list[str]]] = [
    ("gwd",          "gwd_scheme",          ["mcfarlane", "lindzen", "hines", "rayleigh"]),
    ("convection",   "convection_scheme",   ["tiedtke",   "sbm",     "emanuel"]),
    ("turbulence",   "turbulence_scheme",   ["louis",     "tke",     "smagorinsky"]),
    ("microphysics", "microphysics_scheme", ["none",      "sundqvist"]),
    ("cloud",        "cloud_scheme",        ["xu_randall", "sundqvist"]),
]

# Initial winning scheme set (matches the current AIMIP baseline).
# ``surface_bulk_scheme`` is not an ablation DIMENSION here (constant is the
# baseline), but it is threaded so the builder never silently defaults it out
# of step with the rest of the resolved scheme set.
INITIAL = {
    "gwd_scheme":          "mcfarlane",
    "convection_scheme":   "tiedtke",
    "turbulence_scheme":   "louis",
    "surface_bulk_scheme": "constant",
    "microphysics_scheme": "none",
    "cloud_scheme":        "xu_randall",
}


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _build_spectral_config(base_cfg: dict):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig

    loss_kwargs = base_cfg.get("loss", {}) or {}
    loss_config = LossConfig(**{k: v for k, v in loss_kwargs.items()
                                if k in LossConfig._fields})

    return NeuralGCMSpectralConfig(
        n_max=int(base_cfg["n_max"]),
        n_levels=int(base_cfg["nlev"]),
        dt=float(base_cfg["dt"]),
        pe_config=SpectralPEConfig(
            hyperdiff_coeff=2.5e15,
            hyperdiff_order=2,
            time_integrator="ssp_rk3",
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
        ),
        n_epochs=int(base_cfg.get("aimip_n_epochs", 3)),
        lr=float(base_cfg.get("aimip_lr", 3.0e-4)),
        weight_decay=float(base_cfg.get("aimip_weight_decay", 1.0e-5)),
        grad_clip_norm=float(base_cfg.get("aimip_grad_clip", 1.0)),
        optimizer=str(base_cfg.get("aimip_optimizer", "adamw")),
        warmup_steps=int(base_cfg.get("aimip_warmup", 5)),
        early_stop_patience=int(base_cfg.get("aimip_patience", 2)),
        early_stop_min_delta=float(base_cfg.get("aimip_min_delta", 1.0e-3)),
        n_train_days=int(base_cfg["n_train_days"]),
        start_year=int(base_cfg.get("train_year", 2015)),
        windows=tuple(
            tuple(int(x) for x in w[:3])
            for w in (base_cfg.get("train_windows") or ())
        ) or None,
        loss_config=loss_config,
        log_every=int(base_cfg.get("log_every", 1)),
    )


def _per_var_metrics(pred_grid, target_carry, grid):
    """Per-variable area-weighted RMSE + bias at mid-level."""
    import numpy as np

    weights = jnp.asarray(grid.weights)

    def _rmse(p, t):
        sq = (p - t) ** 2
        return float(jnp.sqrt(jnp.sum(sq * weights[:, None]) / jnp.sum(
            weights[:, None] * jnp.ones(sq.shape[-1:])[None, :]
        )))

    def _bias(p, t):
        d = p - t
        return float(jnp.sum(d * weights[:, None]) / jnp.sum(
            weights[:, None] * jnp.ones(d.shape[-1:])[None, :]
        ))

    nlev = pred_grid["T"].shape[-1]
    mid = nlev // 2
    out = {}
    for v, p_arr, t_arr in (
        ("T",   pred_grid["T"][..., mid],   target_carry.T[..., mid]),
        ("u",   pred_grid["u"][..., mid],   target_carry.u[..., mid]),
        ("v",   pred_grid["v"][..., mid],   target_carry.v[..., mid]),
        ("p_s", pred_grid["p_s"],           target_carry.p_s),
    ):
        out[v] = {"rmse": _rmse(p_arr, t_arr), "bias": _bias(p_arr, t_arr)}
    return out


def _train_and_eval(schemes: dict, spec_cfg, grid, sigma, base_cfg, cache_dir):
    """Train the classical AIMIP pipeline with the given scheme overrides,
    evaluate on the held-out eval windows, return metrics."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        compute_spectral_filter,
        compute_sponge_factor,
        spectral_pe_to_grid,
    )
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.neural_gcm_spectral import (
        _train_spectral_loop,
        load_training_data,
        spectral_rollout,
    )

    params = AIMIPClassicalParams.from_defaults()

    eval_windows = tuple(
        tuple(int(x) for x in w[:3])
        for w in (base_cfg.get("eval_windows") or ())
    ) or None

    train_ic, train_targ, _ic_times = load_training_data(
        spec_cfg, grid, sigma, cache_dir, windows=spec_cfg.windows,
        host_resident=True,   # non-chunked full-dataset load (#1155)
    )

    dt = spec_cfg.dt
    radiation = str(base_cfg.get("aimip_radiation", "gray"))
    rad_update_interval = int(base_cfg.get("aimip_rad_update_interval", 6))
    # Classical-mode radiation pin (campaign_driver, D1): ablations are
    # classical scheme-swap runs, so they hold radiation at rrtmgp unless
    # the config carries the explicit escape.
    from legoesm.training.campaign_driver import validate_classical_radiation
    validate_classical_radiation(
        radiation,
        smoke=bool(base_cfg.get("smoke", False)),
        allow_non_rrtmgp=bool(base_cfg.get("allow_non_rrtmgp", False)),
    )

    def _make_physics_fn(p, grid_):
        return make_aimip_classical_spectral_physics(
            p, grid_, dt,
            radiation=radiation,
            rad_update_interval_steps=rad_update_interval,
            # This driver's ``microphysics`` axis includes "none" — dropping a
            # family IS one of the arms it is comparing. The completeness gate
            # guards models that claim to be complete, so an ablation sweep
            # declares the waiver instead of tripping over it.
            allow_unfilled_families=True,
            **schemes,
        )

    params_trained, loss_history = _train_spectral_loop(
        params, _make_physics_fn,
        grid, sigma, train_ic, train_targ, spec_cfg,
        host_staged=True,   # dataset loaded host-resident above (#1155)
    )

    # Eval on held-out windows.
    eval_cfg = spec_cfg._replace(
        n_train_days=int(base_cfg.get("n_eval_days", 8)),
        start_year=int(base_cfg.get("eval_year", 2017)),
        windows=eval_windows,
    )
    ic_states, target_carries, _ic_times = load_training_data(
        eval_cfg, grid, sigma, cache_dir, windows=eval_windows,
    )

    pe_config = spec_cfg.pe_config
    sponge_factor = None
    if pe_config.sponge_tau > 0:
        sponge_factor = compute_sponge_factor(
            sigma.sigma_full, pe_config.sponge_sigma,
            pe_config.sponge_tau, spec_cfg.dt,
        )
    spectral_filter = None
    if pe_config.spectral_filter_strength > 0:
        spectral_filter = compute_spectral_filter(
            grid.ls, grid.n_max,
            order=pe_config.spectral_filter_order,
            cutoff_fraction=pe_config.spectral_filter_strength,
        )
    n_steps_per_day = int(86400 / spec_cfg.dt)

    physics_fn = make_aimip_classical_spectral_physics(
        params_trained, grid, dt,
        radiation=radiation,
        rad_update_interval_steps=rad_update_interval,
        **schemes,
        # Same waiver as the training call above: one arm of this sweep
        # deliberately runs without microphysics.
        allow_unfilled_families=True,
    )

    per_var_acc = {v: {"rmse": [], "bias": []} for v in ("T", "u", "v", "p_s")}
    eval_losses = []
    for ic, target in zip(ic_states, target_carries):
        pred = spectral_rollout(
            ic, physics_fn, grid, sigma, pe_config,
            spec_cfg.dt, n_steps_per_day,
            sponge_factor, spectral_filter,
        )
        from legoesm.training.neural_gcm_spectral import (
            spectral_state_vs_carry_loss,
        )
        eval_losses.append(float(spectral_state_vs_carry_loss(
            pred, target, grid, sigma, jnp.asarray(sigma.sigma_full),
            spec_cfg.loss_config,
        )))
        m = _per_var_metrics(spectral_pe_to_grid(pred, grid, sigma), target, grid)
        for v in per_var_acc:
            per_var_acc[v]["rmse"].append(m[v]["rmse"])
            per_var_acc[v]["bias"].append(m[v]["bias"])

    per_var_mean = {
        v: {
            "rmse": float(sum(per_var_acc[v]["rmse"]) / len(per_var_acc[v]["rmse"])),
            "bias": float(sum(per_var_acc[v]["bias"]) / len(per_var_acc[v]["bias"])),
        }
        for v in per_var_acc
    }

    return {
        "loss_history": [float(x) for x in loss_history],
        "eval_loss_mean": float(sum(eval_losses) / max(1, len(eval_losses))),
        "n_eval_samples": len(eval_losses),
        "metrics": per_var_mean,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path,
        default=Path("config/aimip/aimip_suite.yaml"),
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results/aimip_001"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not jax.config.x64_enabled:
        print("JAX_ENABLE_X64=1 required", file=sys.stderr)
        sys.exit(1)

    suite = _load_yaml(args.suite)
    base = _load_yaml(Path(suite["base"]))
    cache_dir = base.get("cache_dir", ".cache/aimip_era5")

    spec_cfg = _build_spectral_config(base)

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top,
    )

    args.results.mkdir(parents=True, exist_ok=True)
    scorecard_path = args.results / "aimip_ablation_scorecard.json"
    scorecard = {
        "stages": [],
        "current": dict(INITIAL),
        "attempts": {},
    }

    current = dict(INITIAL)

    # ------------------------------------------------------------------
    # Baseline (current scheme set, fresh train).
    # ------------------------------------------------------------------
    logger.info(f"{'='*60}\nBASELINE: {current}\n{'='*60}")
    t0 = time.time()
    try:
        baseline_metrics = _train_and_eval(
            current, spec_cfg, grid, sigma, base, cache_dir,
        )
        baseline_metrics["status"] = "ok"
        baseline_metrics["schemes"] = dict(current)
        baseline_metrics["seconds"] = time.time() - t0
    except Exception as e:
        logger.exception("baseline failed")
        baseline_metrics = {
            "status": "failed", "error": str(e),
            "schemes": dict(current),
            "seconds": time.time() - t0,
        }
    scorecard["attempts"]["baseline"] = baseline_metrics
    with scorecard_path.open("w") as fh:
        json.dump(scorecard, fh, indent=2)

    best_objective = (
        baseline_metrics.get("metrics", {}).get("T", {}).get("rmse", float("inf"))
        + abs(baseline_metrics.get("metrics", {}).get("T", {}).get("bias", 0.0))
        if baseline_metrics["status"] == "ok" else float("inf")
    )
    logger.info(
        f"baseline: T RMSE={baseline_metrics.get('metrics',{}).get('T',{}).get('rmse','?')} "
        f"T bias={baseline_metrics.get('metrics',{}).get('T',{}).get('bias','?')} "
        f"objective={best_objective:.4f}"
    )

    # ------------------------------------------------------------------
    # Stage loop.
    # ------------------------------------------------------------------
    for stage_name, dim, candidates in ABLATIONS:
        stage_results: dict[str, dict] = {}
        for cand in candidates:
            if cand == current[dim]:
                # Already evaluated as the previous winner (or baseline).
                continue
            schemes = dict(current)
            schemes[dim] = cand
            label = f"{stage_name}_{cand}"
            logger.info(f"{'='*60}\n[{label}] schemes={schemes}\n{'='*60}")
            t0 = time.time()
            try:
                m = _train_and_eval(schemes, spec_cfg, grid, sigma, base, cache_dir)
                m["status"] = "ok"
                m["schemes"] = dict(schemes)
                m["seconds"] = time.time() - t0
            except Exception as e:
                logger.exception(f"[{label}] failed")
                m = {
                    "status": "failed", "error": str(e),
                    "schemes": dict(schemes),
                    "seconds": time.time() - t0,
                }
            stage_results[label] = m
            scorecard["attempts"][label] = m
            with scorecard_path.open("w") as fh:
                json.dump(scorecard, fh, indent=2)

        # Pick stage winner: minimize (T RMSE + |T bias|) among ok attempts.
        ok_attempts = {
            k: v for k, v in stage_results.items() if v["status"] == "ok"
        }
        if ok_attempts:
            def _obj(m):
                T = m["metrics"]["T"]
                return T["rmse"] + abs(T["bias"])
            best_label, best_metrics = min(
                ok_attempts.items(), key=lambda kv: _obj(kv[1]),
            )
            best_obj = _obj(best_metrics)
            if best_obj < best_objective:
                # Stage winner improves over current.
                winner = best_metrics["schemes"][dim]
                current[dim] = winner
                best_objective = best_obj
                logger.info(
                    f"[stage {stage_name}] winner: {winner} "
                    f"(obj {best_obj:.4f})"
                )
            else:
                logger.info(
                    f"[stage {stage_name}] no improvement over "
                    f"current ({current[dim]}); keeping current."
                )
        else:
            logger.warning(f"[stage {stage_name}] all attempts failed")

        scorecard["stages"].append({
            "stage": stage_name,
            "current_after": dict(current),
            "objective_after": best_objective,
        })
        scorecard["current"] = dict(current)
        with scorecard_path.open("w") as fh:
            json.dump(scorecard, fh, indent=2)

    logger.info(f"FINAL winning scheme set: {current}")
    logger.info(f"FINAL objective (T RMSE + |T bias|): {best_objective:.4f}")


if __name__ == "__main__":
    main()
