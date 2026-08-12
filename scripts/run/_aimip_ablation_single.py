#!/usr/bin/env python
"""Single-variant AIMIP classical ablation worker.

Runs one (scheme set) combination, trains from defaults end-to-end on
the windowed AIMIP dataset, evaluates against the held-out eval
windows, and writes per-variable metrics to an output JSON.

Designed to be invoked once per variant by an outer shell loop so
each variant runs in a fresh Python/JAX process — avoids
cumulative JAX compilation-cache memory pressure across variants.

Usage::

    JAX_ENABLE_X64=1 python scripts/_aimip_ablation_single.py \\
        --label gwd_lindzen \\
        --gwd-scheme lindzen \\
        --convection-scheme tiedtke \\
        --turbulence-scheme louis \\
        --microphysics-scheme none \\
        --cloud-scheme xu_randall \\
        --suite config/aimip/aimip_suite.yaml \\
        --out results/aimip_001/ablation/gwd_lindzen.json
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

from legoesm.ml.loss import latitude_weighted_bias, latitude_weighted_rmse


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
    weights = jnp.asarray(grid.weights)

    def _rmse(p, t):
        return float(latitude_weighted_rmse(p, t, weights))

    def _bias(p, t):
        return float(latitude_weighted_bias(p, t, weights))

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gwd-scheme",          default="mcfarlane")
    parser.add_argument("--convection-scheme",   default="tiedtke")
    parser.add_argument("--turbulence-scheme",   default="louis")
    parser.add_argument("--surface-bulk-scheme", default="constant")
    parser.add_argument("--microphysics-scheme", default="none")
    parser.add_argument("--cloud-scheme",        default="xu_randall")
    parser.add_argument("--suite", type=Path,
                        default=Path("config/aimip/aimip_suite.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(f"aimip-{args.label}")
    if not jax.config.x64_enabled:
        print("JAX_ENABLE_X64=1 required", file=sys.stderr)
        sys.exit(1)

    suite = _load_yaml(args.suite)
    base = _load_yaml(Path(suite["base"]))
    cache_dir = base.get("cache_dir", ".cache/aimip_era5")

    spec_cfg = _build_spectral_config(base)

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        compute_spectral_filter,
        compute_sponge_factor,
        spectral_pe_to_grid,
    )
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.neural_gcm_spectral import (
        _train_spectral_loop,
        load_training_data,
        spectral_rollout,
        spectral_state_vs_carry_loss,
    )

    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top)

    schemes = {
        "convection_scheme":   args.convection_scheme,
        "turbulence_scheme":   args.turbulence_scheme,
        "surface_bulk_scheme": args.surface_bulk_scheme,
        "gwd_scheme":          args.gwd_scheme,
        "microphysics_scheme": args.microphysics_scheme,
        "cloud_scheme":        args.cloud_scheme,
        # One arm of a scheme ablation may drop a family on purpose (the CLI
        # accepts --microphysics-scheme none); declare it rather than trip the
        # completeness gate, which guards models that claim to be complete.
        "allow_unfilled_families": True,
    }
    logger.info(f"[{args.label}] schemes = {schemes}")

    payload = {"label": args.label, "schemes": dict(schemes)}
    t0 = time.time()
    try:
        # ---- Train ----
        params = AIMIPClassicalParams.from_defaults()
        train_ic, train_targ, _ic_times = load_training_data(
            spec_cfg, grid, sigma, cache_dir, windows=spec_cfg.windows,
            host_resident=True,   # non-chunked full-dataset load (#1155)
        )
        dt = spec_cfg.dt
        radiation = str(base.get("aimip_radiation", "rrtmgp"))
        rad_update_interval = int(base.get("aimip_rad_update_interval", 6))

        def _make_physics_fn(p, grid_):
            return make_aimip_classical_spectral_physics(
                p, grid_, dt,
                radiation=radiation,
                rad_update_interval_steps=rad_update_interval,
                **schemes,
            )

        params_trained, loss_history = _train_spectral_loop(
            params, _make_physics_fn,
            grid, sigma, train_ic, train_targ, spec_cfg,
            host_staged=True,   # dataset loaded host-resident above (#1155)
        )

        # ---- Eval ----
        eval_windows = tuple(
            tuple(int(x) for x in w[:3])
            for w in (base.get("eval_windows") or ())
        ) or None
        eval_cfg = spec_cfg._replace(
            n_train_days=int(base.get("n_eval_days", 8)),
            start_year=int(base.get("eval_year", 2017)),
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
        )

        per_var_acc = {v: {"rmse": [], "bias": []}
                       for v in ("T", "u", "v", "p_s")}
        eval_losses = []
        for ic, target in zip(ic_states, target_carries):
            pred = spectral_rollout(
                ic, physics_fn, grid, sigma, pe_config,
                spec_cfg.dt, n_steps_per_day,
                sponge_factor, spectral_filter,
            )
            eval_losses.append(float(spectral_state_vs_carry_loss(
                pred, target, grid, sigma, jnp.asarray(sigma.sigma_full),
                spec_cfg.loss_config,
            )))
            m = _per_var_metrics(
                spectral_pe_to_grid(pred, grid, sigma), target, grid,
            )
            for v in per_var_acc:
                per_var_acc[v]["rmse"].append(m[v]["rmse"])
                per_var_acc[v]["bias"].append(m[v]["bias"])

        per_var_mean = {
            v: {
                "rmse": float(sum(per_var_acc[v]["rmse"])
                              / len(per_var_acc[v]["rmse"])),
                "bias": float(sum(per_var_acc[v]["bias"])
                              / len(per_var_acc[v]["bias"])),
            }
            for v in per_var_acc
        }

        payload.update({
            "status": "ok",
            "loss_history": [float(x) for x in loss_history],
            "eval_loss_mean": float(
                sum(eval_losses) / max(1, len(eval_losses))
            ),
            "metrics": per_var_mean,
            "n_eval_samples": len(eval_losses),
            "seconds": time.time() - t0,
        })
    except Exception as exc:
        logger.exception("variant failed")
        payload.update({
            "status": "failed",
            "error": repr(exc),
            "seconds": time.time() - t0,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info(f"[{args.label}] wrote {args.out}  status={payload['status']}")


if __name__ == "__main__":
    main()
