#!/usr/bin/env python
"""AIMIP intercomparison driver for legoESM.

Trains and evaluates a fixed set of atmospheric-physics variants on
the same ERA5 IC/target window, writing a unified scorecard JSON.

All variants share the spectral primitive-equation dynamical core
(``SpectralPrimitiveEquationModel``) on a Gaussian grid so the
comparison reflects the choice of physics representation rather than
the dycore.  The variants are:

* ``classical`` — Tiedtke convection, Louis turbulence, surface bulk
  fluxes, McFarlane gravity-wave drag, Xu-Randall cloud fraction.
  Tunables exposed via :class:`legoesm.training.aimip_params.AIMIPClassicalParams`
  and trained end-to-end through the differentiable spectral PE
  rollout.
* ``column_nn`` — column MLP physics (Rasp et al., 2018 style) via
  :func:`legoesm.training.neural_gcm_spectral.train_column_mlp_spectral`.
* ``sfno_physics`` — SFNO replaces the gridded physics step via
  :func:`legoesm.training.neural_gcm_spectral.train_neural_gcm_spectral`.
* ``sfno_full`` — alias of ``sfno_physics`` (kept for forward-compat
  once an end-to-end spectral-SFNO mode lands).

Usage
-----
::

    JAX_ENABLE_X64=1 python scripts/run_aimip.py \\
        --suite config/aimip/aimip_suite.yaml [--smoke]

The ``--smoke`` flag overrides the suite to T21 / 8 levels / 1 epoch /
2 train days so the full intercomparison fits in a few minutes for
end-to-end verification.

Optimization
------------
All variants flow gradients through ``eqx.filter_value_and_grad`` and
default to MUON via :func:`legoesm.ml.training.create_optimizer`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml


logger = logging.getLogger("aimip")


from legoesm.driver.config import AIMIP_VARIANTS as _AIMIP_VARIANTS_FULL
from legoesm.ml.loss import latitude_weighted_bias, latitude_weighted_rmse

# Run-time variants (drop the empty string which means "not AIMIP").
_VALID_VARIANTS = tuple(v for v in _AIMIP_VARIANTS_FULL if v)


# ----------------------------------------------------------------------
# YAML loading + overlay merge
# ----------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Shallow dict merge (overlay keys override base)."""
    out = dict(base)
    out.update(overlay)
    return out


def _apply_smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg.update(dict(
        n_max=21,
        nlev=8,
        n_train_days=2,
        n_eval_days=1,
        aimip_n_epochs=1,
        aimip_warmup=1,
        dt=1800.0,
        # Force the cheap radiation backend + single-day rollout for
        # smoke runs.  The full AIMIP base config defaults to RRTMGP +
        # spatial surface fields for production training, but those
        # add ~10x compile cost and aren't useful for smoke-level
        # end-to-end verification.
        aimip_radiation="gray",
        aimip_spatial_surface=False,
        aimip_rollout_days=1,
    ))
    return cfg


# ----------------------------------------------------------------------
# Variant dispatch
# ----------------------------------------------------------------------

def _build_spectral_config(cfg: dict[str, Any]):
    """Translate AIMIP YAML dict into NeuralGCMSpectralConfig."""
    from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
    from legoesm.training.losses import LossConfig

    loss_kwargs = cfg.get("loss", {}) or {}
    loss_config = LossConfig(**{k: v for k, v in loss_kwargs.items()
                                if k in LossConfig._fields})

    sfno_embed = int(cfg.get("sfno_embed_dim", 128))
    sfno_n_blocks = int(cfg.get("sfno_n_blocks", 4))
    sfno_mlp_expansion = int(cfg.get("sfno_mlp_expansion", 4))

    return NeuralGCMSpectralConfig(
        n_max=int(cfg["n_max"]),
        n_levels=int(cfg["nlev"]),
        dt=float(cfg["dt"]),
        pe_config=SpectralPEConfig(
            hyperdiff_coeff=2.5e15,
            hyperdiff_order=2,
            time_integrator="ssp_rk3",
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
        ),
        sfno_embed_dim=sfno_embed,
        sfno_n_blocks=sfno_n_blocks,
        sfno_mlp_expansion=sfno_mlp_expansion,
        n_epochs=int(cfg["aimip_n_epochs"]),
        lr=float(cfg["aimip_lr"]),
        weight_decay=float(cfg["aimip_weight_decay"]),
        grad_clip_norm=float(cfg["aimip_grad_clip"]),
        optimizer=str(cfg.get("aimip_optimizer", "muon")),
        warmup_steps=int(cfg.get("aimip_warmup", 100)),
        n_train_days=int(cfg["n_train_days"]),
        start_year=int(cfg.get("train_year", cfg.get("years", [2015])[0])),
        early_stop_patience=int(cfg.get("aimip_patience", 0)),
        early_stop_min_delta=float(cfg.get("aimip_min_delta", 1.0e-3)),
        windows=tuple(
            tuple(int(x) for x in w[:3])
            for w in (cfg.get("train_windows") or ())
        ) or None,
        rollout_days=int(cfg.get("aimip_rollout_days", 1)),
        rollout_hours=int(cfg.get("aimip_rollout_hours", 0)),
        spatial_lr_scale=float(cfg.get("aimip_spatial_lr_scale", 1.0)),
        rad_update_interval=int(cfg.get("aimip_rad_update_interval", 1)),
        loss_config=loss_config,
        log_every=int(cfg.get("log_every", 1)),
        checkpoint_dir=str(Path(cfg["output_dir"]) / cfg["aimip_variant"]),
    )


def _train_variant(variant: str, cfg: dict[str, Any], cache_dir: str):
    """Dispatch on AIMIP variant.  Returns (trained_model, loss_history)."""
    spec_cfg = _build_spectral_config(cfg)

    if variant == "classical":
        return _train_aimip_classical(spec_cfg, cache_dir, cfg=cfg)

    if variant == "column_nn":
        from legoesm.training.neural_gcm_spectral import (
            train_column_mlp_spectral,
        )
        return train_column_mlp_spectral(
            config=spec_cfg,
            cache_dir=cache_dir,
            seed=int(cfg.get("nn_seed", 0)),
            hidden_dim=int(cfg.get("nn_hidden_dim", 256)),
            n_layers=int(cfg.get("nn_n_layers", 4)),
        )

    if variant in ("sfno_physics", "sfno_full"):
        from legoesm.training.neural_gcm_spectral import (
            train_neural_gcm_spectral,
        )
        return train_neural_gcm_spectral(
            config=spec_cfg,
            cache_dir=cache_dir,
            seed=int(cfg.get("sfno_seed", 0)),
        )

    raise ValueError(f"Unknown AIMIP variant: {variant!r}")


def _train_aimip_classical(spec_cfg, cache_dir: str, *, cfg: dict | None = None):
    cfg = cfg or {}
    """Train the AIMIP classical variant (Tiedtke/Louis/Surface/McFarlane/XR).

    Builds a Gaussian grid + spectral PE dycore, loads ERA5 daily
    IC/target pairs, and runs the shared ``_train_spectral_loop`` with
    :class:`AIMIPClassicalParams` and
    :func:`make_aimip_classical_spectral_physics` as the
    ``make_physics_fn`` factory.
    """
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.neural_gcm_spectral import (
        _train_spectral_loop,
        load_training_data,
    )

    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top,
    )

    spatial_surface = bool(cfg.get("aimip_spatial_surface", False))
    params = AIMIPClassicalParams.from_defaults(
        spatial_surface=spatial_surface,
        spatial_init_std=float(cfg.get("aimip_spatial_init_std", 0.0)),
        spatial_seed=int(cfg.get("aimip_spatial_seed", 0)),
    )
    n_scalar = len(params.raw_values)
    n_spatial = (
        params.spatial_surface.n_trainable() if params.spatial_surface else 0
    )
    logger.info(
        f"AIMIPClassicalParams: {n_scalar} trainable scheme knobs"
        + (
            f" + {n_spatial} spatial coefs across "
            f"{len(params.spatial_surface.fields)} surface fields"
            if spatial_surface else ""
        )
    )

    ic_states, target_carries = load_training_data(
        spec_cfg, grid, sigma, cache_dir,
        windows=spec_cfg.windows,
    )

    dt = spec_cfg.dt
    radiation = str(cfg.get("aimip_radiation", "gray"))
    rad_update_interval = int(cfg.get("aimip_rad_update_interval", 6))

    # Derive the land mask from surface geopotential (phis > 0 over
    # land).  Static across samples so we extract it once.  Using a
    # soft sigmoid keeps the lat-lon surface-parameter gradients
    # smooth across coastlines (vs. a hard step that would clip them).
    land_mask = None
    if spatial_surface and target_carries:
        from legoesm.training.aimip_spatial import land_mask_from_phis
        land_mask = land_mask_from_phis(
            jnp.asarray(target_carries[0].phis), smooth=True,
        )

    # When rad gating is on (``aimip_rad_update_interval > 1``),
    # ``make_aimip_classical_spectral_physics`` returns a
    # ``(non_rad_fn, rad_fn)`` tuple and the rollout uses lax.cond
    # to compute the RRTMGP step periodically.  Otherwise the legacy
    # single-callable path runs.
    split_rad = rad_update_interval > 1

    def _make_physics_fn(p, grid_):
        return make_aimip_classical_spectral_physics(
            p, grid_, dt,
            radiation=radiation,
            rad_update_interval_steps=rad_update_interval,
            land_mask=land_mask,
            split_rad=split_rad,
        )

    return _train_spectral_loop(
        params, _make_physics_fn,
        grid, sigma, ic_states, target_carries, spec_cfg,
    )


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def _evaluate_variant(
    variant: str,
    trained_model,
    cfg: dict[str, Any],
    cache_dir: str,
) -> dict[str, float]:
    """Evaluate trained model on held-out ERA5 days.

    Returns a dict of mean ``LossConfig``-weighted error + per-variable
    RMSE/bias on the eval window.  Both share the same loss as the
    training objective so the scorecard is directly comparable across
    variants.
    """
    from legoesm.atmosphere.dynamics.spectral_pe import (
        _compute_spectral_filter,
        _compute_sponge_factor,
    )
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.aimip_params import (
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.neural_gcm_spectral import (
        load_training_data,
        make_column_mlp_spectral_physics,
        make_sfno_spectral_physics,
        spectral_rollout,
        spectral_state_vs_carry_loss,
    )

    spec_cfg = _build_spectral_config(cfg)
    eval_windows_raw = cfg.get("eval_windows") or ()
    eval_windows = tuple(
        tuple(int(x) for x in w[:3]) for w in eval_windows_raw
    ) or None
    eval_cfg = spec_cfg._replace(
        n_train_days=int(cfg.get("n_eval_days", 2)),
        start_year=int(cfg.get("eval_year", spec_cfg.start_year)),
        windows=eval_windows,
    )

    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top,
    )
    ic_states, target_carries = load_training_data(
        eval_cfg, grid, sigma, cache_dir, windows=eval_cfg.windows,
    )

    pe_config = spec_cfg.pe_config
    sponge_factor = None
    if pe_config.sponge_tau > 0:
        sponge_factor = _compute_sponge_factor(
            sigma.sigma_full, pe_config.sponge_sigma,
            pe_config.sponge_tau, spec_cfg.dt,
        )
    spectral_filter = None
    if pe_config.spectral_filter_strength > 0:
        spectral_filter = _compute_spectral_filter(
            grid.ls, grid.n_max,
            order=pe_config.spectral_filter_order,
            cutoff_fraction=pe_config.spectral_filter_strength,
        )

    sigma_full = jnp.asarray(sigma.sigma_full)
    n_steps_per_day = int(86400 / spec_cfg.dt)
    eval_rollout_days = int(getattr(spec_cfg, "rollout_days", 1) or 1)
    eval_rollout_hours_cfg = int(getattr(spec_cfg, "rollout_hours", 0) or 0)
    eval_rollout_hours = (
        eval_rollout_hours_cfg
        if eval_rollout_hours_cfg > 0
        else eval_rollout_days * 24
    )
    n_steps_eval = int(round(eval_rollout_hours * 3600.0 / spec_cfg.dt))
    eval_rad_interval = int(cfg.get("aimip_rad_update_interval", 1))

    # Build the per-variant physics_fn (model is frozen for eval).
    eval_physics_pair = None  # (non_rad_fn, rad_fn) when split active
    if variant == "classical":
        eval_land_mask = None
        if bool(cfg.get("aimip_spatial_surface", False)) and target_carries:
            from legoesm.training.aimip_spatial import land_mask_from_phis
            eval_land_mask = land_mask_from_phis(
                jnp.asarray(target_carries[0].phis), smooth=True,
            )
        eval_split_rad = eval_rad_interval > 1
        built = make_aimip_classical_spectral_physics(
            trained_model, grid, spec_cfg.dt,
            radiation=str(cfg.get("aimip_radiation", "gray")),
            rad_update_interval_steps=eval_rad_interval,
            land_mask=eval_land_mask,
            split_rad=eval_split_rad,
        )
        if isinstance(built, tuple):
            eval_physics_pair = built
            physics_fn = built[0]  # non-rad; rad threaded separately
        else:
            physics_fn = built
    elif variant == "column_nn":
        physics_fn = make_column_mlp_spectral_physics(trained_model, grid)
    elif variant in ("sfno_physics", "sfno_full"):
        physics_fn = make_sfno_spectral_physics(trained_model, grid)
    else:
        raise ValueError(f"Unknown variant in eval: {variant!r}")

    from legoesm.atmosphere.dynamics.spectral_pe import spectral_pe_to_grid

    losses: list[float] = []
    per_var_rmse: dict[str, list[float]] = {k: [] for k in ("T", "u", "v", "p_s")}
    per_var_bias: dict[str, list[float]] = {k: [] for k in ("T", "u", "v", "p_s")}

    weights = jnp.asarray(grid.weights)

    for ic, target in zip(ic_states, target_carries):
        if eval_physics_pair is not None:
            non_rad_fn, rad_fn = eval_physics_pair
            pred = spectral_rollout(
                ic, non_rad_fn, grid, sigma, pe_config,
                spec_cfg.dt, n_steps_eval,
                sponge_factor, spectral_filter,
                rad_physics_fn=rad_fn,
                rad_update_interval=eval_rad_interval,
            )
        else:
            pred = spectral_rollout(
                ic, physics_fn, grid, sigma, pe_config,
                spec_cfg.dt, n_steps_eval,
                sponge_factor, spectral_filter,
            )
        losses.append(float(
            spectral_state_vs_carry_loss(
                pred, target, grid, sigma, sigma_full, spec_cfg.loss_config,
            )
        ))

        # Per-variable area-weighted RMSE/bias on the Gaussian grid.
        pred_grid = spectral_pe_to_grid(pred, grid, sigma)
        # Target carry stores 3D arrays with axes (n_lat, n_lon, nlev).
        targets_3d = {
            "T": jnp.asarray(target.T),
            "u": jnp.asarray(target.u),
            "v": jnp.asarray(target.v),
        }
        for name, t_arr in targets_3d.items():
            p_arr = pred_grid[name]
            # Mid-level (nlev//2) cross-section for the scorecard so
            # RMSE numbers are comparable to WeatherBench T@500 hPa.
            mid = p_arr.shape[-1] // 2
            per_var_rmse[name].append(
                float(latitude_weighted_rmse(p_arr[..., mid], t_arr[..., mid], weights))
            )
            per_var_bias[name].append(
                float(latitude_weighted_bias(p_arr[..., mid], t_arr[..., mid], weights))
            )
        # Surface pressure (2D).
        p_s_target = jnp.asarray(target.p_s)
        per_var_rmse["p_s"].append(
            float(latitude_weighted_rmse(pred_grid["p_s"], p_s_target, weights))
        )
        per_var_bias["p_s"].append(
            float(latitude_weighted_bias(pred_grid["p_s"], p_s_target, weights))
        )

    def _agg(lst: list[float]) -> dict[str, float]:
        if not lst:
            return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(sum(lst) / len(lst)),
            "min": float(min(lst)),
            "max": float(max(lst)),
        }

    return {
        "n_eval_days": len(losses),
        "loss": _agg(losses),
        "rmse": {k: _agg(v) for k, v in per_var_rmse.items()},
        "bias": {k: _agg(v) for k, v in per_var_bias.items()},
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, required=True,
        help="Path to the AIMIP suite manifest YAML.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Override config with a minimal smoke-test profile.",
    )
    parser.add_argument(
        "--variants", type=str, default="",
        help=(
            "Comma-separated subset of suite variants to run "
            "(overrides the suite manifest list when set)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not jax.config.x64_enabled:
        print(
            "ERROR: JAX x64 mode required for spectral transforms.\n"
            "Re-run with: JAX_ENABLE_X64=1 python scripts/run_aimip.py ...",
            file=sys.stderr,
        )
        sys.exit(1)

    suite = _load_yaml(args.suite)
    base = _load_yaml(Path(suite["base"]))
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = list(suite.get("variants") or ())
    if not variants:
        raise ValueError(f"Suite {args.suite} lists no variants")
    for v in variants:
        if v not in _VALID_VARIANTS:
            raise ValueError(
                f"Unknown AIMIP variant {v!r}; "
                f"expected one of {_VALID_VARIANTS}"
            )

    output_dir = Path(suite.get("output_dir", base.get("output_dir", "results/aimip")))
    if args.smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = base.get("cache_dir", ".cache/aimip_era5")

    results: dict[str, Any] = {
        "suite": str(args.suite),
        "smoke": args.smoke,
        "variants": {},
    }

    for variant in variants:
        overlay = _load_yaml(
            args.suite.parent / f"variant_{variant}.yaml"
        )
        cfg = _merge(base, overlay)
        cfg["aimip_variant"] = variant
        cfg["output_dir"] = str(output_dir)
        if args.smoke:
            cfg = _apply_smoke_overrides(cfg)

        logger.info("=" * 60)
        logger.info(f"AIMIP variant: {variant}")
        logger.info("=" * 60)

        t0 = time.time()
        model, loss_history = _train_variant(variant, cfg, cache_dir)
        train_elapsed = time.time() - t0

        eval_metrics = _evaluate_variant(variant, model, cfg, cache_dir)
        ckpt_path = output_dir / variant / "params.eqx"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        from legoesm.ml.training import save_checkpoint
        save_checkpoint(model, ckpt_path)

        results["variants"][variant] = {
            "train_loss_history": [float(x) for x in loss_history],
            "train_seconds": train_elapsed,
            "eval_metrics": eval_metrics,
            "checkpoint": str(ckpt_path),
        }
        logger.info(
            f"{variant}: train_loss[-1]={loss_history[-1]:.6f}, "
            f"eval_loss={eval_metrics['loss']['mean']:.6f}, "
            f"RMSE T={eval_metrics['rmse']['T']['mean']:.3f}K "
            f"u={eval_metrics['rmse']['u']['mean']:.3f}m/s "
            f"v={eval_metrics['rmse']['v']['mean']:.3f}m/s "
            f"p_s={eval_metrics['rmse']['p_s']['mean']:.1f}Pa, "
            f"train_time={train_elapsed:.1f}s"
        )

    # Merge with any existing scorecard so multiple --variants invocations
    # share one results/.../aimip_scorecard.json.
    scorecard_path = output_dir / "aimip_scorecard.json"
    if scorecard_path.exists():
        with scorecard_path.open() as fh:
            existing = json.load(fh)
        existing_variants = existing.get("variants", {}) if isinstance(existing, dict) else {}
        merged = dict(existing_variants)
        merged.update(results["variants"])
        results["variants"] = merged
    with scorecard_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Wrote AIMIP scorecard: {scorecard_path}")


if __name__ == "__main__":
    main()
