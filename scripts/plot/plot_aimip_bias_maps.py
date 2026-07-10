#!/usr/bin/env python
"""Plot AIMIP per-variant bias maps for historical + held-out periods.

For each AIMIP variant under ``results/aimip_001/<variant>/params.eqx``
this script:

1. Loads the trained model.
2. Samples one day per year from ``historical_years`` (default 2015,
   2016 — the AIMIP training years) and ``future_years``
   (default 2017..2025 — the held-out years).
3. Runs a single-day spectral rollout from each sampled IC and
   computes ``pred - target`` for T (mid-level), u (mid-level),
   v (mid-level), and surface pressure.
4. Averages biases per grid point within each period and saves a
   per-variable PNG to the results directory.

Outputs (under ``results/aimip_001/``):

* ``aimip_bias_T_map.png``     — 3 rows (variants) × 2 cols (historical,
                                 future) of mid-level T bias [K].
* ``aimip_bias_u_map.png``     — same layout, mid-level u bias [m/s].
* ``aimip_bias_v_map.png``     — same layout, mid-level v bias [m/s].
* ``aimip_bias_p_s_map.png``   — same layout, surface pressure bias [Pa].

Usage::

    JAX_ENABLE_X64=1 .venv/bin/python scripts/plot_aimip_bias_maps.py \\
        --results results/aimip_001 \\
        --suite   config/aimip/aimip_suite.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


logger = logging.getLogger("aimip-bias-maps")

_VARIABLES = ("T", "u", "v", "p_s")
_VAR_UNITS = {"T": "K", "u": "m/s", "v": "m/s", "p_s": "Pa"}
_VAR_VMAX = {"T": 8.0, "u": 12.0, "v": 12.0, "p_s": 3000.0}
# RMSE map colour scale (positive only; per-cell RMSE in same units).
_VAR_VMAX_RMSE = {"T": 8.0, "u": 12.0, "v": 12.0, "p_s": 3000.0}


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _build_spectral_config(base_cfg: dict):
    from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig

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
        sfno_embed_dim=int(base_cfg.get("sfno_embed_dim", 128)),
        sfno_n_blocks=int(base_cfg.get("sfno_n_blocks", 4)),
        sfno_mlp_expansion=int(base_cfg.get("sfno_mlp_expansion", 4)),
        n_epochs=1,
        lr=float(base_cfg.get("aimip_lr", 3.0e-4)),
        weight_decay=float(base_cfg.get("aimip_weight_decay", 1.0e-5)),
        grad_clip_norm=float(base_cfg.get("aimip_grad_clip", 1.0)),
        optimizer=str(base_cfg.get("aimip_optimizer", "adamw")),
        warmup_steps=int(base_cfg.get("aimip_warmup", 5)),
        n_train_days=1,
        start_year=2015,
        loss_config=LossConfig(),
    )


def _build_grid_and_sigma(spec_cfg):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top,
    )
    return grid, sigma


# ----------------------------------------------------------------------
# Variant model loaders
# ----------------------------------------------------------------------

def _load_classical(ckpt_path: Path):
    from legoesm.training.aimip_params import AIMIPClassicalParams
    from legoesm.ml.training import load_checkpoint

    template = AIMIPClassicalParams.from_defaults()
    return load_checkpoint(template, ckpt_path)


def _load_column_nn(ckpt_path: Path, spec_cfg):
    from legoesm.atmosphere.physics.learned_column import build_column_physics
    from legoesm.ml.training import load_checkpoint

    template = build_column_physics(
        nlev=spec_cfg.n_levels,
        hidden_dim=256,
        n_layers=4,
        residual_scale=0.01,
        key=jax.random.PRNGKey(0),
    )
    return load_checkpoint(template, ckpt_path)


def _load_sfno(ckpt_path: Path, spec_cfg, grid, variant="sfno_physics"):
    from legoesm.ml.channel_packing import PE3DChannelSpec
    from legoesm.ml.sfno import SFNO, SFNOConfig
    from legoesm.ml.training import load_checkpoint
    from legoesm.training.neural_gcm_spectral import N_SFNO_FORCING_CHANNELS
    # sfno_physics inputs carry the surface-forcing planes (T_sfc/sic/
    # insolation); sfno_full (whole-atmosphere emulator) stays state-only.
    n_forcing = N_SFNO_FORCING_CHANNELS if variant == "sfno_physics" else 0

    spec = PE3DChannelSpec(nlev=spec_cfg.n_levels)
    template = SFNO(
        SFNOConfig(
            in_channels=spec.n_channels + n_forcing,
            out_channels=spec.n_channels,
            embed_dim=spec_cfg.sfno_embed_dim,
            n_blocks=spec_cfg.sfno_n_blocks,
            mlp_expansion=spec_cfg.sfno_mlp_expansion,
            residual_prediction=False,
        ),
        grid,
        key=jax.random.PRNGKey(0),
    )
    return load_checkpoint(template, ckpt_path)


def _physics_fn_for(variant: str, model, grid, spec_cfg, base_cfg):
    from legoesm.training.aimip_params import (
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.neural_gcm_spectral import (
        make_column_mlp_spectral_physics,
        make_sfno_spectral_physics,
    )

    if variant == "classical":
        return make_aimip_classical_spectral_physics(
            model, grid, spec_cfg.dt,
            radiation=str(base_cfg.get("aimip_radiation", "gray")),
            rad_update_interval_steps=int(
                base_cfg.get("aimip_rad_update_interval", 6),
            ),
        )
    if variant == "column_nn":
        return make_column_mlp_spectral_physics(model, grid)
    if variant in ("sfno_physics", "sfno_full"):
        return make_sfno_spectral_physics(model, grid)
    raise ValueError(f"Unknown variant: {variant!r}")


# ----------------------------------------------------------------------
# Bias computation
# ----------------------------------------------------------------------

def _build_year_windows(years: list[int], n_days_per_year: int) -> list[tuple]:
    """One window per year (winter window, Jan).  ``n_days_per_year`` IC/target
    pairs per year for spatial averaging."""
    return [(int(y), 0, int(n_days_per_year)) for y in years]


def _compute_bias_for_variant(
    variant: str,
    model,
    grid,
    sigma,
    spec_cfg,
    base_cfg,
    historical_years: list[int],
    future_years: list[int],
    n_days_per_year: int,
    cache_dir: str,
):
    """Return dict[period -> dict[var -> (n_lat, n_lon) bias array]]."""
    from legoesm.atmosphere.dynamics.spectral_pe import (
        compute_spectral_filter,
        compute_sponge_factor,
        spectral_pe_to_grid,
    )
    from legoesm.training.neural_gcm_spectral import (
        load_training_data,
        spectral_rollout,
    )

    physics_fn = _physics_fn_for(variant, model, grid, spec_cfg, base_cfg)

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
    # Honor the training rollout horizon for eval so bias/RMSE maps
    # reflect the model's actual forecast skill at its supervised
    # horizon.  v12 uses 6-hour pairs; if the config sets
    # ``aimip_rollout_hours`` we use that, else fall back to a 1-day
    # rollout (legacy behaviour).  ERA5 cadence is 6 h so the
    # period_cfg loader also needs rollout_hours threaded in.
    n_steps_per_day = int(86400 / spec_cfg.dt)
    rollout_hours_cfg = int(base_cfg.get("aimip_rollout_hours", 0) or 0)
    rollout_hours_eval = rollout_hours_cfg if rollout_hours_cfg > 0 else 24
    n_steps_eval = int(round(rollout_hours_eval * 3600.0 / spec_cfg.dt))

    out: dict[str, dict[str, np.ndarray]] = {}
    for period_name, years in (
        ("historical", historical_years),
        ("future", future_years),
    ):
        windows = _build_year_windows(years, n_days_per_year)
        period_cfg = spec_cfg._replace(
            n_train_days=sum(w[2] for w in windows),
            windows=tuple(windows),
            rollout_hours=rollout_hours_eval,
        )
        logger.info(
            f"[{variant}] loading {period_name} windows "
            f"({len(years)} years × {n_days_per_year} days)..."
        )
        ic_states, target_carries, _ic_times = load_training_data(
            period_cfg, grid, sigma, cache_dir, windows=windows,
        )

        sum_bias = {v: None for v in _VARIABLES}
        sum_sq = {v: None for v in _VARIABLES}
        count = 0
        for ic, target in zip(ic_states, target_carries):
            pred = spectral_rollout(
                ic, physics_fn, grid, sigma, pe_config,
                spec_cfg.dt, n_steps_eval,
                sponge_factor, spectral_filter,
            )
            pred_grid = spectral_pe_to_grid(pred, grid, sigma)
            nlev = pred_grid["T"].shape[-1]
            mid = nlev // 2
            sample = {
                "T":   np.asarray(pred_grid["T"][..., mid] - target.T[..., mid]),
                "u":   np.asarray(pred_grid["u"][..., mid] - target.u[..., mid]),
                "v":   np.asarray(pred_grid["v"][..., mid] - target.v[..., mid]),
                "p_s": np.asarray(pred_grid["p_s"]         - target.p_s),
            }
            for v in _VARIABLES:
                sum_bias[v] = sample[v] if sum_bias[v] is None else (sum_bias[v] + sample[v])
                sample_sq = sample[v] ** 2
                sum_sq[v] = sample_sq if sum_sq[v] is None else (sum_sq[v] + sample_sq)
            count += 1
        # Per-cell RMSE = sqrt(mean(error**2)) across samples.
        out[period_name] = {
            "bias": {v: sum_bias[v] / max(count, 1) for v in _VARIABLES},
            "rmse": {
                v: np.sqrt(sum_sq[v] / max(count, 1)) for v in _VARIABLES
            },
        }
        logger.info(
            f"[{variant}] {period_name}: averaged over {count} samples"
        )
    return out


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _plot_one_variable(
    var: str,
    biases: dict[str, dict[str, dict[str, np.ndarray]]],
    grid,
    out_path: Path,
    historical_label: str = "2015-2016",
    future_label: str = "2017-2022",
    metric: str = "bias",
):
    """Map: rows = variants, cols = [historical, future]; metric = bias or rmse."""
    variants = list(biases.keys())
    n_rows = len(variants)
    n_cols = 2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(11, 2.6 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[None, :]

    lon = np.rad2deg(np.asarray(grid.lon))
    lat = np.rad2deg(np.asarray(grid.lat))
    if metric == "bias":
        vmax = _VAR_VMAX[var]
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = "RdBu_r"
        cbar_label = f"{var} bias (pred − target) [{_VAR_UNITS[var]}]"
        title_metric = "bias"
    else:  # rmse
        vmax = _VAR_VMAX_RMSE[var]
        norm = None
        cmap = "viridis"
        cbar_label = f"{var} RMSE (per-cell) [{_VAR_UNITS[var]}]"
        title_metric = "RMSE"

    im = None
    for i, variant in enumerate(variants):
        for j, period in enumerate(("historical", "future")):
            arr = biases[variant][period][metric][var]
            ax = axes[i, j]
            if metric == "rmse":
                im = ax.pcolormesh(
                    lon, lat, arr, cmap=cmap, vmin=0.0, vmax=vmax,
                    shading="auto",
                )
            else:
                im = ax.pcolormesh(
                    lon, lat, arr, cmap=cmap, norm=norm, shading="auto",
                )
            ax.set_title(f"{variant} — {period}", fontsize=9)
            if j == 0:
                ax.set_ylabel("lat [°]")
            if i == n_rows - 1:
                ax.set_xlabel("lon [°]")

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes, orientation="horizontal", shrink=0.7,
            pad=0.02, aspect=40,
        )
        cbar.set_label(cbar_label)

    fig.suptitle(
        f"AIMIP — mid-level {var} {title_metric} map "
        f"(historical {historical_label} / future {future_label})",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/aimip_001"),
        help="Directory containing <variant>/params.eqx checkpoints.",
    )
    parser.add_argument(
        "--suite", type=Path,
        default=Path("config/aimip/aimip_suite.yaml"),
    )
    parser.add_argument(
        "--n-days-per-year", type=int, default=2,
        help="IC/target pairs per year per period (more = better average).",
    )
    parser.add_argument(
        "--historical-years", type=int, nargs="+",
        default=[2015, 2016],
    )
    parser.add_argument(
        "--future-years", type=int, nargs="+",
        default=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not jax.config.x64_enabled:
        raise SystemExit(
            "JAX_ENABLE_X64=1 required for spectral transforms."
        )

    suite = _load_yaml(args.suite)
    base = _load_yaml(Path(suite["base"]))
    cache_dir = base.get("cache_dir", ".cache/aimip_era5")

    spec_cfg = _build_spectral_config(base)
    grid, sigma = _build_grid_and_sigma(spec_cfg)

    biases: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for variant in ("classical", "column_nn", "sfno_physics"):
        ckpt = args.results / variant / "params.eqx"
        if not ckpt.exists():
            logger.warning(f"Missing checkpoint, skipping: {ckpt}")
            continue
        logger.info(f"[{variant}] loading {ckpt}")
        if variant == "classical":
            model = _load_classical(ckpt)
        elif variant == "column_nn":
            model = _load_column_nn(ckpt, spec_cfg)
        else:
            model = _load_sfno(ckpt, spec_cfg, grid, variant)

        biases[variant] = _compute_bias_for_variant(
            variant, model, grid, sigma, spec_cfg, base,
            args.historical_years, args.future_years,
            args.n_days_per_year, cache_dir,
        )

    if not biases:
        raise SystemExit("No checkpoints found; nothing to plot.")

    args.results.mkdir(parents=True, exist_ok=True)
    historical_label = f"{min(args.historical_years)}-{max(args.historical_years)}"
    future_label = f"{min(args.future_years)}-{max(args.future_years)}"
    for var in _VARIABLES:
        # Bias map.
        bias_path = args.results / f"aimip_bias_{var}_map.png"
        _plot_one_variable(
            var, biases, grid, bias_path,
            historical_label=historical_label,
            future_label=future_label,
            metric="bias",
        )
        logger.info(f"Wrote {bias_path}")
        # RMSE map.
        rmse_path = args.results / f"aimip_rmse_{var}_map.png"
        _plot_one_variable(
            var, biases, grid, rmse_path,
            historical_label=historical_label,
            future_label=future_label,
            metric="rmse",
        )
        logger.info(f"Wrote {rmse_path}")


if __name__ == "__main__":
    main()
