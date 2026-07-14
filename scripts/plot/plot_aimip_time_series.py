#!/usr/bin/env python
"""Plot AIMIP per-variant global-mean temperature time series.

For each AIMIP variant under ``results/<run>/<variant>/params.eqx``
this script:

1. Loads the trained model.
2. Loads a sequence of ``n_days + 1`` consecutive ERA5 daily snapshots
   starting at ``--start-year``, ``--start-day``.
3. Initializes the spectral state from snapshot 0 (the IC).
4. Runs ``n_days`` successive 1-day spectral rollouts, accumulating the
   area-weighted global-mean mid-level T at every saved step.
5. Records the same daily means from the ERA5 snapshots for reference.
6. Plots the four curves (classical, column_nn, sfno_physics, ERA5)
   over time -- the canonical AIMIP-paper-style T(t) diagnostic.

Outputs (under ``--results``):

* ``aimip_T_timeseries.png`` -- global-mean mid-level T [K] vs day.
* ``aimip_T_timeseries.csv`` -- raw data.

Usage::

    JAX_ENABLE_X64=1 .venv/bin/python scripts/plot_aimip_time_series.py \\
        --results results/aimip_001_v10 \\
        --suite   config/aimip/aimip_suite.yaml \\
        --start-year 2017 --start-day 0 --n-days 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("aimip-timeseries")

_VARIANTS = ("classical", "column_nn", "sfno_physics")
_VARIANT_COLOURS = {
    "classical": "tab:blue",
    "column_nn": "tab:orange",
    "sfno_physics": "tab:green",
    "era5": "k",
}


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _build_spectral_config(base_cfg: dict):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
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
# Variant model loaders (mirror plot_aimip_bias_maps.py)
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
# Diagnostics
# ----------------------------------------------------------------------

def _area_weighted_mid_T(field_T: np.ndarray, lat_weights: np.ndarray) -> float:
    """Global mid-level T mean weighted by Gaussian latitude weights."""
    nlev = field_T.shape[-1]
    mid = nlev // 2
    zonal = np.mean(field_T[..., mid], axis=-1)  # (n_lat,)
    return float(np.sum(zonal * lat_weights) / np.sum(lat_weights))


def _global_mean_3d(field_3d: np.ndarray, lat_weights: np.ndarray,
                    lev_w: np.ndarray) -> float:
    """Global mean of a (n_lat, n_lon, nlev) field with lat + level weights."""
    zonal = np.mean(field_3d, axis=1)  # (n_lat, nlev)
    lev_mean = np.sum(zonal * lev_w, axis=-1) / np.sum(lev_w)  # (n_lat,)
    return float(np.sum(lev_mean * lat_weights) / np.sum(lat_weights))


def _run_variant_series(
    variant: str,
    model,
    grid,
    sigma,
    spec_cfg,
    base_cfg,
    ic_state,
    n_days: int,
    lat_weights: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Return (mid_T series, mass-weighted column-mean T series) over n_days+1 days."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        compute_spectral_filter,
        compute_sponge_factor,
        spectral_pe_to_grid,
    )
    from legoesm.training.neural_gcm_spectral import spectral_rollout

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
    n_steps_per_day = int(86400 / spec_cfg.dt)

    # Uniform level weights for column-mean diagnostic (mass weighting
    # would need ps from each rollout state; keeping it simple here).
    lev_w = np.ones(spec_cfg.n_levels)

    fields0 = spectral_pe_to_grid(ic_state, grid, sigma)
    mid_T_series = [_area_weighted_mid_T(np.asarray(fields0["T"]), lat_weights)]
    col_T_series = [_global_mean_3d(np.asarray(fields0["T"]), lat_weights, lev_w)]

    state = ic_state
    for d in range(n_days):
        state = spectral_rollout(
            state, physics_fn, grid, sigma, pe_config,
            spec_cfg.dt, n_steps_per_day,
            sponge_factor, spectral_filter,
        )
        fields = spectral_pe_to_grid(state, grid, sigma)
        mid_T_series.append(
            _area_weighted_mid_T(np.asarray(fields["T"]), lat_weights)
        )
        col_T_series.append(
            _global_mean_3d(np.asarray(fields["T"]), lat_weights, lev_w)
        )
        if (d + 1) % 5 == 0:
            logger.info(
                f"[{variant}] day {d+1:3d}/{n_days}: "
                f"mid_T={mid_T_series[-1]:.3f} K, col_T={col_T_series[-1]:.3f} K"
            )
    return mid_T_series, col_T_series


def _era5_reference_series(
    spec_cfg,
    grid,
    sigma,
    cache_dir: str,
    start_year: int,
    start_day: int,
    n_days: int,
    lat_weights: np.ndarray,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Return (days, era5 mid-T series, era5 column-mean T series)."""
    from legoesm.training.neural_gcm_spectral import (
        load_training_data, spectral_pe_to_grid,
    )

    # load_training_data with a single window of n_days returns
    # ``n_days`` IC/target pairs across ``n_days + 1`` consecutive
    # daily snapshots (carry[d] -> spectral IC, carry[d+1] -> target).
    # We use IC + targets to get the full n_days+1 days of ERA5.
    windows = [(int(start_year), int(start_day), int(n_days))]
    period_cfg = spec_cfg._replace(
        n_train_days=n_days, windows=tuple(windows),
    )
    ic_states, target_carries, _ic_times = load_training_data(
        period_cfg, grid, sigma, cache_dir, windows=windows,
    )
    # IC0 spectral -> grid for day-0 anchor.
    fields0 = spectral_pe_to_grid(ic_states[0], grid, sigma)
    lev_w = np.ones(spec_cfg.n_levels)
    mid_T = [_area_weighted_mid_T(np.asarray(fields0["T"]), lat_weights)]
    col_T = [_global_mean_3d(np.asarray(fields0["T"]), lat_weights, lev_w)]
    for target in target_carries:
        T_arr = np.asarray(target.T)
        mid_T.append(_area_weighted_mid_T(T_arr, lat_weights))
        col_T.append(_global_mean_3d(T_arr, lat_weights, lev_w))
    days = np.arange(n_days + 1, dtype=float)
    return days, mid_T, col_T, ic_states[0]


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _plot_series(
    days: np.ndarray,
    series: dict[str, list[float]],
    out_path: Path,
    title: str,
    ylabel: str,
):
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for name, values in series.items():
        ax.plot(
            days, values, color=_VARIANT_COLOURS.get(name, "gray"),
            label=name, linewidth=1.8 if name == "era5" else 1.4,
            linestyle="--" if name == "era5" else "-",
        )
    ax.set_xlabel("forecast day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/aimip_001"),
        help="Directory containing <variant>/params.eqx checkpoints.",
    )
    parser.add_argument(
        "--suite", type=Path, default=Path("config/aimip/aimip_suite.yaml"),
    )
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--start-day", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    suite = _load_yaml(args.suite)
    base = _load_yaml(Path(suite["base"]))
    cache_dir = base.get("cache_dir", ".cache/aimip_era5")

    spec_cfg = _build_spectral_config(base)
    grid, sigma = _build_grid_and_sigma(spec_cfg)
    lat_weights = np.asarray(grid.weights)

    logger.info(
        f"Time-series: n_days={args.n_days} "
        f"from {args.start_year}@day{args.start_day}, "
        f"grid T{spec_cfg.n_max} L{spec_cfg.n_levels}"
    )

    # ERA5 reference + IC for the rollouts.
    days, era5_mid_T, era5_col_T, ic_state = _era5_reference_series(
        spec_cfg, grid, sigma, cache_dir,
        args.start_year, args.start_day, args.n_days, lat_weights,
    )

    mid_T_series = {"era5": era5_mid_T}
    col_T_series = {"era5": era5_col_T}
    for variant in _VARIANTS:
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

        mid_T, col_T = _run_variant_series(
            variant, model, grid, sigma, spec_cfg, base,
            ic_state, args.n_days, lat_weights,
        )
        mid_T_series[variant] = mid_T
        col_T_series[variant] = col_T

    args.results.mkdir(parents=True, exist_ok=True)
    out_mid = args.results / "aimip_T_mid_timeseries.png"
    _plot_series(
        days, mid_T_series, out_mid,
        title=(
            f"AIMIP — global-mean mid-level T "
            f"({args.start_year} day {args.start_day} + {args.n_days}-day rollout)"
        ),
        ylabel="mid-level T [K]",
    )
    logger.info(f"Wrote {out_mid}")

    out_col = args.results / "aimip_T_col_timeseries.png"
    _plot_series(
        days, col_T_series, out_col,
        title=(
            f"AIMIP — global-mean column T "
            f"({args.start_year} day {args.start_day} + {args.n_days}-day rollout)"
        ),
        ylabel="column-mean T [K]",
    )
    logger.info(f"Wrote {out_col}")

    # CSV dump for downstream analysis.
    csv_path = args.results / "aimip_T_timeseries.csv"
    with csv_path.open("w") as fh:
        cols = ["day"] + [f"mid_T_{k}" for k in mid_T_series] + [
            f"col_T_{k}" for k in col_T_series
        ]
        fh.write(",".join(cols) + "\n")
        for d_idx in range(len(days)):
            row = [f"{float(days[d_idx]):.0f}"]
            for k in mid_T_series:
                row.append(f"{mid_T_series[k][d_idx]:.4f}")
            for k in col_T_series:
                row.append(f"{col_T_series[k][d_idx]:.4f}")
            fh.write(",".join(row) + "\n")
    logger.info(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
