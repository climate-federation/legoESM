"""Unified CLI for the local SFNO slab workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.s2s.paths import SFNO_SLAB_RESULTS_ROOT
from legoesm.ml.s2s.sfno_slab import (
    ArcoSSTCacheConfig,
    CHAOSBENCH_ATMOS_VARS,
    CHAOSBENCH_DATA_DIR,
    CHAOSBENCH_PRESSURE_LEVELS,
    DEFAULT_ARCO_SST_CACHE_PATH,
    DEFAULT_ARCO_SST_STATS_PATH,
    DEFAULT_ARCO_ERA5_STORE,
    ArcoSurfaceForcingConfig,
    ChaosBenchS2SConfig,
    LAND_SEA_MASK_VAR,
    S2SSlabCouplingConfig,
    S2SStochasticConfig,
    S2STrainingConfig,
    available_s2s_dates,
    build_sfno_from_checkpoint,
    build_target_grid,
    cast_model_to_float32,
    compute_daily_metrics,
    count_trainable_parameters,
    coupled_rollout_to_dataset,
    create_s2s_training_iterator,
    estimate_sfno_parameter_count,
    extra_input_channels,
    load_training_metadata,
    metadata_path_for_checkpoint,
    postprocess_campaign_center_crps,
    postprocess_ensemble_rollouts,
    parse_window_specs,
    rollout_to_dataset,
    sample_index_from_date,
    save_training_metadata,
    select_available_fields,
    resolve_s2s_sample_dates,
    summarize_window_metrics,
    train_sfno_s2s,
    prepare_arco_surface_forcing,
    prepare_arco_sst_cache,
    write_metric_rows,
)
from legoesm.ml.training import load_checkpoint


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _format_param_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def _metadata_get(metadata: dict[str, Any], section: str, key: str, default=None):
    section_value = metadata.get(section, {})
    if isinstance(section_value, dict) and key in section_value:
        return section_value[key]
    return default


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-years",
        default="1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015",
        help="Comma-separated training years",
    )
    parser.add_argument(
        "--val-years",
        default="2016,2017,2018,2019,2020,2021",
        help="Comma-separated validation years",
    )
    parser.add_argument(
        "--atmosphere-vars",
        default=",".join(CHAOSBENCH_ATMOS_VARS),
        help="Comma-separated atmospheric variables to predict",
    )
    parser.add_argument("--land-vars", default="", help="Comma-separated LRA5 forcing variables")
    parser.add_argument(
        "--ocean-vars",
        default="sosstsst",
        help="Comma-separated ORAS5 forcing variables; these remain input-only forcings",
    )
    parser.add_argument(
        "--ocean-source",
        choices=("oras5", "arco_sst"),
        default="oras5",
        help="Ocean forcing source for training and rollout data loading",
    )
    parser.add_argument(
        "--arco-sst-cache-path",
        default=DEFAULT_ARCO_SST_CACHE_PATH,
        help="Path to the preprocessed ARCO daily SST cache",
    )
    parser.add_argument(
        "--arco-sst-stats-path",
        default=DEFAULT_ARCO_SST_STATS_PATH,
        help="Path to the ARCO SST normalization-statistics cache",
    )
    parser.add_argument(
        "--add-land-sea-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append a static land_sea_mask forcing channel derived from SST coverage",
    )
    parser.add_argument("--lead-time", type=int, default=1, help="Forecast lead offset in days")
    parser.add_argument("--n-steps", type=int, default=42, help="Number of daily target steps loaded from disk")
    parser.add_argument(
        "--train-rollout-steps",
        type=int,
        default=1,
        help="Number of autoregressive daily steps to backpropagate during training",
    )
    parser.add_argument("--gaussian-n-max", type=int, default=79, help="Gaussian-grid truncation for SFNO training")
    parser.add_argument("--embed-dim", type=int, default=32, help="SFNO embedding dimension")
    parser.add_argument("--n-blocks", type=int, default=4, help="Number of SFNO processor blocks")
    parser.add_argument("--mlp-expansion", type=int, default=4, help="SFNO MLP expansion factor")
    parser.add_argument(
        "--residual-prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the big residual skip from input atmosphere channels to output channels",
    )
    parser.add_argument(
        "--tendency-prediction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the model on atmospheric tendencies and integrate them during rollout",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=1, help="Torch DataLoader worker count")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Torch DataLoader prefetch factor per worker",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin host memory in the Torch DataLoader path",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--total-steps", type=int, default=10_000, help="Total optimizer steps")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--validation-every", type=int, default=100, help="Validation interval")
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=32,
        help="Number of validation batches to average per checkpoint decision",
    )
    parser.add_argument("--log-every", type=int, default=100, help="Logging interval")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for the model")
    parser.add_argument("--ensemble-size", type=int, default=1, help="Number of independently seeded checkpoints to train")
    parser.add_argument(
        "--stage-name",
        default="stage1",
        help="Label for this training stage, used in metadata and W&B naming",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start from for staged training",
    )
    parser.add_argument(
        "--train-ensemble-members",
        type=int,
        default=4,
        help="Number of stochastic members sampled per batch for the afCRPS objective",
    )
    parser.add_argument(
        "--noise-channels",
        type=int,
        default=1,
        help="Number of smooth Gaussian noise channels concatenated to the model input",
    )
    parser.add_argument(
        "--noise-lat",
        type=int,
        default=8,
        help="Latitude resolution of the coarse stochastic noise field before interpolation",
    )
    parser.add_argument(
        "--noise-lon",
        type=int,
        default=16,
        help="Longitude resolution of the coarse stochastic noise field before interpolation",
    )
    parser.add_argument(
        "--time-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a normalized lead-time channel to the model input",
    )
    parser.add_argument(
        "--afcrps-alpha",
        type=float,
        default=0.95,
        help="Finite-ensemble adjustment factor used in the almost-fair CRPS loss",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=30_000_000,
        help="Hard parameter-count cap; set <=0 to disable",
    )
    parser.add_argument(
        "--dry-run-stats",
        action="store_true",
        help="Instantiate the model, print shapes/parameter count, and exit without training",
    )
    parser.add_argument(
        "--save-intermediate-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain numbered step_*.eqx checkpoints during training",
    )
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain final.eqx in addition to the best validation checkpoint",
    )
    parser.add_argument(
        "--member-offset",
        type=int,
        default=0,
        help="Offset used when naming member_XX directories; useful for Slurm arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SFNO_SLAB_RESULTS_ROOT,
        help="Checkpoint root directory",
    )
    parser.add_argument("--wandb-project", default=None, help="Optional Weights & Biases project name")
    parser.add_argument("--wandb-entity", default=None, help="Optional Weights & Biases entity")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="Weights & Biases mode; disabled by default",
    )


def _add_rollout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained SFNO checkpoint")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata.json path")
    parser.add_argument("--data-dir", default=None, help="Override dataset root; defaults to metadata or ChaosBench root")
    parser.add_argument("--years", default=None, help="Comma-separated dataset years for sample discovery")
    parser.add_argument("--atmosphere-vars", default=None, help="Comma-separated atmospheric variables to predict")
    parser.add_argument("--pressure-levels", default=None, help="Comma-separated pressure levels")
    parser.add_argument("--land-vars", default=None, help="Comma-separated LRA5 forcing variables")
    parser.add_argument("--ocean-vars", default=None, help="Comma-separated ORAS5 forcing variables")
    parser.add_argument(
        "--ocean-source",
        choices=("oras5", "arco_sst"),
        default=None,
        help="Override the ocean forcing source used by the training metadata",
    )
    parser.add_argument(
        "--arco-sst-cache-path",
        default=None,
        help="Override the ARCO SST cache path recorded in metadata",
    )
    parser.add_argument(
        "--arco-sst-stats-path",
        default=None,
        help="Override the ARCO SST stats path recorded in metadata",
    )
    parser.add_argument(
        "--add-land-sea-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether a land_sea_mask forcing channel is appended",
    )
    parser.add_argument("--lead-time", type=int, default=None, help="Forecast lead offset in days")
    parser.add_argument("--n-steps", type=int, default=None, help="Number of daily rollout steps")
    parser.add_argument("--gaussian-n-max", type=int, default=None, help="Gaussian-grid truncation")
    parser.add_argument("--embed-dim", type=int, default=None, help="SFNO embedding dimension")
    parser.add_argument("--n-blocks", type=int, default=None, help="Number of SFNO processor blocks")
    parser.add_argument("--mlp-expansion", type=int, default=None, help="SFNO MLP expansion factor")
    parser.add_argument(
        "--residual-prediction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override residual prediction flag from metadata",
    )
    parser.add_argument(
        "--tendency-prediction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override tendency-prediction flag from metadata",
    )
    parser.add_argument("--sample-index", type=int, default=None, help="Dataset sample index to roll out")
    parser.add_argument("--sample-date", default=None, help="Dataset init date in YYYYMMDD format")
    parser.add_argument(
        "--stochastic-seed",
        type=int,
        default=0,
        help="Seed controlling the stochastic member sampled from the checkpoint",
    )
    parser.add_argument("--output-path", type=Path, default=None, help="Output NetCDF path for the rollout")


def _add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rollout-path", type=Path, required=True, help="Path to a saved rollout NetCDF")
    parser.add_argument(
        "--fields",
        default="t-850,z-500,q-700,sosstsst",
        help="Comma-separated channel labels to evaluate; defaults to headline fields",
    )
    parser.add_argument(
        "--windows",
        default="wk3_4=15:28,wk5_6=29:42",
        help="Comma-separated lead windows like wk3_4=15:28,wk5_6=29:42",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for evaluation CSVs")


def _add_coupled_rollout_arguments(parser: argparse.ArgumentParser) -> None:
    _add_rollout_arguments(parser)
    parser.add_argument(
        "--mode",
        choices=("coupled", "uncoupled", "both"),
        default="both",
        help="Whether to run coupled, uncoupled, or paired daily rollouts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for coupled-rollout NetCDF outputs; defaults to checkpoint parent / inference_<init> / seed_XX",
    )
    parser.add_argument(
        "--surface-source",
        choices=("arco", "chaosbench"),
        default="arco",
        help="Surface forcing source for slab coupling",
    )
    parser.add_argument(
        "--surface-forcing-path",
        type=Path,
        default=None,
        help="Optional precomputed surface_forcing.nc path; defaults to on-the-fly preparation for ARCO",
    )
    parser.add_argument(
        "--surface-output-path",
        type=Path,
        default=None,
        help="Optional path to write the prepared ARCO surface forcing dataset",
    )
    parser.add_argument(
        "--era5-store",
        default=DEFAULT_ARCO_ERA5_STORE,
        help="ARCO ERA5 store used when --surface-source=arco",
    )


def _add_ensemble_inference_arguments(parser: argparse.ArgumentParser) -> None:
    _add_rollout_arguments(parser)
    parser.add_argument("--n-members", type=int, default=5, help="Number of stochastic members to sample")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First stochastic seed; members are sampled from seed_start .. seed_start + n_members - 1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Inference root; defaults to checkpoint parent / inference_<init>",
    )
    parser.add_argument(
        "--surface-source",
        choices=("arco", "chaosbench"),
        default="arco",
        help="Surface forcing source for slab coupling",
    )
    parser.add_argument(
        "--surface-forcing-path",
        type=Path,
        default=None,
        help="Optional precomputed surface_forcing.nc path; defaults to on-the-fly preparation for ARCO",
    )
    parser.add_argument(
        "--surface-output-path",
        type=Path,
        default=None,
        help="Optional path to write the prepared ARCO surface forcing dataset",
    )
    parser.add_argument(
        "--era5-store",
        default=DEFAULT_ARCO_ERA5_STORE,
        help="ARCO ERA5 store used when --surface-source=arco",
    )
    parser.add_argument(
        "--fields",
        default="t-850,z-500,q-700,sosstsst",
        help="Comma-separated fields to score",
    )
    parser.add_argument(
        "--windows",
        default="wk3_4=15:28,wk5_6=29:42",
        help="Comma-separated lead windows like wk3_4=15:28,wk5_6=29:42",
    )
    parser.add_argument(
        "--snapshot-days",
        default="1,7,21,42",
        help="Comma-separated lead days to plot in representative-member grids",
    )
    parser.add_argument(
        "--plot-members",
        default="0,1",
        help="Comma-separated member indices to show in representative-member grids and GIFs",
    )


def _add_prepare_surface_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-time", required=True, help="Initialization time, e.g. 2022-01-01T00:00:00")
    parser.add_argument("--forecast-days", type=int, default=42, help="Number of daily forecast steps")
    parser.add_argument("--gaussian-n-max", type=int, default=79, help="Target Gaussian grid truncation")
    parser.add_argument("--output-path", type=Path, required=True, help="Output NetCDF path for the surface forcing")
    parser.add_argument("--era5-store", default=DEFAULT_ARCO_ERA5_STORE, help="ARCO ERA5 store")


def _add_prepare_sst_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-time", required=True, help="Start date/time, e.g. 1979-01-01T00:00:00")
    parser.add_argument("--end-time", required=True, help="End date/time, e.g. 2022-12-31T00:00:00")
    parser.add_argument("--output-path", type=Path, required=True, help="Output Zarr path for daily SST cache")
    parser.add_argument(
        "--stats-output-path",
        type=Path,
        required=True,
        help="Output Zarr path for mean/sigma stats",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="Optional ChaosBench ORAS5 reference file for the target 1.5-degree grid",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=31,
        help="Number of daily snapshots per ARCO read/write chunk",
    )
    parser.add_argument("--era5-store", default=DEFAULT_ARCO_ERA5_STORE, help="ARCO ERA5 store")


def _add_postprocess_ensemble_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coupled", type=Path, nargs="+", required=True, help="Coupled rollout NetCDFs, one per member")
    parser.add_argument("--uncoupled", type=Path, nargs="+", required=True, help="Uncoupled rollout NetCDFs, one per member")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for ensemble metrics and plots")
    parser.add_argument(
        "--fields",
        default="t-850,z-500,q-700,sosstsst",
        help="Comma-separated fields to score",
    )
    parser.add_argument(
        "--windows",
        default="wk3_4=15:28,wk5_6=29:42",
        help="Comma-separated lead windows like wk3_4=15:28,wk5_6=29:42",
    )
    parser.add_argument(
        "--snapshot-days",
        default="1,7,21,42",
        help="Comma-separated lead days to plot in representative-member grids",
    )
    parser.add_argument(
        "--plot-members",
        default="0,1",
        help="Comma-separated member indices to show in the representative-member grids",
    )


def _add_postprocess_center_crps_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--campaign-dir",
        dest="campaign_dirs",
        type=Path,
        action="append",
        required=True,
        help="Campaign directory containing metrics/case_ensemble_daily_metrics.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for center-comparison metrics and plots")
    parser.add_argument(
        "--fields",
        default="t-850,z-500,q-700",
        help="Comma-separated fields to compare against ChaosBench center baselines",
    )
    parser.add_argument(
        "--windows",
        default="wk3_4=15:28,wk5_6=29:42",
        help="Comma-separated lead windows like wk3_4=15:28,wk5_6=29:42",
    )


def _run_train(args: argparse.Namespace) -> None:
    atmosphere_vars = _parse_csv_strings(args.atmosphere_vars)
    land_vars = _parse_csv_strings(args.land_vars)
    ocean_vars = _parse_csv_strings(args.ocean_vars)
    if args.add_land_sea_mask and LAND_SEA_MASK_VAR not in ocean_vars:
        ocean_vars = (*ocean_vars, LAND_SEA_MASK_VAR)
    train_years = _parse_csv_ints(args.train_years)
    val_years = _parse_csv_ints(args.val_years)
    stochastic_config = S2SStochasticConfig(
        ensemble_members=int(args.train_ensemble_members),
        noise_channels=int(args.noise_channels),
        noise_lat=int(args.noise_lat),
        noise_lon=int(args.noise_lon),
        use_time_signal=bool(args.time_signal),
        afcrps_alpha=float(args.afcrps_alpha),
    )
    model_residual_prediction = bool(args.residual_prediction)
    if args.tendency_prediction and model_residual_prediction:
        print(
            "Disabling residual_prediction because tendency_prediction is enabled.",
            flush=True,
        )
        model_residual_prediction = False

    train_data_config = ChaosBenchS2SConfig(
        years=train_years,
        atmosphere_vars=atmosphere_vars,
        land_vars=land_vars,
        ocean_vars=ocean_vars,
        ocean_source=args.ocean_source,
        arco_sst_cache_path=args.arco_sst_cache_path,
        arco_sst_stats_path=args.arco_sst_stats_path,
        n_steps=args.n_steps,
        lead_time=args.lead_time,
        gaussian_n_max=args.gaussian_n_max,
    )
    val_data_config = ChaosBenchS2SConfig(
        years=val_years,
        atmosphere_vars=atmosphere_vars,
        land_vars=land_vars,
        ocean_vars=ocean_vars,
        ocean_source=args.ocean_source,
        arco_sst_cache_path=args.arco_sst_cache_path,
        arco_sst_stats_path=args.arco_sst_stats_path,
        n_steps=args.n_steps,
        lead_time=args.lead_time,
        gaussian_n_max=args.gaussian_n_max,
    )

    target_grid = build_target_grid(args.gaussian_n_max)
    n_atmos_channels = len(atmosphere_vars) * len(train_data_config.pressure_levels)
    n_forcing_channels = len(land_vars) + len(ocean_vars)
    n_extra_channels = extra_input_channels(stochastic_config)
    estimated_params = estimate_sfno_parameter_count(
        n_sh=target_grid.grid.n_sh,
        in_channels=n_atmos_channels + n_forcing_channels + n_extra_channels,
        out_channels=n_atmos_channels,
        embed_dim=args.embed_dim,
        n_blocks=args.n_blocks,
        mlp_expansion=args.mlp_expansion,
    )

    print(
        "S2S SFNO setup:"
        f" grid={target_grid.grid.n_lat}x{target_grid.grid.n_lon}"
        f" n_sh={target_grid.grid.n_sh}"
        f" atmos_channels={n_atmos_channels}"
        f" forcing_channels={n_forcing_channels}"
        f" stochastic_members={stochastic_config.ensemble_members}"
        f" noise_channels={stochastic_config.noise_channels}"
        f" time_signal={int(stochastic_config.use_time_signal)}"
        f" rollout_steps={min(args.train_rollout_steps, args.n_steps)}"
        f" estimated_params={_format_param_count(estimated_params)}"
    )

    for member_index in range(args.ensemble_size):
        member_id = args.member_offset + member_index
        member_seed = args.seed + member_id
        model = SFNO(
            SFNOConfig(
                in_channels=n_atmos_channels + n_forcing_channels + n_extra_channels,
                out_channels=n_atmos_channels,
                embed_dim=args.embed_dim,
                n_blocks=args.n_blocks,
                mlp_expansion=args.mlp_expansion,
                residual_prediction=model_residual_prediction,
            ),
            grid=target_grid.grid,
            key=jax.random.PRNGKey(member_seed),
        )
        model = cast_model_to_float32(model)
        if args.init_checkpoint is not None:
            print(
                f"Warm-starting member_{member_id:02d} from {args.init_checkpoint}",
                flush=True,
            )
            model = load_checkpoint(model, args.init_checkpoint)
            model = cast_model_to_float32(model)
        actual_params = count_trainable_parameters(model)
        print(f"member_{member_id:02d}: actual_params={_format_param_count(actual_params)}")
        if args.max_params > 0 and actual_params > args.max_params:
            raise ValueError(
                f"Model has {actual_params} parameters, exceeding max_params={args.max_params}. "
                "Reduce embed_dim / n_blocks or raise --max-params."
            )
        if args.dry_run_stats:
            continue

        member_dir = args.output_dir / f"member_{member_id:02d}"
        metadata = {
            "data": {
                "data_dir": str(train_data_config.data_dir),
                "atmosphere_vars": list(atmosphere_vars),
                "pressure_levels": list(train_data_config.pressure_levels),
                "land_vars": list(land_vars),
                "ocean_vars": list(ocean_vars),
                "ocean_source": str(args.ocean_source),
                "arco_sst_cache_path": str(args.arco_sst_cache_path),
                "arco_sst_stats_path": str(args.arco_sst_stats_path),
                "lead_time": int(args.lead_time),
                "n_steps": int(args.n_steps),
                "gaussian_n_max": int(args.gaussian_n_max),
                "available_years": sorted(set(train_years) | set(val_years)),
            },
            "model": {
                "embed_dim": int(args.embed_dim),
                "n_blocks": int(args.n_blocks),
                "mlp_expansion": int(args.mlp_expansion),
                "residual_prediction": bool(model_residual_prediction),
                "tendency_prediction": bool(args.tendency_prediction),
            },
            "stochastic": {
                "ensemble_members": int(stochastic_config.ensemble_members),
                "noise_channels": int(stochastic_config.noise_channels),
                "noise_lat": int(stochastic_config.noise_lat),
                "noise_lon": int(stochastic_config.noise_lon),
                "use_time_signal": bool(stochastic_config.use_time_signal),
                "afcrps_alpha": float(stochastic_config.afcrps_alpha),
            },
            "training": {
                "stage_name": str(args.stage_name),
                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "prefetch_factor": int(args.prefetch_factor),
                "pin_memory": bool(args.pin_memory),
                "train_rollout_steps": int(min(args.train_rollout_steps, args.n_steps)),
                "validation_batches": int(args.validation_batches),
                "lr": float(args.lr),
                "warmup_steps": int(args.warmup_steps),
                "total_steps": int(args.total_steps),
                "weight_decay": float(args.weight_decay),
                "grad_clip_norm": float(args.grad_clip_norm),
                "save_best_checkpoint": True,
                "save_final_checkpoint": bool(args.save_final_checkpoint),
                "save_step_checkpoints": bool(args.save_intermediate_checkpoints),
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_mode": args.wandb_mode,
                "tendency_prediction": bool(args.tendency_prediction),
            },
        }
        save_training_metadata(member_dir / "metadata.json", metadata)

        training_config = S2STrainingConfig(
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            total_steps=args.total_steps,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            n_autoregressive_steps=min(args.train_rollout_steps, args.n_steps),
            grad_clip_norm=args.grad_clip_norm,
            checkpoint_dir=str(member_dir),
            checkpoint_every=args.checkpoint_every,
            validation_every=args.validation_every,
            validation_batches=args.validation_batches,
            save_best_checkpoint=True,
            save_final_checkpoint=args.save_final_checkpoint,
            save_step_checkpoints=args.save_intermediate_checkpoints,
            stochastic_seed=member_seed,
            stochastic_config=stochastic_config,
            tendency_prediction=bool(args.tendency_prediction),
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
            wandb_run_name=(
                f"{args.output_dir.name}-{args.stage_name}-member_{member_id:02d}"
                f"-bs{args.batch_size}-ar{min(args.train_rollout_steps, args.n_steps)}"
                f"-stoch{stochastic_config.ensemble_members}"
                f"-noise{stochastic_config.noise_channels}"
                f"-n{args.n_steps}"
                f"-tend{int(bool(args.tendency_prediction))}"
            ),
            wandb_group=f"{args.output_dir.name}-{args.stage_name}",
        )

        train_iterator = create_s2s_training_iterator(
            train_data_config,
            batch_size=args.batch_size,
            seed=member_seed,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
        val_iterator = create_s2s_training_iterator(
            val_data_config,
            batch_size=args.batch_size,
            seed=member_seed + 10_000,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )

        print(
            f"Training member_{member_id:02d} with {n_atmos_channels} atmospheric channels "
            f"and {n_forcing_channels} forcing channels "
            f"(num_workers={args.num_workers})",
            flush=True,
        )
        train_sfno_s2s(
            model,
            target_grid.grid,
            train_iterator,
            training_config,
            val_iterator=val_iterator,
            log_every=args.log_every,
        )

    if args.dry_run_stats:
        print("Dry run complete.")


def _resolve_rollout_settings(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path | None]:
    metadata_path = args.metadata or metadata_path_for_checkpoint(args.checkpoint)
    metadata = load_training_metadata(metadata_path) if metadata_path.exists() else {}

    data_settings = {
        "data_dir": args.data_dir or _metadata_get(metadata, "data", "data_dir", CHAOSBENCH_DATA_DIR),
        "atmosphere_vars": _parse_csv_strings(args.atmosphere_vars)
        if args.atmosphere_vars is not None else tuple(_metadata_get(metadata, "data", "atmosphere_vars", CHAOSBENCH_ATMOS_VARS)),
        "pressure_levels": _parse_csv_ints(args.pressure_levels)
        if args.pressure_levels is not None else tuple(_metadata_get(metadata, "data", "pressure_levels", CHAOSBENCH_PRESSURE_LEVELS)),
        "land_vars": _parse_csv_strings(args.land_vars)
        if args.land_vars is not None else tuple(_metadata_get(metadata, "data", "land_vars", ())),
        "ocean_vars": _parse_csv_strings(args.ocean_vars)
        if args.ocean_vars is not None else tuple(_metadata_get(metadata, "data", "ocean_vars", ("sosstsst",))),
        "ocean_source": args.ocean_source
        if args.ocean_source is not None else str(_metadata_get(metadata, "data", "ocean_source", "oras5")),
        "arco_sst_cache_path": args.arco_sst_cache_path
        if args.arco_sst_cache_path is not None else str(_metadata_get(metadata, "data", "arco_sst_cache_path", DEFAULT_ARCO_SST_CACHE_PATH)),
        "arco_sst_stats_path": args.arco_sst_stats_path
        if args.arco_sst_stats_path is not None else str(_metadata_get(metadata, "data", "arco_sst_stats_path", DEFAULT_ARCO_SST_STATS_PATH)),
        "lead_time": args.lead_time if args.lead_time is not None else int(_metadata_get(metadata, "data", "lead_time", 1)),
        "n_steps": args.n_steps if args.n_steps is not None else int(_metadata_get(metadata, "data", "n_steps", 42)),
        "gaussian_n_max": args.gaussian_n_max if args.gaussian_n_max is not None else int(_metadata_get(metadata, "data", "gaussian_n_max", 79)),
    }
    if args.add_land_sea_mask is not None:
        if args.add_land_sea_mask and LAND_SEA_MASK_VAR not in data_settings["ocean_vars"]:
            data_settings["ocean_vars"] = (*data_settings["ocean_vars"], LAND_SEA_MASK_VAR)
        if (not args.add_land_sea_mask) and LAND_SEA_MASK_VAR in data_settings["ocean_vars"]:
            data_settings["ocean_vars"] = tuple(
                var for var in data_settings["ocean_vars"] if var != LAND_SEA_MASK_VAR
            )
    if args.years is not None:
        years = _parse_csv_ints(args.years)
    elif args.sample_date is not None:
        years = (int(args.sample_date[:4]),)
    else:
        years = tuple(_metadata_get(metadata, "data", "available_years", ()))
    if not years:
        raise ValueError("Rollout requires dataset years via metadata, --years, or --sample-date.")
    data_settings["years"] = years

    model_settings = {
        "embed_dim": args.embed_dim if args.embed_dim is not None else int(_metadata_get(metadata, "model", "embed_dim", 32)),
        "n_blocks": args.n_blocks if args.n_blocks is not None else int(_metadata_get(metadata, "model", "n_blocks", 4)),
        "mlp_expansion": args.mlp_expansion if args.mlp_expansion is not None else int(_metadata_get(metadata, "model", "mlp_expansion", 4)),
        "residual_prediction": args.residual_prediction if args.residual_prediction is not None else bool(_metadata_get(metadata, "model", "residual_prediction", True)),
        "tendency_prediction": args.tendency_prediction if args.tendency_prediction is not None else bool(_metadata_get(metadata, "model", "tendency_prediction", _metadata_get(metadata, "training", "tendency_prediction", False))),
    }
    stochastic_settings = {
        "ensemble_members": int(_metadata_get(metadata, "stochastic", "ensemble_members", 1)),
        "noise_channels": int(_metadata_get(metadata, "stochastic", "noise_channels", 0)),
        "noise_lat": int(_metadata_get(metadata, "stochastic", "noise_lat", 8)),
        "noise_lon": int(_metadata_get(metadata, "stochastic", "noise_lon", 16)),
        "use_time_signal": bool(_metadata_get(metadata, "stochastic", "use_time_signal", False)),
        "afcrps_alpha": float(_metadata_get(metadata, "stochastic", "afcrps_alpha", 0.95)),
    }
    return data_settings, model_settings, stochastic_settings, metadata_path if metadata_path.exists() else None


def _resolve_sample_selection(
    args: argparse.Namespace,
    data_config: ChaosBenchS2SConfig,
) -> tuple[int, str, str]:
    if args.sample_date is not None:
        sample_dates = available_s2s_dates(data_config)
        sample_index = sample_index_from_date(sample_dates, data_config, args.sample_date)
        sample_tag = args.sample_date
        sample_date = args.sample_date
    else:
        sample_index = 0 if args.sample_index is None else int(args.sample_index)
        sample_tag = f"sample_{sample_index:04d}"
        sample_date = resolve_s2s_sample_dates(data_config, sample_index=sample_index)[0]
    return sample_index, sample_tag, sample_date


def _default_inference_root(
    checkpoint: Path,
    sample_tag: str,
    output_dir: Path | None,
) -> Path:
    return output_dir or (checkpoint.parent / f"inference_{sample_tag}")


def _prepare_surface_forcing_for_rollout(
    *,
    args: argparse.Namespace,
    data_config: ChaosBenchS2SConfig,
    sample_date: str,
) -> xr.Dataset | None:
    if args.surface_source != "arco":
        return None
    if args.surface_forcing_path is not None:
        return xr.open_dataset(args.surface_forcing_path)
    return prepare_arco_surface_forcing(
        start_time=np.datetime64(f"{sample_date[:4]}-{sample_date[4:6]}-{sample_date[6:8]}T00:00:00"),
        forecast_days=data_config.n_steps,
        gaussian_n_max=data_config.gaussian_n_max,
        config=ArcoSurfaceForcingConfig(era5_store=args.era5_store),
        output_path=args.surface_output_path,
    )


def _run_rollout(args: argparse.Namespace) -> None:
    data_settings, model_settings, stochastic_settings, metadata_path = _resolve_rollout_settings(args)
    data_config = ChaosBenchS2SConfig(
        years=tuple(int(year) for year in data_settings["years"]),
        data_dir=str(data_settings["data_dir"]),
        atmosphere_vars=tuple(data_settings["atmosphere_vars"]),
        pressure_levels=tuple(int(level) for level in data_settings["pressure_levels"]),
        land_vars=tuple(data_settings["land_vars"]),
        ocean_vars=tuple(data_settings["ocean_vars"]),
        ocean_source=str(data_settings["ocean_source"]),
        arco_sst_cache_path=str(data_settings["arco_sst_cache_path"]),
        arco_sst_stats_path=str(data_settings["arco_sst_stats_path"]),
        lead_time=int(data_settings["lead_time"]),
        n_steps=int(data_settings["n_steps"]),
        gaussian_n_max=int(data_settings["gaussian_n_max"]),
    )

    n_atmos_channels = len(data_config.atmosphere_vars) * len(data_config.pressure_levels)
    n_forcing_channels = len(data_config.land_vars) + len(data_config.ocean_vars)
    stochastic_config = S2SStochasticConfig(
        ensemble_members=int(stochastic_settings["ensemble_members"]),
        noise_channels=int(stochastic_settings["noise_channels"]),
        noise_lat=int(stochastic_settings["noise_lat"]),
        noise_lon=int(stochastic_settings["noise_lon"]),
        use_time_signal=bool(stochastic_settings["use_time_signal"]),
        afcrps_alpha=float(stochastic_settings["afcrps_alpha"]),
    )
    model, _ = build_sfno_from_checkpoint(
        args.checkpoint,
        gaussian_n_max=data_config.gaussian_n_max,
        in_channels=n_atmos_channels + n_forcing_channels + extra_input_channels(stochastic_config),
        out_channels=n_atmos_channels,
        embed_dim=int(model_settings["embed_dim"]),
        n_blocks=int(model_settings["n_blocks"]),
        mlp_expansion=int(model_settings["mlp_expansion"]),
        residual_prediction=bool(model_settings["residual_prediction"]),
    )

    sample_index, sample_tag, _ = _resolve_sample_selection(args, data_config)

    ds = rollout_to_dataset(
        model,
        data_config,
        sample_index=sample_index,
        stochastic_seed=int(args.stochastic_seed),
        stochastic_config=stochastic_config,
        tendency_prediction=bool(model_settings["tendency_prediction"]),
    )
    if metadata_path is not None:
        ds.attrs["metadata_path"] = str(metadata_path)
    output_path = args.output_path
    if output_path is None:
        output_path = _default_inference_root(args.checkpoint, sample_tag, None) / f"seed_{int(args.stochastic_seed):02d}" / "rollout.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    print(f"Saved rollout: {output_path}")


def _run_coupled_rollout(args: argparse.Namespace) -> None:
    data_settings, model_settings, stochastic_settings, metadata_path = _resolve_rollout_settings(args)
    data_config = ChaosBenchS2SConfig(
        years=tuple(int(year) for year in data_settings["years"]),
        data_dir=str(data_settings["data_dir"]),
        atmosphere_vars=tuple(data_settings["atmosphere_vars"]),
        pressure_levels=tuple(int(level) for level in data_settings["pressure_levels"]),
        land_vars=tuple(data_settings["land_vars"]),
        ocean_vars=tuple(data_settings["ocean_vars"]),
        ocean_source=str(data_settings["ocean_source"]),
        arco_sst_cache_path=str(data_settings["arco_sst_cache_path"]),
        arco_sst_stats_path=str(data_settings["arco_sst_stats_path"]),
        lead_time=int(data_settings["lead_time"]),
        n_steps=int(data_settings["n_steps"]),
        gaussian_n_max=int(data_settings["gaussian_n_max"]),
    )

    n_atmos_channels = len(data_config.atmosphere_vars) * len(data_config.pressure_levels)
    n_forcing_channels = len(data_config.land_vars) + len(data_config.ocean_vars)
    stochastic_config = S2SStochasticConfig(
        ensemble_members=int(stochastic_settings["ensemble_members"]),
        noise_channels=int(stochastic_settings["noise_channels"]),
        noise_lat=int(stochastic_settings["noise_lat"]),
        noise_lon=int(stochastic_settings["noise_lon"]),
        use_time_signal=bool(stochastic_settings["use_time_signal"]),
        afcrps_alpha=float(stochastic_settings["afcrps_alpha"]),
    )
    model, _ = build_sfno_from_checkpoint(
        args.checkpoint,
        gaussian_n_max=data_config.gaussian_n_max,
        in_channels=n_atmos_channels + n_forcing_channels + extra_input_channels(stochastic_config),
        out_channels=n_atmos_channels,
        embed_dim=int(model_settings["embed_dim"]),
        n_blocks=int(model_settings["n_blocks"]),
        mlp_expansion=int(model_settings["mlp_expansion"]),
        residual_prediction=bool(model_settings["residual_prediction"]),
    )

    sample_index, sample_tag, sample_date = _resolve_sample_selection(args, data_config)
    surface_forcing = _prepare_surface_forcing_for_rollout(
        args=args,
        data_config=data_config,
        sample_date=sample_date,
    )

    output_dir = _default_inference_root(args.checkpoint, sample_tag, args.output_dir) / f"seed_{int(args.stochastic_seed):02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    coupling_config = S2SSlabCouplingConfig()

    for coupled in (True, False) if args.mode == "both" else (args.mode == "coupled",):
        ds = coupled_rollout_to_dataset(
            model,
            data_config,
            sample_index=sample_index,
            coupled=coupled,
            surface_forcing=surface_forcing,
            config=coupling_config,
            stochastic_seed=int(args.stochastic_seed),
            stochastic_config=stochastic_config,
            tendency_prediction=bool(model_settings["tendency_prediction"]),
        )
        if metadata_path is not None:
            ds.attrs["metadata_path"] = str(metadata_path)
        suffix = "coupled" if coupled else "uncoupled"
        output_path = output_dir / f"{suffix}.nc"
        ds.to_netcdf(output_path)
        print(f"Saved {suffix} rollout: {output_path}")


def _run_ensemble_inference(args: argparse.Namespace) -> None:
    data_settings, model_settings, stochastic_settings, metadata_path = _resolve_rollout_settings(args)
    data_config = ChaosBenchS2SConfig(
        years=tuple(int(year) for year in data_settings["years"]),
        data_dir=str(data_settings["data_dir"]),
        atmosphere_vars=tuple(data_settings["atmosphere_vars"]),
        pressure_levels=tuple(int(level) for level in data_settings["pressure_levels"]),
        land_vars=tuple(data_settings["land_vars"]),
        ocean_vars=tuple(data_settings["ocean_vars"]),
        ocean_source=str(data_settings["ocean_source"]),
        arco_sst_cache_path=str(data_settings["arco_sst_cache_path"]),
        arco_sst_stats_path=str(data_settings["arco_sst_stats_path"]),
        lead_time=int(data_settings["lead_time"]),
        n_steps=int(data_settings["n_steps"]),
        gaussian_n_max=int(data_settings["gaussian_n_max"]),
    )

    n_atmos_channels = len(data_config.atmosphere_vars) * len(data_config.pressure_levels)
    n_forcing_channels = len(data_config.land_vars) + len(data_config.ocean_vars)
    stochastic_config = S2SStochasticConfig(
        ensemble_members=int(stochastic_settings["ensemble_members"]),
        noise_channels=int(stochastic_settings["noise_channels"]),
        noise_lat=int(stochastic_settings["noise_lat"]),
        noise_lon=int(stochastic_settings["noise_lon"]),
        use_time_signal=bool(stochastic_settings["use_time_signal"]),
        afcrps_alpha=float(stochastic_settings["afcrps_alpha"]),
    )
    model, _ = build_sfno_from_checkpoint(
        args.checkpoint,
        gaussian_n_max=data_config.gaussian_n_max,
        in_channels=n_atmos_channels + n_forcing_channels + extra_input_channels(stochastic_config),
        out_channels=n_atmos_channels,
        embed_dim=int(model_settings["embed_dim"]),
        n_blocks=int(model_settings["n_blocks"]),
        mlp_expansion=int(model_settings["mlp_expansion"]),
        residual_prediction=bool(model_settings["residual_prediction"]),
    )

    sample_index, sample_tag, sample_date = _resolve_sample_selection(args, data_config)
    inference_root = _default_inference_root(args.checkpoint, sample_tag, args.output_dir)
    inference_root.mkdir(parents=True, exist_ok=True)
    surface_forcing = _prepare_surface_forcing_for_rollout(
        args=args,
        data_config=data_config,
        sample_date=sample_date,
    )
    coupling_config = S2SSlabCouplingConfig()
    coupled_paths: list[Path] = []
    uncoupled_paths: list[Path] = []

    for member_offset in range(int(args.n_members)):
        stochastic_seed = int(args.seed_start) + member_offset
        seed_dir = inference_root / f"seed_{stochastic_seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for coupled, output_paths in ((True, coupled_paths), (False, uncoupled_paths)):
            ds = coupled_rollout_to_dataset(
                model,
                data_config,
                sample_index=sample_index,
                coupled=coupled,
                surface_forcing=surface_forcing,
                config=coupling_config,
                stochastic_seed=stochastic_seed,
                stochastic_config=stochastic_config,
                tendency_prediction=bool(model_settings["tendency_prediction"]),
            )
            if metadata_path is not None:
                ds.attrs["metadata_path"] = str(metadata_path)
            suffix = "coupled" if coupled else "uncoupled"
            output_path = seed_dir / f"{suffix}.nc"
            ds.to_netcdf(output_path)
            output_paths.append(output_path)
            print(f"Saved {suffix} rollout: {output_path}")

    postprocess_ensemble_rollouts(
        coupled_paths=coupled_paths,
        uncoupled_paths=uncoupled_paths,
        output_dir=inference_root,
        fields=_parse_csv_strings(args.fields),
        windows=parse_window_specs(args.windows),
        snapshot_days=list(_parse_csv_ints(args.snapshot_days)),
        plot_member_indices=list(_parse_csv_ints(args.plot_members)),
    )
    print(f"Saved ensemble inference to {inference_root}")


def _run_prepare_surface(args: argparse.Namespace) -> None:
    ds = prepare_arco_surface_forcing(
        start_time=args.start_time,
        forecast_days=args.forecast_days,
        gaussian_n_max=args.gaussian_n_max,
        config=ArcoSurfaceForcingConfig(era5_store=args.era5_store),
        output_path=args.output_path,
    )
    print(f"Prepared ARCO surface forcing: {args.output_path}")
    print(
        "Surface forcing summary:"
        f" lead_days={ds.sizes['lead_day']}"
        f" grid={ds.sizes['latitude']}x{ds.sizes['longitude']}"
    )


def _run_prepare_sst_cache(args: argparse.Namespace) -> None:
    ds, stats = prepare_arco_sst_cache(
        start_time=args.start_time,
        end_time=args.end_time,
        config=ArcoSSTCacheConfig(era5_store=args.era5_store),
        output_path=args.output_path,
        stats_output_path=args.stats_output_path,
        reference_path=args.reference_path,
        chunk_days=int(args.chunk_days),
    )
    print(f"Prepared ARCO SST cache: {args.output_path}")
    print(f"Prepared ARCO SST stats: {args.stats_output_path}")
    print(
        "SST cache summary:"
        f" time={ds.sizes['time']}"
        f" grid={ds.sizes['latitude']}x{ds.sizes['longitude']}"
        f" mean={float(stats['mean'].sel(param='sosstsst').item()):.4f}"
        f" sigma={float(stats['sigma'].sel(param='sosstsst').item()):.4f}"
    )


def _run_postprocess_ensemble(args: argparse.Namespace) -> None:
    fields = _parse_csv_strings(args.fields)
    snapshot_days = list(_parse_csv_ints(args.snapshot_days))
    plot_members = list(_parse_csv_ints(args.plot_members))
    windows = parse_window_specs(args.windows)
    postprocess_ensemble_rollouts(
        coupled_paths=args.coupled,
        uncoupled_paths=args.uncoupled,
        output_dir=args.output_dir,
        fields=fields,
        windows=windows,
        snapshot_days=snapshot_days,
        plot_member_indices=plot_members,
    )
    print(f"Saved ensemble postprocessing to {args.output_dir}")


def _run_postprocess_center_crps(args: argparse.Namespace) -> None:
    fields = _parse_csv_strings(args.fields)
    windows = parse_window_specs(args.windows)
    postprocess_campaign_center_crps(
        campaign_dirs=args.campaign_dirs,
        output_dir=args.output_dir,
        fields=fields,
        windows=windows,
    )
    print(f"Saved center-comparison CRPS postprocessing to {args.output_dir}")


def _run_eval(args: argparse.Namespace) -> None:
    ds = xr.open_dataset(args.rollout_path)
    requested_fields = _parse_csv_strings(args.fields) if args.fields else ()
    selected_fields = select_available_fields(ds, requested=list(requested_fields) if requested_fields else None)
    if not selected_fields:
        raise ValueError("None of the requested fields are available in the rollout dataset.")
    daily_rows = compute_daily_metrics(ds, fields=selected_fields)
    window_rows = summarize_window_metrics(daily_rows, windows=parse_window_specs(args.windows))

    output_dir = args.output_dir or (args.rollout_path.parent / "metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metric_rows(daily_rows, output_dir / "daily_metrics.csv")
    if window_rows:
        write_metric_rows(window_rows, output_dir / "window_metrics.csv")
    else:
        print("No requested lead windows overlap this rollout; skipped window_metrics.csv")
    print(f"Saved evaluation to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for local SFNO training and future slab-coupled workflows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train one or more SFNO S2S checkpoints with SST as an input-only forcing",
    )
    _add_train_arguments(train_parser)
    train_parser.set_defaults(func=_run_train)

    rollout_parser = subparsers.add_parser(
        "rollout",
        help="Run one denormalized SFNO forecast rollout from a saved checkpoint",
    )
    _add_rollout_arguments(rollout_parser)
    rollout_parser.set_defaults(func=_run_rollout)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Compute compact RMSE/MAE summary CSVs from a saved rollout",
    )
    _add_eval_arguments(eval_parser)
    eval_parser.set_defaults(func=_run_eval)

    coupled_parser = subparsers.add_parser(
        "coupled-rollout",
        help="Run daily SFNO slab-ocean coupled or uncoupled rollouts from a saved checkpoint",
    )
    _add_coupled_rollout_arguments(coupled_parser)
    coupled_parser.set_defaults(func=_run_coupled_rollout)

    ensemble_inference_parser = subparsers.add_parser(
        "ensemble-inference",
        help="Run multi-seed stochastic coupled and uncoupled inference and postprocess into one inference_<init> directory",
    )
    _add_ensemble_inference_arguments(ensemble_inference_parser)
    ensemble_inference_parser.set_defaults(func=_run_ensemble_inference)

    prepare_surface_parser = subparsers.add_parser(
        "prepare-surface",
        help="Prepare ARCO daily surface forcing on the SFNO Gaussian grid",
    )
    _add_prepare_surface_arguments(prepare_surface_parser)
    prepare_surface_parser.set_defaults(func=_run_prepare_surface)

    prepare_sst_cache_parser = subparsers.add_parser(
        "prepare-sst-cache",
        help="Prepare a daily ARCO SST cache and matching normalization stats for SFNO training",
    )
    _add_prepare_sst_cache_arguments(prepare_sst_cache_parser)
    prepare_sst_cache_parser.set_defaults(func=_run_prepare_sst_cache)

    postprocess_parser = subparsers.add_parser(
        "postprocess-ensemble",
        help="Compute ensemble metrics and representative-member plots from paired rollout files",
    )
    _add_postprocess_ensemble_arguments(postprocess_parser)
    postprocess_parser.set_defaults(func=_run_postprocess_ensemble)

    postprocess_center_parser = subparsers.add_parser(
        "postprocess-center-crps",
        help="Compare aggregated SFNO campaign CRPS against ChaosBench center baselines",
    )
    _add_postprocess_center_crps_arguments(postprocess_center_parser)
    postprocess_center_parser.set_defaults(func=_run_postprocess_center_crps)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["build_parser", "main"]
