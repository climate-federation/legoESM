"""Canonical SFNO S2S package.

This subpackage contains the subseasonal SFNO training, rollout, coupling,
and evaluation workflow and is separate from the older one-step WB2/ERA5
training path in :mod:`legoesm.ml`.
"""

from legoesm.ml.sfno_s2s.config import (
    CHAOSBENCH_ATMOS_VARS,
    CHAOSBENCH_DATA_DIR,
    CHAOSBENCH_PRESSURE_LEVELS,
    ChaosBenchS2SConfig,
    DEFAULT_ARCO_SST_CACHE_PATH,
    DEFAULT_ARCO_SST_STATS_PATH,
    LAND_SEA_MASK_VAR,
)
from legoesm.ml.sfno_s2s.coupling import (
    S2SSlabCouplingConfig,
    coupled_rollout_to_dataset,
)
from legoesm.ml.sfno_s2s.data import (
    SFNOS2SDataset,
    atmospheric_param_labels,
    available_s2s_dates,
    create_s2s_dataloader,
    create_s2s_training_iterator,
    denormalize_atmospheric_channels,
    denormalize_forcing_channels,
    forcing_channel_labels,
    load_normalization_bundle,
    load_s2s_sample,
    load_surface_sequence,
    normalize_atmospheric_channels,
    normalize_forcing_channels,
    resolve_s2s_sample_dates,
)
from legoesm.ml.sfno_s2s.preparation import (
    ArcoSSTCacheConfig,
    ArcoSurfaceForcingConfig,
    DEFAULT_ARCO_ERA5_STORE,
    prepare_arco_sst_cache,
    prepare_arco_surface_forcing,
)
# postprocess requires optional imageio; lazy-import via __getattr__ below.
from legoesm.ml.sfno_s2s.evaluation import (
    aggregate_long_records,
    compute_ensemble_daily_metrics,
    compute_daily_metrics,
    parse_window_specs,
    select_available_fields,
    summarize_window_metrics,
    write_metric_rows,
)
from legoesm.ml.sfno_s2s.regrid import (
    TargetGridSpec,
    build_target_grid,
    regrid_channels_to_gaussian,
)
from legoesm.ml.sfno_s2s.rollout import (
    build_sfno_from_checkpoint,
    load_training_metadata,
    metadata_path_for_checkpoint,
    rollout_to_dataset,
    sample_index_from_date,
    save_training_metadata,
)
from legoesm.ml.sfno_s2s.training import (
    S2SStochasticConfig,
    S2STrainingConfig,
    cast_model_to_float32,
    count_trainable_parameters,
    estimate_sfno_parameter_count,
    extra_input_channels,
    predict_next_atmosphere,
    rollout_with_forcing,
    train_s2s_step,
    train_sfno_s2s,
    validate_s2s_step,
)

_POSTPROCESS_NAMES = {
    "open_rollout_members",
    "postprocess_campaign_center_crps",
    "postprocess_ensemble_rollouts",
}


def __getattr__(name):
    if name in _POSTPROCESS_NAMES:
        from legoesm.ml.sfno_s2s import postprocess
        return getattr(postprocess, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CHAOSBENCH_ATMOS_VARS",
    "CHAOSBENCH_DATA_DIR",
    "CHAOSBENCH_PRESSURE_LEVELS",
    "ChaosBenchS2SConfig",
    "DEFAULT_ARCO_SST_CACHE_PATH",
    "DEFAULT_ARCO_SST_STATS_PATH",
    "LAND_SEA_MASK_VAR",
    "S2SSlabCouplingConfig",
    "S2SStochasticConfig",
    "S2STrainingConfig",
    "SFNOS2SDataset",
    "TargetGridSpec",
    "ArcoSurfaceForcingConfig",
    "ArcoSSTCacheConfig",
    "DEFAULT_ARCO_ERA5_STORE",
    "atmospheric_param_labels",
    "available_s2s_dates",
    "aggregate_long_records",
    "build_target_grid",
    "build_sfno_from_checkpoint",
    "cast_model_to_float32",
    "count_trainable_parameters",
    "create_s2s_dataloader",
    "create_s2s_training_iterator",
    "compute_ensemble_daily_metrics",
    "compute_daily_metrics",
    "coupled_rollout_to_dataset",
    "denormalize_atmospheric_channels",
    "denormalize_forcing_channels",
    "estimate_sfno_parameter_count",
    "extra_input_channels",
    "forcing_channel_labels",
    "load_normalization_bundle",
    "load_training_metadata",
    "load_s2s_sample",
    "load_surface_sequence",
    "metadata_path_for_checkpoint",
    "open_rollout_members",
    "postprocess_campaign_center_crps",
    "prepare_arco_sst_cache",
    "prepare_arco_surface_forcing",
    "postprocess_ensemble_rollouts",
    "predict_next_atmosphere",
    "normalize_atmospheric_channels",
    "normalize_forcing_channels",
    "parse_window_specs",
    "regrid_channels_to_gaussian",
    "resolve_s2s_sample_dates",
    "rollout_with_forcing",
    "rollout_to_dataset",
    "sample_index_from_date",
    "save_training_metadata",
    "select_available_fields",
    "summarize_window_metrics",
    "train_s2s_step",
    "train_sfno_s2s",
    "validate_s2s_step",
    "write_metric_rows",
]
