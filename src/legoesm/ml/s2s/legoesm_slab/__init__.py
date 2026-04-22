"""Initialized pure-physics legoESM slab S2S workflow."""

from legoesm.ml.s2s.legoesm_slab.campaign import (
    campaign_dir_for_year,
    campaign_dir_name,
    case_dir_for_start,
    filter_start_times_to_year,
    generate_semimonthly_start_times,
    surface_forcing_filename,
)
from legoesm.ml.s2s.legoesm_slab.metrics import (
    DEFAULT_FIELD_SPECS,
    DEFAULT_METRICS,
    DEFAULT_METRIC_SPECS,
    DEFAULT_S2S_WINDOWS,
    FieldSpec,
    LeadTimeWindow,
    MetricSpec,
    aggregate_long_records,
    aggregate_metric_table,
    align_truth_to_forecast,
    available_metric_names,
    build_daily_metric_table,
    compute_forecast_delta_table,
    load_long_record_csv,
    load_metric_csv,
    resolve_metric_specs,
    save_metric_summary_csv,
    save_metric_table_csv,
    score_forecast_metrics,
)
from legoesm.ml.s2s.legoesm_slab.preparation import (
    DEFAULT_ARCO_ERA5_STORE,
    DEFAULT_PRESSURE_LEVELS,
    PreparedCaseMetadata,
    PreparationConfig,
    build_control_surface_forcing_dataset,
    load_case_metadata,
    prepare_legoesm_case,
)
from legoesm.ml.s2s.legoesm_slab.rollout import (
    CaseRunSummary,
    DailyForecastCollector,
    ForecastExportConfig,
    InitializedCoupledSlabDriver,
    run_legoesm_case,
    wrap_relative_forcing_getter,
)
from legoesm.ml.s2s.legoesm_slab.teleconnections import (
    classify_enso_phase,
    compute_mjo_wind_shear_proxy,
    compute_nino34_series,
    weighted_region_mean,
)

_POSTPROCESS_NAMES = {"postprocess_campaign", "postprocess_case", "EXPERIMENTS", "FIELD_PLOT_INFO"}


def __getattr__(name):
    if name in _POSTPROCESS_NAMES:
        from legoesm.ml.s2s.legoesm_slab import postprocess

        return getattr(postprocess, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_ARCO_ERA5_STORE",
    "DEFAULT_FIELD_SPECS",
    "DEFAULT_METRICS",
    "DEFAULT_METRIC_SPECS",
    "DEFAULT_PRESSURE_LEVELS",
    "DEFAULT_S2S_WINDOWS",
    "CaseRunSummary",
    "DailyForecastCollector",
    "FieldSpec",
    "ForecastExportConfig",
    "InitializedCoupledSlabDriver",
    "LeadTimeWindow",
    "MetricSpec",
    "PreparedCaseMetadata",
    "PreparationConfig",
    "aggregate_long_records",
    "aggregate_metric_table",
    "align_truth_to_forecast",
    "available_metric_names",
    "build_control_surface_forcing_dataset",
    "build_daily_metric_table",
    "campaign_dir_for_year",
    "campaign_dir_name",
    "case_dir_for_start",
    "classify_enso_phase",
    "compute_forecast_delta_table",
    "compute_mjo_wind_shear_proxy",
    "compute_nino34_series",
    "filter_start_times_to_year",
    "generate_semimonthly_start_times",
    "load_case_metadata",
    "load_long_record_csv",
    "load_metric_csv",
    "postprocess_campaign",
    "postprocess_case",
    "prepare_legoesm_case",
    "resolve_metric_specs",
    "run_legoesm_case",
    "save_metric_summary_csv",
    "save_metric_table_csv",
    "score_forecast_metrics",
    "surface_forcing_filename",
    "weighted_region_mean",
    "wrap_relative_forcing_getter",
]
