"""Metric helpers for initialized legoESM slab forecasts.

The pure-physics legoESM slab workflow uses the same field definitions and
windowed deterministic metrics as the NeuralGCM slab workflow so that paired
case and campaign summaries stay comparable across the S2S family.
"""

from legoesm.ml.s2s.neuralgcm_slab.evaluation import (
    DEFAULT_S2S_WINDOWS,
    LeadTimeWindow,
    aggregate_long_records,
    aggregate_metric_table,
    compute_skill_score_records,
    load_long_record_csv,
    load_metric_csv,
    save_metric_summary_csv,
    summarize_metric_csv,
    summarize_metric_directory,
)
from legoesm.ml.s2s.neuralgcm_slab.metrics import (
    DEFAULT_FIELD_SPECS,
    DEFAULT_METRICS,
    DEFAULT_METRIC_SPECS,
    FieldSpec,
    MetricSpec,
    align_truth_to_forecast,
    available_metric_names,
    build_daily_metric_table,
    compute_forecast_delta_table,
    resolve_metric_specs,
    save_metric_table_csv,
    score_forecast_metrics,
    weighted_mae,
    weighted_rmse,
)

__all__ = [
    "DEFAULT_FIELD_SPECS",
    "DEFAULT_METRICS",
    "DEFAULT_METRIC_SPECS",
    "DEFAULT_S2S_WINDOWS",
    "FieldSpec",
    "LeadTimeWindow",
    "MetricSpec",
    "aggregate_long_records",
    "aggregate_metric_table",
    "align_truth_to_forecast",
    "available_metric_names",
    "build_daily_metric_table",
    "compute_forecast_delta_table",
    "compute_skill_score_records",
    "load_long_record_csv",
    "load_metric_csv",
    "resolve_metric_specs",
    "save_metric_summary_csv",
    "save_metric_table_csv",
    "score_forecast_metrics",
    "summarize_metric_csv",
    "summarize_metric_directory",
    "weighted_mae",
    "weighted_rmse",
]
