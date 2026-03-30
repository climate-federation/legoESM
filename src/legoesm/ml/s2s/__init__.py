"""Shared slab S2S namespace for model-specific workflows and common utilities."""

from legoesm.ml.s2s.paths import (
    NEURALGCM_SLAB_RESULTS_ROOT,
    SFNO_SLAB_RESULTS_ROOT,
    S2S_RESULTS_ROOT,
)
from legoesm.ml.s2s.plotting import EXPERIMENT_COLORS, frame_to_image, plot_latlon_map

__all__ = [
    "EXPERIMENT_COLORS",
    "NEURALGCM_SLAB_RESULTS_ROOT",
    "SFNO_SLAB_RESULTS_ROOT",
    "S2S_RESULTS_ROOT",
    "frame_to_image",
    "plot_latlon_map",
]
