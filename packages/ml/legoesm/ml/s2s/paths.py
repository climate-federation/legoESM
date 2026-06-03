"""Canonical result roots for slab subseasonal workflows."""

from pathlib import Path

S2S_RESULTS_ROOT = Path("results/ml/s2s")
SFNO_SLAB_RESULTS_ROOT = S2S_RESULTS_ROOT / "sfno_slab"
NEURALGCM_SLAB_RESULTS_ROOT = S2S_RESULTS_ROOT / "neuralgcm_slab"

__all__ = [
    "NEURALGCM_SLAB_RESULTS_ROOT",
    "SFNO_SLAB_RESULTS_ROOT",
    "S2S_RESULTS_ROOT",
]
