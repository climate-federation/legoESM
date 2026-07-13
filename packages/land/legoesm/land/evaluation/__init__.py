"""Land-surface evaluation pipeline for legoESM.

A shared, config-driven harness for scoring land/canopy offline runs against
Fortran references, other schemes, and tower observations.  See
``docs/land/evaluation_pipeline.md`` for the collaborator guide.

Layers:

* :mod:`~legoesm.land.evaluation.metrics` — pure-numpy comparison metrics
  and ILAMB-style unit-interval scores (shared by every script).
* :mod:`~legoesm.land.evaluation.fluxio` — CLM-ML ``.out`` column schema +
  loader.
* :mod:`~legoesm.land.evaluation.scorecard` — variable→group→overall score
  aggregation to JSON.
* :mod:`~legoesm.land.evaluation.references` — reference-dataset resolution
  (in-tree / shared bundle / fetch).
* :mod:`~legoesm.land.evaluation.recipe` — declarative YAML cases + driver.
"""
from __future__ import annotations

from legoesm.land.evaluation import (
    fluxio,
    metrics,
    recipe,
    references,
    scorecard,
)

__all__ = ["metrics", "fluxio", "scorecard", "references", "recipe"]
