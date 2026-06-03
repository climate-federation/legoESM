"""Canopy state container for the CLM-ML-JAX multilayer canopy scheme."""

from __future__ import annotations

from typing import Any, NamedTuple


class CanopyState(NamedTuple):
    """Prognostic state for the CLM-ML-JAX multilayer canopy scheme.

    Carries the ``mlcanopy_type`` instance between legoESM timesteps so
    that leaf temperatures, leaf water potentials, and within-canopy
    air profiles are properly initialised on each call.  Typed as
    ``Any`` to avoid importing ``clm_ml_jax`` at module load time
    (optional dependency — only present when the ``canopy`` extra is
    installed).

    Shape conventions follow ``mlcanopy_type``: all arrays use
    1-based patch indexing where ``np = endp + 1`` covers indices
    ``1..ncol`` (index 0 is allocated but unused).
    """

    #: mlcanopy_type instance from clm_ml_jax (Any to keep import lazy).
    mlcanopy: Any
