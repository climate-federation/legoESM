"""Canopy state container for the CLM-ML-JAX multilayer canopy scheme."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import numpy as np


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

    .. note::
        ``t_a10_arr`` is a plain NumPy array (``np.ndarray``), **not** a
        JAX array.  It lives on the Python host and is intentionally kept
        off the JAX computation graph (no gradient flows through it).
    """

    #: mlcanopy_type instance from clm_ml_jax (Any to keep import lazy).
    mlcanopy: Any

    #: 10-day running mean of near-surface air temperature [K], shape (ncol,), float64.
    #: Plain NumPy array (not JAX) — lives on the Python host between steps.
    #: Carried explicitly here (not read from mlcanopy.tacclim_forcing) to
    #: avoid double-filtering: MLCanopyFluxes copies the input t_a10_patch
    #: directly into tacclim_forcing on output — reading it back would apply
    #: the exponential filter twice per step (effective e-folding ~20d not 10d).
    t_a10_arr: "np.ndarray | None" = None  # shape (ncol,), float64; None → cold start
