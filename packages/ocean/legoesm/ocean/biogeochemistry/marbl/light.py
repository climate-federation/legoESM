"""Photosynthetically available radiation in the water column (MARBL compute_PAR).

Morel & Maritorena (2001) chlorophyll-dependent attenuation, propagated
layer by layer; PAR below ``PAR_threshold`` is set to zero.  Sub-columns
(e.g. ice-covered / open fractions) carry their own surface shortwave and
are combined with ``PAR_col_frac`` weights by the consumers.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.ocean.biogeochemistry.marbl.config import MarblConfig


class PARProfile(NamedTuple):
    interface: jnp.ndarray   # (n_subcol, ..., nlev+1) PAR at layer interfaces [W/m^2], k=0 surface
    avg: jnp.ndarray         # (n_subcol, ..., nlev) layer-mean PAR [W/m^2]
    KPARdz: jnp.ndarray      # (..., nlev) optical thickness of each layer
    col_frac: jnp.ndarray    # (n_subcol, ...)


def compute_par(
    surf_shortwave: jnp.ndarray,
    col_frac: jnp.ndarray,
    total_chl: jnp.ndarray,
    dz: jnp.ndarray,
    cfg: MarblConfig,
) -> PARProfile:
    """PAR profile for each sub-column.

    Parameters
    ----------
    surf_shortwave : (n_subcol, ...) net shortwave entering the ocean [W/m^2]
    col_frac : (n_subcol, ...) sub-column fractions
    total_chl : (..., nlev) total chlorophyll [mg/m^3]
    dz : (..., nlev) layer thickness [m]
    """
    PAR_in = jnp.where(col_frac > 0.0, cfg.f_qsw_par * surf_shortwave, 0.0)
    PAR_in = jnp.where(PAR_in < cfg.PAR_threshold, 0.0, PAR_in)

    chl = jnp.maximum(total_chl, 0.02)
    # Morel & Maritorena (2001) two-regime fit, K in 1/m
    K = jnp.where(chl < 0.13224, 0.0919 * chl ** 0.3536, 0.1131 * chl ** 0.4562)
    KPARdz = K * dz

    # interface PAR: PAR_in * exp(-cumsum(KPARdz)), thresholded to zero
    optical_depth = jnp.cumsum(KPARdz, axis=-1)
    interface_below = PAR_in[..., None] * jnp.exp(-optical_depth)[None]
    interface_below = jnp.where(interface_below < cfg.PAR_threshold, 0.0, interface_below)
    interface = jnp.concatenate([PAR_in[..., None], interface_below], axis=-1)

    # layer mean: PAR_top (1 - exp(-KPARdz)) / KPARdz; KPARdz = 0 only for
    # zero-thickness (masked) layers, where the mean is the top value.
    KPARdz_safe = jnp.where(KPARdz > 0.0, KPARdz, 1.0)
    attenuation = jnp.where(KPARdz > 0.0, -jnp.expm1(-KPARdz) / KPARdz_safe, 1.0)
    avg = interface[..., :-1] * attenuation[None]
    return PARProfile(interface=interface, avg=avg, KPARdz=KPARdz, col_frac=col_frac)
