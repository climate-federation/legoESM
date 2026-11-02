"""MARBL nitrification and water-column denitrification (compute_nitrif, compute_denitrif)."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.biogeochemistry.marbl.config import MarblConfig
from legoesm.ocean.biogeochemistry.marbl.light import PARProfile

_DPS = 1.0 / constants.seconds_per_day


def compute_nitrification(NH4: jnp.ndarray, PAR: PARProfile, cfg: MarblConfig) -> jnp.ndarray:
    """Light-inhibited nitrification [mmol N/m^3/s].

    Nitrification proceeds where PAR at the layer bottom is below
    ``parm_nitrif_par_lim``; in the layer where PAR crosses the limit it is
    tapered by the fraction of the layer below the crossing depth.
    """
    lim = cfg.parm_nitrif_par_lim
    PAR_top = PAR.interface[..., :-1]          # (nsub, ..., nlev)
    PAR_bot = PAR.interface[..., 1:]
    base = cfg.parm_kappa_nitrif_per_day * _DPS * NH4
    dark_bottom = PAR_bot < lim
    crossing = dark_bottom & (PAR_top > lim)
    KPARdz_safe = jnp.where(PAR.KPARdz > 0.0, PAR.KPARdz, 1.0)
    taper = jnp.log(jnp.maximum(PAR_bot, cfg.PAR_threshold) / lim) / (-KPARdz_safe)
    per_sub = jnp.where(dark_bottom, base * jnp.where(crossing, taper, 1.0), 0.0)
    w = jnp.where(PAR.col_frac > 0.0, PAR.col_frac, 0.0)[..., None]
    return jnp.where(NH4 > 0.0, jnp.sum(w * per_sub, axis=0), 0.0)


def compute_denitrification(
    O2: jnp.ndarray, NO3: jnp.ndarray,
    DOC_remin: jnp.ndarray, DOCr_remin: jnp.ndarray, POC_remin: jnp.ndarray,
    other_remin: jnp.ndarray, sed_denitrif: jnp.ndarray,
    POCremin_refract: float, cfg: MarblConfig,
) -> jnp.ndarray:
    """Water-column denitrification under low O2 [mmol N/m^3/s]."""
    work = jnp.clip(((cfg.parm_o2_min + cfg.parm_o2_min_delta) - O2) / cfg.parm_o2_min_delta,
                    0.0, 1.0)
    denitrif = work * ((DOC_remin + DOCr_remin + POC_remin * (1.0 - POCremin_refract)
                        - other_remin) / cfg.denitrif_C_N - sed_denitrif)
    return jnp.where(NO3 < 1.0, NO3 * denitrif, denitrif)
