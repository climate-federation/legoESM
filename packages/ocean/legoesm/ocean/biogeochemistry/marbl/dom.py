"""MARBL dissolved organic matter (compute_dissolved_organic_matter)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.biogeochemistry.marbl.config import MarblConfig
from legoesm.ocean.biogeochemistry.marbl.light import PARProfile

_DPS = 1.0 / constants.seconds_per_day


class DOMTerms(NamedTuple):
    DOC_prod: jnp.ndarray
    DON_prod: jnp.ndarray
    DOP_prod: jnp.ndarray
    DOC_remin: jnp.ndarray
    DON_remin: jnp.ndarray
    DOP_remin: jnp.ndarray
    DOCr_remin: jnp.ndarray
    DONr_remin: jnp.ndarray
    DOPr_remin: jnp.ndarray


def top_layer_photo_factor(PAR: PARProfile, dz_top: jnp.ndarray, cfg: MarblConfig) -> jnp.ndarray:
    """sum_subcol col_frac ln(PAR_in) 0.4373 (10 m / dz_1) for PAR_in > 1 W/m^2, shape (...,).

    Shared by refractory-DOM photodegradation and ligand photochemistry.
    """
    PAR_in = PAR.interface[..., 0]
    lit = (PAR.col_frac > 0.0) & (PAR_in > 1.0)
    ln_PAR = jnp.log(jnp.where(lit, PAR_in, 1.0))
    per_sub = jnp.where(lit, PAR.col_frac * ln_PAR * cfg.DOMr_photo_lnPAR_coeff, 0.0)
    return jnp.sum(per_sub, axis=0) * (cfg.DOMr_photo_ref_depth / dz_top)


def compute_dom(
    DOC: jnp.ndarray, DON: jnp.ndarray, DOP: jnp.ndarray,
    DOCr: jnp.ndarray, DONr: jnp.ndarray, DOPr: jnp.ndarray,
    zoo_loss_doc: jnp.ndarray,
    auto_loss_doc: jnp.ndarray, auto_graze_doc: jnp.ndarray,
    remaining_N_don: jnp.ndarray, remaining_P_dop: jnp.ndarray,
    PAR: PARProfile, dz: jnp.ndarray, cfg: MarblConfig,
) -> DOMTerms:
    """DOM production and remineralization; DOM pools (..., nlev), per-PFT (3, ..., nlev).

    Zooplankton are not grazed in the CESM2 matrix, so zoo_graze_doc = 0.
    """
    DOC_prod = zoo_loss_doc + jnp.sum(auto_loss_doc, axis=0) + jnp.sum(auto_graze_doc, axis=0)
    DON_prod = cfg.Q * zoo_loss_doc + jnp.sum(remaining_N_don, axis=0)
    DOP_prod = cfg.Qp_zoo * zoo_loss_doc + jnp.sum(remaining_P_dop, axis=0)

    # semi-labile remineralization: "light" rate where layer-mean PAR > 1 W/m^2
    w = PAR.col_frac[..., None]
    light = (PAR.col_frac[..., None] > 0.0) & (PAR.avg > cfg.DOM_remin_light_PAR)
    dark = (PAR.col_frac[..., None] > 0.0) & ~light
    f_light = jnp.sum(jnp.where(light, w, 0.0), axis=0)
    f_dark = jnp.sum(jnp.where(dark, w, 0.0), axis=0)
    DOC_reminR = (f_light * cfg.DOC_reminR_light_per_day + f_dark * cfg.DOC_reminR_dark_per_day) * _DPS
    DON_reminR = (f_light * cfg.DON_reminR_light_per_day + f_dark * cfg.DON_reminR_dark_per_day) * _DPS
    DOP_reminR = (f_light * cfg.DOP_reminR_light_per_day + f_dark * cfg.DOP_reminR_dark_per_day) * _DPS

    # refractory pools, photodegradation added in the top layer only
    photo = top_layer_photo_factor(PAR, dz[..., 0], cfg) * cfg.DOMr_reminR_photo_per_day * _DPS
    top = jnp.zeros_like(DOCr).at[..., 0].set(1.0)
    DOCr_reminR = cfg.DOCr_reminR0_per_day * _DPS + top * photo[..., None]
    DONr_reminR = cfg.DONr_reminR0_per_day * _DPS + top * photo[..., None]
    DOPr_reminR = cfg.DOPr_reminR0_per_day * _DPS + top * photo[..., None]

    return DOMTerms(
        DOC_prod=DOC_prod, DON_prod=DON_prod, DOP_prod=DOP_prod,
        DOC_remin=DOC * DOC_reminR, DON_remin=DON * DON_reminR, DOP_remin=DOP * DOP_reminR,
        DOCr_remin=DOCr * DOCr_reminR, DONr_remin=DONr * DONr_reminR,
        DOPr_remin=DOPr * DOPr_reminR,
    )
