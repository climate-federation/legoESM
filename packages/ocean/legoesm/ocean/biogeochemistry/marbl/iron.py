"""MARBL iron binding, scavenging rates and ligand budget.

Ports the one-ligand branch (Lig_cnt = 1) of compute_scavenging and
compute_Lig_terms from ``marbl_interior_tendency_mod.F90``.  The level-local
scavenging *rate* depends on the sinking particle flux and is applied
inside the particle scan (``particles.py``); this module provides the
level-independent pieces.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.biogeochemistry.marbl.config import MarblConfig
from legoesm.ocean.biogeochemistry.marbl.dom import top_layer_photo_factor
from legoesm.ocean.biogeochemistry.marbl.light import PARProfile

_DPS = 1.0 / constants.seconds_per_day


class IronSpeciation(NamedTuple):
    Fefree: jnp.ndarray   # unbound Fe [mmol/m^3]
    FeLig1: jnp.ndarray   # ligand-bound Fe [mmol/m^3]


def iron_speciation(Fe: jnp.ndarray, Lig: jnp.ndarray, cfg: MarblConfig) -> IronSpeciation:
    """Free and ligand-bound iron from K Fef^2 + (1 + K (Lt - Fet)) Fef - Fet = 0.

    The positive root is written as 2 Fet / (p1 + sqrt(p1^2 + 4 K Fet)), which
    equals MARBL's (-p1 + sqrt(...)) / (2 K) without the cancellation at
    small Fe, and is exactly 0 at Fe = 0.
    """
    K = cfg.KFeLig1
    p1 = 1.0 + K * (Lig - Fe)
    Fefree = 2.0 * Fe / (p1 + jnp.sqrt(p1 * p1 + 4.0 * K * Fe))
    FeLig1 = K * Fefree * Lig / (1.0 + K * Fefree)
    return IronSpeciation(Fefree=Fefree, FeLig1=FeLig1)


def doc_scavenging_factor(DOC: jnp.ndarray, DOCr: jnp.ndarray) -> jnp.ndarray:
    """Sinking-mass enhancement (1.6 DOC + DOCr) / DOCr where both are positive, else 1."""
    pos = (DOC > 0.0) & (DOCr > 0.0)
    DOCr_safe = jnp.where(pos, DOCr, 1.0)
    return jnp.where(pos, (DOC * 1.6 + DOCr) / DOCr_safe, 1.0)


class LigandTerms(NamedTuple):
    Lig_prod: jnp.ndarray
    Lig_photochem: jnp.ndarray
    Lig_deg: jnp.ndarray
    Lig_loss: jnp.ndarray


def compute_ligand(
    Lig: jnp.ndarray,
    Lig_prod_vent: jnp.ndarray,
    POC_remin: jnp.ndarray,
    DOC_prod: jnp.ndarray,
    Lig_scavenge: jnp.ndarray,
    photoFe_total: jnp.ndarray,
    PAR: PARProfile,
    dz: jnp.ndarray,
    cfg: MarblConfig,
) -> LigandTerms:
    """Ligand production (remineralization, DOM production, vents) and losses."""
    Lig_prod = Lig_prod_vent + (POC_remin + DOC_prod) * cfg.remin_to_Lig

    rate_top = top_layer_photo_factor(PAR, dz[..., 0], cfg) * cfg.Lig_photochem_per_day * _DPS
    top = jnp.zeros_like(Lig).at[..., 0].set(1.0)
    Lig_photochem = top * Lig * rate_top[..., None]

    Lig_deg = POC_remin * cfg.parm_Lig_degrade_rate0
    uptake_and_deg = cfg.Lig_uptake_ratio * photoFe_total + Lig_deg
    taper = jnp.where(Lig < cfg.Lig_low_thres, Lig / cfg.Lig_low_thres, 1.0)
    Lig_loss = Lig_scavenge + Lig_photochem + uptake_and_deg * taper
    return LigandTerms(Lig_prod=Lig_prod, Lig_photochem=Lig_photochem,
                       Lig_deg=Lig_deg, Lig_loss=Lig_loss)
