"""MARBL zooplankton: losses, grazing and routing (CESM2 grazing matrix).

Ports compute_Zprime, compute_grazing and compute_routing from
``marbl_interior_tendency_mod.F90``.

Grazing assumption: the CESM2 grazing matrix has one zooplankton and three
prey classes, each containing exactly one autotroph and no zooplankton,
all with the Michaelis-Menten functional form.  For that matrix MARBL's
prey-class aggregation reduces exactly to
    auto_graze_i = z_umax_i Tfunc_zoo zooC Pprime_i / (Pprime_i + z_grz_i),
which is what is implemented here.  Other grazing matrices are not
supported by this port.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.biogeochemistry.marbl.autotrophs import (
    EPS_C, EPS_T_INV, loss_threshold_factor, pft_mask, temperature_function,
)
from legoesm.ocean.biogeochemistry.marbl.config import SP, MarblConfig, per_pft

_DPS = 1.0 / constants.seconds_per_day


class ZooTerms(NamedTuple):
    Zprime: jnp.ndarray
    zoo_loss: jnp.ndarray        # basal + bulk
    zoo_loss_poc: jnp.ndarray
    zoo_loss_doc: jnp.ndarray
    zoo_loss_dic: jnp.ndarray
    x_graze_zoo: jnp.ndarray     # grazing gain of zooplankton
    f_zoo_detr: jnp.ndarray


class GrazingTerms(NamedTuple):
    """Per-prey (3, ..., nlev) grazing and routing terms."""
    auto_graze: jnp.ndarray
    auto_graze_zoo: jnp.ndarray
    auto_graze_poc: jnp.ndarray
    auto_graze_doc: jnp.ndarray
    auto_graze_dic: jnp.ndarray
    remaining_P_pop: jnp.ndarray
    remaining_P_dop: jnp.ndarray
    remaining_P_dip: jnp.ndarray
    remaining_N_pon: jnp.ndarray
    remaining_N_don: jnp.ndarray
    remaining_N_din: jnp.ndarray
    remaining_Fe_pfe: jnp.ndarray
    remaining_Fe_dfe: jnp.ndarray


def compute_zooplankton(
    zooC: jnp.ndarray,
    Pprime: jnp.ndarray,
    QCaCO3: jnp.ndarray,
    Qp: jnp.ndarray,
    Qfe: jnp.ndarray,
    auto_loss: jnp.ndarray,
    auto_loss_poc: jnp.ndarray,
    auto_agg: jnp.ndarray,
    temperature: jnp.ndarray,
    zt: jnp.ndarray,
    cfg: MarblConfig,
) -> tuple[ZooTerms, GrazingTerms]:
    """Zooplankton losses, grazing on the three autotrophs and routing.

    zooC, temperature, zt: (..., nlev); per-PFT inputs: (3, ..., nlev).
    """
    trailing = zooC.ndim
    dt = zooC.dtype
    pp = lambda v: per_pft(v, trailing, dt)  # noqa: E731
    is_sp = pft_mask(SP, trailing, dt)

    # --- zooplankton losses (compute_Zprime) ---------------------------------------
    Tfunc_zoo = temperature_function(temperature, cfg.Tref_zoo, cfg.Q_10)
    f_loss = loss_threshold_factor(zt, cfg.thres_z1_zoo, cfg.thres_z2_zoo)
    Zprime = jnp.maximum(zooC - f_loss * cfg.zoo_loss_thres, 0.0)
    zoo_loss_basal = cfg.basal_respiration_rate_per_day * _DPS * Zprime * Tfunc_zoo
    zoo_loss_bulk = (cfg.z_mort_0_per_day * _DPS * Zprime
                     + cfg.z_mort2_0_per_day * _DPS * Zprime ** cfg.zoo_mort2_exp) * Tfunc_zoo
    zoo_loss = zoo_loss_basal + zoo_loss_bulk

    # --- grazing (Michaelis-Menten, one autotroph per prey class) ------------------
    prey = Pprime
    graze = jnp.where(
        prey > 0.0,
        pp(cfg.z_umax_0_per_day) * _DPS * Tfunc_zoo * zooC * prey / (prey + pp(cfg.z_grz)),
        0.0)
    graze_zoo = pp(cfg.graze_zoo) * graze
    graze_poc_sp = graze * jnp.maximum(
        cfg.caco3_poc_min * QCaCO3,
        jnp.minimum(cfg.spc_poc_fac * (Pprime + 0.6) ** 1.6, cfg.f_graze_sp_poc_lim))
    graze_poc = jnp.where(is_sp > 0.0, graze_poc_sp, pp(cfg.graze_poc) * graze)
    graze_doc = pp(cfg.graze_doc) * graze
    graze_dic = graze - (graze_zoo + graze_poc + graze_doc)
    x_graze_zoo = jnp.sum(graze_zoo, axis=0)

    food = graze + EPS_C * EPS_T_INV
    f_zoo_detr = jnp.sum(pp(cfg.f_zoo_detr) * food, axis=0) / jnp.sum(food, axis=0)

    # --- zooplankton loss routing (compute_routing) ----------------------------------
    zoo_loss_poc = f_zoo_detr * zoo_loss_bulk
    zoo_loss_doc = (1.0 - cfg.parm_labile_ratio) * (1.0 - f_zoo_detr) * zoo_loss_bulk
    zoo_loss_dic = cfg.parm_labile_ratio * (1.0 - f_zoo_detr) * zoo_loss_bulk + zoo_loss_basal

    # --- P, Fe, N routing of autotroph losses ---------------------------------------
    to_poc = graze_poc + auto_loss_poc + auto_agg
    total = graze + auto_loss + auto_agg

    def split(quota, zoo_quota):
        sinking = to_poc * quota
        remaining = total * quota - graze_zoo * zoo_quota - sinking
        # reduce the sinking part where the remaining part would be negative
        sinking = jnp.where(remaining < 0.0, sinking + remaining, sinking)
        remaining = jnp.maximum(remaining, 0.0)
        return sinking, remaining

    P_pop, P_rem = split(Qp, cfg.Qp_zoo)
    Fe_pfe, Fe_rem = split(Qfe, cfg.Qfe_zoo)
    N_pon, N_rem = split(cfg.Q, cfg.Q)

    zoo = ZooTerms(
        Zprime=Zprime, zoo_loss=zoo_loss, zoo_loss_poc=zoo_loss_poc,
        zoo_loss_doc=zoo_loss_doc, zoo_loss_dic=zoo_loss_dic,
        x_graze_zoo=x_graze_zoo, f_zoo_detr=f_zoo_detr,
    )
    grazing = GrazingTerms(
        auto_graze=graze, auto_graze_zoo=graze_zoo, auto_graze_poc=graze_poc,
        auto_graze_doc=graze_doc, auto_graze_dic=graze_dic,
        remaining_P_pop=P_pop, remaining_P_dop=cfg.f_toDOP * P_rem,
        remaining_P_dip=(1.0 - cfg.f_toDOP) * P_rem,
        remaining_N_pon=N_pon, remaining_N_don=cfg.f_toDON * N_rem,
        remaining_N_din=(1.0 - cfg.f_toDON) * N_rem,
        remaining_Fe_pfe=Fe_pfe, remaining_Fe_dfe=Fe_rem,
    )
    return zoo, grazing
