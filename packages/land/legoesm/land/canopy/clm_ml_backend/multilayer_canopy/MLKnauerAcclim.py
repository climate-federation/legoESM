"""Knauer et al. (2023) C3 thermal acclimation for the multilayer canopy (``acclim_type == 2``; DifferBESS v10, plan Part C).

The ONE source of these constants is DifferBESS ``process/Photosynthesis_FvCB_Knauer.py`` (``_C3_*``); the table below is a
plain-float copy so the backend has no DifferBESS import, and ``tests/land/unit/test_clm_ml_acclim.py`` pins every entry.
Vcmax and Jmax temperature responses become the peaked Arrhenius with growth-temperature-dependent activation energy and
entropy; the Jmax25 / Vcmax25 ratio depends on the growth and home temperatures. Rd, the kinetics, electron transport,
co-limitation, the nitrogen profile and the stomatal solve are unchanged (plan D1). C4 keeps CLM's constants (not ported).
"""
from __future__ import annotations

import jax.numpy as jnp

KNAUER_C3 = dict(
    HA_V_INTERCEPT=42.6e3, HA_V_SLOPE_TG=1.14e3,            # [J mol-1], [J mol-1 K-1]
    HA_J=40.71e3,                                          # [J mol-1]
    HD_V=200000.0, HD_J=200000.0,                          # [J mol-1]
    DS_V_INTERCEPT=645.13, DS_V_SLOPE_TG=-0.38,            # [J mol-1 K-1]
    DS_J_INTERCEPT=658.77, DS_J_SLOPE_TH=-0.84, DS_J_SLOPE_TGTH=-0.52,
    BJV_INTERCEPT=2.56, BJV_SLOPE_TH=-0.0375, BJV_SLOPE_TGTH=-0.0202,
    TG_LOW=3.0, TG_HIGH=37.0, TH_LOW=3.0, TH_HIGH=37.0,     # [degC] clips of the linear forms (DifferBESS _KNAUER_T*)
)
KNAUER_C3_HD = KNAUER_C3["HD_V"]   # Hd_V == Hd_J == 200 kJ mol-1
KNAUER_RGAS = 8.314   # [J K-1 mol-1] DifferBESS _R: the Vcmax / Jmax responses under acclim_type 2 use it (CLM-ML's rgas = 8.31446 elsewhere)


def knauer_c3_leaf_params(tg_C, th_C):
    """(vcmaxha, jmaxha, vcmaxse, jmaxse, j2v) for the growth temperature ``tg_C`` and home temperature ``th_C`` [degC],
    both clipped to [3, 37] degC inside (DifferBESS ``knauer_c3_acclimation``). Returns JAX scalars (``tg_C`` may be traced)."""
    k = KNAUER_C3
    tg = jnp.clip(jnp.asarray(tg_C, dtype=jnp.float64), k["TG_LOW"], k["TG_HIGH"])
    th = jnp.clip(jnp.asarray(th_C, dtype=jnp.float64), k["TH_LOW"], k["TH_HIGH"])
    vcmaxha = k["HA_V_INTERCEPT"] + k["HA_V_SLOPE_TG"] * tg
    jmaxha = jnp.asarray(k["HA_J"], dtype=jnp.float64) + 0.0 * tg
    vcmaxse = k["DS_V_INTERCEPT"] + k["DS_V_SLOPE_TG"] * tg
    jmaxse = k["DS_J_INTERCEPT"] + k["DS_J_SLOPE_TH"] * th + k["DS_J_SLOPE_TGTH"] * (tg - th)
    j2v = k["BJV_INTERCEPT"] + k["BJV_SLOPE_TH"] * th + k["BJV_SLOPE_TGTH"] * (tg - th)
    return vcmaxha, jmaxha, vcmaxse, jmaxse, j2v
