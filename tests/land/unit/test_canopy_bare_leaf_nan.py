"""Bare ground: the canopy residual must stay finite next to its root.

A leaf with no area has no energy balance to invert; its "new" leaf
temperature came back as -0.71 K on hot bare desert ground, and the canopy-air
humidity update evaluated the saturation humidity there, which overflows to
NaN for a 1e-4 K change of the canopy-air temperature.  The solver rejected
every step into that region and stalled short of the root.

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.solver import (CanopyForcingBundle, _canopy_residual,
                                         solve_canopy_closure_diag)

jax.config.update("jax_enable_x64", True)

# Recorded from a production AMIP run (res6): bare desert ground near noon,
# soil 33 K above the air.
_DESERT = dict(
    LAI=0.0, SZA=13.681223119823454, La=375.77227783203125, epsf=0.97, epss=0.96,
    fSun=0.0, APAR_Sun=0.0, APAR_Sh=0.0, Vcmax25_Sun=0.0, Vcmax25_Sh=0.0,
    Vcmax25_C4Sun=0.0, Vcmax25_C4Sh=0.0, ASW_Sun=0.0, ASW_Sh=0.0,
    ASW_Soil=900.8282451329029, Ts_bc=341.9416422486664, Ca=415.0,
    Ps=94226.26267555617, Ta=309.03550697672847, lam=constants.L_v,
    Cp=constants.c_pd, rhoa=1.0644671083707788, Tv_atm=309.9363985900491,
    q_atm=0.004796515430928461, m=0.7414458801318919, b0=0.0008238287557021022,
    alf=0.3, TgC=25.0, fC4=0.0, fStress_soil=0.9933396987820982,
    ur=4.094671468428906, CI=0.75, z0m=0.010000000000000004, displa=0.0,
    z0=67.44706593998977, cv=0.0135, d_leaf=0.025,
    r_soil_surface=2026.5393718767527, fwet=0.0)
_B = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _DESERT.items()})
_TA = _DESERT["Ta"]
_COLD = jnp.asarray([_TA, _TA, 0.7 * 415.0, 0.7 * 415.0, _TA, 0.007388308287991176])
# Where the former solver stopped (canopy air 24.5 K above the air).
_STOP = np.array([333.398, 333.398, 415.0, 415.0, 333.5524, 0.0086])


def test_residual_is_finite_around_the_former_stop_point():
    cfg = CanopyConfig()
    for dTc in (-1e-1, -1e-2, -1e-4, 0.0, 1e-4, 1e-2):
        x = _STOP.copy()
        x[4] += dTc
        r = _canopy_residual(jnp.asarray(x), _B, cfg.LE_module, cfg.stomatal_model,
                             cfg.le_cap_mode, cfg.use_ta_for_photosynthesis,
                             cfg.rh_cap_smoothing_width, cfg.zeta_cap_smoothing_width,
                             cfg.most_n_iters)
        assert bool(jnp.all(jnp.isfinite(r))), dTc


def test_bare_desert_column_converges_from_the_cold_start():
    x, _n, conv = solve_canopy_closure_diag(_COLD, _B, CanopyConfig())[:3]
    assert bool(conv)
    assert _TA < float(x[4]) < _DESERT["Ts_bc"]      # canopy air between air and soil
    assert abs(float(x[0] - x[4])) < 1e-3 and abs(float(x[1] - x[4])) < 1e-3  # leaves pinned


def test_gradient_through_the_bare_desert_solve_is_finite_and_matches_fd():
    def tc_root(Ta):
        b = _B._replace(Ta=Ta, Tv_atm=Ta + (_DESERT["Tv_atm"] - _TA))
        return solve_canopy_closure_diag(_COLD, b, CanopyConfig())[0][4]
    g = float(jax.grad(tc_root)(jnp.asarray(_TA)))
    h = 1e-3
    fd = (float(tc_root(jnp.asarray(_TA + h))) - float(tc_root(jnp.asarray(_TA - h)))) / (2 * h)
    assert np.isfinite(g)
    assert abs(g - fd) < 1e-3 * max(1.0, abs(fd))
