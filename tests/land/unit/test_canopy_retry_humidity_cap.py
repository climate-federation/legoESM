"""Retry seed's canopy-air humidity is capped below saturation.

The cold-start humidity (qsat(T_soil) + q_atm)/2 at canopy air = air
temperature is supersaturated whenever the soil is warmer than the air.  With
the dew limit the first iterates are then held at saturation, the
canopy-humidity row of the Jacobian loses its leaf and soil dependence, and the
solve runs past the iteration cap although the root is far below saturation.

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import legoesm.land.surface_scheme.two_leaf_canopy as tlc
from legoesm import constants
from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.energy_balance import saturation_specific_humidity
from legoesm.land.canopy.stability import sat_specific_humidity
from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure_diag
from tests.land.unit.test_canopy_warm_start import _canopy_call

jax.config.update("jax_enable_x64", True)

# Recorded from a production AMIP day (res6, 16 ranks): sunny vegetated column
# left unsolved after the last pass; soil 6.5 K warmer than the air.
_DAY = dict(
    LAI=2.088326176673121, SZA=21.92006941256998, La=293.2599792480469, epsf=0.97,
    epss=0.96, fSun=0.5064953929765732, APAR_Sun=1118.4534095609952,
    APAR_Sh=127.91143779795452, Vcmax25_Sun=75.23062252546784,
    Vcmax25_Sh=71.00819269683488, Vcmax25_C4Sun=0.0, Vcmax25_C4Sh=0.0,
    ASW_Sun=386.91466638055243, ASW_Sh=68.80203158866097, ASW_Soil=450.0332001284308,
    Ts_bc=293.56158447265625, Ca=415.0, Ps=99686.09238755958, Ta=287.0351511455946,
    lam=2452583.721630859, Cp=constants.c_pd, rhoa=1.2124619150949498,
    Tv_atm=288.46769984345275, q_atm=0.008211755165812115, m=9.0, b0=0.01, alf=0.3,
    TgC=25.0, fC4=0.0, fStress_soil=0.9999468834105985, ur=5.079471859386667, CI=0.75,
    z0m=0.04332818123065903, displa=0.24552636030706787, z0=62.64548354267194,
    cv=0.0135, d_leaf=0.025, r_soil_surface=293.84107348884726, fwet=0.0,
    lam_soil=2452583.721630859)
_T_RETRY = _DAY["Ta"] - tlc._RETRY_COOLING_K
_Q_UNCAPPED = 0.011641474862931192        # (qsat(Ts) + q_atm)/2, recorded seed


def _solve(q0):
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _DAY.items()})
    x0 = jnp.asarray([_T_RETRY, _T_RETRY, 290.5, 290.5, _T_RETRY, q0])
    # The production AMIP iteration cap (the recorded run's canopy config).
    return solve_canopy_closure_diag(x0, b, CanopyConfig(max_iters=10))


def test_supersaturated_retry_seed_stalls_and_the_capped_one_converges():
    qs = float(sat_specific_humidity(jnp.asarray(_T_RETRY), jnp.asarray(_DAY["Ps"])))
    assert _Q_UNCAPPED > qs                              # the seed is supersaturated
    assert not bool(_solve(_Q_UNCAPPED)[2])
    x, _n, conv = _solve(min(_Q_UNCAPPED, tlc._RETRY_RH_CAP * qs))[:3]
    assert bool(conv)
    assert float(x[5]) < float(saturation_specific_humidity(x[4], jnp.asarray(_DAY["Ps"])))


def test_retry_pass_uses_the_capped_humidity(monkeypatch):
    """Stand-in solver that reports convergence exactly when it is seeded by the
    retry pass (canopy air not at the air temperature) with canopy air at most
    95% saturated.  Without the cap at least one column's retry seed is above that and fails."""
    real = tlc.solve_canopy_closure_diag

    def picky(x0, bun, cc):
        x, n_it, conv, *rest = real(x0, bun, cc)
        qs = sat_specific_humidity(x0[4], bun.Ps)   # the curve the cap uses
        return (x, n_it, (x0[4] != bun.Ta) & (x0[5] <= 0.95 * qs * (1 + 1e-6)), *rest)   # float32 seed

    monkeypatch.setattr(tlc, "solve_canopy_closure_diag", picky)
    assert bool(np.all(np.asarray(_canopy_call(ncol=4).converged)))
    monkeypatch.setattr(tlc, "_RETRY_RH_CAP", 10.0)
    assert not bool(np.all(np.asarray(_canopy_call(ncol=4).converged)))
