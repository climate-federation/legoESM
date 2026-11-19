"""Retry seed for canopy columns the cold start cannot solve.

On humid, light-wind nights the cold start (leaf and canopy air at the air
temperature) sits on the unstable side of the neutral-to-stable switch, where
the canopy residual has a local minimum that is not a root; the root lies past
the ridge, 1.5-2.2 K colder.  The retry pass therefore restarts slightly colder.

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import legoesm.land.surface_scheme.two_leaf_canopy as tlc
from legoesm import constants
from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure_diag
from tests.land.unit.test_canopy_warm_start import _canopy_call

jax.config.update("jax_enable_x64", True)

# Recorded from a production AMIP run (res6, 16 ranks): a dense canopy at night,
# air relative humidity 0.92, wind 1.0 m/s.
_NIGHT = dict(
    LAI=4.242029016153242, SZA=90.0, La=397.4852600097656, epsf=0.97, epss=0.96,
    fSun=0.00022359994708651707, APAR_Sun=0.0, APAR_Sh=0.0,
    Vcmax25_Sun=1.1536169692759934, Vcmax25_Sh=218.26662474520515,
    Vcmax25_C4Sun=0.0, Vcmax25_C4Sh=0.0, ASW_Sun=0.0, ASW_Sh=0.0, ASW_Soil=0.0,
    Ts_bc=296.69907605131897, Ca=415.0, Ps=97133.3618974805, Ta=297.2745210754717,
    lam=constants.L_v, Cp=constants.c_pd, rhoa=1.14072077397212,
    Tv_atm=300.51845674817014, q_atm=0.017954620436591545, m=8.45303901428033,
    b0=0.00939226557142259, alf=0.3, TgC=25.0, fC4=0.0,
    fStress_soil=0.9929684328870212, ur=1.0013755109486073, CI=0.75,
    z0m=1.8700695611198663, displa=16.705954746004142, z0=64.88022823463504,
    cv=0.0135, d_leaf=0.025, r_soil_surface=426.1079555640294, fwet=0.0)
_TA = _NIGHT["Ta"]
_COLD = [_TA, _TA, 290.5, 290.5, _TA, 0.01836278186548695]


def _solve(dT):
    x0 = jnp.asarray(_COLD).at[jnp.array([0, 1, 4])].add(-dT)
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _NIGHT.items()})
    return solve_canopy_closure_diag(x0, b, CanopyConfig())


def test_cold_start_is_trapped_and_the_cooled_start_converges():
    assert not bool(_solve(0.0)[2])
    x, n_it, conv = _solve(tlc._RETRY_COOLING_K)[:3]
    assert bool(conv)
    assert float(x[4]) < _TA            # the root is the stable, colder branch


def test_retry_pass_uses_the_cooled_start(monkeypatch):
    """Stand-in solver that never converges from the cold start itself (canopy
    air seeded exactly at the air temperature): the canopy call converges with
    the cooled retry and not without it."""
    real = tlc.solve_canopy_closure_diag

    def picky(x0, bun, cc):
        x, n_it, conv, *rest = real(x0, bun, cc)
        return (x, n_it, conv & (x0[4] != bun.Ta), *rest)

    monkeypatch.setattr(tlc, "solve_canopy_closure_diag", picky)
    assert bool(np.all(np.asarray(_canopy_call(ncol=4).converged)))
    monkeypatch.setattr(tlc, "_RETRY_COOLING_K", 0.0)
    assert not bool(np.any(np.asarray(_canopy_call(ncol=4).converged)))
