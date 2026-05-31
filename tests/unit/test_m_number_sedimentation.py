"""Rain + ice NUMBER sedimentation tests (iter-23).

gSAM ``MICRO_M2005`` sediments hydrometeor NUMBER with the number-weighted
fall speed (UNR=ARN·Γ(1+BR)/LAMR^BR, UNI=AIN·Γ(1+BI)/LAMI^BI), which is SLOWER
than the mass-weighted UMR/UMI — so number falls with the mass (keeping the PSD
consistent in a column) but with size-sorting (big particles, carrying more
mass, fall faster). legoESM previously sedimented only the mass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice


jax.config.update("jax_enable_x64", True)

_T = 290.0
_P_FULL = jnp.asarray([[5.0e4, 7.0e4, 9.0e4]])
_P_HALF = jnp.asarray([[4.0e4, 6.0e4, 8.0e4, 1.0e5]])
_RHO = _P_FULL / (constants.R_d * _T)
_DZ = jnp.full((1, 3), 300.0)
# Isolate sedimentation: saturated (no rain evap) + no self-collection.
_CFG = MorrisonConfig(rain_selfcoll_scheme="legacy", k_sc=0.0)


def _run(q_r=2.0e-3, N_r=1.0e4):
    """Rain only in the TOP level (index 0); saturated, no self-collection."""
    qsat = saturation_mixing_ratio(jnp.full((1, 3), _T), _P_FULL)
    z = jnp.zeros((1, 3))
    hm = HydrometeorState(
        q_c=z, q_r=z.at[0, 0].set(q_r), q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=z.at[0, 0].set(N_r), N_i=z,
    )
    return morrison_microphysics(
        jnp.full((1, 3), _T), qsat, hm, _P_FULL, _P_HALF,
        _RHO, _DZ, 20.0, _CFG)


def test_number_sediments_down_the_column():
    """Rain number leaves the top level and arrives in the level below — it
    falls WITH the mass (both dN_r and dq_r have the same sign pattern)."""
    out = _run()
    dN_r = out.dN_r_dt[0]
    dq_r = out.dq_r_dt[0]
    assert float(dN_r[0]) < 0.0 and float(dN_r[1]) > 0.0
    assert float(dq_r[0]) < 0.0 and float(dq_r[1]) > 0.0   # mass falls too


def test_number_conserved_in_column():
    """With rain confined to the interior (no flux out the bottom in one
    step), the column-integrated number change is ~0 — sedimentation only
    REDISTRIBUTES number (per-volume N_r: Σ dN_r·dz ≈ 0)."""
    out = _run()
    col = float(jnp.sum(out.dN_r_dt[0] * _DZ[0]))
    n_init = 1.0e4 * float(_DZ[0, 0])          # column number scale
    assert abs(col) < 1.0e-6 * n_init          # negligible vs the number present


def test_size_sorting_number_falls_slower_than_mass():
    """The number-weighted fall speed is slower than the mass-weighted one,
    so the number lags the mass slightly: |dN_r/N_r| < |dq_r/q_r| at the
    shedding level (mass leaves faster than number ⇒ mean drop size at the
    source level decreases)."""
    q_r, N_r = 2.0e-3, 1.0e4
    out = _run(q_r=q_r, N_r=N_r)
    frac_mass = abs(float(out.dq_r_dt[0, 0])) / q_r
    frac_num = abs(float(out.dN_r_dt[0, 0])) / N_r
    assert frac_num < frac_mass                # number falls slower


def test_ice_number_also_sediments():
    """Cloud-ice number (per-mass) sediments via UNI — top loses, below
    gains. Ice-SUBsaturated air (0.5·q_sat_ice) so deposition/PRCI don't fire
    and only sedimentation moves the ice number."""
    qsat_i = saturation_mixing_ratio_ice(jnp.full((1, 3), 240.0), _P_FULL)
    z = jnp.zeros((1, 3))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=z.at[0, 0].set(5.0e-4), q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=z.at[0, 0].set(1.0e5),
    )
    out = morrison_microphysics(
        jnp.full((1, 3), 240.0), 0.5 * qsat_i, hm, _P_FULL, _P_HALF,
        _RHO, _DZ, 20.0, _CFG)
    assert float(out.dN_i_dt[0, 0]) < 0.0 and float(out.dN_i_dt[0, 1]) > 0.0


def test_number_sedimentation_ad_safe():
    """grad of the column number tendency wrt rain mass is finite."""
    qsat = saturation_mixing_ratio(jnp.full((1, 3), _T), _P_FULL)

    def loss(q_r0):
        z = jnp.zeros((1, 3))
        hm = HydrometeorState(
            q_c=z, q_r=z.at[0, 0].set(q_r0), q_i=z, q_s=z, q_g=z,
            N_c=z, N_r=z.at[0, 0].set(1.0e4), N_i=z,
        )
        out = morrison_microphysics(
            jnp.full((1, 3), _T), qsat, hm, _P_FULL, _P_HALF,
            _RHO, _DZ, 20.0, _CFG)
        return jnp.sum(out.dN_r_dt)

    for q0 in (0.0, 2.0e-3):
        g = jax.grad(loss)(jnp.asarray(q0))
        assert bool(jnp.isfinite(g))
