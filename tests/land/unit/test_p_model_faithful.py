"""Faithfulness pin of the P-model optimality core against a scalar oracle.

Independent transcription (math module, literal coefficients) of:

- Stocker et al. (2020) GMD eq. 8-9 optimal chi with the least-cost xi,
- the Wang et al. (2017) "wang17" Jmax-limitation (c* = 0.41),
- coordination Vcmax = phi0(T)*Iabs*mj'/mc and Jmax = 4*phi0(T)*Iabs*
  sqrt((1-k)/k) (the rearranged rpmodel form),
- the Bernacchi (2003) phi0(T) quadratic (rpmodel ftemp_kphio),
- Huber et al. (2009) viscosity with Fisher & Dial (1975) Tumlirz density
  (rpmodel viscosity_h2o / density_h2o),

compared at rel 1e-9 under x64 over a (T, VPD, ca, Ps) grid.  The oracle
includes the module's documented smooth guards (softplus VPD/mj floors) so the
comparison is exact; the 25 degC normalisation deliberately reuses the repo's
own Kattge & Knorr responses (module docstring, departure 3) and is therefore
pinned THROUGH those functions rather than re-derived.  Constant canaries pin
the transcribed literals; a non-vacuity probe shows the pin fails when a
coefficient is perturbed.
"""

from __future__ import annotations

import math

import jax
import pytest

_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _force_x64():
    """Per-test float64 for the rel-1e-9 pins; restore the process-entry state."""
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", _ENTRY_X64)


import jax.numpy as jnp                                              # noqa: E402
import numpy as np                                                   # noqa: E402

from legoesm import constants                                        # noqa: E402
from legoesm.land import p_model as pm                               # noqa: E402
from legoesm.land import leaf_biophysics as lb                       # noqa: E402
from legoesm.land.canopy.photosynthesis import (                     # noqa: E402
    jmax_temperature_response,
    vcmax_temperature_response,
)

# --- independent oracle literals (Bernacchi 2001; Stocker 2020; Wang 2017) ---
# The C<->K offset and gas constant are unit conversions, not the physics being
# pinned; they come from constants (themselves canaried in test_leaf_biophysics).
_O_TFREEZE = float(constants.T_freeze)
_O_R = float(constants.R_universal)
_O_KC25, _O_KO25, _O_GS25, _O_O2 = 404.9, 278400.0, 42.75, 209000.0
_O_HA_KC, _O_HA_KO, _O_HA_GS = 79430.0, 36380.0, 37830.0
_O_CSTAR = 0.41
_O_BETA = 146.0
_O_KPHIO = 0.081785
_O_QUAD = (0.352, 0.022, -3.4e-4)
_O_PATM = 101325.0  # const-ok: independent oracle literal (standard atmosphere)


def _o_arrh(T, Ha):
    Tref = _O_TFREEZE + 25.0
    return math.exp(Ha * (T - Tref) / (Tref * _O_R * T))


def _o_density(T, P):
    tc = T - _O_TFREEZE
    lam = 1788.316 + 21.55053 * tc - 0.4695911 * tc**2 \
        + 3.096363e-3 * tc**3 - 7.341182e-6 * tc**4
    po = 5918.499 + 58.05267 * tc - 1.1253317 * tc**2 \
        + 6.6123869e-3 * tc**3 - 1.4661625e-5 * tc**4
    vinf = (0.6980547 - 7.435626e-4 * tc + 3.704258e-5 * tc**2
            - 6.315724e-7 * tc**3 + 9.829576e-9 * tc**4
            - 1.197269e-10 * tc**5 + 1.005461e-12 * tc**6
            - 5.437898e-15 * tc**7 + 1.69946e-17 * tc**8
            - 2.295063e-20 * tc**9)
    return 1e3 / (vinf + lam / (po + 1e-5 * P))


_O_H = ((0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0),
        (0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573),
        (-0.281378, -0.906851, -0.772479, -0.489837, -0.257040, 0.0),
        (0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0),
        (-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0),
        (0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264))


def _o_viscosity(T, P):
    tbar = T / 647.096
    rbar = _o_density(T, P) / 322.0
    mu0 = 1e2 * math.sqrt(tbar) / (
        1.67752 + 2.20462 / tbar + 0.6366564 / tbar**2 - 0.241605 / tbar**3)
    ctbar = 1.0 / tbar - 1.0
    mu1 = 0.0
    for i in range(6):
        inner = sum(_O_H[j][i] * (rbar - 1.0) ** j for j in range(7))
        mu1 += ctbar**i * inner
    return mu0 * math.exp(rbar * mu1) * 1e-6


def _o_softfloor(x, floor, width):
    return floor + width * math.log1p(math.exp((x - floor) / width))


def _o_chi_pack(T, D, ca_ppm, P, beta, vpd_min, mj_eps, mj_w):
    gs = _O_GS25 * 1e-6 * P * _o_arrh(T, _O_HA_GS)
    kc = _O_KC25 * 1e-6 * P * _o_arrh(T, _O_HA_KC)
    ko = _O_KO25 * 1e-6 * P * _o_arrh(T, _O_HA_KO)
    K = kc * (1.0 + (_O_O2 * 1e-6 * P) / ko)
    eta = _o_viscosity(T, P) / _o_viscosity(_O_TFREEZE + 25.0, _O_PATM)
    xi = math.sqrt(beta * (K + gs) / (1.6 * eta))
    Df = _o_softfloor(D, vpd_min, vpd_min)
    ca = ca_ppm * 1e-6 * P
    chi = gs / ca + (1.0 - gs / ca) * xi / (xi + math.sqrt(Df))
    ci = chi * ca
    mj = (ci - gs) / (ci + 2.0 * gs)
    mj_safe = _o_softfloor(mj, _O_CSTAR + mj_eps, mj_w)
    k = (_O_CSTAR / mj_safe) ** (2.0 / 3.0)
    mj_over_mc = (ci + K) / (ci + 2.0 * gs)
    return chi, xi, k, mj_over_mc


def _o_phi0(T, kphio):
    tc = T - _O_TFREEZE
    a0, a1, a2 = _O_QUAD
    quad = a0 + a1 * tc + a2 * tc * tc
    quad = 0.01 * math.log1p(math.exp(quad / 0.01))
    return kphio * quad


_GRID = [
    (298.15, 1000.0, 400.0, 101325.0),
    (283.15, 400.0, 380.0, 101325.0),
    (308.15, 3000.0, 400.0, 95000.0),
    (293.15, 800.0, 300.0, 70000.0),   # high altitude
    (275.15, 200.0, 400.0, 101325.0),  # cold, humid
]


def _mk_state(T, D, ca, P, iabs=400.0):
    one = jnp.ones((1,), dtype=jnp.float64)
    return pm.PModelAcclimState(
        iabs_mean=iabs * one, t_mean_K=T * one,
        vpd_mean_pa=D * one, co2_mean_ppm=ca * one, ps_ema=P * one)


def test_viscosity_matches_oracle():
    for T in (275.15, 283.15, 298.15, 308.15):
        for P in (70000.0, 101325.0):
            got = float(lb.water_viscosity(jnp.float64(T), jnp.float64(P)))
            assert got == pytest.approx(_o_viscosity(T, P), rel=1e-9)


def test_optimal_chi_matches_oracle():
    cfg = pm.PModelConfig()
    for T, D, ca, P in _GRID:
        chi, xi, ci, gs, K = pm.optimal_chi(
            jnp.float64(T), jnp.float64(D), jnp.float64(ca), jnp.float64(P), cfg)
        o_chi, o_xi, _, _ = _o_chi_pack(
            T, D, ca, P, _O_BETA, cfg.vpd_min_pa, cfg.mj_floor_eps, cfg.mj_floor_width)
        assert float(chi) == pytest.approx(o_chi, rel=1e-9)
        assert float(xi) == pytest.approx(o_xi, rel=1e-9)


def test_capacities_match_oracle():
    cfg = pm.PModelConfig()
    iabs = 400.0
    for T, D, ca, P in _GRID:
        caps = pm.acclimated_capacities(_mk_state(T, D, ca, P, iabs), cfg)
        _, xi, k, mj_over_mc = _o_chi_pack(
            T, D, ca, P, _O_BETA, cfg.vpd_min_pa, cfg.mj_floor_eps, cfg.mj_floor_width)
        phi0 = _o_phi0(T, _O_KPHIO)
        # Departure 3 (documented): 25 degC normalisation THROUGH the repo's
        # own K&K responses, evaluated at the same growth temperature.
        f_v = float(vcmax_temperature_response(jnp.float64(T), jnp.float64(T - _O_TFREEZE)))
        f_j = float(jmax_temperature_response(jnp.float64(T), jnp.float64(T - _O_TFREEZE)))
        o_vcmax25 = phi0 * iabs * math.sqrt(1.0 - k) * mj_over_mc / f_v
        o_rjv25 = 4.0 / (math.sqrt(k) * mj_over_mc) * (f_v / f_j)
        o_g1 = xi / math.sqrt(1000.0)
        assert float(caps.vcmax25_leaf[0]) == pytest.approx(o_vcmax25, rel=1e-9)
        assert float(caps.rjv25[0]) == pytest.approx(o_rjv25, rel=1e-9)
        assert float(caps.g1_kpa[0]) == pytest.approx(o_g1, rel=1e-9)


def test_constant_canaries():
    assert pm.C_STAR == 0.41
    assert pm._KPHIO_QUAD_C3 == (0.352, 0.022, -3.4e-4)
    cfg = pm.PModelConfig()
    assert cfg.beta_cost == 146.0
    assert cfg.kphio == 0.081785
    assert cfg.tau_acclim_s == 15.0 * 86400.0
    # Huber reference viscosity check value: 890.02 uPa s at 298.15 K, 1 atm.
    mu = float(lb.water_viscosity(jnp.float64(298.15), jnp.float64(_O_PATM)))
    assert mu == pytest.approx(890.02e-6, rel=1e-4)


def test_non_vacuity_perturbed_oracle_fails():
    cfg = pm.PModelConfig()
    T, D, ca, P = _GRID[0]
    chi, *_ = pm.optimal_chi(
        jnp.float64(T), jnp.float64(D), jnp.float64(ca), jnp.float64(P), cfg)
    o_chi_bad, *_ = _o_chi_pack(
        T, D, ca, P, _O_BETA * 1.001, cfg.vpd_min_pa, cfg.mj_floor_eps,
        cfg.mj_floor_width)
    assert float(chi) != pytest.approx(o_chi_bad, rel=1e-9)
    # And the SUT side: a perturbed module config must break the pin too.
    o_chi, *_ = _o_chi_pack(
        T, D, ca, P, _O_BETA, cfg.vpd_min_pa, cfg.mj_floor_eps,
        cfg.mj_floor_width)
    chi_bad, *_ = pm.optimal_chi(
        jnp.float64(T), jnp.float64(D), jnp.float64(ca), jnp.float64(P),
        pm.PModelConfig(beta_cost=146.0 * 1.001))
    assert float(chi_bad) != pytest.approx(o_chi, rel=1e-9)


def test_ema_recurrence_exact():
    # The acclimation update is the documented linear recurrence: one daylight
    # step from a known state reproduces the closed-form convex combination.
    cfg = pm.PModelConfig()
    s = _mk_state(293.15, 900.0, 400.0, _O_PATM, iabs=300.0)
    dt = 1800.0
    s2 = pm.advance_pmodel_acclim(
        s, T_K=jnp.full((1,), 300.0), ppfd=jnp.full((1,), 600.0),
        vpd_pa=jnp.full((1,), 1200.0), co2_ppm=jnp.full((1,), 410.0),
        ps_pa=jnp.full((1,), _O_PATM), cfg=cfg, dt=dt)
    alpha = dt / cfg.tau_acclim_s
    gain = 1.0 - math.exp(-alpha * 600.0 / (300.0 + cfg.ppfd_ref_floor))
    assert float(s2.iabs_mean[0]) == pytest.approx(
        300.0 + gain * (600.0 - 300.0), rel=1e-12)
    assert float(s2.t_mean_K[0]) == pytest.approx(
        293.15 + gain * (300.0 - 293.15), rel=1e-12)
    assert float(s2.vpd_mean_pa[0]) == pytest.approx(
        900.0 + gain * (1200.0 - 900.0), rel=1e-12)


def test_grid_capacities_finite_and_positive():
    cfg = pm.PModelConfig()
    for T, D, ca, P in _GRID:
        caps = pm.acclimated_capacities(_mk_state(T, D, ca, P), cfg)
        for leaf in caps:
            assert np.all(np.isfinite(np.asarray(leaf)))
        assert float(caps.vcmax25_leaf[0]) > 0.0
        assert 0.0 < float(caps.chi[0]) < 1.0


# --- C4 (rpmodel c4 method; Cai & Prentice 2020 quadratic) -----------------
_O_BETA_C4 = 146.0 / 9.0
_O_QUAD_C4 = (-0.064, 0.03, -0.000464)
# Collatz C4 kernel constants (CLM5): independent oracle literals.
_O_Q10_C4, _O_S1, _O_S2, _O_S3, _O_S4 = 2.0, 0.3, 313.15, 0.2, 288.15


def _o_phi0_c4(T, kphio_c4):
    tc = T - _O_TFREEZE
    a0, a1, a2 = _O_QUAD_C4
    quad = a0 + a1 * tc + a2 * tc * tc
    quad = 0.01 * math.log1p(math.exp(quad / 0.01))
    return kphio_c4 * quad


def _o_c4_response(T):
    tref = _O_TFREEZE + 25.0
    q10 = _O_Q10_C4 ** ((T - tref) / 10.0)
    fH = 1.0 + math.exp(_O_S1 * (T - _O_S2))
    fL = 1.0 + math.exp(_O_S3 * (_O_S4 - T))
    return q10 / (fH * fL)


def test_c4_capacities_match_oracle():
    from legoesm.land.p_model import acclimated_capacities_c4
    cfg = pm.PModelConfig()
    iabs = 400.0
    for T, D, ca, P in _GRID:
        caps4 = acclimated_capacities_c4(_mk_state(T, D, ca, P, iabs), cfg)
        o_chi, o_xi, _, _ = _o_chi_pack(
            T, D, ca, P, _O_BETA_C4, cfg.vpd_min_pa, cfg.mj_floor_eps,
            cfg.mj_floor_width)
        mprime = math.sqrt(1.0 - 0.41 ** (2.0 / 3.0))
        o_v4 = (_o_phi0_c4(T, cfg.kphio_c4) * iabs * mprime
                / _o_c4_response(T))
        assert float(caps4.chi_c4[0]) == pytest.approx(o_chi, rel=1e-9)
        assert float(caps4.g1_c4_kpa[0]) == pytest.approx(
            o_xi / math.sqrt(1000.0), rel=1e-9)
        assert float(caps4.vcmax25_c4_leaf[0]) == pytest.approx(o_v4, rel=1e-9)


def test_c4_constant_canaries():
    cfg = pm.PModelConfig()
    assert cfg.beta_cost_c4 == pytest.approx(146.0 / 9.0, rel=0)
    assert cfg.kphio_c4 == 1.0
    assert pm._KPHIO_QUAD_C4 == (-0.064, 0.03, -0.000464)
    # The host Collatz response is deliberately UN-normalised at 25 C
    # (fH*fL active there): pin the reference value the inversion divides by.
    from legoesm.land.canopy.photosynthesis import c4_vcmax_temperature_response
    r25 = float(c4_vcmax_temperature_response(jnp.float64(_O_TFREEZE + 25.0)))
    assert r25 == pytest.approx(_o_c4_response(_O_TFREEZE + 25.0), rel=1e-12)
    assert 0.85 < r25 < 0.90  # ~0.87, NOT 1


def test_c4_non_vacuity():
    from legoesm.land.p_model import acclimated_capacities_c4
    cfg = pm.PModelConfig()
    T, D, ca, P = _GRID[0]
    caps4 = acclimated_capacities_c4(_mk_state(T, D, ca, P), cfg)
    o_chi_bad, *_ = _o_chi_pack(
        T, D, ca, P, _O_BETA_C4 * 1.001, cfg.vpd_min_pa, cfg.mj_floor_eps,
        cfg.mj_floor_width)
    assert float(caps4.chi_c4[0]) != pytest.approx(o_chi_bad, rel=1e-9)
