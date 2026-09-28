"""Unit tests for the NEMO-parity NCAR bulk algorithm (bulk_flux_omip).

Strategy
--------
1. An INDEPENDENT NumPy transcription of NEMO 5.0.1
   ``sbcblk_algo_ncar.F90``/``sbc_phy.F90`` (written directly from the
   Fortran, no JAX, no shared code) cross-checks the JAX implementation on a
   seeded random ocean-like input field — a transcription bug in either copy
   shows up as a mismatch.
2. Physical-limit checks: neutral reduction to the LY09 neutral coefficients,
   stability monotonicity, Goff curve / moist density / L_vap reference
   values, flux sign conventions, evap-lh consistency.
3. JAX health: jit + grad (differentiability through the fixed-point loop).
4. Legacy scheme pinning + dispatch hardening (unknown algo raises).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.ocean.bulk_flux_omip import (  # noqa: E402
    air_sea_fluxes,
    exner_potential_temperature,
    latent_heat_vaporization_sst as latent_heat_vaporization,
    moist_air_cp,
    ncar_transfer_coefficients,
    potential_air_temperature_10m,
    pressure_at_height,
    rho_air_moist,
    seawater_q_sat,
)
from legoesm.thermo import saturation_vapor_pressure_goff  # noqa: E402


# ---------------------------------------------------------------------------
# Independent NumPy mirror of NEMO sbcblk_algo_ncar + sbc_phy preprocessing
# (zt = zu = 10 m), transcribed directly from the Fortran.  Deliberately
# uses NEMO's literal values (NOT legoesm.constants) so a wrong constant in
# the implementation cannot hide.
# ---------------------------------------------------------------------------

_VK = 0.4
_GRAV = 9.80665                       # const-ok: NEMO phycst grav, test mirror
_RCTV0 = 461.495 / 287.05 - 1.0       # const-ok: NEMO R_vap/R_dry, test mirror
_REPS0 = 287.05 / 461.495             # const-ok: NEMO R_dry/R_vap, test mirror
_RGAS = 8.314510                      # NEMO R_gas
_MDRY = 28.9647e-3                    # NEMO rmm_dryair
_MWAT = 18.0153e-3                    # NEMO rmm_water
_GAMMA_DRY = _RGAS / (_MDRY * 1005.0)  # NEMO rgamma_dry
_RT0 = 273.15                         # const-ok: NEMO rt0 in the test mirror


def _np_e_sat_goff(T):
    """NEMO sbc_phy e_sat (Goff 1957), straight from the Fortran."""
    T = np.maximum(T, 180.0)
    rt = _RT0 / T
    return 100.0 * 10.0 ** (  # satcurve-ok: independent NEMO-Goff test mirror
        10.79574 * (1.0 - rt) - 5.028 * np.log10(T / _RT0)
        + 1.50475e-4 * (1.0 - 10.0 ** (-8.2969 * (T / _RT0 - 1.0)))
        + 0.42873e-3 * (10.0 ** (4.76955 * (1.0 - rt)) - 1.0) + 0.78614)


def _np_q_sat(T, p):
    e = _np_e_sat_goff(T)
    return _REPS0 * e / (p - (1.0 - _REPS0) * e)


def _np_pres_temp(q, slp, z, T_abs):
    """NEMO pres_temp (absolute-T branch, 3 iterations) — UNCLIPPED
    w = q/q_sat exactly like the Fortran, so it can falsify the
    implementation's garbage-cell guard on the physical domain."""
    T = np.maximum(T_abs, 180.0)
    p = np.asarray(slp, dtype=np.float64).copy()
    for _ in range(3):
        qs = _np_q_sat(T, p)
        w = q / qs
        xm = (1.0 - w) * _MDRY + w * _MWAT
        p = slp * np.exp(-_GRAV * xm * z / (_RGAS * T))
    return p


def _np_theta_exner(T, p):
    return T * (1.0e5 / p) ** _GAMMA_DRY


def _np_rho_air(T_abs, q, p):
    R_dry = 287.05  # const-ok: NEMO R_dry in the independent test mirror
    return np.maximum(p / (R_dry * T_abs * (1.0 + _RCTV0 * q)), 0.8)


def _np_cd_n10(w):
    w = np.maximum(w, 0.5)
    cd = (2.7 / w + 0.142 + w / 13.09 - 3.14807e-10 * w ** 6) * 1e-3
    cd = np.where(w >= 33.0, 2.34e-3, cd)
    return np.maximum(cd, 0.1e-3)


def _np_psi_m(z):
    z = np.clip(z, -10.0, 10.0)
    x2 = np.sqrt(np.abs(1.0 - 16.0 * z))
    x2 = np.maximum(x2, 1.0)
    x = np.sqrt(x2)
    unst = (2.0 * np.log(0.5 * (1.0 + x)) + np.log(0.5 * (1.0 + x2))
            - 2.0 * np.arctan(x) + 0.5 * np.pi)
    stab = -5.0 * z
    return np.where(z > 0.0, stab, unst)


def _np_psi_h(z):
    z = np.clip(z, -10.0, 10.0)
    x2 = np.sqrt(np.abs(1.0 - 16.0 * z))
    x2 = np.maximum(x2, 1.0)
    unst = 2.0 * np.log(0.5 * (1.0 + x2))
    stab = -5.0 * z
    return np.where(z > 0.0, stab, unst)


def _np_turb_ncar(theta, q, sst, ssq, wind, nb_iter=5):
    """Direct Fortran transcription (vector NumPy), zt == zu == 10 m."""
    Ub = np.maximum(wind, 0.5)
    dTv = theta * (1.0 + _RCTV0 * q) - sst * (1.0 + _RCTV0 * ssq)
    stab = np.where(dTv > 0.0, 1.0, 0.0)
    CdN = _np_cd_n10(Ub)
    sqCdN = np.sqrt(CdN)
    Cd = CdN.copy()
    Ce = np.maximum(34.6e-3 * sqCdN, 0.1e-3)
    Ch = np.maximum(1e-3 * sqCdN * (18.0 * stab + 32.7 * (1.0 - stab)), 0.1e-3)
    sqCd = sqCdN.copy()
    dt_ = theta - sst
    dq_ = q - ssq
    for _ in range(nb_iter):
        us = sqCd * Ub
        ts = Ch / sqCd * dt_
        qs = Ce / sqCd * dq_
        zqa = 1.0 + _RCTV0 * q
        invL = (_GRAV * _VK * (ts * zqa + _RCTV0 * theta * qs)
                / np.maximum(us * us * theta * zqa, 1e-9))
        invL = np.clip(invL, -200.0, 200.0)
        zeta = np.clip(10.0 * invL, -10.0, 10.0)
        psim = _np_psi_m(zeta)
        UN10 = np.maximum(0.25, Ub * (1.0 + sqCd / _VK * psim))
        CdN = _np_cd_n10(UN10)
        sqCdN = np.sqrt(CdN)
        denom = 1.0 - sqCdN / _VK * psim
        Cd = np.maximum(CdN / (denom * denom), 0.1e-3)
        sqCd = np.sqrt(Cd)
        zh = -_np_psi_h(zeta) / _VK / sqCdN
        ratio = sqCd / sqCdN
        stab = np.where(zeta > 0.0, 1.0, 0.0)
        ChN = np.maximum(1e-3 * sqCdN * (18.0 * stab + 32.7 * (1.0 - stab)),
                         0.1e-3)
        CeN = np.maximum(34.6e-3 * sqCdN, 0.1e-3)
        Ch = np.maximum(ChN * ratio / (1.0 + ChN * zh), 0.1e-3)
        Ce = np.maximum(CeN * ratio / (1.0 + CeN * zh), 0.1e-3)
    return Cd, Ch, Ce, Ub


def _np_fluxes(u, v, T_air, q_air, sst, slp):
    """Full NEMO open-water flux path in the NumPy mirror: preprocessing
    (pres_temp/theta_exner/ssq) -> turb_ncar -> BULK_FORMULA."""
    wind = np.sqrt(u * u + v * v + 1e-12)
    ssq = 0.98 * _np_q_sat(sst, slp)  # satcurve-ok: NEMO rdct_qsat_salt mirror
    p10 = _np_pres_temp(q_air, slp, 10.0, T_air)
    theta_air = _np_theta_exner(T_air, p10)
    theta_sst = _np_theta_exner(sst, slp)
    Cd, Ch, Ce, Ub = _np_turb_ncar(theta_air, q_air, theta_sst, ssq, wind)
    rho = _np_rho_air(T_air, q_air, p10)
    Urho = Ub * np.maximum(rho, 1.0)
    tau_x = -Urho * Cd * u
    tau_y = -Urho * Cd * v
    L_vap = 2.501e6 - (4218.0 - 1846.0) * (theta_sst - _RT0)  # const-ok: Kirchhoff L_v(T) mirror
    cp_a = 1005.0 + 1860.0 * q_air
    zevap = Urho * Ce * (q_air - ssq)
    sh = Urho * Ch * (theta_air - theta_sst) * cp_a
    lh = L_vap * zevap
    return tau_x, tau_y, sh, lh, -zevap


def _random_marine_inputs(n=4096, seed=7):
    rng = np.random.default_rng(seed)
    sst = rng.uniform(271.3, 303.0, n)                  # K
    theta = sst + rng.uniform(-8.0, 8.0, n)             # K (potential, at 10 m)
    q = rng.uniform(1e-4, 0.022, n)                     # kg/kg
    slp = rng.uniform(95000.0, 104000.0, n)             # Pa
    # ssq from the INDEPENDENT mirror (not the implementation under test)
    ssq = 0.98 * _np_q_sat(sst, slp)  # satcurve-ok: NEMO mirror
    wind = rng.uniform(0.0, 38.0, n)                    # m/s incl. calm+cyclone
    return theta, q, sst, ssq, wind, slp


# ---------------------------------------------------------------------------
# 1. Cross-implementation equivalence
# ---------------------------------------------------------------------------

def test_ncar_coefficients_match_independent_numpy_mirror():
    theta, q, sst, ssq, wind, _slp = _random_marine_inputs()
    Cd_np, Ch_np, Ce_np, Ub_np = _np_turb_ncar(theta, q, sst, ssq, wind)
    Cd, Ch, Ce, Ub = ncar_transfer_coefficients(
        jnp.asarray(theta), jnp.asarray(q), jnp.asarray(sst),
        jnp.asarray(ssq), jnp.asarray(wind))
    assert np.allclose(np.asarray(Ub), Ub_np, rtol=0, atol=0)
    assert np.allclose(np.asarray(Cd), Cd_np, rtol=1e-12, atol=1e-15)
    assert np.allclose(np.asarray(Ch), Ch_np, rtol=1e-12, atol=1e-15)
    assert np.allclose(np.asarray(Ce), Ce_np, rtol=1e-12, atol=1e-15)


def test_full_flux_path_matches_independent_numpy_mirror():
    """END-TO-END parity: air_sea_fluxes(algo='ncar') vs the NumPy Fortran
    mirror INCLUDING NEMO's preprocessing (pres_temp 10-m pressure, Exner
    potential air temperature, Exner potential SST, mirror ssq/rho/L_vap).
    Catches a transcription bug anywhere in the chain (codex round-1
    finding: the earlier gamma_moist shortcut + absolute-SST mix)."""
    rng = np.random.default_rng(23)
    n = 2048
    sst = rng.uniform(271.5, 302.0, n)
    T_air = sst + rng.uniform(-6.0, 6.0, n)             # ABSOLUTE at 10 m
    q = rng.uniform(2e-4, 0.020, n)
    slp = rng.uniform(96000.0, 104000.0, n)
    # Keep q within w = q/q_sat <= 1.5 (up to 150% RH): the random draw can
    # otherwise pair cold air with tropical humidity (w ~ 9, unphysical),
    # where the implementation's w<=2 garbage guard INTENTIONALLY departs
    # from the unclipped Fortran mirror.  Supersaturation parity has its own
    # dedicated test (q = 1.3 q_sat).
    q = np.minimum(q, 1.5 * _np_q_sat(T_air, slp))
    u = rng.uniform(-20.0, 20.0, n)
    v = rng.uniform(-20.0, 20.0, n)
    ref = _np_fluxes(u, v, T_air, q, sst, slp)
    out = air_sea_fluxes(
        u10=jnp.asarray(u), v10=jnp.asarray(v), T_air_K=jnp.asarray(T_air),
        q_air=jnp.asarray(q), T_sfc_K=jnp.asarray(sst),
        slp_Pa=jnp.asarray(slp))
    names = ("tau_x", "tau_y", "sh", "lh", "evap")
    for nm, a, b in zip(names, out, ref):
        assert np.allclose(np.asarray(a), b, rtol=1e-10, atol=1e-12), nm


def test_ncar_coefficient_magnitudes_physical():
    """Coefficients stay in the physically-plausible bulk range."""
    theta, q, sst, ssq, wind, _slp = _random_marine_inputs(seed=11)
    Cd, Ch, Ce, _ = ncar_transfer_coefficients(
        jnp.asarray(theta), jnp.asarray(q), jnp.asarray(sst),
        jnp.asarray(ssq), jnp.asarray(wind))
    for C in (Cd, Ch, Ce):
        a = np.asarray(C)
        assert np.isfinite(a).all()
        assert (a >= 0.1e-3 - 1e-12).all()
        # Calm-wind CdN reaches 5.54e-3 (NEMO 2.7/U term) and strong
        # instability amplifies it by up to 1/(1-sqrt(CdN)/k psi_m)^2 ~ 3.6x
        # at zeta=-10 -> bound ~2e-2; anything above is a real blunder.
        assert (a <= 2.5e-2).all()


# ---------------------------------------------------------------------------
# 2. Physical limits
# ---------------------------------------------------------------------------

def test_near_neutral_reduces_to_neutral_coefficients():
    """theta -> SST, q -> ssq  =>  Cd -> CdN, Ce -> 34.6e-3 sqrt(CdN)."""
    from legoesm.core.bulk_flux import large_yeager_neutral_cd
    wind = jnp.asarray([2.0, 5.0, 8.0, 12.0, 20.0])
    sst = jnp.full(wind.shape, 288.0)
    ssq = seawater_q_sat(sst)
    # tiny UNSTABLE offset (exact zero sits on the Fortran SIGN() knife edge)
    theta = sst - 1e-7
    Cd, Ch, Ce, Ub = ncar_transfer_coefficients(theta, ssq, sst, ssq, wind)
    CdN = large_yeager_neutral_cd(Ub, nemo_parity=True)
    assert np.allclose(np.asarray(Cd), np.asarray(CdN), rtol=1e-5)
    assert np.allclose(np.asarray(Ce),
                       34.6e-3 * np.sqrt(np.asarray(CdN)), rtol=1e-5)
    # near-neutral from the unstable side -> 32.7 coefficient
    assert np.allclose(np.asarray(Ch),
                       32.7e-3 * np.sqrt(np.asarray(CdN)), rtol=1e-5)


def test_stability_monotonicity():
    """Stable boundary layer suppresses transfer; unstable enhances it."""
    wind = jnp.asarray(7.0)
    sst = jnp.asarray(288.0)
    ssq = seawater_q_sat(sst)
    Cd_u, Ch_u, Ce_u, _ = ncar_transfer_coefficients(
        sst - 4.0, ssq, sst, ssq, wind)      # unstable (air colder)
    Cd_n, Ch_n, Ce_n, _ = ncar_transfer_coefficients(
        sst - 1e-7, ssq, sst, ssq, wind)     # ~neutral
    Cd_s, Ch_s, Ce_s, _ = ncar_transfer_coefficients(
        sst + 4.0, ssq, sst, ssq, wind)      # stable (air warmer)
    assert float(Cd_u) > float(Cd_n) > float(Cd_s)
    assert float(Ce_u) > float(Ce_n) > float(Ce_s)
    assert float(Ch_u) > float(Ch_s)


def test_goff_curve_reference_values():
    """Goff (1957) curve: ~611 Pa at 0 degC, ~3537 Pa at 300 K (Goff values),
    and within 0.5% of the Tetens default over the marine range."""
    e0 = float(saturation_vapor_pressure_goff(jnp.asarray(constants.T_freeze)))
    assert abs(e0 - 611.0) < 2.5
    e300 = float(saturation_vapor_pressure_goff(jnp.asarray(300.0)))
    assert abs(e300 - 3537.0) < 35.0
    from legoesm.thermo import saturation_vapor_pressure
    T = jnp.linspace(271.0, 305.0, 35)
    rel = np.abs(np.asarray(saturation_vapor_pressure_goff(T))
                 / np.asarray(saturation_vapor_pressure(T)) - 1.0)
    assert rel.max() < 0.005


def test_seawater_q_sat_salt_factor_and_pressure():
    T = jnp.asarray(293.0)
    q_std = float(seawater_q_sat(T))
    e_s = float(saturation_vapor_pressure_goff(T))
    # NEMO reps0 = R_dry/R_vap with the NEMO gas constants (NOT
    # constants.epsilon, which uses legoESM's R_v and differs in the 5th
    # digit -- the implementation is NEMO-parity by design).
    q_pure = _REPS0 * e_s / (constants.p_atm_std - (1.0 - _REPS0) * e_s)
    assert q_std == pytest.approx(0.98 * q_pure, rel=1e-12)
    # lower pressure -> higher specific humidity at saturation
    assert float(seawater_q_sat(T, 95000.0)) > q_std


def test_rho_air_moist_reference_and_floor():
    rho_dry = float(rho_air_moist(288.15, 0.0, constants.p_atm_std))
    # rel 2e-6: the implementation floors q at 1e-6 like NEMO (rctv0*1e-6
    # ~ 6e-7 relative shift vs the exact dry-air value).
    assert rho_dry == pytest.approx(
        constants.p_atm_std / (constants.R_d * 288.15), rel=2e-6)
    # moisture lightens air
    assert float(rho_air_moist(288.15, 0.015, constants.p_atm_std)) < rho_dry
    # NEMO floor at 0.8
    assert float(rho_air_moist(400.0, 0.0, 50000.0)) >= 0.8


def test_l_vap_and_cp_air_reference():
    assert float(latent_heat_vaporization(constants.T_freeze)) == (
        pytest.approx(constants.L_v, rel=1e-12))
    assert float(latent_heat_vaporization(constants.T_freeze + 25.0)) == (
        pytest.approx(constants.L_v - (constants.c_pw - constants.c_pv) * 25.0, rel=1e-12))
    assert float(moist_air_cp(0.0)) == pytest.approx(
        constants.c_p_dry_air_nemo, rel=1e-12)
    assert float(moist_air_cp(0.01)) == pytest.approx(
        constants.c_p_dry_air_nemo + 0.01 * constants.c_p_vapor_nemo,
        rel=1e-12)


def test_pressure_at_height_supersaturated_matches_unclipped_nemo():
    """SUPERSATURATED air (q = 1.3 q_sat, RH 130%): the implementation must
    still match the UNCLIPPED NEMO formula bit-for-bit — the garbage guard
    clips only at w > 2, beyond any physical supersaturation (codex
    round-2 MED: a clip at 1.0 would silently diverge from NEMO here)."""
    T, slp = 275.0, 101000.0
    q_sat0 = float(_np_q_sat(np.float64(T), np.float64(slp)))
    q_super = 1.3 * q_sat0
    p_impl = float(pressure_at_height(
        jnp.asarray(q_super), jnp.asarray(slp), 10.0, jnp.asarray(T)))
    p_ref = float(_np_pres_temp(np.float64(q_super), np.float64(slp), 10.0,
                                np.float64(T)))
    assert p_impl == pytest.approx(p_ref, rel=1e-13)
    # and the guard still protects true garbage (land cell at the T floor)
    p_junk = float(pressure_at_height(
        jnp.asarray(0.01), jnp.asarray(slp), 10.0, jnp.asarray(150.0)))
    assert np.isfinite(p_junk) and 0.0 < p_junk <= slp


def test_pressure_at_height_and_exner_theta():
    """NEMO preprocessing reference values: the 10-m pressure sits ~120 Pa
    below slp; theta at 10 m exceeds the absolute T by ~0.09-0.11 K (the
    adiabatic 10-m lift); the potential SST at slp>1e5 is COLDER than the
    absolute SST (Exner reference 1e5 Pa)."""
    T, q, slp = 288.0, 0.008, 101325.0
    p10 = float(pressure_at_height(jnp.asarray(q), jnp.asarray(slp), 10.0,
                                   jnp.asarray(T)))
    assert 80.0 < slp - p10 < 160.0
    th_air = float(exner_potential_temperature(jnp.asarray(T),
                                               jnp.asarray(p10)))
    th_sst = float(exner_potential_temperature(jnp.asarray(T),
                                               jnp.asarray(slp)))
    # both referenced to 1e5 Pa: at slp > 1e5 the Exner factor < 1...
    assert th_sst < T
    # ...but the AIR-SEA pair (same Exner reference) keeps the ~+0.1 K
    # 10-m adiabatic separation that gamma*z used to approximate.
    assert 0.05 < th_air - th_sst < 0.16
    # cross-check vs the independent mirror
    assert p10 == pytest.approx(
        float(_np_pres_temp(np.float64(q), np.float64(slp), 10.0,
                            np.float64(T))), rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Flux-level checks
# ---------------------------------------------------------------------------

def test_fluxes_signs_and_consistency():
    tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
        u10=jnp.asarray(8.0), v10=jnp.asarray(-3.0),
        T_air_K=jnp.asarray(285.0), q_air=jnp.asarray(0.006),
        T_sfc_K=jnp.asarray(291.0), slp_Pa=jnp.asarray(101000.0))
    # atmospheric convention: tau opposes the wind components
    assert float(tau_x) < 0.0 and float(tau_y) > 0.0
    # warm ocean under cold dry air: ocean loses sensible + latent heat
    assert float(sh) < 0.0 and float(lh) < 0.0 and float(evap) > 0.0
    # evap-lh consistency: NEMO evaluates L_vap at the POTENTIAL SST
    # (BULK_FORMULA pTs = zsspt), not the absolute SST.
    theta_sst = exner_potential_temperature(jnp.asarray(291.0),
                                            jnp.asarray(101000.0))
    L = float(latent_heat_vaporization(theta_sst))
    assert float(evap) == pytest.approx(-float(lh) / L, rel=1e-12)


def test_flux_magnitudes_tropical_reference():
    """Trade-wind regime: latent O(100 W/m2), sensible O(10 W/m2),
    stress O(0.05 Pa) -- guards against unit / coefficient blunders."""
    tau_x, _, sh, lh, evap = air_sea_fluxes(
        u10=jnp.asarray(7.0), v10=jnp.asarray(0.0),
        T_air_K=jnp.asarray(299.0), q_air=jnp.asarray(0.016),
        T_sfc_K=jnp.asarray(300.5), slp_Pa=jnp.asarray(101000.0))
    assert 0.02 < abs(float(tau_x)) < 0.15
    assert 40.0 < -float(lh) < 250.0
    assert 0.0 < -float(sh) < 40.0
    assert 1.5e-5 < float(evap) < 1.0e-4


def test_air_sea_fluxes_jit_and_grad():
    """The fixed-point loop is unrolled + jnp.where-branched: jit and grad
    must work and give finite gradients (AD safety for DA/tuning)."""
    def loss(sst):
        _, _, sh, lh, _ = air_sea_fluxes(
            u10=jnp.asarray(6.0), v10=jnp.asarray(2.0),
            T_air_K=jnp.asarray(287.5),  # const-ok: air temperature [K]
            q_air=jnp.asarray(0.008),
            T_sfc_K=sst, slp_Pa=jnp.asarray(101325.0))
        return sh + lh

    g = jax.grad(loss)(jnp.asarray(290.0))
    assert np.isfinite(float(g))
    # heat loss increases (more negative q) as the ocean warms
    assert float(g) < 0.0
    lj = jax.jit(loss)(jnp.asarray(290.0))
    assert np.isfinite(float(lj))
    assert float(lj) == pytest.approx(float(loss(jnp.asarray(290.0))),
                                      rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Legacy scheme + dispatch hardening
# ---------------------------------------------------------------------------

def test_legacy_two_coeff_scheme_pinned():
    """algo='ly09_2coeff' reproduces the exact pre-NCAR formula."""
    u, v = 10.0, 0.0
    T_air, q_air, sst, q_sfc, rho = 290.0, 0.005, 295.0, 0.012, 1.2
    tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
        u10=jnp.asarray(u), v10=jnp.asarray(v), T_air_K=jnp.asarray(T_air),
        q_air=jnp.asarray(q_air), T_sfc_K=jnp.asarray(sst),
        q_sfc=jnp.asarray(q_sfc), rho_air=jnp.asarray(rho),
        algo="ly09_2coeff")
    from legoesm.core.bulk_flux import large_yeager_neutral_cd
    w = np.sqrt(u * u + v * v + 1e-12)
    Cd = float(large_yeager_neutral_cd(jnp.asarray(w)))   # legacy clip bounds
    Ch = 1.46e-3  # unstable: sst > T_air
    assert float(tau_x) == pytest.approx(-rho * Cd * w * u, rel=1e-12)
    assert float(sh) == pytest.approx(
        rho * constants.c_pd * Ch * w * (T_air - sst), rel=1e-12)
    assert float(lh) == pytest.approx(
        rho * constants.L_v * Ch * w * (q_air - q_sfc), rel=1e-12)
    assert float(evap) == pytest.approx(-float(lh) / constants.L_v, rel=1e-12)


def test_unknown_algo_raises():
    with pytest.raises(ValueError, match="Unknown OMIP bulk algo"):
        air_sea_fluxes(
            u10=jnp.asarray(1.0), v10=jnp.asarray(0.0),
            T_air_K=jnp.asarray(280.0), q_air=jnp.asarray(0.005),
            T_sfc_K=jnp.asarray(281.0), algo="nope")


def test_legacy_requires_q_sfc():
    with pytest.raises(ValueError, match="requires an explicit"):
        air_sea_fluxes(
            u10=jnp.asarray(1.0), v10=jnp.asarray(0.0),
            T_air_K=jnp.asarray(280.0), q_air=jnp.asarray(0.005),
            T_sfc_K=jnp.asarray(281.0), algo="ly09_2coeff")
