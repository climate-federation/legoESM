"""NEMO turb_ncar zt /= zu shift (air T/q at 2 m, wind at 10 m) in the K0 card's NCAR path.

Oracle: a scalar fp64 NumPy transcription of sbcblk_algo_ncar.F90 turb_ncar
(lines 113-217, NEMO 5.0.1 round-55 WORK copy) and its helpers cd_n10_ncar,
psi_m_ncar, psi_h_ncar (same file) and One_on_L / UN10_from_CD / z0_from_Cd
(sbc_phy.F90), written from the Fortran, not from the JAX port.  RULE 5: the
hard limit (q_zu_width = 0) matches the oracle to 1e-12; the smooth default
records its per-case gap; check_grads runs through the shift at the default.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean import bulk_flux_omip as bf

from legoesm import constants

jax.config.update("jax_enable_x64", True)

VK, GRAV = constants.kappa_von_karman, constants.g_nemo
RCTV0 = constants.R_v_nemo / constants.R_d - 1.0
CX_MIN = 0.1e-3


def _sign(a, b):                       # Fortran SIGN(a, b)
    return abs(a) if b >= 0.0 else -abs(a)


def _cd_n10(w):
    gt33 = 0.5 + _sign(0.5, w - 33.0)
    cd = 1e-3 * ((1.0 - gt33) * (2.7 / w + 0.142 + w / 13.09 - 3.14807e-10 * w ** 6) + gt33 * 2.34)
    return max(cd, CX_MIN)


def _psi_m(z):
    x2 = max(math.sqrt(abs(1.0 - 16.0 * z)), 1.0)
    x = math.sqrt(x2)
    unst = (2.0 * math.log((1.0 + x) * 0.5) + math.log((1.0 + x2) * 0.5)
            - 2.0 * math.atan(x) + math.pi * 0.5)
    st = 0.5 + _sign(0.5, z)
    return st * (-5.0 * z) + (1.0 - st) * unst


def _psi_h(z):
    x2 = max(math.sqrt(abs(1.0 - 16.0 * z)), 1.0)
    st = 0.5 + _sign(0.5, z)
    return st * (-5.0 * z) + (1.0 - st) * 2.0 * math.log(0.5 * (1.0 + x2))


def _clamp10(z):
    return _sign(min(abs(z), 10.0), z)


def turb_ncar(zt, zu, sst, t_zt, ssq, q_zt, u_zu, nbit=5):
    """sbcblk_algo_ncar.F90 turb_ncar, ln_cdgw = F; returns Cd, Ch, Ce, t_zu, q_zu."""
    eq = abs(zu - zt) < 0.01
    ub = max(0.5, u_zu)
    dtv = t_zt * (1.0 + RCTV0 * q_zt) - sst * (1.0 + RCTV0 * ssq)
    stab = 0.5 + _sign(0.5, dtv)
    cdn = _cd_n10(ub)
    scdn = math.sqrt(cdn)
    cd = cdn
    ce = max(1e-3 * (34.6 * scdn), CX_MIN)
    ch = max(1e-3 * scdn * (18.0 * stab + 32.7 * (1.0 - stab)), CX_MIN)
    scd = scdn
    t_zu, q_zu = max(t_zt, 180.0), max(q_zt, 1e-6)
    for _ in range(nbit):
        tst = ch / scd * (t_zu - sst)
        qst = ce / scd * (q_zu - ssq)
        us = scd * ub
        zqa = 1.0 + RCTV0 * q_zu
        inv_l = GRAV * VK * (tst * zqa + RCTV0 * t_zu * qst) / max(us * us * t_zu * zqa, 1e-9)
        inv_l = _sign(min(abs(inv_l), 200.0), inv_l)
        zeta_u = _clamp10(zu * inv_l)
        if not eq:
            zeta_t = _clamp10(zt * inv_l)
            shift = math.log(zt / zu) + _psi_h(zeta_u) - _psi_h(zeta_t)
            t_zu = t_zt - tst / VK * shift
            q_zu = max(0.0, q_zt - qst / VK * shift)
        psim = _psi_m(zeta_u)
        z0 = zu * math.exp(-(VK / math.sqrt(cd) + psim))
        un10 = max(0.25, math.sqrt(cd) * ub / VK * math.log(10.0 / z0))
        cdn = _cd_n10(un10)
        scdn = math.sqrt(cdn)
        t1 = 1.0 + scdn / VK * (math.log(zu / 10.0) - psim)
        cd = max(cdn / (t1 * t1), CX_MIN)
        scd = math.sqrt(cd)
        t0 = (math.log(zu / 10.0) - _psi_h(zeta_u)) / VK / scdn
        t2 = scd / scdn
        st = 0.5 + _sign(0.5, zeta_u)
        chn = 1e-3 * scdn * (18.0 * st + 32.7 * (1.0 - st))
        cen = 1e-3 * (34.6 * scdn)
        ch = max(chn * t2 / (1.0 + chn * t0), CX_MIN)
        ce = max(cen * t2 / (1.0 + cen * t0), CX_MIN)
    return cd, ch, ce, t_zu, q_zu


# (potential air T at zt [K], q_zt, potential SST [K], ssq, wind [m/s])
CASES = {
    "unstable": (288.0, 0.008, 293.0, 0.0145, 7.0),
    "stable": (291.0, 0.010, 285.0, 0.0095, 5.0),
    "low_wind_unstable": (283.0, 0.005, 290.0, 0.0125, 0.8),
    "strong_wind": (300.0, 0.017, 301.5, 0.0230, 25.0),
    "dry_floor_binds": (270.0, 1.0e-9, 285.0, 0.0090, 12.0),
    # q_zt bisected on the oracle so the hard q_zu = 3.3e-9, inside the 1e-8 band
    "dry_in_smoothing_band": (270.0, 0.0007020222010943271, 285.0, 0.0090, 12.0),
    # warm air over ice-cold sea, calm: |zt/L| reaches ~336, so the zeta_t clamp binds
    "very_stable_zeta_t_clamped": (300.0, 0.004, 271.5, 0.0030, 0.5),
    # cold air over a warm sea, calm: zt/L reaches about -35 (the -10 bound binds)
    "very_unstable_zeta_t_clamped": (260.0, 0.001, 300.0, 0.026, 0.5),
}
FIELDS = ("Cd", "Ch", "Ce", "theta_zu", "q_zu")


@pytest.fixture(autouse=True, scope="module")
def _fp64():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(previous)


def _port(case, zt_m, width):
    t, q, sst, ssq, u = (jnp.asarray(v, jnp.float64) for v in CASES[case])
    cd, ch, ce, _, theta_zu, q_zu = bf._ncar_transfer_coefficients_exact(
        t, q, sst, ssq, u, nb_iter=5, zt_m=zt_m, q_zu_width=width)
    return np.array([float(x) for x in (cd, ch, ce, theta_zu, q_zu)])


def _rel(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


@pytest.mark.parametrize("case", sorted(CASES))
def test_hard_limit_matches_the_fortran_transcription(case):
    t, q, sst, ssq, u = CASES[case]
    ref = np.array(turb_ncar(2.0, 10.0, sst, t, ssq, q, u))
    got = _port(case, 2.0, 0.0)
    assert np.all(_rel(got, ref) <= 1e-12) or np.allclose(got, ref, rtol=1e-12, atol=1e-18), (
        dict(zip(FIELDS, _rel(got, ref))))


def test_the_shift_does_something_and_the_floor_binds_in_the_dry_case():
    t, q, sst, ssq, u = CASES["unstable"]
    shifted = turb_ncar(2.0, 10.0, sst, t, ssq, q, u)
    unshifted = turb_ncar(10.0, 10.0, sst, t, ssq, q, u)
    assert abs(shifted[3] - unshifted[3]) > 1e-2 and abs(shifted[4] - unshifted[4]) > 1e-5
    t, q, sst, ssq, u = CASES["dry_floor_binds"]
    assert turb_ncar(2.0, 10.0, sst, t, ssq, q, u)[4] == 0.0


# Smooth default (q_zu_width = card default 1e-8 kg/kg): measured relative gap
# to the hard NEMO form per case, fields (Cd, Ch, Ce, theta_zu, q_zu), measured
# 2026-11-18.  Only a shifted humidity within the band moves: 3.3e-9 -> 1.82e-9
# kg/kg (45 % of a value 1e-6 of typical humidity); everything else is exact.
DEFAULT_GAPS = {
    "dry_floor_binds": [0.0, 0.0, 0.0, 0.0, 0.0],
    "dry_in_smoothing_band": [0.0, 0.0, 0.0, 0.0, 0.448899999994],
    "low_wind_unstable": [0.0, 0.0, 0.0, 0.0, 0.0],
    "stable": [0.0, 0.0, 0.0, 0.0, 0.0],
    "strong_wind": [0.0, 0.0, 0.0, 0.0, 0.0],
    "unstable": [0.0, 0.0, 0.0, 0.0, 0.0],
    "very_stable_zeta_t_clamped": [0.0, 0.0, 0.0, 0.0, 0.0],
    "very_unstable_zeta_t_clamped": [0.0, 0.0, 0.0, 0.0, 0.0],
}


def test_smooth_default_gap_per_case():
    width = bf.NemoStandaloneK0OceanCard._field_defaults["q_zu_floor_width_kg_kg"]
    for case, gaps in DEFAULT_GAPS.items():
        got, hard = _port(case, 2.0, width), _port(case, 2.0, 0.0)
        np.testing.assert_allclose(_rel(got, hard), gaps, rtol=1e-6, atol=1e-12, err_msg=case)
    assert set(DEFAULT_GAPS) == set(CASES)


@pytest.mark.parametrize("case", ["unstable", "stable", "dry_in_smoothing_band"])
def test_gradient_through_the_shift_at_the_default(case):
    """d(q_zu)/d(q_zt) and d(theta_zu)/d(theta_zt) at the card's default width.

    In the band case the final q_zu sits inside the smoothing band, so the
    derivative runs through positive_part_c1; the q step is 1e-3 of the width."""
    width = bf.NemoStandaloneK0OceanCard._field_defaults["q_zu_floor_width_kg_kg"]
    t, q, sst, ssq, u = CASES[case]

    def out(t_air, q_air):
        _, _, _, _, th, qz = bf._ncar_transfer_coefficients_exact(
            t_air, q_air, jnp.float64(sst), jnp.float64(ssq), jnp.float64(u),
            nb_iter=5, zt_m=2.0, q_zu_width=width)
        return th, qz

    check_grads(lambda tt: out(tt, jnp.float64(q))[0], (jnp.float64(t),),
                order=1, modes=("rev",), eps=1e-6)
    check_grads(lambda qq: out(jnp.float64(t), qq)[1], (jnp.float64(q),),
                order=1, modes=("rev",), eps=1e-3 * width)
    dq = float(jax.grad(lambda qq: out(jnp.float64(t), qq)[1])(jnp.float64(q)))
    assert np.isfinite(dq) and dq > 0.0


def test_zt_equal_zu_is_bit_identical_to_the_default_call():
    for case in CASES:
        t, q, sst, ssq, u = (jnp.asarray(v, jnp.float64) for v in CASES[case])
        old = bf._ncar_transfer_coefficients_exact(t, q, sst, ssq, u, nb_iter=5)
        new = bf._ncar_transfer_coefficients_exact(t, q, sst, ssq, u, nb_iter=5,
                                                   zt_m=10.0, q_zu_width=1e-8)
        for a, b in zip(old, new, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_zt_not_zu_without_a_width_or_with_bad_values_refuses():
    t, q, sst, ssq, u = (jnp.asarray(v, jnp.float64) for v in CASES["unstable"])
    with pytest.raises(ValueError, match="q_zu_width"):
        bf._ncar_transfer_coefficients_exact(t, q, sst, ssq, u, nb_iter=5, zt_m=2.0)
    for zt, w in ((-2.0, 1e-8), (2.0, -1e-8), (2.0, np.inf), (np.inf, 1e-8), (10.0, -1.0)):
        with pytest.raises(ValueError, match="zt_m > 0"):
            bf._ncar_transfer_coefficients_exact(t, q, sst, ssq, u, nb_iter=5,
                                                 zt_m=zt, q_zu_width=w)


def test_turbulent_path_takes_air_temperature_potential_at_zt():
    """sbcblk.F90:612: theta_air_zt = theta_exner(T, pres_temp(q, slp, rn_zqt, T))."""
    f = jnp.float64
    t_abs, q, slp = f(288.0), f(0.008), f(101325.0)
    st = bf._nemo_ncar_turbulent(f(7.0), f(0.0), t_abs, q, f(293.0), slp, f(1.0), 1.0,
                                 nb_iter=5, zt_m=2.0, q_zu_width=1e-8)
    p2 = bf.pressure_at_height(q, slp, 2.0, t_abs, nemo_source_exact=True)
    expect = bf.exner_potential_temperature(t_abs, p2, nemo_source_exact=True)
    assert float(st.theta_air) == float(expect)
    st10 = bf._nemo_ncar_turbulent(f(7.0), f(0.0), t_abs, q, f(293.0), slp, f(1.0), 1.0, nb_iter=5)
    assert float(st.theta_air) < float(st10.theta_air)         # 2 m pressure is higher
    assert abs(float(st.sensible) - float(st10.sensible)) > 1.0  # the shift reaches the fluxes
    # sbcblk.F90:862-871: air density at rn_zu from (theta_zu, q_zu); BULK_FORMULA at zu
    sst = f(293.0)
    th_sst = bf.exner_potential_temperature(sst, slp, nemo_source_exact=True)
    ssq = bf.seawater_q_sat(sst, slp, nemo_source_exact=True)
    _, ch, _, ub, th_zu, q_zu = bf._ncar_transfer_coefficients_exact(
        st.theta_air, q, th_sst, ssq, f(7.0), nb_iter=5, zt_m=2.0, q_zu_width=1e-8)
    p_zu, t_abs_zu = bf._pressure_from_potential(q_zu, slp, 10.0, th_zu)
    rho = bf.rho_air_moist(t_abs_zu, q_zu, p_zu, nemo_source_exact=True)
    sens = (ub * max(float(rho), 1.0) * ch * (th_zu - th_sst)
            * bf.moist_air_cp(q_zu, nemo_source_exact=True))
    np.testing.assert_allclose(float(st.rhoa), float(rho), rtol=1e-13)
    np.testing.assert_allclose(float(st.sensible), float(sens), rtol=1e-12)


def test_card_and_public_entry_points_pass_the_heights_on():
    """The executing K0 program (_nemo_orca2_k0_surface_compiled) hands the card's
    zt_m / q_zu_width to the NCAR path.  The public literal boundary
    nemo_blk_oce_ncar keeps its no-defaults signature (test_ncar_bulk) and stays
    CORE-only (air T/q at 10 m)."""
    import inspect

    src = inspect.getsource(inspect.unwrap(bf._nemo_orca2_k0_surface_compiled))
    assert "zt_m=card.zt_m" in src and "q_zu_width=card.q_zu_floor_width_kg_kg" in src
    s = inspect.getsource(bf._nemo_blk_oce_ncar)
    assert "zt_m=zt_m, q_zu_width=q_zu_width" in s


def test_card_carries_zt_and_width_and_core_card_is_10m():
    d = bf.NemoStandaloneK0OceanCard._field_defaults
    assert d["zt_m"] == 10.0 and d["q_zu_floor_width_kg_kg"] == pytest.approx(1e-8)
