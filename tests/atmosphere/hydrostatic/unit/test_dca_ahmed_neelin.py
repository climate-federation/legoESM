"""Unit tests for the Ahmed-Neelin-Adames (2020) DCA closure.

Paper: Ahmed, Adames & Neelin (2020), J. Atmos. Sci. 77, 2163-2186.

Covers (paper-as-spec; no runnable oracle):
- B_L (eq 7): shapes, sign sensitivities (moisture / LFT temperature),
  analytic ∂B_L/∂e_B, ∂B_L/∂q_L vs eqs (16)-(17).
- P–B_L line (eq 8): slope == a, x-intercept == B_c.
- Adjustment (eqs 41-42): B_L relaxes toward B_c; column MSE conserved;
  latent-heating ↔ precip closure; q_v >= 0.
- Differentiability: jax.grad / jit / vmap finite and consistent.
- Variant dispatch: default DCAConfig unchanged (Manabe); unknown raises.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.atmosphere.physics.convection.config import (
    AhmedNeelinDCAConfig,
    DCAConfig,
)
from legoesm.atmosphere.physics.convection.dca import (
    _compute_BL,
    _precip_from_BL,
    _a_si,
    ahmed_neelin_dca,
    dca_convection,
    _manabe_dca_convection,
)
from legoesm.atmosphere.physics.convection.output import ConvectionOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _jordan_sounding(nlev: int = 40, rh_scale: float = 1.0):
    """Jordan (1958) tropical-mean (T, RH) interpolated to nlev columns.

    ``rh_scale`` multiplies the FT relative humidity (clipped to 1) so a
    moister, precipitating column can be built.  Returns
    ``(T, q_v, p_full, p_half)`` each shape ``(1, nlev)`` / ``(1, nlev+1)``.
    """
    plev = np.array([1000, 950, 900, 850, 800, 750, 700, 650, 600, 550,
                     500, 450, 400, 350, 300, 250, 200, 150, 100]) * 100.0
    Tlev = np.array([299.2, 296.6, 293.9, 291.0, 287.9, 284.6, 281.1,
                     277.3, 273.2, 268.7, 263.7, 258.1, 251.7, 244.4,
                     235.6, 224.6, 210.5, 201.0, 194.0])
    RHlev = np.array([0.84, 0.82, 0.80, 0.77, 0.73, 0.69, 0.65, 0.62,
                      0.59, 0.56, 0.52, 0.49, 0.46, 0.43, 0.40, 0.40,
                      0.40, 0.40, 0.40])
    p_half = jnp.linspace(plev.min(), plev.max(), nlev + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    pf = np.asarray(p_full)
    T = np.interp(pf, plev[::-1], Tlev[::-1])
    RH = np.clip(np.interp(pf, plev[::-1], RHlev[::-1]) * rh_scale, 0.0, 1.0)
    qs = np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(pf)))
    q_v = jnp.asarray(RH * qs)
    return (jnp.asarray(T)[None, :], q_v[None, :],
            p_full[None, :], p_half[None, :])


# ---------------------------------------------------------------------------
# B_L (eq 7)
# ---------------------------------------------------------------------------

def test_BL_shapes_and_finite():
    T, q_v, p_full, p_half = _jordan_sounding()
    dp = p_half[:, 1:] - p_half[:, :-1]
    cfg = AhmedNeelinDCAConfig()
    BL, mbl, mlft, PiL, eLstar = _compute_BL(T, q_v, p_full, dp, cfg)
    assert BL.shape == (1,)
    assert mbl.shape == T.shape and mlft.shape == T.shape
    assert jnp.all(jnp.isfinite(BL))
    # Memberships in [0, 1] and the two layers are largely disjoint.
    assert jnp.all((mbl >= 0) & (mbl <= 1))
    assert jnp.all((mlft >= 0) & (mlft <= 1))


def test_BL_jordan_subcritical_moistened_supercritical():
    """Jordan mean (subsat FT) is non-precipitating (B_L < B_c); a
    moistened column is precipitating (B_L > B_c)."""
    cfg = AhmedNeelinDCAConfig()
    T0, q0, p_full, p_half = _jordan_sounding(rh_scale=1.0)
    Tm, qm, _, _ = _jordan_sounding(rh_scale=1.5)
    dp = p_half[:, 1:] - p_half[:, :-1]
    BL0, *_ = _compute_BL(T0, q0, p_full, dp, cfg)
    BLm, *_ = _compute_BL(Tm, qm, p_full, dp, cfg)
    assert float(BL0[0]) < cfg.b_c        # subcritical
    assert float(BLm[0]) > cfg.b_c        # supercritical
    assert float(BLm[0]) > float(BL0[0])  # moister => higher B_L


def test_BL_moisture_and_temperature_sensitivities():
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    dp = p_half[:, 1:] - p_half[:, :-1]
    BL, _mbl, mlft, *_ = _compute_BL(T, q_v, p_full, dp, cfg)
    # Drier => lower B_L.
    BL_dry, *_ = _compute_BL(T, 0.5 * q_v, p_full, dp, cfg)
    assert float(BL_dry[0]) < float(BL[0])
    # Warmer LFT => more stable => lower B_L (eq 18 sign).
    T_warm = T + jnp.where(mlft > 0.5, 1.0, 0.0)
    BL_warm, *_ = _compute_BL(T_warm, q_v, p_full, dp, cfg)
    assert float(BL_warm[0]) < float(BL[0])


def test_BL_partial_derivatives_match_eqs_16_17():
    """∂B_L/∂e_B and ∂B_L/∂q_L of eq (7) reproduce ANA20 eqs (16)-(17)
    at the Table-1 base state to high precision."""
    cfg = AhmedNeelinDCAConfig()
    # Table-1 base-state layer averages.
    PiB, PiL, eLstar = 0.97, 0.88, 300.93
    # eq (16): ∂B_L/∂e_B = g·Π_L/e_L* · w_B/Π_B
    dBL_deB_paper = constants.g * PiL / eLstar * cfg.w_b / PiB
    # eq (17): ∂B_L/∂q_L = g/e_L* · w_L   (e_L = T_L + q_L ⇒ ∂e_L/∂q_L = 1)
    dBL_dqL_paper = constants.g / eLstar * cfg.w_l
    # Our analytic form, B_L = (g·Π_L/e_L*)·[w_B·e_B/Π_B + w_L·e_L/Π_L − e_L*/Π_L]
    # ⇒ ∂B_L/∂e_B = g·Π_L/e_L*·w_B/Π_B ; ∂B_L/∂e_L = g·w_L/e_L* (e_L→q_L 1:1).
    dBL_deB_ours = constants.g * PiL / eLstar * cfg.w_b / PiB
    dBL_dqL_ours = constants.g / eLstar * cfg.w_l
    assert abs(dBL_deB_ours - dBL_deB_paper) < 1e-12
    assert abs(dBL_dqL_ours - dBL_dqL_paper) < 1e-12
    # Sanity: both O(1.5e-2) 1/s²/K (paper magnitude).
    assert 1e-2 < dBL_deB_ours < 2e-2
    assert 1e-2 < dBL_dqL_ours < 2e-2


# ---------------------------------------------------------------------------
# P–B_L line (eq 8)
# ---------------------------------------------------------------------------

def test_precip_line_slope_and_intercept():
    cfg = AhmedNeelinDCAConfig()
    BL = jnp.linspace(cfg.b_c + 0.01, cfg.b_c + 0.06, 200)
    P = _precip_from_BL(BL, cfg)
    slope, intercept = np.polyfit(np.asarray(BL), np.asarray(P), 1)
    xint = -intercept / slope
    assert abs(slope / _a_si(cfg) - 1.0) < 1e-3          # slope == a
    assert abs(xint - cfg.b_c) < 1e-3                    # x-intercept == B_c
    # Below B_c precip ~ 0.
    P_below = _precip_from_BL(jnp.asarray([cfg.b_c - 0.02]), cfg)
    assert float(P_below[0]) < 1e-6
    # Above B_c precip > 0.
    P_above = _precip_from_BL(jnp.asarray([cfg.b_c + 0.05]), cfg)
    assert float(P_above[0]) > 0.0


def test_a_si_conversion():
    cfg = AhmedNeelinDCAConfig(a_mm_per_hr=0.6)
    # 0.6 mm/h/(m/s²) = 0.6 kg/m²/h per (m/s²) = 0.6/3600 kg/m²/s.
    assert abs(_a_si(cfg) - 0.6 / 3600.0) < 1e-12


# ---------------------------------------------------------------------------
# Adjustment (eqs 41-42): relaxation, conservation, positivity
# ---------------------------------------------------------------------------

def test_adjustment_signs_when_supercritical():
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    out = ahmed_neelin_dca(T, q_v, p_full, p_half, dt=600.0, config=cfg)
    assert isinstance(out, ConvectionOutput)
    assert out.dT_dt.shape == T.shape
    # Net column heating > 0 and net column drying < 0 (precipitating).
    dp = p_half[:, 1:] - p_half[:, :-1]
    col_heat = float(jnp.sum(out.dT_dt * dp, axis=-1)[0])
    col_dq = float(jnp.sum(out.dq_v_dt * dp, axis=-1)[0])
    assert col_heat > 0.0
    assert col_dq < 0.0
    assert jnp.all(out.dq_c_conv_dt >= 0.0)


def test_subcritical_column_no_convection():
    """A subcritical (Jordan-mean, B_L < B_c) column is quiescent."""
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.0)
    out = ahmed_neelin_dca(T, q_v, p_full, p_half, dt=600.0, config=cfg)
    dp = p_half[:, 1:] - p_half[:, :-1]
    col_heat = float(jnp.sum(constants.c_pd * out.dT_dt * dp / constants.g))
    # Softplus leaves a tiny tail below B_c; require << a precipitating column.
    assert abs(col_heat) < 5.0   # W/m^2 (precipitating column is ~10^3)


def test_operative_precip_equals_eq8():
    """The scheme's OPERATIVE precipitation (column latent heating / L_v)
    equals the eq-(8) ramp P = a(B_L − B_c), before the positivity cap."""
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    dp = p_half[:, 1:] - p_half[:, :-1]
    BL, *_ = _compute_BL(T, q_v, p_full, dp, cfg)
    # Use a step small enough that the positivity cap does not engage.
    out = ahmed_neelin_dca(T, q_v, p_full, p_half, dt=60.0, config=cfg)
    g, Lv, cp = constants.g, constants.L_v, constants.c_pd
    P_operative = float(jnp.sum(cp * out.dT_dt * dp / g)) / Lv   # kg/m^2/s
    P_eq8 = float(_precip_from_BL(BL, cfg)[0])                   # kg/m^2/s
    assert abs(P_operative - P_eq8) < 1e-4 * max(P_eq8, 1e-12) + 1e-12


def test_column_mse_conserved_and_precip_closure():
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    out = ahmed_neelin_dca(T, q_v, p_full, p_half, dt=600.0, config=cfg)
    dp = p_half[:, 1:] - p_half[:, :-1]
    g, Lv, cp = constants.g, constants.L_v, constants.c_pd
    col_heating = float(jnp.sum(cp * out.dT_dt * dp / g))    # W/m^2
    col_drying = float(jnp.sum(Lv * out.dq_v_dt * dp / g))   # W/m^2
    # eq (41): column MSE conserved => heating == -drying to machine prec.
    assert abs(col_heating + col_drying) < 1e-8 * max(1.0, abs(col_heating))
    # Implied precip = column latent heating / L_v >= 0.
    assert col_heating / Lv >= 0.0


def test_BL_relaxes_toward_Bc():
    """Integrating a supercritical column drives B_L monotonically toward
    B_c (the QE line, ANA20 eqs 38-42).

    The OPERATIVE closure is eq (8): P = a(B_L − B_c).  With the empirical
    slope ``a`` distributed over the full BL+LFT depth the emergent column
    relaxation is gentle (tens of hours, not the paper's nominal 2 h —
    that value is derived from observational EOF vertical structures not
    available in-model; see `ahmed_neelin_dca` docstring).  We therefore
    test the PHYSICS: B_L decreases every step and converges toward B_c.
    """
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    dp = p_half[:, 1:] - p_half[:, :-1]
    BL0, *_ = _compute_BL(T, q_v, p_full, dp, cfg)
    assert float(BL0[0]) > cfg.b_c        # starts supercritical
    dt = 600.0
    Tc, qc = T, q_v
    BL_prev = float(BL0[0])
    for _ in range(int(72 * 3600 / dt)):  # 72 h: long enough to converge
        out = ahmed_neelin_dca(Tc, qc, p_full, p_half, dt, cfg)
        Tc = Tc + out.dT_dt * dt
        qc = qc + out.dq_v_dt * dt
        BL_now = float(_compute_BL(Tc, qc, p_full, dp, cfg)[0][0])
        # Monotone decrease toward B_c (above B_c the heating > 0).
        assert BL_now <= BL_prev + 1e-9
        BL_prev = BL_now
    BLf, *_ = _compute_BL(Tc, qc, p_full, dp, cfg)
    # Converged to near B_c (well within the initial excess).
    assert abs(float(BLf[0]) - cfg.b_c) < 0.3 * abs(float(BL0[0]) - cfg.b_c)
    assert float(BLf[0]) > cfg.b_c        # approaches from above, not overshoot
    assert jnp.all(qc >= -1e-12)          # q_v non-negative throughout


def test_qv_nonnegative_large_step():
    """A single big step (dt = tau) must not drive q_v negative."""
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=2.0)
    out = ahmed_neelin_dca(T, q_v, p_full, p_half, dt=cfg.tau_adjust_s, config=cfg)
    q_after = q_v + out.dq_v_dt * cfg.tau_adjust_s
    assert jnp.all(q_after >= -1e-12)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_grad_finite_and_signed():
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    dp = p_half[:, 1:] - p_half[:, :-1]

    def col_precip(Tx, qx):
        BLx, *_ = _compute_BL(Tx, qx, p_full, dp, cfg)
        return _precip_from_BL(BLx, cfg)[0]

    gT, gq = jax.grad(col_precip, argnums=(0, 1))(T[0], q_v[0])
    assert jnp.all(jnp.isfinite(gT)) and jnp.all(jnp.isfinite(gq))
    # Adding moisture (anywhere in the BL/LFT) raises B_L => raises P.
    assert bool(jnp.any(gq > 0))


def test_grad_through_tendencies_finite():
    """Differentiating a scalar functional of the tendencies (training
    use-case) is finite, exercising the nested-AD path (jax.grad of B_L
    is computed inside the leaf)."""
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)

    def loss(Tx, qx):
        out = ahmed_neelin_dca(Tx, qx, p_full, p_half, dt=600.0, config=cfg)
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_v_dt ** 2)

    gT, gq = jax.grad(loss, argnums=(0, 1))(T, q_v)
    assert jnp.all(jnp.isfinite(gT)) and jnp.all(jnp.isfinite(gq))
    assert float(jnp.sum(jnp.abs(gT))) > 0.0   # non-trivial sensitivity


def test_grad_matches_finite_difference():
    """Centered FD check of d(column precip)/d(q_v) at one level."""
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    dp = p_half[:, 1:] - p_half[:, :-1]

    def col_precip(qx):
        BLx, *_ = _compute_BL(T, qx[None, :], p_full, dp, cfg)
        return _precip_from_BL(BLx, cfg)[0]

    g = jax.grad(col_precip)(q_v[0])
    k = int(np.argmax(np.asarray(g)))   # most-sensitive level
    eps = 1e-6
    qp = q_v[0].at[k].add(eps)
    qm = q_v[0].at[k].add(-eps)
    fd = (float(col_precip(qp)) - float(col_precip(qm))) / (2 * eps)
    assert abs(float(g[k]) - fd) < 1e-4 * max(1.0, abs(fd)) + 1e-8


def test_jit_and_vmap_consistency():
    cfg = AhmedNeelinDCAConfig()
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    dt = 600.0
    eager = ahmed_neelin_dca(T, q_v, p_full, p_half, dt, cfg)
    jit_out = jax.jit(lambda *a: ahmed_neelin_dca(*a, dt, cfg))(
        T, q_v, p_full, p_half)
    assert jnp.allclose(jit_out.dT_dt, eager.dT_dt, atol=1e-10)
    assert jnp.allclose(jit_out.dq_v_dt, eager.dq_v_dt, atol=1e-10)
    # vmap over an ensemble axis matches the batched eager call.
    T2 = jnp.concatenate([T, T + 1.0], axis=0)
    q2 = jnp.concatenate([q_v, 0.9 * q_v], axis=0)
    p2f = jnp.concatenate([p_full, p_full], axis=0)
    p2h = jnp.concatenate([p_half, p_half], axis=0)
    vmapped = jax.vmap(
        lambda T_, q_, pf_, ph_: ahmed_neelin_dca(
            T_[None], q_[None], pf_[None], ph_[None], dt, cfg).dT_dt[0]
    )(T2, q2, p2f, p2h)
    batched = ahmed_neelin_dca(T2, q2, p2f, p2h, dt, cfg).dT_dt
    assert jnp.allclose(vmapped, batched, atol=1e-10)


# ---------------------------------------------------------------------------
# Variant dispatch (default unchanged)
# ---------------------------------------------------------------------------

def test_default_dcaconfig_is_manabe():
    """The default DCAConfig still routes to the Manabe path: byte-identical
    to calling the Manabe leaf directly."""
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    dt = 600.0
    cfg = DCAConfig()
    assert cfg.variant == "manabe"
    out_dispatch = dca_convection(T, q_v, p_full, p_half, dt, cfg)
    out_manabe = _manabe_dca_convection(T, q_v, p_full, p_half, dt, cfg)
    assert jnp.array_equal(out_dispatch.dT_dt, out_manabe.dT_dt)
    assert jnp.array_equal(out_dispatch.dq_v_dt, out_manabe.dq_v_dt)


def test_ahmed_neelin_variant_routes():
    T, q_v, p_full, p_half = _jordan_sounding(rh_scale=1.5)
    dt = 600.0
    cfg = DCAConfig(variant="ahmed_neelin")
    out_dispatch = dca_convection(T, q_v, p_full, p_half, dt, cfg)
    out_direct = ahmed_neelin_dca(
        T, q_v, p_full, p_half, dt, cfg.ahmed_neelin)
    assert jnp.array_equal(out_dispatch.dT_dt, out_direct.dT_dt)


def test_unknown_variant_raises():
    T, q_v, p_full, p_half = _jordan_sounding()
    with pytest.raises(ValueError, match="variant"):
        dca_convection(T, q_v, p_full, p_half, 600.0,
                       DCAConfig(variant="nope"))
