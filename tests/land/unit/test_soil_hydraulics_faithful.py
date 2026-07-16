"""Soil-hydraulics retention/conductivity ORACLE-FAITHFULNESS tests.

``soil_hydraulics`` implements six pluggable water-retention + hydraulic-
conductivity models.  The three FAITHFUL closed forms (Clapp-Hornberger 1978,
van Genuchten-Mualem 1980, Brooks-Corey 1964) are pinned here against an
INDEPENDENT NumPy transcription of their canonical papers (rel 1e-9), plus a
cross-check of Clapp-Hornberger against the on-disk gSAM Simple-Land-Model
reference code ``SLM/soil_proc.f90``.

The existing tests (test_multilayer_land.py CH round-trip + vG-K monotonicity,
the Richards integration tests) never pin the retention/conductivity FORMS or
their exponents against the oracle — only bounds / inverse-consistency.  These
pin psi = psi_sat*Se^(-b), K = K_sat*Se^(2b+3), the vG-Mualem structure, the BC
Se^(3+2/lambda) exponent, and the constants; canary the Se-clip and vG
saturation-guard departures; and check AD-safety (x64 + float32) at the dry and
saturated boundaries where the guards engage.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.land.soil_hydraulics import (                        # noqa: E402
    SoilHydraulicsConfig,
    clapp_hornberger_psi, clapp_hornberger_theta,
    clapp_hornberger_K, clapp_hornberger_C,
    van_genuchten_Se, van_genuchten_theta, van_genuchten_psi,
    van_genuchten_K, van_genuchten_C,
    brooks_corey_Se, brooks_corey_theta, brooks_corey_psi, brooks_corey_K,
    theta_from_psi, hydraulic_conductivity,
)

_CFG = SoilHydraulicsConfig(
    retention_curve="clapp_hornberger",
    theta_r=0.078, theta_sat=0.43, alpha_vg=3.6, n_vg=1.56, K_sat=2.89e-6,
    psi_sat=-0.478, b_ch=5.39, psi_b=-0.478, lambda_bc=0.186)


# --- Clapp & Hornberger (1978) closed form -------------------------------------


def _ch_oracle(theta, cfg):
    """Independent Clapp & Hornberger (1978): (psi, K, C=dtheta/dpsi)."""
    Se = np.asarray(theta) / cfg.theta_sat
    psi = cfg.psi_sat * Se ** (-cfg.b_ch)
    K = cfg.K_sat * Se ** (2.0 * cfg.b_ch + 3.0)
    C = -cfg.theta_sat / (cfg.b_ch * cfg.psi_sat) * Se ** (cfg.b_ch + 1.0)
    return psi, K, C


def test_clapp_hornberger_matches_paper_oracle():
    """psi/K/C and the theta<->psi round trip match the CAMPBELL/CH power-law
    branch to round-off over the physical unsaturated range (Se-clip inactive).
    NOTE: Clapp & Hornberger (1978) also define a short parabolic near-saturation
    section; legoESM (like CLM/Noah) adopts the pure Campbell power law
    everywhere, which is the branch pinned here."""
    theta = np.linspace(0.1, 0.99, 12) * _CFG.theta_sat
    psi_o, K_o, C_o = _ch_oracle(theta, _CFG)
    psi_p = np.asarray(clapp_hornberger_psi(jnp.asarray(theta), _CFG))
    K_p = np.asarray(clapp_hornberger_K(jnp.asarray(theta), _CFG))
    C_p = np.asarray(clapp_hornberger_C(jnp.asarray(theta), _CFG))
    assert np.allclose(psi_p, psi_o, rtol=1e-9, atol=0.0)
    assert np.allclose(K_p, K_o, rtol=1e-9, atol=0.0)
    assert np.allclose(C_p, C_o, rtol=1e-9, atol=0.0)
    # theta(psi) inverts psi(theta).
    theta_back = np.asarray(clapp_hornberger_theta(jnp.asarray(psi_o), _CFG))
    assert np.allclose(theta_back, theta, rtol=1e-9, atol=0.0)


def test_clapp_hornberger_matches_gsam_slm_diffusivity_and_velocity():
    """Cross-check CH against the on-disk gSAM SLM soil_proc.f90.  The Fortran
    forms a depth-weighted TWO-NODE interface average; for an unfrozen,
    UNIFORM-Se interface that average reduces to Se**p, so its soil-water
    DIFFUSIVITY ks*B*|psi_sat|*Se^(B+2)/poro equals legoESM K*|dpsi/dtheta| =
    K/|C|, and its pore VELOCITY ks*Se^(2B+2)/poro equals K/(poro*Se).  This
    fixes the K exponent (2b+3) and the diffusivity exponent (b+2) against the
    reference model, not just the paper."""
    for Se in (0.2, 0.5, 0.8):
        theta = Se * _CFG.theta_sat
        K = float(clapp_hornberger_K(jnp.array(theta), _CFG))
        C = float(clapp_hornberger_C(jnp.array(theta), _CFG))
        D_slm = (_CFG.K_sat * _CFG.b_ch * abs(_CFG.psi_sat)
                 * Se ** (_CFG.b_ch + 2.0) / _CFG.theta_sat)
        assert K / abs(C) == pytest.approx(D_slm, rel=1e-9, abs=0.0)          # diffusivity
        v_slm = _CFG.K_sat * Se ** (2.0 * _CFG.b_ch + 2.0) / _CFG.theta_sat
        assert K / (_CFG.theta_sat * Se) == pytest.approx(v_slm, rel=1e-9, abs=0.0)  # velocity


def test_clapp_hornberger_K_exponent_is_2b_plus_3():
    """Non-vacuity: the CH conductivity exponent is exactly 2b+3, not 2b+2 or
    2b+4 (a regression to either moves K by >1% at Se=0.5)."""
    Se = 0.5
    theta = Se * _CFG.theta_sat
    K = float(clapp_hornberger_K(jnp.array(theta), _CFG))
    assert K == pytest.approx(_CFG.K_sat * Se ** (2 * _CFG.b_ch + 3), rel=1e-12, abs=0.0)
    assert abs(K - _CFG.K_sat * Se ** (2 * _CFG.b_ch + 2)) / K > 0.01
    assert abs(K - _CFG.K_sat * Se ** (2 * _CFG.b_ch + 4)) / K > 0.01


def test_clapp_hornberger_se_clip_departs_at_extreme_dry():
    """DEPARTURE: legoESM floors Se at 1e-6, so at theta->0 K plateaus at
    K_sat*(1e-6)^(2b+3) instead of the pure power law's 0 limit."""
    K0 = float(clapp_hornberger_K(jnp.array(0.0), _CFG))
    assert K0 == pytest.approx(_CFG.K_sat * (1e-6) ** (2 * _CFG.b_ch + 3), rel=1e-9, abs=0.0)


# --- van Genuchten-Mualem (1980) closed form -----------------------------------


def _vg_oracle(psi, cfg):
    """Independent van Genuchten (1980) eqs 8-9: (Se, theta, K, C=dtheta/dpsi)."""
    psi = np.asarray(psi)
    m = 1.0 - 1.0 / cfg.n_vg
    ap = np.abs(cfg.alpha_vg * psi)
    Se = (1.0 + ap ** cfg.n_vg) ** (-m)
    theta = cfg.theta_r + (cfg.theta_sat - cfg.theta_r) * Se
    K = cfg.K_sat * np.sqrt(Se) * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2
    C = (cfg.alpha_vg * m * cfg.n_vg * ap ** (cfg.n_vg - 1.0)
         * (cfg.theta_sat - cfg.theta_r) * (1.0 + ap ** cfg.n_vg) ** (-m - 1.0))
    return Se, theta, K, C


def test_van_genuchten_matches_paper_oracle():
    """Se/theta/K/C and the psi(theta) inverse match the vG-Mualem 1980 closed
    form to round-off for moderately negative psi (guards/clips inactive)."""
    psi = np.linspace(-5.0, -0.1, 12)
    Se_o, theta_o, K_o, C_o = _vg_oracle(psi, _CFG)
    assert np.allclose(np.asarray(van_genuchten_Se(jnp.asarray(psi), _CFG)),
                       Se_o, rtol=1e-9, atol=0.0)
    assert np.allclose(np.asarray(van_genuchten_theta(jnp.asarray(psi), _CFG)),
                       theta_o, rtol=1e-9, atol=0.0)
    assert np.allclose(np.asarray(van_genuchten_K(jnp.asarray(psi), _CFG)),
                       K_o, rtol=1e-9, atol=0.0)
    assert np.allclose(np.asarray(van_genuchten_C(jnp.asarray(psi), _CFG)),
                       C_o, rtol=1e-9, atol=0.0)
    # psi(theta) inverts theta(psi).
    psi_back = np.asarray(van_genuchten_psi(jnp.asarray(theta_o), _CFG))
    assert np.allclose(psi_back, psi, rtol=1e-9, atol=0.0)


def test_van_genuchten_K_at_saturation_is_ksat_exact():
    """The where-before-pow AD guard keeps the forward EXACT at saturation:
    psi=0 -> Se=1 -> Mualem inner=1 -> K=K_sat (canaries the guard's forward
    value, distinct from a naive pow that would give 0*inf)."""
    K = float(van_genuchten_K(jnp.array(0.0), _CFG))
    assert K == pytest.approx(_CFG.K_sat, rel=1e-12, abs=0.0)


# --- Brooks & Corey (1964) closed form -----------------------------------------


def _bc_oracle(psi, cfg):
    """Independent Brooks & Corey (1964): (Se, theta, K)."""
    psi = np.asarray(psi)
    Se = (np.abs(cfg.psi_b) / np.abs(psi)) ** cfg.lambda_bc
    theta = cfg.theta_r + (cfg.theta_sat - cfg.theta_r) * Se
    K = cfg.K_sat * Se ** (3.0 + 2.0 / cfg.lambda_bc)
    return Se, theta, K


def test_brooks_corey_matches_paper_oracle():
    """Se/theta/K and the psi(theta) inverse match the Brooks-Corey 1964 closed
    form (Brooks-Corey/Burdine K exponent eta = (2+3 lambda)/lambda = 3+2/lambda)
    for psi below air entry."""
    psi = np.linspace(-5.0, -1.0, 12)          # more negative than psi_b=-0.478
    Se_o, theta_o, K_o = _bc_oracle(psi, _CFG)
    assert np.allclose(np.asarray(brooks_corey_Se(jnp.asarray(psi), _CFG)),
                       Se_o, rtol=1e-9, atol=0.0)
    assert np.allclose(np.asarray(brooks_corey_theta(jnp.asarray(psi), _CFG)),
                       theta_o, rtol=1e-9, atol=0.0)
    assert np.allclose(np.asarray(brooks_corey_K(jnp.asarray(psi), _CFG)),
                       K_o, rtol=1e-9, atol=0.0)
    psi_back = np.asarray(brooks_corey_psi(jnp.asarray(theta_o), _CFG))
    assert np.allclose(psi_back, psi, rtol=1e-9, atol=0.0)


def test_brooks_corey_K_exponent_is_3_plus_2_over_lambda():
    """Non-vacuity: the BC conductivity exponent is 3+2/lambda (Burdine), moving
    it to 3+1/lambda or 3+3/lambda changes K by >1% at Se=0.5."""
    Se = 0.5
    # Build a psi that gives this Se: Se = (|psi_b|/|psi|)^lambda.
    psi = -abs(_CFG.psi_b) / Se ** (1.0 / _CFG.lambda_bc)
    K = float(brooks_corey_K(jnp.array(psi), _CFG))
    assert K == pytest.approx(_CFG.K_sat * Se ** (3.0 + 2.0 / _CFG.lambda_bc), rel=1e-9, abs=0.0)
    assert abs(K - _CFG.K_sat * Se ** (3.0 + 1.0 / _CFG.lambda_bc)) / K > 0.01
    assert abs(K - _CFG.K_sat * Se ** (3.0 + 3.0 / _CFG.lambda_bc)) / K > 0.01


# --- Campbell == Clapp-Hornberger dispatch -------------------------------------


def test_campbell_dispatches_to_clapp_hornberger():
    """Campbell (1974) IS the CH power law; the 'campbell' dispatch must be
    byte-identical to 'clapp_hornberger' for theta(psi) and K."""
    cfg_ch = _CFG._replace(retention_curve="clapp_hornberger")
    cfg_cb = _CFG._replace(retention_curve="campbell")
    psi = jnp.linspace(-5.0, -0.2, 8)
    theta = _CFG.theta_sat * jnp.linspace(0.2, 0.9, 8)
    assert jnp.array_equal(theta_from_psi(psi, cfg_ch), theta_from_psi(psi, cfg_cb))
    assert jnp.array_equal(
        hydraulic_conductivity(psi, theta, cfg_ch),
        hydraulic_conductivity(psi, theta, cfg_cb))


# --- AD-safety (x64 and float32) at dry / saturated boundaries -----------------


def test_soil_hydraulics_grad_finite_x64():
    """grad of each K wrt its argument is finite at the guard boundaries AND at
    representative interior states.  Genuine guard cases: CH theta=1e-8 (Se<1e-6
    floor engages), vG psi=0 (Se=1 where-before-pow saturation guard) and vG
    psi=-1e22 (Se collapses onto the 1e-12 sqrt floor — asserted below), BC psi=0
    (the |psi| denominator floored to 1e-10).  Representative states: CH
    mid/saturated, vG/BC psi=-0.5 and -100 (BC has no Se<=1e-12 floor — its
    guards are the ratio |psi|>=1e-10 and the Se in [0,1] clip)."""
    for th in (1.0e-8, 0.2 * _CFG.theta_sat, _CFG.theta_sat):
        g = jax.grad(lambda t: clapp_hornberger_K(t, _CFG))(jnp.array(th))
        assert bool(jnp.isfinite(g)), ("CH", th, g)
    # The deep-dry vG case must actually reach the 1e-12 sqrt floor (Se decays
    # only as |psi|^-(n-1)=|psi|^-0.56, so |psi| ~ 1e22 is needed, not 1e6).
    assert float(van_genuchten_Se(jnp.array(-1.0e22), _CFG)) < 1e-12
    for ps in (0.0, -0.5, -100.0, -1.0e22):
        gv = jax.grad(lambda p: van_genuchten_K(p, _CFG))(jnp.array(ps))
        assert bool(jnp.isfinite(gv)), ("vG", ps, gv)
    for ps in (0.0, -0.5, -100.0):
        gb = jax.grad(lambda p: brooks_corey_K(p, _CFG))(jnp.array(ps))
        assert bool(jnp.isfinite(gb)), ("BC", ps, gb)


def test_van_genuchten_K_grad_finite_float32_at_saturation():
    """The vG-K where-before-pow guard must yield a finite gradient at psi=0
    (Se=1) in float32 too — a naive (1-Se^(1/m))^m would 0*inf -> NaN there."""
    _was = getattr(jax.config, "jax_enable_x64", False)
    jax.config.update("jax_enable_x64", False)
    try:
        g = jax.grad(lambda p: van_genuchten_K(p, _CFG))(jnp.float32(0.0))
        assert g.dtype == jnp.float32
        assert bool(jnp.isfinite(g))
    finally:
        jax.config.update("jax_enable_x64", _was)
