"""NEMO ``bn2`` (S-EOS) Brunt-Väisälä ``N²`` — exact-transcription tests.

Validates :func:`legoesm.ocean.eos.compute_buoyancy_frequency_nemo_bn2` and its
``rab`` companion :func:`legoesm.ocean.eos.nemo_seos_alpha_beta` against an
independent NumPy transcription of NEMO ``eosbn2.F90`` (``rab_3d_t`` +
``bn2_t``, ``np_seos`` branch) on an analytic S-EOS column — to machine
precision — plus the ``n2_mode`` dispatch-raise hardening.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    NemoSEOSConfig,
    compute_buoyancy_frequency_nemo_bn2,
    nemo_seos_alpha_beta,
    nemo_seos_eos,
)


def _numpy_bn2(T, S, gdept, gdepw_int, cfg, g):
    """Independent NumPy transcription of NEMO eosbn2.F90 bn2_t (np_seos).

    Loops the NEMO w-point recurrence directly (no vectorised broadcasting)
    so a broadcasting/index slip in the JAX version cannot be masked.
    """
    T = np.asarray(T); S = np.asarray(S); gdept = np.asarray(gdept)
    nlev = T.shape[-1]
    # rab alpha/beta at each T-cell's own depth (rab_3d_t np_seos).
    zt = T - cfg.T0; zs = S - cfg.S0; zh = gdept
    alpha = (cfg.a0 * (1.0 + cfg.lambda1 * zt + cfg.mu1 * zh)
             + cfg.nu * zs) / cfg.rho0
    beta = (cfg.b0 * (1.0 - cfg.lambda2 * zs - cfg.mu2 * zh)
            - cfg.nu * zt) / cfg.rho0
    out = np.zeros(T.shape[:-1] + (nlev - 1,))
    for i in range(nlev - 1):
        jk_up, jk_lo = i, i + 1  # NEMO jk-1 (upper), jk (lower)
        zrw = (gdepw_int[i] - gdept[jk_lo]) / (gdept[jk_up] - gdept[jk_lo])
        zaw = alpha[..., jk_lo] * (1.0 - zrw) + alpha[..., jk_up] * zrw
        zbw = beta[..., jk_lo] * (1.0 - zrw) + beta[..., jk_up] * zrw
        e3w = gdept[jk_lo] - gdept[jk_up]
        out[..., i] = g * (
            zaw * (T[..., jk_up] - T[..., jk_lo])
            - zbw * (S[..., jk_up] - S[..., jk_lo])
        ) / e3w
    return out


def _column():
    """A stably-and-unstably layered analytic S-EOS column (nlev=8)."""
    rng = np.random.default_rng(0)
    nlev = 8
    gdepw = np.array([0., 10., 25., 55., 105., 190., 350., 650., 1200.])
    gdept = 0.5 * (gdepw[:-1] + gdepw[1:]) + rng.uniform(-2, 2, nlev)  # non-midpoint
    gdept = np.sort(gdept)
    gdepw_int = gdepw[1:-1]  # interior w-interfaces (nlev-1,)
    # Warm/salty surface, cool/fresh deep — plus one inverted (unstable) pair.
    T = np.linspace(18.0, 3.0, nlev) + rng.uniform(-0.5, 0.5, nlev)
    S = np.linspace(35.5, 34.6, nlev) + rng.uniform(-0.05, 0.05, nlev)
    # Force a statically unstable interface 3: cold-over-warm with the
    # salinity held neutral there so alpha*ΔT (destabilising) dominates.
    T[3] = T[4] - 3.0
    S[3] = S[4]
    return T, S, gdept, gdepw_int


def test_alpha_beta_matches_analytic_seos_derivative():
    """rab alpha,beta == analytic (T,S)-derivatives of the S-EOS density / rho0."""
    import jax
    cfg = NemoSEOSConfig()
    T, S, depth = 12.3, 34.7, 250.0
    p = cfg.rho0 * constants.g * depth  # so nemo_seos_eos recovers zh = depth
    drdT = float(jax.grad(lambda t: nemo_seos_eos(t, S, p, cfg))(T))
    drdS = float(jax.grad(lambda s: nemo_seos_eos(T, s, p, cfg))(S))
    a, b = nemo_seos_alpha_beta(jnp.array(T), jnp.array(S), jnp.array(depth), cfg)
    assert np.allclose(float(a), -drdT / cfg.rho0, rtol=0, atol=1e-15)
    assert np.allclose(float(b), drdS / cfg.rho0, rtol=0, atol=1e-15)


def test_alpha_beta_cross_terms_full_nemo_namelist():
    """Non-zero nu/lambda2/mu2 (NEMO source-default S-EOS set) still equal the
    analytic jax.grad of the density polynomial — locks the cross terms the
    DINO defaults (nu=lambda2=mu2=0) never exercise."""
    import jax
    cfg = NemoSEOSConfig(a0=0.1655, b0=0.76554, lambda1=0.05952,
                         lambda2=5.4914e-4, mu1=1.4970e-4, mu2=1.109e-5,
                         nu=2.4341e-3)
    T, S, depth = 7.6, 34.2, 812.0
    p = cfg.rho0 * constants.g * depth
    drdT = float(jax.grad(lambda t: nemo_seos_eos(t, S, p, cfg))(T))
    drdS = float(jax.grad(lambda s: nemo_seos_eos(T, s, p, cfg))(S))
    a, b = nemo_seos_alpha_beta(jnp.array(T), jnp.array(S), jnp.array(depth), cfg)
    assert np.allclose(float(a), -drdT / cfg.rho0, rtol=0, atol=1e-15)
    assert np.allclose(float(b), drdS / cfg.rho0, rtol=0, atol=1e-15)


def test_bn2_matches_numpy_transcription():
    """Vectorised JAX bn2 == looped NumPy NEMO transcription to machine eps."""
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    g = constants.g
    ref = _numpy_bn2(T, S, gdept, gdepw_int, cfg, g)
    got = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
        jnp.asarray(gdepw_int), cfg=cfg, g=g))
    assert got.shape == (len(gdept) - 1,)
    assert np.allclose(got, ref, rtol=0, atol=1e-18), np.max(np.abs(got - ref))


def test_bn2_batched_broadcasts():
    """Leading batch dims + (nlev,) depth ladder broadcast correctly."""
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    Tb = np.broadcast_to(T, (4, 3, len(T))).copy()
    Sb = np.broadcast_to(S, (4, 3, len(S))).copy()
    got = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(Tb), jnp.asarray(Sb), jnp.asarray(gdept),
        jnp.asarray(gdepw_int), cfg=cfg))
    ref = _numpy_bn2(Tb, Sb, gdept, gdepw_int, cfg, constants.g)
    assert got.shape == (4, 3, len(gdept) - 1)
    assert np.allclose(got, ref, atol=1e-18)


def test_bn2_sign_is_convection_trigger():
    """The forced inverted pair (interface 3) is statically UNSTABLE (N²<0)."""
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    n2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
        jnp.asarray(gdepw_int), cfg=cfg))
    assert n2[3] < 0.0                       # unstable interface fires
    assert (n2[:3] > 0.0).all()              # stable stratification above


def test_convection_n2_mode_dispatch_raises():
    """Unknown enhanced-diffusion n2_mode raises (dispatch hardening)."""
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        convective_K_A_flag,
    )
    from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
    rho = jnp.ones((2, 2, 4))
    dz = jnp.ones((4,)); J = jnp.ones((2, 2))
    with pytest.raises(ValueError, match="n2_mode"):
        convective_K_A_flag(rho, dz, J,
                            EnhancedDiffusionConfig(n2_mode="bogus"))


def test_nemo_bn2_requires_depth_ladders():
    """enhanced-diffusion nemo_bn2 without the ladders raises (no silent fallback)."""
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        convective_K_A_flag,
    )
    from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
    rho = jnp.ones((2, 2, 4)); dz = jnp.ones((4,)); J = jnp.ones((2, 2))
    T = jnp.ones((2, 2, 4)); S = jnp.ones((2, 2, 4))
    with pytest.raises(ValueError, match="nemo_bn2"):
        convective_K_A_flag(rho, dz, J,
                            EnhancedDiffusionConfig(n2_mode="nemo_bn2"),
                            T=T, S=S)  # t_depth / w_depth missing


def test_other_consumers_reject_nemo_bn2_loudly():
    """Richardson does not thread the bn2 depth ladders and must raise, and
    _shared.compute_N2 without the ladders must raise (catke pass-through) —
    no consumer may silently mis-handle nemo_bn2."""
    from legoesm.ocean.physics.vertical_mixing.richardson import (
        richardson_vertical_mixing,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        RichardsonVerticalMixingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
    with pytest.raises(ValueError, match="n2_mode"):
        richardson_vertical_mixing(
            jnp.ones((2, 2, 4)), jnp.ones((2, 2, 4)), jnp.ones((2, 2, 4)),
            jnp.ones((2, 2, 4)), jnp.ones((2, 2, 4)), None, jnp.ones((2, 2)),
            RichardsonVerticalMixingConfig(n2_mode="nemo_bn2"))
    with pytest.raises(ValueError, match="nemo_bn2"):
        compute_N2(jnp.ones((2, 2, 4)), jnp.ones((2, 2, 3)), 1026.0,
                   T_cell=jnp.ones((2, 2, 4)), S_cell=jnp.ones((2, 2, 4)),
                   n2_mode="nemo_bn2")  # ladders missing


def test_shared_compute_N2_nemo_bn2_branch():
    """_shared.compute_N2 routes nemo_bn2 to the eos bn2 and matches it."""
    from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    Tj = jnp.asarray(T[None, :]); Sj = jnp.asarray(S[None, :])
    n2 = np.asarray(compute_N2(
        jnp.zeros_like(Tj), jnp.ones((1, len(T) - 1)), cfg.rho0,
        T_cell=Tj, S_cell=Sj,
        t_depth=jnp.asarray(gdept), w_depth=jnp.asarray(gdepw_int),
        n2_mode="nemo_bn2"))
    ref = _numpy_bn2(T[None, :], S[None, :], gdept, gdepw_int, cfg, constants.g)
    assert np.allclose(n2, ref, atol=1e-18)
