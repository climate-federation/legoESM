"""Faithfulness pins for the convective-adjustment enhanced-diffusion K-mapping.

Target: ``convective_K_A_flag`` in
``legoesm.ocean.physics.convection.enhanced_diffusion`` — the Oceananigans
``ConvectiveAdjustmentVerticalDiffusivity`` (Klinger et al. 1996) convective
adjustment: a large vertical tracer diffusivity ``K_conv`` (and an independent
momentum viscosity ``nu_conv``) wherever the water column is statically unstable
(``N^2 < 0``), background values elsewhere.

Most-trustful source
--------------------
Oceananigans ``ConvectiveAdjustmentVerticalDiffusivity`` (Ramadhan et al. 2020);
Klinger et al. (1996) JPO 26.  With ``N^2 = -(g/rho_0) d(rho)/dz`` at interior
interfaces:

  hard   (smooth_transition=False): K = where(N^2 < 0, K_conv, K_bg)
  smooth (smooth_transition=True):  K = K_bg + (K_conv - K_bg) * sigmoid(-N^2 * s)

The HARD branch is the Oceananigans stable/unstable coefficient selection; the
SMOOTH (sigmoid) branch is legoESM's differentiable extension of it (not
Oceananigans behaviour).  (A/flag analogous with nu_conv/nu_bg and 1/0 or the
sigmoid.)  Dry interfaces (zero actual thickness) get zero mixing.

The existing test_enhanced_diffusion_momentum.py is behavioral (0 exact-magnitude
assertions); this pins the hard + smooth K/A/flag mappings against an independent
N^2 oracle, the neutral (N^2=0) sigmoid midpoint, dry masking, the n2_mode
dispatch, coefficient plumbing, and differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.ocean.eos import rho_0
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.enhanced_diffusion import convective_K_A_flag

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_G = float(constants.g)
_DZ = jnp.asarray([10.0, 20.0, 30.0, 40.0])


def _cfg(smooth, sharp=1000.0, k_conv=1.0, k_bg=1e-5, nu_conv=0.5, nu_bg=2e-5,
         n2_mode="insitu"):
    return EnhancedDiffusionConfig(
        smooth_transition=smooth, sigmoid_sharpness=sharp,
        K_conv=k_conv, K_bg=k_bg, nu_conv=nu_conv, nu_bg=nu_bg, n2_mode=n2_mode)


def _n2_oracle(rho, dz_ref, jac):
    rho = np.asarray(rho, dtype=float)
    dz_actual = np.asarray(dz_ref, dtype=float) * jac
    dz_iface = 0.5 * (dz_actual[:-1] + dz_actual[1:])
    drho_dz = (rho[:-1] - rho[1:]) / dz_iface
    return -(_G / rho_0) * drho_dz


def _call(rho, cfg, jac=1.0, dz_ref=_DZ):
    K, A, flag = convective_K_A_flag(jnp.asarray(rho, dtype=jnp.float64),
                                     jnp.asarray(dz_ref, dtype=jnp.float64),
                                     jnp.asarray(jac, dtype=jnp.float64), cfg)
    return np.asarray(K), np.asarray(A), np.asarray(flag)


# ---------------------------------------------------------------------------
# 1. Hard mode: K = where(N^2 < 0, conv, bg).
# ---------------------------------------------------------------------------
def test_hard_mode_mixed_column():
    # rho = [1025, 1027, 1026, 1028] -> interfaces: stable, UNSTABLE, stable.
    rho = [1025.0, 1027.0, 1026.0, 1028.0]
    cfg = _cfg(smooth=False)
    K, A, flag = _call(rho, cfg)
    n2 = _n2_oracle(rho, _DZ, 1.0)
    exp_K = np.where(n2 < 0.0, 1.0, 1e-5)
    exp_A = np.where(n2 < 0.0, 0.5, 2e-5)
    exp_flag = np.where(n2 < 0.0, 1.0, 0.0)
    np.testing.assert_allclose(K, exp_K, rtol=1e-12)
    np.testing.assert_allclose(A, exp_A, rtol=1e-12)
    np.testing.assert_allclose(flag, exp_flag, rtol=1e-12)
    assert exp_flag.tolist() == [0.0, 1.0, 0.0]     # the crafted regime is discriminating


def test_hard_all_stable_and_all_unstable():
    cfg = _cfg(smooth=False)
    # increasing rho with depth (denser below) -> stable -> K_bg.
    Ks, _, fs = _call([1025.0, 1026.0, 1027.0, 1028.0], cfg)
    np.testing.assert_allclose(Ks, 1e-5, rtol=1e-12)
    assert np.all(fs == 0.0)
    # decreasing rho (denser on top, inversion) -> unstable -> K_conv.
    Ku, _, fu = _call([1028.0, 1027.0, 1026.0, 1025.0], cfg)
    np.testing.assert_allclose(Ku, 1.0, rtol=1e-12)
    assert np.all(fu == 1.0)


# ---------------------------------------------------------------------------
# 2. Smooth (sigmoid) mode.
# ---------------------------------------------------------------------------
def test_smooth_mode_sigmoid_form():
    rho = [1025.0, 1026.0, 1025.4, 1026.2]          # mixed, small gradients
    cfg = _cfg(smooth=True, sharp=1000.0)
    K, A, flag = _call(rho, cfg)
    n2 = _n2_oracle(rho, _DZ, 1.0)
    sig = 1.0 / (1.0 + np.exp(n2 * 1000.0))         # sigmoid(-n2*s)
    np.testing.assert_allclose(K, 1e-5 + (1.0 - 1e-5) * sig, rtol=1e-10)
    np.testing.assert_allclose(A, 2e-5 + (0.5 - 2e-5) * sig, rtol=1e-10)
    np.testing.assert_allclose(flag, sig, rtol=1e-10)
    assert np.any((sig > 0.05) & (sig < 0.95))       # a genuinely intermediate sigmoid


def test_smooth_neutral_gives_midpoint():
    # rho constant -> N^2 = 0 exactly -> sigmoid(0) = 0.5 -> K = (K_bg+K_conv)/2.
    cfg = _cfg(smooth=True, sharp=1e6)               # sharpness irrelevant at N^2=0
    K, A, flag = _call([1026.0, 1026.0, 1026.0, 1026.0], cfg)
    np.testing.assert_allclose(K, 0.5 * (1e-5 + 1.0), rtol=1e-12)
    np.testing.assert_allclose(A, 0.5 * (2e-5 + 0.5), rtol=1e-12)
    np.testing.assert_allclose(flag, 0.5, rtol=1e-12)


def test_smooth_limits_saturate():
    cfg = _cfg(smooth=True, sharp=1e6)
    # strong inversion -> sigmoid -> 1 -> K_conv.
    Ku, _, _ = _call([1030.0, 1026.0, 1022.0, 1018.0], cfg)
    np.testing.assert_allclose(Ku, 1.0, rtol=1e-6)
    # strong stable -> sigmoid -> 0 -> K_bg.
    Ks, _, _ = _call([1018.0, 1022.0, 1026.0, 1030.0], cfg)
    np.testing.assert_allclose(Ks, 1e-5, rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. Dry masking, dispatch, plumbing, differentiability.
# ---------------------------------------------------------------------------
def test_dry_column_zero_everywhere():
    # jacobian <= 0 -> dz_actual <= 0 -> every interface dry -> K=A=flag=0.
    cfg = _cfg(smooth=False)
    K, A, flag = _call([1028.0, 1027.0, 1026.0, 1025.0], cfg, jac=0.0)
    assert np.all(K == 0.0) and np.all(A == 0.0) and np.all(flag == 0.0)


def test_negative_jacobian_all_dry():
    # jacobian < 0 -> dz_actual < 0 -> the ``<= 0`` mask (not ``== 0``) zeros all.
    cfg = _cfg(smooth=False)
    K, A, flag = _call([1028.0, 1027.0, 1026.0, 1025.0], cfg, jac=-0.5)
    assert np.all(K == 0.0) and np.all(A == 0.0) and np.all(flag == 0.0)


def test_interface_local_dry_mask_or():
    # A single zero-thickness reference layer (dz_ref[1]=0) makes BOTH interfaces
    # touching it dry via dry_iface = (dz[:-1]<=0)|(dz[1:]<=0), leaving the far
    # interface wet -> K = [0, 0, K_conv] for an unstable column.
    cfg = _cfg(smooth=False)
    dz_ref = jnp.asarray([10.0, 0.0, 30.0, 40.0])
    K, _, flag = _call([1028.0, 1027.0, 1026.0, 1025.0], cfg, jac=1.0, dz_ref=dz_ref)
    np.testing.assert_allclose(K, [0.0, 0.0, 1.0], rtol=1e-12)     # OR mask + interface-local
    np.testing.assert_allclose(flag, [0.0, 0.0, 1.0], rtol=1e-12)


def test_wet_jacobian_scales_n2():
    # A non-unit POSITIVE Jacobian enters N^2 through dz_actual = dz_ref*jac; the
    # smooth output must match the oracle computed with the SAME jac (an impl
    # ignoring jac in the denominator would fail).
    rho = [1025.0, 1026.0, 1025.4, 1026.2]
    cfg = _cfg(smooth=True, sharp=1000.0)
    K, _, _ = _call(rho, cfg, jac=0.5)
    n2 = _n2_oracle(rho, _DZ, 0.5)
    sig = 1.0 / (1.0 + np.exp(n2 * 1000.0))
    np.testing.assert_allclose(K, 1e-5 + (1.0 - 1e-5) * sig, rtol=1e-10)


def test_insitu_signed_mode_accepted():
    # 'insitu_signed' is an allowed mode (must not raise); on this scheme it goes
    # through the same in-situ N^2 path as 'insitu'.
    rho = [1025.0, 1027.0, 1026.0, 1028.0]
    k_insitu, _, _ = _call(rho, _cfg(smooth=False, n2_mode="insitu"))
    k_signed, _, _ = _call(rho, _cfg(smooth=False, n2_mode="insitu_signed"))
    np.testing.assert_allclose(k_signed, k_insitu, rtol=1e-12)


def test_hard_neutral_is_background_strict_lt():
    # N^2 = 0 (constant rho): the hard mask is STRICT ``N^2 < 0``, so neutral
    # stability gets background K/A and zero flag (a ``<= 0`` bug would pick
    # convective coefficients here).
    cfg = _cfg(smooth=False)
    K, A, flag = _call([1026.0, 1026.0, 1026.0, 1026.0], cfg)
    np.testing.assert_allclose(K, 1e-5, rtol=1e-12)
    np.testing.assert_allclose(A, 2e-5, rtol=1e-12)
    assert np.all(flag == 0.0)


def test_n2_mode_dispatch_raises():
    cfg = _cfg(smooth=False, n2_mode="quasi")
    try:
        _call([1025.0, 1026.0, 1027.0, 1028.0], cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError for n2_mode='quasi'")


def test_adiabatic_mode_requires_t_s_p():
    # n2_mode='adiabatic' without T/S/p_cell must raise (not silently fall back).
    cfg = _cfg(smooth=False, n2_mode="adiabatic")
    try:
        _call([1025.0, 1026.0, 1027.0, 1028.0], cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError for adiabatic without T/S/p")


def test_config_plumbing():
    # Non-default coefficients flow through the hard mapping independently for
    # tracer (K) and momentum (A).
    rho_u = [1028.0, 1027.0, 1026.0, 1025.0]         # unstable
    rho_s = [1025.0, 1026.0, 1027.0, 1028.0]         # stable
    cfg = _cfg(smooth=False, k_conv=2.5, k_bg=3e-5, nu_conv=1.5, nu_bg=4e-5)
    Ku, Au, _ = _call(rho_u, cfg)
    Ks, As, _ = _call(rho_s, cfg)
    np.testing.assert_allclose(Ku, 2.5, rtol=1e-12)
    np.testing.assert_allclose(Au, 1.5, rtol=1e-12)
    np.testing.assert_allclose(Ks, 3e-5, rtol=1e-12)
    np.testing.assert_allclose(As, 4e-5, rtol=1e-12)


def test_smooth_differentiable():
    cfg = _cfg(smooth=True, sharp=1000.0)

    def loss(rho):
        K, A, flag = convective_K_A_flag(rho, _DZ, jnp.asarray(1.0), cfg)
        return jnp.sum(K + A + flag)
    g = jax.grad(loss)(jnp.asarray([1025.0, 1026.0, 1025.4, 1026.2]))
    assert jnp.all(jnp.isfinite(g))
