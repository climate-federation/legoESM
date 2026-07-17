"""Pacanowski & Philander (1981) Richardson-mixing ORACLE-FAITHFULNESS tests.

The ocean PP81 Richardson-number vertical-mixing scheme
(``ocean/physics/vertical_mixing/richardson.py``) implements the canonical PP81
momentum form (CVMix Aug-2012 documentation Eq. 4.4, POP settings) with a
FULL-nu tracer division, but the existing tests only check directional /
conservation properties.  These pin the diffusivity closed form to round-off
against an INDEPENDENT reimplementation and canary the config defaults.

Oracle: the CVMix technical documentation (Griffies et al., Aug 2012) Eqs.
4.4-4.5 give the SHEAR terms nu_shear = nu0/(1+a*Ri)^n and kappa_shear =
nu0/(1+a*Ri)^(n+1) (same nu0, NO background in the equations themselves); CVMix
cites a=5, n=2 as the common POP settings.  Only a and n are taken from CVMix;
the amplitude nu0=K_0 and backgrounds A_bg/K_bg are project/POP defaults (the
CVMix doc does not fix those values).  All are local _O_* literals here.

What is pinned, in order of authority:
1. ``_pp81_diffusivities`` A_v (nu) and K_v (kappa) to rel 1e-12 across a Ri grid
   vs an independent reimplementation (nu = K_0/(1+alpha*Ri)^n + A_bg,
   kappa = nu/(1+alpha*Ri) + K_bg), for BOTH the default config and a non-default
   (K_0, alpha, n, A_bg, K_bg).
2. The DEPARTURE from the canonical PP81 total tracer diffusivity (CVMix Eq. 4.5
   SHEAR term nu0/(1+alpha*Ri)^(n+1), SAME nu0, plus the K_bg background): the
   code's full-nu division re-divides the background nu_b into the tracer, so
   kappa exceeds the canonical tracer by exactly A_bg/(1+alpha*Ri) (pinned, shown
   nonzero).  This algebraic departure is the auditable claim; no separate tracer
   amplitude is asserted.
3. Physical properties: monotone decrease with Ri, maximum (K_0+A_bg) at Ri=0,
   the effective Prandtl Pr = nu/kappa grows with Ri.
4. END-TO-END: the public ``richardson_vertical_mixing`` returns A_v/K_v equal to
   ``_pp81_diffusivities`` at the Ri it computes internally, with a NON-UNIT z*
   Jacobian and a mixed-sign (clipped) Ri column, so a call-site regression in
   the coefficient formula, the Jacobian/dz_actual, or the Ri>=0 clip is caught.
5. ``RichardsonVerticalMixingConfig`` default canary vs the config's declared
   defaults (a=5, n=2 CVMix/POP; K_0, A_bg, K_bg project/POP defaults).
6. The dead ``Pr_t`` no-op warns; x64 AD-finiteness.

Precision: the form carries no precision-policy promotion; the autouse fixture
enables x64 for the round-off pins and restores the entry state.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest
from legoesm.ocean.physics.vertical_mixing.config import RichardsonVerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.richardson import _pp81_diffusivities


@pytest.fixture(autouse=True)
def _force_x64():
    entry_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", entry_x64)


# PP81 constants (NOT copied from the config).  CVMix Aug-2012 (Griffies et al.)
# Eqs. 4.4-4.5 give the SHEAR terms: Eq. 4.4 momentum = nu0/(1+a*Ri)^n, Eq. 4.5
# tracer = nu0/(1+a*Ri)^(n+1) — the tracer numerator is the SAME nu0 as momentum
# (NO separate tracer amplitude, and NO background inside the equations).  Only
# a and n are from CVMix (the common POP settings); nu0 and the backgrounds are
# project/POP defaults.  The canary asserts config K_0==nu0.
_O_NU0 = 5e-3      # shear numerator nu0 [m^2/s] (project default), BOTH Eq.4.4 and 4.5
_O_ALPHA = 5.0     # stability parameter a (CVMix/POP)
_O_N = 2           # exponent n (CVMix/POP)
_O_A_BG = 1e-4     # background viscosity nu_b [m^2/s] (project default)
_O_K_BG = 1e-5     # background diffusivity kappa_b [m^2/s] (project default)


def _av_kv_oracle(Ri, nu0, alpha, n, A_bg, K_bg):
    """Independent PP81 diffusivities, full-nu tracer division (plain Python)."""
    one_plus = 1.0 + alpha * Ri
    A_v = nu0 / one_plus ** n + A_bg
    K_v = A_v / one_plus + K_bg
    return A_v, K_v


def _kappa_canonical_total(Ri, nu0, alpha, n, K_bg):
    """The CANONICAL PP81 total tracer diffusivity: the CVMix (Aug-2012) Eq. 4.5
    SHEAR term nu0/(1+alpha*Ri)^(n+1) PLUS the K_bg background (the background is
    added separately — it is not part of Eq. 4.5).  The shear numerator is the
    SAME nu0 as the momentum (Eq. 4.4) — NOT a separate amplitude.  This is the
    form the code's full-nu division departs from (by +A_bg/(1+alpha*Ri))."""
    return nu0 / (1.0 + alpha * Ri) ** (n + 1) + K_bg


def _a(x):
    return jnp.array(float(x))


_CFG = RichardsonVerticalMixingConfig()
_RI = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0]


# --- diffusivity closed form ------------------------------------------------

@pytest.mark.parametrize("Ri", _RI)
def test_pp81_diffusivities_match_oracle(Ri):
    A_v, K_v = _pp81_diffusivities(_a(Ri), _CFG)
    A_o, K_o = _av_kv_oracle(Ri, _O_NU0, _O_ALPHA, _O_N, _O_A_BG, _O_K_BG)
    assert float(A_v) == pytest.approx(A_o, rel=1e-12, abs=0.0)
    assert float(K_v) == pytest.approx(K_o, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("Ri", _RI)
def test_pp81_diffusivities_match_oracle_nondefault_cfg(Ri):
    """The pin is NOT tied to the classic defaults: a non-default (K_0, alpha, n,
    A_bg, K_bg) config still matches the oracle to round-off (so a regression that
    hard-codes the classic constants is caught)."""
    cfg = RichardsonVerticalMixingConfig(
        K_0=8.0e-3, alpha=3.0, n=3, A_bg=2.0e-4, K_bg=5.0e-6)
    A_v, K_v = _pp81_diffusivities(_a(Ri), cfg)
    A_o, K_o = _av_kv_oracle(Ri, 8.0e-3, 3.0, 3, 2.0e-4, 5.0e-6)
    assert float(A_v) == pytest.approx(A_o, rel=1e-12, abs=0.0)
    assert float(K_v) == pytest.approx(K_o, rel=1e-12, abs=0.0)


def test_public_function_uses_helper_convention():
    """END-TO-END: the returned A_v/K_v from the public richardson_vertical_mixing
    equal _pp81_diffusivities at the Ri the function computes internally.  Ri is
    reconstructed INDEPENDENTLY of _pp81_diffusivities (from the same N^2 and
    shear), so this pins THREE call-site behaviours a regression could silently
    change: (a) the coefficient formula (the returned A_v/K_v must be the helper
    of the reconstructed Ri); (b) the z* Jacobian — a NON-UNIT jac=1.3 enters
    BOTH N^2 and the shear dz, so dropping it or using dz_ref for dz_actual moves
    Ri and breaks the pin; (c) the negative-Ri clip — the column is deliberately
    non-monotonic so at least one interface has N^2<0 (raw Ri<0), exercising
    clip_negative=True (a regression that stopped clipping would diverge).
    This is NOT a full-independence claim: N^2/shear share the public helpers by
    construction (that is the point — the SAME internal inputs, a re-derived Ri)."""
    from legoesm.ocean.eos import compute_buoyancy_frequency, wright_eos
    from legoesm.ocean.physics.vertical_mixing._shared import richardson_number
    from legoesm.ocean.physics.vertical_mixing.richardson import (
        _EPS,
        richardson_vertical_mixing,
    )
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    nlev = 6
    shape = (1, 1, nlev)
    u = jnp.broadcast_to(jnp.linspace(0.3, 0.0, nlev), shape)    # sheared (S^2>0)
    v = jnp.zeros(shape)
    # Non-monotonic T -> mixed-sign N^2 (both a stable stratification AND at
    # least one static instability), so the Ri>=0 clip path is actually hit.
    T = jnp.broadcast_to(jnp.array([4.0, 12.0, 6.0, 14.0, 8.0, 2.0]), shape)
    S = jnp.full(shape, 35.0)
    rho = wright_eos(T, S, jnp.zeros_like(T))
    z_coord = create_z_star_from_thicknesses(jnp.full(nlev, 10.0))
    jac = jnp.full((1, 1), 1.3)                                  # NON-unit z* Jacobian
    cfg = RichardsonVerticalMixingConfig()

    out = richardson_vertical_mixing(u, v, T, S, rho, z_coord, jac, cfg,
                                     apply_diffusion=False)
    # reconstruct the internal Ri from the same N^2 and shear (insitu mode)
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jac)
    dz_actual = z_coord.dz_ref * jac[..., jnp.newaxis]
    Ri_raw = richardson_number(N2, u, v, dz_actual, eps=_EPS, clip_negative=False)
    # self-validate the fixture actually spans the clip: both signs present.
    assert float(jnp.min(Ri_raw)) < 0.0 < float(jnp.max(Ri_raw))
    Ri = richardson_number(N2, u, v, dz_actual, eps=_EPS, clip_negative=True)
    A_exp, K_exp = _pp81_diffusivities(Ri, cfg)
    assert jnp.allclose(out.A_v, A_exp, rtol=1e-12, atol=0.0)
    assert jnp.allclose(out.K_v, K_exp, rtol=1e-12, atol=0.0)


def test_max_mixing_at_ri_zero():
    """At Ri = 0 (the clipped unstable branch) the decay factor is 1, so
    A_v = K_0 + A_bg (maximum) and K_v = (K_0 + A_bg) + K_bg."""
    A_v, K_v = _pp81_diffusivities(_a(0.0), _CFG)
    assert float(A_v) == pytest.approx(_O_NU0 + _O_A_BG, rel=1e-12)
    assert float(K_v) == pytest.approx(_O_NU0 + _O_A_BG + _O_K_BG, rel=1e-12)


def test_monotone_decrease_and_background_floor():
    """A_v, K_v strictly decrease with Ri and asymptote to the backgrounds."""
    A_prev, K_prev = _pp81_diffusivities(_a(0.0), _CFG)
    for Ri in (0.1, 0.5, 2.0, 10.0, 100.0):
        A_v, K_v = _pp81_diffusivities(_a(Ri), _CFG)
        assert float(A_v) < float(A_prev)
        assert float(K_v) < float(K_prev)
        A_prev, K_prev = A_v, K_v
    A_big, K_big = _pp81_diffusivities(_a(1.0e6), _CFG)
    assert float(A_big) == pytest.approx(_O_A_BG, rel=1e-4)
    assert float(K_big) == pytest.approx(_O_K_BG, rel=1e-3)


# --- full-nu vs canonical PP81 (CVMix Eq. 4.5) departure --------------------

@pytest.mark.parametrize("Ri", [0.2, 1.0, 5.0])
def test_full_nu_division_departs_from_canonical_shear(Ri):
    """DEPARTURE canary: the code's tracer kappa (full-nu division) exceeds the
    canonical PP81 total tracer diffusivity (CVMix Aug-2012 Eq. 4.5 SHEAR term
    nu0/(1+alpha*Ri)^(n+1) plus the K_bg background) by exactly A_bg/(1+alpha*Ri).

    The extra term is the background nu_b re-divided into the tracer.  Both forms
    share the SAME nu0 numerator (Eq. 4.5 uses the momentum nu0 — there is no
    separate tracer amplitude in PP81) and the SAME K_bg background, so the
    leading nu0/(1+alpha*Ri)^(n+1) term and K_bg both cancel exactly and the
    residual is the pure background departure — a NONZERO gap (vanishing only as
    A_bg -> 0)."""
    _, K_v = _pp81_diffusivities(_a(Ri), _CFG)
    k_canon = _kappa_canonical_total(Ri, _O_NU0, _O_ALPHA, _O_N, _O_K_BG)
    gap = float(K_v) - k_canon
    assert gap == pytest.approx(_O_A_BG / (1.0 + _O_ALPHA * Ri), rel=1e-11)
    assert gap > 0.0


def test_effective_prandtl_grows_with_ri():
    """Pr = A_v/K_v grows with Ri (stable shear mixes momentum more efficiently
    than tracer) — the physical point of the tracer division."""
    prs = []
    for Ri in (0.0, 0.5, 2.0, 10.0):
        A_v, K_v = _pp81_diffusivities(_a(Ri), _CFG)
        prs.append(float(A_v) / float(K_v))
    assert all(prs[i + 1] > prs[i] for i in range(len(prs) - 1))
    assert prs[0] == pytest.approx(
        (_O_NU0 + _O_A_BG) / (_O_NU0 + _O_A_BG + _O_K_BG), rel=1e-12)


# --- coefficient canary -----------------------------------------------------

def test_config_matches_declared_defaults():
    """Canary the config against the oracle literals: alpha=5, n=2 are the
    CVMix/POP common settings; K_0, A_bg, K_bg are the project/POP defaults.
    A drifted config default trips this."""
    c = RichardsonVerticalMixingConfig()
    assert c.K_0 == _O_NU0
    assert c.alpha == _O_ALPHA
    assert c.n == _O_N
    assert c.A_bg == _O_A_BG
    assert c.K_bg == _O_K_BG


# --- dead Pr_t no-op --------------------------------------------------------

def test_pr_t_is_dead_noop_and_warns():
    """cfg.Pr_t no longer affects the diffusivities (the Ri-Prandtl is built into
    the formula); a non-default value must WARN, not silently do nothing.  Also
    confirm the diffusivities are Pr_t-independent."""
    c_default = RichardsonVerticalMixingConfig()
    c_changed = c_default._replace(Pr_t=3.0)
    A0, K0 = _pp81_diffusivities(_a(0.7), c_default)
    A1, K1 = _pp81_diffusivities(_a(0.7), c_changed)
    assert float(A0) == float(A1) and float(K0) == float(K1)

    # the full-scheme path warns on non-default Pr_t
    from legoesm.ocean.eos import wright_eos
    from legoesm.ocean.physics.vertical_mixing.richardson import richardson_vertical_mixing
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    nlev = 4
    shape = (1, 1, nlev)
    u = jnp.zeros(shape)
    v = jnp.zeros(shape)
    T = jnp.broadcast_to(jnp.linspace(15.0, 5.0, nlev), shape)
    S = jnp.full(shape, 35.0)
    rho = wright_eos(T, S, jnp.zeros_like(T))
    z_coord = create_z_star_from_thicknesses(jnp.full(nlev, 10.0))
    jac = jnp.ones((1, 1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        richardson_vertical_mixing(u, v, T, S, rho, z_coord, jac, c_changed,
                                   apply_diffusion=False)
    assert any("Pr_t" in str(rec.message) for rec in w)


# --- AD-safety --------------------------------------------------------------

def test_pp81_grads_finite():
    """grad of A_v and K_v wrt Ri is finite and negative (decreasing)."""
    gA = jax.grad(lambda Ri: _pp81_diffusivities(Ri, _CFG)[0])(_a(0.5))
    gK = jax.grad(lambda Ri: _pp81_diffusivities(Ri, _CFG)[1])(_a(0.5))
    assert bool(jnp.isfinite(gA)) and float(gA) < 0.0
    assert bool(jnp.isfinite(gK)) and float(gK) < 0.0
