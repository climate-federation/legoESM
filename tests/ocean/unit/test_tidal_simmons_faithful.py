"""Internal-tide mixing ORACLE-FAITHFULNESS tests (Simmons 2004 / Jayne-St-Laurent).

The ocean tidal-mixing scheme (``ocean/physics/vertical_mixing/tidal.py``) is the
Jayne & St-Laurent (2001) / Simmons et al. (2004) bottom-intensified internal-tide
diffusivity.  The existing ``tests/unit/test_tidal_mixing.py`` checks directional /
scaling / bound properties; these pin the closed form to round-off (rel 1e-12)
against a FULLY INDEPENDENT NumPy reimplementation (no SUT helper appears in any
expected value) and canary each departure.

Oracle: the CVMix Aug-2012 documentation (Griffies et al.) Ch. 5, typed as local
``_O_*`` literals (NOT copied from the config), in the module's positive-downward
depth convention (z in [0, H]; z = H at the seafloor):

    kappa = q * Gamma * E * F(z) / (rho * N^2)              (Eq. 5.22)
    F(z)  = e^(-(H-z)/zeta) / [zeta*(1 - e^(-H/zeta))]      (Eq. 5.27)
    q = 1/3 (Eq. 5.19)  Gamma = 0.2 (Eq. 5.14)  zeta = 500 m (Eq. 5.30)
    K_max default 50e-4 = 5e-3 m^2/s (Sec. 5.3.3)

What is pinned:
1. ``_exp_decay_structure`` == exp(-(H-z)/zeta) across a FULL wet column (Eq. 5.27
   numerator, pure oracle with NO clamp); the ``max(H-z,0)`` clamp and the
   ``h_decay >= 1e-6`` floor are canaried SEPARATELY as implementation guards.
2. ``normalize_structure`` integrates F to unity (sum_k F*h == 1); the DISCRETE
   normaliser sum_k exp(-(H-z_k)/zeta)*h_k converges to CVMix's ANALYTIC denominator
   zeta*(1-e^(-H/zeta)) (Eq. 5.27) as dz -> 0 (the discrete-vs-analytic DEPARTURE);
   per-level DRY cells (h_partial<=0) return zero (no spurious K leak).
3. ``compute_tidal_diffusivity`` == Gamma*q*E_BT*F/(rho_0*max(N^2,N2min)), clipped
   [0, K_max], vs a fully independent NumPy reimpl (default + non-default cfg incl.
   a varied rho_0); exact E-linearity, 1/N^2, 1/rho_0; the K_max cap; the N2 floor
   pinned to the oracle at exactly N2min (with a wrong-floor canary); dry/land -> 0.
4. Config-default canary vs CVMix (Gamma=0.2, q=1/3, zeta=500, K_max=5e-3);
   N_squared_min=1e-7 is the module's higher-than-Simmons(1e-8) floor; rho_0 is the
   Boussinesq reference density (Eq. 5.22 writes in-situ rho — a departure).
5. Differentiability is AD-compatible (finite selected-branch subgradients), NOT
   everywhere smooth: grads at the K_max cap and the N2 floor are finite.

Precision: the form carries no precision-policy promotion; the autouse fixture
enables x64 for the round-off pins and restores the entry state.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ocean.physics.vertical_mixing.tidal import (
    TidalMixingConfig,
    _exp_decay_structure,
    compute_tidal_diffusivity,
    normalize_structure,
)

from legoesm import constants


@pytest.fixture(autouse=True)
def _force_x64():
    entry_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", entry_x64)


# CVMix Ch. 5 oracle constants (typed from the doc, NOT the config).
_O_GAMMA = 0.2          # Eq. 5.14 Osborn mixing efficiency
_O_Q = 1.0 / 3.0        # Eq. 5.19 local-dissipation fraction (St-Laurent 2002)
_O_ZETA = 500.0         # Eq. 5.30 exponential decay scale [m]
_O_KMAX = 5.0e-3        # Sec. 5.3.3 default cap 50e-4 [m^2/s]
_O_N2MIN_SIMMONS = 1.0e-8   # Sec. 5.3.5 Simmons suggested floor
_O_N2MIN_MODULE = 1.0e-7    # the module's (stronger) floor — a documented departure
_O_H_DECAY_FLOOR = 1.0e-6   # module AD guard floor on h_decay
_O_RHO_DEFAULT = float(constants.rho_ocean)  # Boussinesq reference density

_CFG = TidalMixingConfig()


def _uniform_column(nlev=10, dz=100.0):
    """(layer_depths, h_partial, H_bathy) for one uniform-dz column, positive-down."""
    h = jnp.full((1, nlev), dz)
    z = jnp.arange(nlev)[None, :] * dz + 0.5 * dz          # cell-centre depths
    H = jnp.full((1,), nlev * dz)
    return z.astype(jnp.float64), h.astype(jnp.float64), H.astype(jnp.float64)


# --- fully independent NumPy oracle (no SUT helper appears below) ------------

def _F_raw_oracle(z, H, zeta):
    """Eq. 5.27 numerator on the wet column: exp(-(H-z)/zeta), NO clamp."""
    return np.exp(-(np.asarray(H)[:, None] - np.asarray(z)) / zeta)


def _F_raw_oracle_rebased(z, H, zeta):
    """The module's numerically-stable numerator: the Eq. 5.27 numerator with the
    ``max(H-z,0)`` clamp and the per-column log-sum-exp rebase (subtract the
    minimum distance-above-bottom).  Proportional to _F_raw_oracle on a wet
    column; the constant cancels in normalisation."""
    dist = np.maximum(np.asarray(H)[:, None] - np.asarray(z), 0.0)
    dist_ref = np.min(dist, axis=-1, keepdims=True)
    return np.exp(-(dist - dist_ref) / zeta)


def _normalize_oracle(F_raw, h):
    """Independent reimpl of the module's DISCRETE normalisation + dry mask.

    Mirrors the documented algorithm: NEGATIVE/zero thicknesses are clamped OUT
    of the integral denominator (``h_wet``) and zeroed on output.
    """
    F_raw = np.asarray(F_raw)
    h = np.asarray(h)
    h_wet = np.where(h > 0.0, h, 0.0)
    integral = np.sum(F_raw * h_wet, axis=-1, keepdims=True)
    safe = np.where(integral > 1.0e-12, integral, 1.0)
    F = np.where(integral > 1.0e-12, F_raw / safe, 0.0)
    return np.where(h > 0.0, F, 0.0)


def _K_oracle(E, F_norm, N2, *, gamma, q, rho0, kmax, n2min):
    """Eq. 5.22 with the K_max cap and N2 floor — all constants explicit."""
    N2s = np.maximum(np.asarray(N2), n2min)
    K = gamma * q * np.asarray(E)[:, None] * np.asarray(F_norm) / (rho0 * N2s)
    return np.clip(K, 0.0, kmax)


def _oracle_K_default(E, z, h, H, N2, cfg=_CFG):
    F = _normalize_oracle(_F_raw_oracle(z, H, cfg.h_decay_m), h)
    return _K_oracle(E, F, N2, gamma=_O_GAMMA, q=_O_Q, rho0=cfg.rho_0,
                     kmax=cfg.K_max, n2min=cfg.N_squared_min)


# --- exponential vertical structure (Eq. 5.27 numerator) --------------------

def test_exp_structure_normalized_matches_eq527():
    """The NORMALISED structure matches CVMix Eq. 5.27 exactly (rebase-invariant),
    and the raw ratio to the deepest cell is exp(-Δdist/zeta) (Eq. 5.27 SHAPE,
    independent of the log-sum-exp rebase constant)."""
    z, h, H = _uniform_column()                     # z < H everywhere (clamp inactive)
    F = np.asarray(normalize_structure(_exp_decay_structure(z, H, _O_ZETA), h))
    np.testing.assert_allclose(F, _normalize_oracle(_F_raw_oracle(z, H, _O_ZETA), h),
                               rtol=1e-12, atol=0.0)
    raw = np.asarray(_exp_decay_structure(z, H, _O_ZETA))
    dist = np.maximum(1000.0 - np.asarray(z)[0], 0.0)      # H = 1000
    ratio_o = np.exp(-(dist - dist.min()) / _O_ZETA)       # vs deepest (max) cell
    np.testing.assert_allclose(raw[0] / raw[0].max(), ratio_o, rtol=1e-12, atol=0.0)
    assert raw[0, -1] >= raw[0, 0]                   # bottom-intensified (deepest = max)


def test_exp_structure_bottom_value_is_one_at_seafloor():
    # At the exact seafloor (z=H) F_raw = exp(0) = 1.
    F = float(np.asarray(_exp_decay_structure(jnp.array([[1000.0]]),
                                              jnp.array([1000.0]), _O_ZETA))[0, 0])
    assert F == pytest.approx(1.0, rel=1e-12)


def test_exp_structure_clamp_guard_below_seafloor():
    """DEPARTURE guard: for z>H (below the seafloor) the max(H-z,0) clamp pins
    F_raw=1, unlike the unclamped Eq. 5.27 numerator exp(-(H-z)/zeta) > 1."""
    z = jnp.array([[1050.0]])                        # 50 m below H
    H = jnp.array([1000.0])
    F = float(np.asarray(_exp_decay_structure(z, H, _O_ZETA))[0, 0])
    assert F == pytest.approx(1.0, rel=1e-12)        # clamped
    unclamped = math.exp(-(1000.0 - 1050.0) / _O_ZETA)   # = exp(0.1) ~ 1.105
    assert not math.isclose(F, unclamped, rel_tol=1e-3)  # clamp changed the result


def test_exp_structure_h_decay_floor_guard():
    """DEPARTURE guard: h_decay below 1e-6 is floored to 1e-6 (pinned against the
    rebased oracle, which mirrors the module's log-sum-exp shift)."""
    z = jnp.array([[1000.0 - 2.0e-6, 1000.0 - 1.0e-6]])
    H = jnp.array([1000.0])
    F = np.asarray(_exp_decay_structure(z, H, 1.0e-9))       # 1e-9 -> floored to 1e-6
    np.testing.assert_allclose(F, _F_raw_oracle_rebased(z, H, _O_H_DECAY_FLOOR),
                               rtol=1e-12, atol=0.0)
    # canary the floor VALUE: the unfloored 1e-9 scale gives a different profile.
    assert not np.allclose(F, _F_raw_oracle_rebased(z, H, 1.0e-9), atol=1e-30)


def test_h_decay_floor_concentrates_not_vanishes():
    """P4.1: a tiny h_decay (floored to 1e-6) on a STANDARD 100 m cell-centred
    column must NOT collapse to zero mixing.  The log-sum-exp rebase concentrates
    all weight in the deepest wet cell (the correct h->0 limit), keeping the
    column integral 1 and K finite/positive at the bottom.  The end-to-end K path
    is exercised at the SAME pathological h_decay (config replaced to 1e-9)."""
    z, h, H = _uniform_column()                      # deepest cell z=950 < H=1000
    F = np.asarray(normalize_structure(
        _exp_decay_structure(z, H, 1.0e-9, h), h))
    assert np.all(np.isfinite(F))
    assert float(np.sum(F[0] * np.asarray(h)[0])) == pytest.approx(1.0, rel=1e-12)
    assert F[0, -1] > 0.0 and int(np.argmax(F[0])) == 9   # concentrated at the bottom
    cfg_tiny = _CFG._replace(h_decay_m=1.0e-9)        # exercise the floor path in K
    K = np.asarray(compute_tidal_diffusivity(
        jnp.array([0.05]), z, h, H, jnp.full((1, 10), 1.0e-4), config=cfg_tiny))
    assert np.all(np.isfinite(K)) and K[0, -1] > 0.0      # no vanish
    assert np.all(K[0, :-1] == 0.0)                       # all weight in deepest cell


def test_wet_over_dry_tiny_h_concentrates_in_deepest_wet_cell():
    """P5.1: a column with WET levels above DRY below-bottom levels, at the tiny-h
    floor, must rebase over the WET part (not the dry dist=0 cells) so the deepest
    WET cell carries the mixing — not collapse to zero.  Exercised end-to-end
    through compute_tidal_diffusivity (which passes h_partial to the rebase)."""
    h = jnp.array([[100.0, 100.0, 100.0, 0.0, 0.0]])   # 3 wet, 2 dry
    z = jnp.array([[50.0, 150.0, 250.0, 350.0, 450.0]])   # dry levels z>H
    H = jnp.array([300.0])
    cfg_tiny = _CFG._replace(h_decay_m=1.0e-9)
    K = np.asarray(compute_tidal_diffusivity(
        jnp.array([0.05]), z, h, H, jnp.full((1, 5), 1.0e-4), config=cfg_tiny))
    assert np.all(np.isfinite(K))
    assert K[0, 2] > 0.0                               # deepest WET cell carries K
    assert np.all(K[0, :2] == 0.0) and np.all(K[0, 3:] == 0.0)   # others (wet-upper + dry) 0


def test_grads_through_structure_finite_wet_over_dry():
    """P6: a VJP that TRAVERSES _exp_decay_structure (grad wrt layer depths / H,
    NOT wrt E — which stays downstream of the structure and would miss the trap)
    must stay finite at the wet-over-dry tiny-h case, where dry cells overflow the
    exp; the DOUBLE-WHERE guard keeps the cotangent finite (a single output mask
    leaves 0*inf=NaN through the discarded branch — codex reproduced d/dH = NaN)."""
    h = jnp.array([[100.0, 100.0, 100.0, 0.0, 0.0]])   # wet-over-dry
    z = jnp.array([[50.0, 150.0, 250.0, 350.0, 450.0]])
    H = jnp.array([300.0])
    N2 = jnp.full((1, 5), 1.0e-4)

    def total_K_z(z_arr, hd):
        return jnp.sum(compute_tidal_diffusivity(
            jnp.array([0.05]), z_arr, h, H, N2, config=_CFG._replace(h_decay_m=hd)))

    # tiny h -> dry cells overflow the exp; the guard must keep grad wrt depths finite.
    assert np.all(np.isfinite(np.asarray(jax.grad(total_K_z)(z, 1.0e-9))))

    # moderate h -> a genuinely NONZERO finite grad through the structure.  sum(K) is
    # structure-INVARIANT (normalization pins sum(F*h)=1), so differentiate a single
    # UNCAPPED wet component K[0,0] instead — its F depends on the depths, giving a
    # MEANINGFUL (>>roundoff) depth sensitivity that proves the VJP is non-vacuous.
    def K_top_z(z_arr, hd):
        return compute_tidal_diffusivity(
            jnp.array([0.05]), z_arr, h, H, N2,
            config=_CFG._replace(h_decay_m=hd))[0, 0]

    g_mod = np.asarray(jax.grad(K_top_z)(z, 500.0))
    assert np.all(np.isfinite(g_mod)) and float(np.max(np.abs(g_mod))) > 1e-12

    # codex's probe: grad wrt H_bathy at tiny h also traverses the rebase -> finite.
    def total_K_H(H_val):
        return jnp.sum(compute_tidal_diffusivity(
            jnp.array([0.05]), z, h, jnp.array([H_val]), N2,
            config=_CFG._replace(h_decay_m=1.0e-9)))

    assert bool(jnp.isfinite(jax.grad(total_K_H)(300.0)))


# --- normalisation: unit integral, discrete-vs-analytic, dry levels ----------

def test_normalize_unit_column_integral():
    z, h, H = _uniform_column()
    F = normalize_structure(_exp_decay_structure(z, H, _O_ZETA), h)
    assert float(jnp.sum(F * h, axis=-1)[0]) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("dz,tol", [(100.0, 2e-3), (10.0, 2e-5), (1.0, 2e-7)])
def test_discrete_normalizer_converges_to_cvmix_analytic(dz, tol):
    """The module's DISCRETE-normalised F converges to CVMix's ANALYTIC Eq. 5.27
    structure exp(-(H-z)/zeta)/[zeta*(1-e^(-H/zeta))] as dz -> 0 — a documented
    departure (the discrete denominator sum_k F_raw*h_k vs the analytic integral
    zeta*(1-e^(-H/zeta)); mid-point quadrature error, shrinking with resolution).
    A COARSE grid must miss the fine tol so convergence is genuinely demonstrated."""
    z, h, H = _uniform_column(nlev=int(1000.0 / dz), dz=dz)
    F_sut = np.asarray(normalize_structure(_exp_decay_structure(z, H, _O_ZETA), h))[0]
    analytic_denom = _O_ZETA * (1.0 - math.exp(-1000.0 / _O_ZETA))   # Eq. 5.27 denom
    F_analytic = _F_raw_oracle(z, H, _O_ZETA)[0] / analytic_denom
    rel_err = float(np.max(np.abs(F_sut - F_analytic) / F_analytic))
    assert rel_err < tol
    if dz >= 100.0:                                  # coarse grid must NOT hit fine tol
        assert rel_err > 2e-7


def test_dry_levels_return_zero_structure_and_K():
    """P1.2: sub-bathymetry dry levels (h_partial=0) must carry F=0 and K=0, not a
    spurious 1/integral leak from the max(H-z,0) clamp."""
    h = jnp.array([[100.0, 100.0, 100.0, 0.0, 0.0, 0.0]])   # 3 wet, 3 dry
    z = jnp.array([[50.0, 150.0, 250.0, 350.0, 450.0, 550.0]])  # dry levels z>H
    H = jnp.array([300.0])
    F = np.asarray(normalize_structure(_exp_decay_structure(z, H, _O_ZETA), h))
    assert np.all(F[0, 3:] == 0.0)                   # dry structure zeroed
    assert np.all(F[0, :3] > 0.0)
    assert float(np.sum(F[0] * np.asarray(h)[0])) == pytest.approx(1.0, rel=1e-12)
    K = np.asarray(compute_tidal_diffusivity(
        jnp.array([0.05]), z, h, H, jnp.full((1, 6), 1.0e-4), config=_CFG))
    assert np.all(K[0, 3:] == 0.0)                   # no K leak in dry levels
    assert np.all(K[0, :3] > 0.0)


def test_negative_thickness_excluded_from_partition():
    """P2.2: a NEGATIVE sentinel thickness (h<0, also 'dry' per the h<=0 rule) must
    be clamped OUT of the integral denominator, so it cannot alter the wet-level
    partition.  Pinned against the INDEPENDENT NumPy oracle (which clamps h) for
    both F and the full K, and cross-checked bit-identical to the h=0 column."""
    z = jnp.array([[50.0, 150.0, 250.0]])
    H = jnp.array([300.0])
    h_clean = jnp.array([[100.0, 100.0, 0.0]])       # third level dry (h=0)
    h_neg = jnp.array([[100.0, 100.0, -100.0]])      # third level dry (h<0)
    E = jnp.array([0.05])
    N2 = jnp.full((1, 3), 1.0e-4)
    F_neg = np.asarray(normalize_structure(_exp_decay_structure(z, H, _O_ZETA), h_neg))
    # Independent oracle (clamps h internally) — NOT a second SUT call.
    F_o = _normalize_oracle(_F_raw_oracle(z, H, _O_ZETA), h_neg)
    np.testing.assert_allclose(F_neg, F_o, rtol=1e-12, atol=0.0)
    assert F_neg[0, 2] == 0.0
    # Full K pinned against the independent oracle with the negative sentinel.
    K = np.asarray(compute_tidal_diffusivity(E, z, h_neg, H, N2, config=_CFG))
    K_o = _K_oracle(E, F_o, N2, gamma=_O_GAMMA, q=_O_Q, rho0=_CFG.rho_0,
                    kmax=_O_KMAX, n2min=_O_N2MIN_MODULE)
    np.testing.assert_allclose(K, K_o, rtol=1e-12, atol=0.0)
    assert K[0, 2] == 0.0
    # Wet-level F invariant to the dry sentinel's sign (== the h=0 column).
    F_clean = np.asarray(normalize_structure(_exp_decay_structure(z, H, _O_ZETA), h_clean))
    np.testing.assert_allclose(F_neg[0, :2], F_clean[0, :2], rtol=1e-12, atol=0.0)


def test_degenerate_near_zero_column_threshold():
    """P3.3: the ``integral > 1e-12`` guard (STRICT >) makes a degenerate near-zero
    column return F=0.  A single seafloor level (F_raw=1) has integral == h, so the
    threshold VALUE 1e-12 is pinned EXACTLY: h==1e-12 -> F=0 (strict >, not >=), and
    the next representable value above -> normalises."""
    z = jnp.array([[1000.0]])                         # at seafloor -> F_raw = 1
    H = jnp.array([1000.0])
    F_at = np.asarray(normalize_structure(
        _exp_decay_structure(z, H, _O_ZETA), jnp.array([[1.0e-12]])))  # integral==1e-12
    assert F_at[0, 0] == 0.0                          # strict > -> exactly-at is dry
    h_above = float(np.nextafter(1.0e-12, np.inf))    # smallest value > 1e-12
    F_above = np.asarray(normalize_structure(
        _exp_decay_structure(z, H, _O_ZETA), jnp.array([[h_above]])))
    assert F_above[0, 0] == pytest.approx(1.0 / h_above, rel=1e-12)    # F_raw/integral


def test_degenerate_column_with_nan_N2_gives_zero():
    """P3.5: a POSITIVE but degenerate column (integral<=1e-12, so F=0) with N^2=NaN
    must still yield K=0 — the F>0 structural-zero guard catches it (the h_partial>0
    guard alone would not, since the thickness is positive)."""
    z = jnp.array([[1000.0]])
    H = jnp.array([1000.0])
    h_tiny = jnp.array([[1.0e-13]])                   # integral < 1e-12 -> F=0
    K = np.asarray(compute_tidal_diffusivity(
        jnp.array([0.05]), z, h_tiny, H, jnp.array([[jnp.nan]]), config=_CFG))
    assert np.all(np.isfinite(K)) and np.all(K == 0.0)


# --- full diffusivity (Eq. 5.22) vs independent oracle ----------------------

def test_compute_tidal_diffusivity_matches_oracle():
    z, h, H = _uniform_column()
    E = jnp.array([0.05])                            # W/m^2 (below the K_max regime)
    N2 = jnp.full((1, 10), 1.0e-4)                   # well above N_squared_min
    K = np.asarray(compute_tidal_diffusivity(E, z, h, H, N2, config=_CFG))
    np.testing.assert_allclose(K, _oracle_K_default(E, z, h, H, N2),
                               rtol=1e-12, atol=0.0)
    assert np.all(K < _CFG.K_max)                    # uncapped regime


def test_compute_tidal_diffusivity_nondefault_cfg_incl_rho0():
    cfg = TidalMixingConfig(Gamma=0.3, q_local=0.25, h_decay_m=300.0, K_max=1e-2,
                            N_squared_min=1e-9, rho_0=1000.0)   # rho_0 varied
    z, h, H = _uniform_column()
    E = jnp.array([0.04])
    N2 = jnp.full((1, 10), 2.0e-4)
    K = np.asarray(compute_tidal_diffusivity(E, z, h, H, N2, config=cfg))
    F = _normalize_oracle(_F_raw_oracle(z, H, cfg.h_decay_m), h)
    K_o = _K_oracle(E, F, N2, gamma=cfg.Gamma, q=cfg.q_local, rho0=cfg.rho_0,
                    kmax=cfg.K_max, n2min=cfg.N_squared_min)
    np.testing.assert_allclose(K, K_o, rtol=1e-12, atol=0.0)


def test_K_linear_in_E():
    z, h, H = _uniform_column()
    N2 = jnp.full((1, 10), 1.0e-4)
    K1 = compute_tidal_diffusivity(jnp.array([0.02]), z, h, H, N2, config=_CFG)
    K2 = compute_tidal_diffusivity(jnp.array([0.04]), z, h, H, N2, config=_CFG)
    assert jnp.allclose(K2, 2.0 * K1, rtol=1e-12)


def test_K_inverse_in_N2():
    z, h, H = _uniform_column()
    E = jnp.array([0.02])                            # both N2 above the floor
    K1 = compute_tidal_diffusivity(E, z, h, H, jnp.full((1, 10), 1.0e-4), config=_CFG)
    K2 = compute_tidal_diffusivity(E, z, h, H, jnp.full((1, 10), 2.0e-4), config=_CFG)
    assert jnp.allclose(K1, 2.0 * K2, rtol=1e-12)    # K ~ 1/N^2


def test_K_inverse_in_rho0():
    z, h, H = _uniform_column()
    E = jnp.array([0.02])
    N2 = jnp.full((1, 10), 1.0e-4)
    K1 = compute_tidal_diffusivity(E, z, h, H, N2,
                                   config=_CFG._replace(rho_0=1000.0))
    K2 = compute_tidal_diffusivity(E, z, h, H, N2,
                                   config=_CFG._replace(rho_0=2000.0))
    assert jnp.allclose(K1, 2.0 * K2, rtol=1e-12)    # K ~ 1/rho_0


def test_Kmax_cap_active():
    z, h, H = _uniform_column()
    E = jnp.array([1.0e3])                           # huge -> saturates the cap
    N2 = jnp.full((1, 10), _CFG.N_squared_min)
    K = compute_tidal_diffusivity(E, z, h, H, N2, config=_CFG)
    assert float(jnp.max(K)) == pytest.approx(_CFG.K_max, rel=1e-12)
    assert jnp.all(K <= _CFG.K_max + 1e-15)


def test_N2_floor_pinned_to_oracle_at_floor():
    """P2.4: below-floor N^2 must reproduce the oracle evaluated at EXACTLY the
    module floor 1e-7 (not merely 'two SUT calls agree'), and a wrong-floor oracle
    must NOT match — pinning the floor VALUE.  Kept below the cap so the floor,
    not the clip, is the operative effect."""
    z, h, H = _uniform_column()
    E = jnp.array([0.001])
    N2_below = jnp.full((1, 10), _O_N2MIN_MODULE * 1e-3)
    K = np.asarray(compute_tidal_diffusivity(E, z, h, H, N2_below, config=_CFG))
    assert np.all(K < _CFG.K_max)                    # floor operative, not the cap
    F = _normalize_oracle(_F_raw_oracle(z, H, _O_ZETA), h)
    K_at_floor = _K_oracle(E, F, N2_below, gamma=_O_GAMMA, q=_O_Q,
                           rho0=_CFG.rho_0, kmax=_O_KMAX, n2min=_O_N2MIN_MODULE)
    np.testing.assert_allclose(K, K_at_floor, rtol=1e-12, atol=0.0)
    # wrong-floor canary: a 4x-higher floor would give a different (smaller) K.
    K_wrong = _K_oracle(E, F, N2_below, gamma=_O_GAMMA, q=_O_Q, rho0=_CFG.rho_0,
                        kmax=_O_KMAX, n2min=_O_N2MIN_MODULE * 4.0)
    assert not np.allclose(K, K_wrong, rtol=1e-6)
    # just-above-floor: unfloored 1/N^2 branch matches the oracle at N2 itself.
    N2_above = jnp.full((1, 10), _O_N2MIN_MODULE * 10.0)
    K2 = np.asarray(compute_tidal_diffusivity(E, z, h, H, N2_above, config=_CFG))
    K2_o = _K_oracle(E, F, N2_above, gamma=_O_GAMMA, q=_O_Q, rho0=_CFG.rho_0,
                     kmax=_O_KMAX, n2min=_O_N2MIN_MODULE)
    np.testing.assert_allclose(K2, K2_o, rtol=1e-12, atol=0.0)


def test_land_mask_and_zero_energy_give_zero():
    z, h, H = _uniform_column()
    N2 = jnp.full((1, 10), 1.0e-4)
    K_land = compute_tidal_diffusivity(
        jnp.array([0.05]), z, h, H, N2, config=_CFG, land_mask=jnp.array([0.0]))
    assert jnp.all(K_land == 0.0)
    K_noE = compute_tidal_diffusivity(jnp.array([0.0]), z, h, H, N2, config=_CFG)
    assert jnp.all(K_noE == 0.0)


def test_dry_and_land_zero_robust_to_nan_N2():
    """P1.1: masked / below-bottom cells commonly carry N^2=NaN.  Dry levels and
    land cells must still be EXACTLY 0 (a multiplicative mask leaves 0/NaN=NaN and
    NaN*0=NaN, contaminating the solver); the ``where`` guards force literal 0."""
    # Dry levels (h=0) with NaN N^2 -> 0, wet levels finite.
    h = jnp.array([[100.0, 100.0, 0.0, 0.0]])
    z = jnp.array([[50.0, 150.0, 250.0, 350.0]])
    H = jnp.array([200.0])
    N2 = jnp.array([[1.0e-4, 1.0e-4, jnp.nan, jnp.nan]])
    K = np.asarray(compute_tidal_diffusivity(jnp.array([0.05]), z, h, H, N2, config=_CFG))
    assert np.all(np.isfinite(K))
    assert np.all(K[0, 2:] == 0.0)                   # dry -> 0 despite NaN N^2
    assert np.all(K[0, :2] > 0.0)
    # Whole-column land with NaN N^2 -> all 0.
    z2, h2, H2 = _uniform_column()
    K_land = np.asarray(compute_tidal_diffusivity(
        jnp.array([0.05]), z2, h2, H2, jnp.full((1, 10), jnp.nan),
        config=_CFG, land_mask=jnp.array([0.0])))
    assert np.all(np.isfinite(K_land)) and np.all(K_land == 0.0)


def test_fractional_land_mask_preserves_amplitude():
    """P2.2: the land mask is MULTIPLICATIVE — a fractional mask (partial coastal
    cell) scales K, it does not round up to full K.  mask=0.5 -> half K; mask=0 -> 0."""
    z, h, H = _uniform_column()
    N2 = jnp.full((1, 10), 1.0e-4)
    E = jnp.array([0.05])
    K_full = compute_tidal_diffusivity(E, z, h, H, N2, config=_CFG)
    K_half = compute_tidal_diffusivity(E, z, h, H, N2, config=_CFG,
                                       land_mask=jnp.array([0.5]))
    assert jnp.allclose(K_half, 0.5 * K_full, rtol=1e-12)


# --- coefficient canaries ---------------------------------------------------

def test_config_defaults_match_cvmix():
    c = TidalMixingConfig()
    assert c.Gamma == _O_GAMMA           # Eq. 5.14
    assert c.q_local == pytest.approx(_O_Q, rel=1e-12)   # Eq. 5.19
    assert c.h_decay_m == _O_ZETA        # Eq. 5.30
    assert c.K_max == _O_KMAX            # Sec. 5.3.3
    assert c.rho_0 == _O_RHO_DEFAULT     # Boussinesq reference density
    # DEPARTURE: the module's N^2 floor is HIGHER than Simmons' suggested 1e-8.
    assert c.N_squared_min == _O_N2MIN_MODULE
    assert c.N_squared_min > _O_N2MIN_SIMMONS


# --- differentiability (AD-compatible, not everywhere smooth) ----------------

def test_tidal_grads_finite_smooth_regime():
    z, h, H = _uniform_column()
    N2 = jnp.full((1, 10), 1.0e-4)

    def total_K(E_val):
        return jnp.sum(compute_tidal_diffusivity(
            jnp.array([E_val]), z, h, H, N2, config=_CFG))

    g = jax.grad(total_K)(0.02)
    assert bool(jnp.isfinite(g)) and float(g) > 0.0


def test_tidal_grads_finite_at_cap_and_floor():
    """P3.4/P3.2/P3.3/P3.9: selected-branch subgradients AT the kinks are finite and
    the kinks are REAL (opposite one-sided slopes).  The cap transition of the
    bottom level is located to ULP resolution in the SUT (not assumed from the
    NumPy oracle), and it is confirmed that ONLY the bottom level caps there."""
    z, h, H = _uniform_column()
    N2min = _CFG.N_squared_min
    N2col = jnp.full((1, 10), N2min)

    def K_bottom(E_val):
        return compute_tidal_diffusivity(
            jnp.array([E_val]), z, h, H, N2col, config=_CFG)[0, -1]

    # Bisect the SUT to ULP: lo = last E with the bottom level strictly UNCAPPED;
    # hi = the first representable E at which the bottom output is no longer < K_max
    # (the first capped/at-cap input — not necessarily an exact pre-clip tie).
    F = _normalize_oracle(_F_raw_oracle(z, H, _O_ZETA), h)
    E_seed = _O_KMAX * _CFG.rho_0 * N2min / (_O_GAMMA * _O_Q * float(F[0, -1]))
    lo, hi = E_seed * 0.5, E_seed * 2.0
    assert float(K_bottom(lo)) < _O_KMAX <= float(K_bottom(hi))
    for _ in range(80):                              # converge to adjacent floats
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if float(K_bottom(mid)) < _O_KMAX:
            lo = mid
        else:
            hi = mid
    assert float(K_bottom(lo)) < _O_KMAX             # lo: bottom strictly uncapped
    assert float(K_bottom(hi)) == pytest.approx(_O_KMAX, rel=1e-12)  # hi: at/above cap
    # ONLY the bottom level caps at the transition (no column-wide saturation):
    K_hi = np.asarray(compute_tidal_diffusivity(
        jnp.array([hi]), z, h, H, N2col, config=_CFG))
    assert np.all(K_hi[0, :-1] < _O_KMAX)

    # Opposite one-sided slopes prove a REAL clip kink of the bottom level: positive
    # on the uncapped side (lo), zero strictly inside the cap (2x the transition E).
    # The subgradient at the first-capped input hi is only required to be FINITE
    # (JAX's clip subgradient is finite on both sides and at the tie).
    g_lo = float(jax.grad(K_bottom)(lo))
    g_capped = float(jax.grad(K_bottom)(E_seed * 2.0))
    g_hi = float(jax.grad(K_bottom)(hi))
    assert np.isfinite(g_lo) and g_lo > 0.0          # uncapped: dK/dE > 0
    assert np.isfinite(g_capped) and abs(g_capped) < 1e-18   # capped: dK/dE = 0
    assert np.isfinite(g_hi)                         # first-capped input: finite subgrad

    # N^2 floor kink: finite subgradient evaluated EXACTLY at N^2 == N_squared_min.
    def K_of_N2(n2_val):
        return jnp.sum(compute_tidal_diffusivity(
            jnp.array([0.001]), z, h, H,
            jnp.full((1, 10), n2_val), config=_CFG))

    assert bool(jnp.isfinite(jax.grad(K_of_N2)(N2min)))
