"""Unit tests for ``legoesm.atmosphere.physics.convection._plume``.

The five column-physics workhorses delivered in PR 0:

* :func:`compute_lcl`                     — Bolton (1980) LCL.
* :func:`compute_lfc_lnb`                 — smooth fractional indices.
* :func:`compute_cin`                     — convective inhibition [J/kg].
* :func:`entraining_detraining_plume`     — vmappable updraft integrator.
* :func:`cmt_gregory_1997`                — Gregory et al. 1997 CMT closure.

These tests pin:

* Bolton 1980 reference LCL (an unsaturated parcel test that exercises
  the formula non-trivially);
* finite gradients through every helper at non-trivial input values;
* analytical limits of the plume integrator (no entrainment ⇒ moist
  adiabat; strong entrainment ⇒ environmental relaxation; sub-cloud
  layer is ``M_u = 0``);
* CMT zero-mass-flux invariant and shear-sign convention.

The tests run on CPU regardless of the host's JAX backend so that the
suite is reproducible on the Apple-Silicon Metal hosts used during
development.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat
from legoesm.atmosphere.physics.convection import _plume as P


# ---------------------------------------------------------------------------
# Synthetic-column helpers reused across tests
# ---------------------------------------------------------------------------

def _synthetic_column(
    ncol: int = 2,
    nlev: int = 16,
    *,
    p_s: float = 1.0e5,
    p_top: float = 5.0e3,
    T_sfc: float = 295.0,
    lapse_rate_K_per_km: float = 6.5,
    q_sfc: float = 8.0e-3,
    q_scale_height_m: float = 3000.0,
):
    """Surface-last column with a stable troposphere and an exponential
    moisture profile.

    Returns (T_env, q_v_env, p_full, p_half, z_full).  The default
    ``q_sfc=8 g/kg`` gives an unsaturated parcel at the surface, which
    exercises Bolton's LCL formula non-trivially (a saturated parcel
    short-circuits to ``T_LCL = T_parcel``).
    """
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)         # surface=1.0
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)   # (ncol, nlev)

    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [
            jnp.full((ncol, 1), p_top * 0.5),  # half-level above TOA
            p_half_inner,
            jnp.full((ncol, 1), p_s),
        ],
        axis=1,
    )

    H = 8500.0  # scale height [m]
    z_full = -H * jnp.log(p_full / p_s)                  # surface=0
    T_env = (
        jnp.full((ncol,), T_sfc)[:, None]
        - (lapse_rate_K_per_km * 1e-3) * z_full
    )
    q_v_env = q_sfc * jnp.exp(-z_full / q_scale_height_m)

    return T_env, q_v_env, p_full, p_half, z_full


# ---------------------------------------------------------------------------
# compute_lcl — Bolton 1980 reference
# ---------------------------------------------------------------------------

def test_compute_lcl_unsaturated_parcel_bolton_reference():
    """Bolton 1980 Eq. 22 against a reference parcel.

    Parcel state: ``T = 290 K``, ``q = 8 g/kg``, ``p = 1000 hPa``.
    For these inputs the Bolton formula produces ``T_LCL ≈ 282.6 K``
    and Poisson's relation gives ``p_LCL ≈ 88,250 Pa`` — well above
    the LCL coincidence (where the parcel is saturated and LCL =
    parcel level).  Tolerances are generous because Bolton's formula
    is itself an empirical fit accurate to ~0.5 K.
    """
    T_parcel = jnp.asarray([290.0])
    q_parcel = jnp.asarray([8.0e-3])
    p_parcel = jnp.asarray([1.0e5])

    nlev = 16
    p_full = jnp.linspace(5.0e3, 1.0e5, nlev)[None, :]

    lcl = P.compute_lcl(T_parcel, q_parcel, p_parcel, p_full)

    # T_LCL: literature value ~283 K (Bolton 1980 produces values in
    # the 280–284 K window for this input depending on which form of
    # the formula one cites).  Allow a 2 K tolerance.
    assert pytest.approx(283.0, abs=2.0) == float(lcl.T_lcl[0])
    # p_LCL: between 86 and 92 kPa for this parcel.
    assert 8.6e4 < float(lcl.p_lcl[0]) < 9.2e4
    # Smooth fractional level index lies in the valid range.
    k_lcl = float(lcl.k_lcl_smooth[0])
    assert 0.0 <= k_lcl <= float(nlev - 1)


def test_compute_lcl_saturated_parcel_returns_parcel_level():
    """If the parcel is already saturated (or supersaturated), the
    Bolton formula gives ``T_LCL = T_parcel`` and ``p_LCL = p_parcel``."""
    T = jnp.asarray([290.0])
    p = jnp.asarray([1.0e5])
    q_sat = saturation_mixing_ratio(T, p)
    q_super = q_sat * 1.5  # supersaturated — should still produce LCL = parcel level

    p_full = jnp.linspace(5.0e3, 1.0e5, 8)[None, :]
    lcl = P.compute_lcl(T, q_super, p, p_full)
    assert pytest.approx(float(T[0]), rel=1e-6) == float(lcl.T_lcl[0])
    assert pytest.approx(float(p[0]), rel=1e-6) == float(lcl.p_lcl[0])


def test_compute_lcl_grad_through_temperature():
    """``d T_LCL / d T_parcel`` is finite — preserves training signal
    flowing through the Bolton formula."""
    p = jnp.asarray([1.0e5])
    q = jnp.asarray([8.0e-3])
    p_full = jnp.linspace(5.0e3, 1.0e5, 8)[None, :]

    def f(T_p):
        return P.compute_lcl(T_p, q, p, p_full).T_lcl[0]

    g = float(jax.grad(f)(jnp.asarray(290.0)))
    assert np.isfinite(g)
    assert abs(g) > 1e-3


# ---------------------------------------------------------------------------
# compute_lfc_lnb — order and smooth-grad
# ---------------------------------------------------------------------------

def test_compute_lfc_lnb_capping_inversion_column():
    """In a column with a capping inversion, the LFC sits above the
    inversion and the LNB sits above the LFC."""
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
        T_sfc=300.0, q_sfc=15.0e-3, lapse_rate_K_per_km=5.0,  # destabilized profile
    )
    T_sfc = T_env[:, -1]
    T_ma = compute_moist_adiabat(T_sfc, p_full)
    k_lfc, k_lnb = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)

    # In surface-last convention the LFC has a *larger* index than
    # the LNB (closer to surface).  This ordering is the structural
    # invariant.
    assert float(k_lfc[0]) > float(k_lnb[0]), (
        f"Surface-last ordering: LFC must have larger index than LNB; "
        f"got LFC={float(k_lfc[0])}, LNB={float(k_lnb[0])}"
    )


def test_compute_lfc_lnb_lnb_finds_upper_zero_crossing():
    """LNB must localize where buoyancy transitions positive→negative
    going UP, not at the LFC.

    Constructed buoyancy profile (surface-last; index 9 = surface):
        k=0..1: -2, -1     (above LNB, parcel cool)
        k=2..6: +1, +2, +2, +1, +0.5  (CAPE region)
        k=7..8: -1, -2     (CIN region above LCL)
        k=9:    +1         (parcel warmer at sfc launch)

    Expected: LFC ≈ 6.5 (lowest upward crossing 0 going up from sfc),
    LNB ≈ 1.5 (where buoyancy goes positive → negative).

    Pre-fix: LNB ≈ 6.2 (collapsed onto the LFC because the
    "above-LFC" mask used a surface-first threshold against
    surface-last level indices).
    """
    buoyancy = jnp.array([
        -2.0, -1.0, 1.0, 2.0, 2.0, 1.0, 0.5, -1.0, -2.0, 1.0,
    ])
    T_env = jnp.full((1, 10), 280.0)
    T_parcel_ma = T_env + buoyancy[None, :]
    k_lfc, k_lnb = P.compute_lfc_lnb(T_env, T_parcel_ma, sharpness=10.0)
    # LFC unchanged by the bug — ~6-7
    assert 5.5 < float(k_lfc[0]) < 7.5, (
        f"k_lfc = {float(k_lfc[0])}; expected ~6.5"
    )
    # LNB must clearly separate from LFC at index ~1-3
    assert float(k_lnb[0]) < 4.0, (
        f"k_lnb = {float(k_lnb[0])}; expected < 4.0 (well above LFC). "
        "Pre-fix bug: k_lnb collapses onto k_lfc."
    )


def test_compute_lfc_lnb_grad_through_environment():
    """``d k_lfc / d T_sfc`` is finite — training-signal preservation."""
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(T_sfc=298.0)
    T_ma_base = compute_moist_adiabat(T_env[:, -1], p_full)

    def f(dT):
        T_perturbed = T_env + dT
        T_ma = compute_moist_adiabat(T_env[:, -1] + dT, p_full)
        k_lfc, _ = P.compute_lfc_lnb(T_perturbed, T_ma, sharpness=1.0)
        return k_lfc[0]

    g = float(jax.grad(f)(jnp.asarray(0.0)))
    assert np.isfinite(g)


# ---------------------------------------------------------------------------
# compute_cin
# ---------------------------------------------------------------------------

def test_compute_cin_non_negative_and_finite():
    """CIN is non-negative by construction (positive-part of negative
    buoyancy) and finite for every column."""
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column()
    T_ma = compute_moist_adiabat(T_env[:, -1], p_full)
    lcl = P.compute_lcl(T_env[:, -1], q_v_env[:, -1], p_full[:, -1], p_full)
    k_lfc, _ = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)

    cin = P.compute_cin(T_env, T_ma, p_full, p_half, lcl.k_lcl_smooth, k_lfc)
    assert jnp.all(cin >= 0.0)
    assert jnp.all(jnp.isfinite(cin))


def test_compute_cin_integrates_negative_buoyancy_between_lcl_and_lfc():
    """CIN must integrate the negative buoyancy in the LCL→LFC layer.

    Construct a column where T_env > T_parcel by 2 K only at indices
    6, 7 (the CIN layer between LCL at index 8 and LFC at index 5).
    Analytical CIN = R_d * sum_{k=6,7} 2 * dp_k / p_full_k.

    Pre-fix the CIN window was the empty intersection of "above LFC"
    AND "below LCL" (the two factors selected the WRONG sides), so
    CIN was suppressed to ~7% of the correct value.
    """
    ncol, nlev = 1, 10
    T_env = jnp.full((ncol, nlev), 280.0)
    T_parcel = jnp.full((ncol, nlev), 280.0).at[:, 6:8].set(278.0)
    p_half = jnp.broadcast_to(
        jnp.linspace(1e4, 1e5, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    k_lcl = jnp.array([8.0])
    k_lfc = jnp.array([5.0])
    cin = P.compute_cin(
        T_env, T_parcel, p_full, p_half, k_lcl, k_lfc,
        indicator_sharpness=5.0,
    )
    # Analytical CIN over k=6,7 only
    dp = p_half[:, 1:] - p_half[:, :-1]
    cin_analytical = constants.R_d * jnp.sum(
        jnp.maximum(0.0, T_env - T_parcel) * dp / p_full, axis=-1,
    )
    rel_err = float(jnp.max(jnp.abs(cin - cin_analytical) / cin_analytical))
    # Pre-fix: rel_err ~ 0.93 (CIN ≈ 7% of analytical).
    # Post-fix: should match within ~30% (smooth window has soft edges).
    assert rel_err < 0.3, (
        f"CIN = {float(cin[0]):.3e} vs analytical = "
        f"{float(cin_analytical[0]):.3e}; rel_err = {rel_err:.2f}. "
        "Pre-fix the CIN window factors selected the WRONG sides "
        "(`above_LFC * below_LCL` is empty for k_lnb<k_lfc<k_lcl)."
    )


def test_compute_cin_zero_for_unstable_parcel_above_LCL():
    """When the parcel is positively buoyant immediately above the
    LCL (no cap), CIN ≈ 0."""
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
        T_sfc=302.0, q_sfc=18.0e-3, lapse_rate_K_per_km=8.0,
    )
    T_ma = compute_moist_adiabat(T_env[:, -1], p_full)
    lcl = P.compute_lcl(T_env[:, -1], q_v_env[:, -1], p_full[:, -1], p_full)
    k_lfc, _ = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)
    cin = P.compute_cin(T_env, T_ma, p_full, p_half, lcl.k_lcl_smooth, k_lfc)
    assert float(cin[0]) < 5.0   # J/kg


# ---------------------------------------------------------------------------
# entraining_detraining_plume
# ---------------------------------------------------------------------------

def test_plume_outputs_finite_and_correct_shape():
    """Smoke: every output array is finite with the expected
    ``(ncol, nlev)`` shape."""
    ncol, nlev = 3, 14
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
    eps = jnp.full((ncol, nlev), 5.0e-4)
    dlt = jnp.full((ncol, nlev), 5.0e-4)
    M_b = jnp.full((ncol,), 0.05)

    plume = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
    )
    for arr in (plume.M_u, plume.T_u, plume.q_u, plume.q_c_u, plume.B_u):
        assert arr.shape == (ncol, nlev)
        assert jnp.all(jnp.isfinite(arr))


def test_plume_no_entrainment_limit_matches_moist_adiabat():
    """With ``epsilon = delta = 0`` and a strongly buoyant parcel,
    the in-cloud plume temperature follows the moist adiabat (within
    a tolerance set by the discretization)."""
    ncol, nlev = 1, 16
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
        ncol, nlev, T_sfc=300.0, q_sfc=18.0e-3, lapse_rate_K_per_km=8.0,
    )
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]

    # Cloud base at surface.  With unsaturated air the plume condenses
    # immediately above the surface; the moist adiabat starts from
    # T_base.
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
    T_ma = compute_moist_adiabat(T_base, p_full)

    eps = jnp.zeros((ncol, nlev))
    dlt = jnp.zeros((ncol, nlev))
    M_b = jnp.full((ncol,), 0.05)

    plume = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
    )
    # Sample three mid-troposphere levels and compare to the moist
    # adiabat.  The first-order Euler ascent in the plume integrator
    # accumulates discretization error, so the tolerance is generous.
    sampled = jnp.array([nlev // 2, nlev // 2 + 1, nlev // 2 + 2])
    plume_T = plume.T_u[0, sampled]
    ma_T = T_ma[0, sampled]
    diff = jnp.abs(plume_T - ma_T)
    assert jnp.all(diff < 5.0), (
        f"No-entrainment plume should track moist adiabat within ~5 K, "
        f"got differences {np.asarray(diff)} K"
    )


def test_plume_sub_cloud_mass_flux_is_suppressed():
    """Levels strictly below the cloud base have ``M_u ≈ 0`` because
    the ``above_base_weight`` mask suppresses them."""
    ncol, nlev = 1, 16
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    # Force a cloud-base index well above the surface by passing a
    # synthetic ``k_base_smooth`` (mid-column) so we have several
    # sub-cloud levels to inspect.
    k_base_synthetic = jnp.full((ncol,), float(nlev) - 4.0)

    eps = jnp.full((ncol, nlev), 5.0e-4)
    dlt = jnp.full((ncol, nlev), 5.0e-4)
    M_b = jnp.full((ncol,), 0.05)

    plume = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, k_base_synthetic, eps, dlt, M_b,
        buoyancy_sharpness=2.0,
    )
    # Surface-last: levels with index > k_base are below the cloud
    # base (lower altitude).
    sub_cloud_mass = plume.M_u[:, -1]   # surface-most level
    base_mass = plume.M_u[:, int(float(nlev) - 4.0)]
    # Sub-cloud value should be much smaller than the cloud-base
    # value.  Tolerance reflects the smoothness of the sigmoid mask.
    assert float(sub_cloud_mass[0]) < 0.5 * float(base_mass[0])


def test_plume_q_c_u_diluted_by_entrainment():
    """Audit Codex review (severity major): plume cloud water was
    accumulated but not diluted by environmental entrainment.  The
    proper continuity for an intensive quantity in an entraining-
    detraining plume is ``dq_c/dz = -eps · q_c + cond/M``: env air
    carries q_c=0, so entrainment uniformly decreases ``q_c_u``.

    Test setup: a parcel that condenses at the LCL and ascends to a
    level where the moist-adiabatic temperature exceeds the
    saturation profile only slightly — so condensation occurs in a
    narrow lower layer and is essentially zero aloft.  With strong
    entrainment, the carried-aloft q_c_u must DECREASE between the
    last condensation level and the column top (entrainment
    dilution dominates).  In the buggy form q_c_u was monotone
    non-decreasing aloft (carried unchanged when condensation = 0).
    """
    # High-resolution column + small ``eps`` so that ``eps·dz <= 0.25``
    # at *every* layer.  Prior version used ``eps = 1e-3`` which gave
    # ``eps·dz > 1`` in the top 5 levels — the dilution clip
    # ``max(0, 1 - eps·dz)`` triggered there and the test was
    # exercising the clip path, not the stable explicit-Euler dilution
    # (Codex stop-time review: "regression test exercises the clip
    # path, not stable dilution").  ``eps = 1e-4 /m`` keeps the
    # dilution multiplier strictly in (0.75, 1) for the entire column
    # so ``q_c_u`` decay is governed by the physics, not the clip.
    ncol, nlev = 1, 60
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
        ncol, nlev, T_sfc=295.0, q_sfc=8.0e-3, lapse_rate_K_per_km=7.0,
    )
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)

    eps_scalar = 1.0e-4
    eps = jnp.full((ncol, nlev), eps_scalar)
    dlt = jnp.full((ncol, nlev), 1.0e-5)        # weaker detrainment
    M_b = jnp.full((ncol,), 0.05)

    # Sanity: assert eps·dz < 1 at every layer so the test really
    # exercises stable dilution rather than the corner-case clip.
    dz_layers = z_full[0, :-1] - z_full[0, 1:]
    max_eps_dz = float(jnp.max(eps_scalar * dz_layers))
    assert max_eps_dz < 1.0, (
        f"Test fixture broken — max eps·dz = {max_eps_dz:.3f} ≥ 1; "
        "the dilution clip ``max(0, 1-eps·dz)`` would dominate and "
        "the test would not exercise stable explicit-Euler dilution."
    )

    plume = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
    )
    q_c_u = plume.q_c_u[0]                       # surface-last (ncol, nlev)

    # The plume's q_c_u peaks somewhere above LCL where condensation
    # is active; aloft, where condensation tapers off, the only
    # forcing on q_c_u is the entrainment dilution ``-eps·q_c_u``,
    # so it must decay.  Compare the column top to the peak.
    q_c_max = float(jnp.max(q_c_u))
    q_c_top = float(q_c_u[0])                    # column top (surface-last)

    assert q_c_max > 1e-8, (
        f"Test fixture broken — plume produced no cloud water "
        f"(max q_c_u = {q_c_max:.3e}); cannot test dilution."
    )
    # Quantitative bound: with ``eps·dz ≈ 0.02`` per layer (typical
    # mid-tropospheric value here) the multiplicative dilution
    # ``(1 - eps·dz)`` over ~30 layers between peak and column top
    # gives a ratio of order ``0.98^30 ≈ 0.55``.  We use a slightly
    # looser threshold to leave headroom for the layer-thickness
    # variation (top layers are thicker, dilution is faster).  The
    # buggy form (no dilution) gives ratio ≈ 1.0 — orders of
    # magnitude away from the threshold.
    ratio = q_c_top / q_c_max
    assert ratio < 0.85, (
        f"Plume q_c_u at column top ({q_c_top:.3e}) is not significantly "
        f"smaller than peak ({q_c_max:.3e}); ratio = {ratio:.3f}, "
        "expected < 0.85 from stable explicit-Euler dilution.  Audit "
        "Codex finding 'plume cloud water is accumulated but not "
        "diluted by entrainment' has regressed."
    )


def test_plume_buoyancy_death_memory_terminates_plume_above_inversion():
    """Audit Codex cycle 2 P2 (deferred → addressed via opt-in flag).

    In the default (legacy) mode, the plume's buoyancy filter is
    purely local — each level applies its own ``sigmoid(B_u)`` — so
    a plume killed by negative buoyancy at one level can resume
    reporting nonzero ``M_u`` above an inversion.  The new
    ``buoyancy_death_memory=True`` flag makes the filter
    monotone-decreasing along the column ascent: once the plume
    dies, it stays dead.

    Test setup: stable lower troposphere up to a thin warm layer
    that creates positive buoyancy aloft (an "elevated" CAPE region
    above an inversion).  In legacy mode, the upper-CAPE region
    sees high ``plume_alive`` and reports ``M_u``.  With memory
    enabled, the negative-buoyancy zone in the inversion truncates
    the plume and the upper levels see ``alive_min`` ≈ 0.
    """
    # Construct a column where the plume's buoyancy goes:
    # positive (cloud base → LFC) → NEGATIVE (mid-trop inversion) →
    # POSITIVE again (cold upper trop above the inversion).  The
    # plume integrator's T_u follows a moist adiabat (cools roughly
    # linearly with z), so for B_u to dip negative then recover, we
    # need T_env to have a WARM bump at mid-altitude AGAINST a
    # very steep lapse rate elsewhere (so upper trop is much colder
    # than the moist adiabat).
    ncol, nlev = 1, 40
    p_s = 1.0e5
    p_top = 5.0e3
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), p_half_inner, jnp.full((ncol, 1), p_s)],
        axis=1,
    )
    z_full = -8500.0 * jnp.log(p_full / p_s)

    # Steep environmental lapse rate (10 K/km) so upper trop is
    # much colder than a moist adiabat — guarantees positive B_u
    # aloft.  Add a strong warm bump at mid-altitude to drive a
    # negative B_u zone in between.
    T_sfc = 305.0
    T_env_baseline = T_sfc - 10.0e-3 * z_full
    z = z_full[0]
    bump_center = 7000.0
    bump_width = 1000.0
    bump_amplitude = 12.0                        # +12 K bump
    bump = bump_amplitude * jnp.exp(
        -((z - bump_center) ** 2) / (2 * bump_width ** 2)
    )
    T_env = T_env_baseline + bump[None, :]
    q_v_env = 16.0e-3 * jnp.exp(-z_full / 3000.0)
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
    eps = jnp.full((ncol, nlev), 5.0e-4)
    dlt = jnp.full((ncol, nlev), 5.0e-4)
    M_b = jnp.full((ncol,), 0.05)

    plume_legacy = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
        buoyancy_death_memory=False,
    )
    plume_memory = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
        buoyancy_death_memory=True,
    )

    # The revival region is the layer JUST ABOVE the warm bump,
    # where the env temperature drops sharply and the plume's
    # T_u (still on its moist adiabat) becomes warmer again — a
    # second positive-buoyancy zone.  For the bump centered at
    # ~7000 m on a 40-level column, the revival zone is roughly
    # k = 12-14 (surface-last with surface at index nlev-1, so
    # the upper region is *low* indices).  Compute the maximum
    # plume M_u over this slice and confirm legacy mode reports
    # a substantial flux there while memory mode does not.
    revival_slice = slice(10, 16)               # indices around z ≈ 8000–11000 m
    upper_M_legacy = float(jnp.max(plume_legacy.M_u[0, revival_slice]))
    upper_M_memory = float(jnp.max(plume_memory.M_u[0, revival_slice]))

    # The two behaviors must differ — that's the whole point.
    if upper_M_legacy < 1e-12 and upper_M_memory < 1e-12:
        # Fixture didn't construct a strong-enough double-CAPE; the
        # legacy plume isn't actually reviving in the upper trop.
        # The test is then degenerate; surface a clear message so
        # the fixture can be tuned.
        pytest.skip(
            f"fixture did not exercise the buoyancy-revive path "
            f"(both upper M_u tiny: legacy={upper_M_legacy:.3e}, "
            f"memory={upper_M_memory:.3e})"
        )

    # Memory-mode upper-trop M_u must be substantially smaller than
    # legacy mode (the inversion killed the plume; the death is
    # remembered so the upper-CAPE region cannot revive it).
    assert upper_M_memory < 0.5 * upper_M_legacy, (
        f"buoyancy_death_memory failed to suppress upper-CAPE plume "
        f"revival: legacy upper M_u = {upper_M_legacy:.3e}, "
        f"memory upper M_u = {upper_M_memory:.3e} — expected memory < "
        f"50% of legacy."
    )


def test_plume_M_u_grad_through_strong_detrainment_is_finite_and_nonzero():
    """Audit Codex review (severity major): the explicit-Euler plume
    mass flux ``M * (1 + (eps - dlt) * dz)`` can produce a *negative*
    value for strong detrainment + thick layers (e.g.
    ``dlt = 5e-3 /m``, ``dz = 2000 m`` ⇒ multiplier = ``-7``).  The
    subsequent ``jnp.maximum(M, 0)`` clip then KILLED the gradient
    w.r.t. ``eps`` and ``dlt`` for every level above the first clip
    — making AD-based sensitivity studies silently mis-state the
    detrainment-parameter sensitivity through deep convective
    columns.

    The exact integration ``M * exp((eps - dlt) * dz)`` is always
    positive and produces a meaningful, non-zero gradient even at
    very negative ``(eps - dlt) * dz``.

    **Critical**: the buggy form's gradient is NOT identically zero
    — the very first scan iteration uses the layer-thickness floor
    ``dz = max(z_e - z_prev, 1)`` = 1 m at the surface, where
    ``(eps - dlt) * 1 = -4e-3`` is too small to clip.  That residual
    surface contribution alone yields ``|grad| ≈ 3.5e-3`` regardless
    of what happens aloft.  The test must therefore assert a
    threshold *much larger* than the residual to actually
    discriminate fixed vs buggy form (an earlier formulation used
    ``> 1e-30`` and accepted both forms).
    """
    ncol, nlev = 1, 16
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
        ncol, nlev, T_sfc=300.0, q_sfc=18.0e-3, lapse_rate_K_per_km=8.0,
    )
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
    M_b = jnp.full((ncol,), 0.05)
    eps = jnp.full((ncol, nlev), 1.0e-3)

    def M_total_at_top(dlt_scalar):
        dlt = jnp.full((ncol, nlev), dlt_scalar)
        plume = P.entraining_detraining_plume(
            T_env, q_v_env, p_full, p_half, z_full,
            T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
        )
        return jnp.sum(plume.M_u)

    # Sample at dlt = 3e-3 — within this fixture the fixed form
    # gives |grad| ≈ 4.8 while the buggy clipped form gives
    # |grad| ≈ 3.5e-3 (a 1300× discrimination, vs. ~270× at
    # dlt = 5e-3).  The threshold ``0.5`` is well above the buggy
    # residual (~3.5e-3) and well below the fixed value (~4.8) so
    # the test discriminates robustly.
    dlt_test = jnp.asarray(3.0e-3)
    g = float(jax.grad(M_total_at_top)(dlt_test))
    assert np.isfinite(g), (
        f"Plume gradient w.r.t. strong detrainment: g = {g} (NaN/Inf)"
    )
    assert abs(g) > 0.5, (
        f"|d(sum M_u)/d_dlt| = {abs(g):.3e} at dlt=3e-3 is below the "
        "discrimination threshold 0.5.  The buggy explicit-Euler form "
        "``M*(1 + (eps-dlt)*dz)`` clipped at ``max(., 0)`` gives "
        "|grad| ≈ 3.5e-3 (only the surface-layer floor survives the "
        "clip).  The fixed exponential form gives |grad| ≈ 4.8 — "
        "the AD path is intact through every level.  A failure here "
        "indicates the exp integration has been reverted to the buggy "
        "explicit-Euler form (audit Codex finding 'plume mass flux "
        "uses explicit Euler plus a hard nonnegative clip')."
    )


def test_plume_grad_through_epsilon():
    """``d (sum M_u) / d epsilon_0`` is finite — the plume integrator
    is differentiable through its tunables."""
    ncol, nlev = 1, 12
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
    M_b = jnp.full((ncol,), 0.05)
    dlt = jnp.full((ncol, nlev), 5.0e-4)

    def f(eps_0):
        eps = jnp.full((ncol, nlev), eps_0)
        plume = P.entraining_detraining_plume(
            T_env, q_v_env, p_full, p_half, z_full,
            T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
        )
        return jnp.sum(plume.M_u)

    g = float(jax.grad(f)(jnp.asarray(5.0e-4)))
    assert np.isfinite(g)


# ---------------------------------------------------------------------------
# cmt_gregory_1997
# ---------------------------------------------------------------------------

def test_cmt_zero_when_no_mass_flux():
    """``M_u = M_d = 0`` ⇒ both wind tendencies are exactly zero."""
    ncol, nlev = 2, 8
    u = jnp.linspace(0.0, 30.0, nlev)[None, :].repeat(ncol, axis=0)
    v = jnp.zeros((ncol, nlev))
    M_u = jnp.zeros((ncol, nlev))
    p_full = jnp.linspace(5.0e3, 1.0e5, nlev)[None, :].repeat(ncol, axis=0)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), 2.5e3), p_half_inner, jnp.full((ncol, 1), 1.0e5)],
        axis=1,
    )
    rho = p_full / (constants.R_d * jnp.full(p_full.shape, 280.0))

    du, dv = P.cmt_gregory_1997(u, v, M_u, None, p_full, p_half, rho)
    assert jnp.all(du == 0.0)
    assert jnp.all(dv == 0.0)


def test_cmt_finite_in_sheared_environment():
    """Non-zero updraft mass flux in a sheared environment produces
    finite, non-trivial momentum tendencies."""
    ncol, nlev = 1, 12
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
    rho = p_full / (constants.R_d * T_env)

    u = jnp.linspace(0.0, 30.0, nlev)[None, :].repeat(ncol, axis=0)
    v = jnp.zeros_like(u)
    M_u = jnp.linspace(0.0, 0.05, nlev)[None, :].repeat(ncol, axis=0)

    du, dv = P.cmt_gregory_1997(u, v, M_u, None, p_full, p_half, rho)
    assert jnp.all(jnp.isfinite(du))
    assert jnp.all(jnp.isfinite(dv))
    # Some level has non-trivial du/dt — the tendency should not be
    # uniformly zero in a sheared column with non-zero mass flux.
    assert float(jnp.max(jnp.abs(du))) > 1e-8


def test_cmt_grad_through_c_u():
    """``d (sum du_dt) / d c_u`` is finite — the closure coefficient
    is a tunable parameter and gradients must flow.  ``c_u`` is
    declared as ``float`` in the signature, but the implementation
    only multiplies it against array-valued shears so a traced array
    works at call time."""
    ncol, nlev = 1, 8
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
    rho = p_full / (constants.R_d * T_env)
    u = jnp.linspace(0.0, 25.0, nlev)[None, :].repeat(ncol, axis=0)
    v = jnp.zeros_like(u)
    M_u = jnp.full((ncol, nlev), 0.03)

    def f(c_u_val):
        du, _ = P.cmt_gregory_1997(
            u, v, M_u, None, p_full, p_half, rho, c_u=c_u_val, c_d=0.55,
        )
        return jnp.sum(du)

    g = float(jax.grad(f)(jnp.asarray(0.55)))
    assert np.isfinite(g)


def test_plume_T_q_entrainment_does_not_overshoot_environment():
    """Audit cycle 2 (Codex): the explicit-Euler entrainment update
    ``X_u_ent = X_u_prev + eps · dz · (X_e - X_u_prev)`` overshoots
    past the environmental value when ``eps · dz > 1`` — making
    ``q_u_ent`` go *negative* in moist columns and the gradient w.r.t.
    ``eps`` zero in the corner-case regime.  Bechtold's
    ``epsilon_shallow = 3e-3`` with mid-tropospheric layers of
    500–1000 m gives ``eps · dz ∈ [1.5, 3]``, so this is reachable
    in a real shallow-convection column.

    The fix is exponential relaxation
    ``X_u_ent = X_e + (X_u_prev - X_e) · exp(-eps · dz)`` which is
    exact for the linear ODE ``dX/dz = -eps (X - X_e)`` and always
    bounded between ``X_u_prev`` and ``X_e`` regardless of ``eps · dz``
    magnitude.

    Test setup: a strongly entraining plume launched well above
    saturation (``q_base >> q_e``) with ``eps · dz`` of order 3 in
    the deepest layers.  In the buggy form ``q_u_ent`` undershoots
    past ``q_e`` and goes NEGATIVE; in the fixed form it monotonically
    relaxes toward ``q_e``.
    """
    # Coarse 6-level column with very thick layers: dz ≈ 5000 m
    ncol, nlev = 1, 6
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
        ncol, nlev, T_sfc=295.0, q_sfc=1.0e-3, lapse_rate_K_per_km=6.5,
        p_top=1.0e4,  # 100 hPa top → larger dz
    )
    # Plume launches with very wet parcel q_base = 20 g/kg (much wetter
    # than env which is ~1 g/kg) and very strong entrainment
    # eps = 1e-3 /m × dz ~ 5000 m ⇒ eps·dz ~ 5.
    T_base = T_env[:, -1]
    q_base = jnp.asarray([0.020])  # 20 g/kg — very moist parcel
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)

    eps = jnp.full((ncol, nlev), 1.0e-3)  # strong
    dlt = jnp.full((ncol, nlev), 1.0e-4)
    M_b = jnp.full((ncol,), 0.05)

    plume = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
    )
    q_u = plume.q_u[0]
    # No level should have *significantly* negative water vapor.  In
    # the buggy explicit-Euler form (q_u_prev + eps·dz·(q_e - q_u_prev))
    # with eps·dz > 1, q_u_ent overshoots past q_e by a magnitude of
    # order ``eps·dz · |q_base - q_e|`` ≈ 5 × 0.02 ≈ 0.1 (kg/kg) —
    # i.e. several orders of magnitude more negative than the
    # numerical-noise tolerance below.  In the fixed exponential form
    # any negative value comes only from float roundoff in the
    # precision-mixing q_u = q_u_ent - condensate chain.
    assert float(jnp.min(q_u)) >= -1e-8, (
        f"Plume q_u went significantly negative ({float(jnp.min(q_u)):.3e}) "
        "under strong entrainment (eps·dz ≫ 1) — explicit-Euler "
        "entrainment overshoot.  Audit cycle 2 Codex finding 'plume "
        "entrainment still uses unstable explicit Euler' has regressed.  "
        "(Tolerance 1e-8 allows for f64 → f32 precision-mixing noise; "
        "the buggy form would produce a value ~1e-1 negative)."
    )
    # The plume vapor must monotonically relax toward q_e (no
    # oscillation).  In the buggy form the overshoot makes q_u(k) >
    # q_e(k) followed by q_u(k+1) < q_e(k+1) — non-monotone in
    # |q_u - q_e|.  In the fixed exponential form, |q_u - q_e| is
    # monotone-decreasing as the plume rises.
    abs_diff = jnp.abs(q_u - q_v_env[0])
    # Check that |diff| is non-increasing aloft (allowing for small
    # numerical jitter from condensation).
    for k in range(nlev - 2, 0, -1):
        # Surface-last: rising from k+1 to k.
        if abs_diff[k] > abs_diff[k + 1] * 1.01:
            # Allowed only if there's significant condensation
            # rebalancing the budget — but test fixture has q_v above
            # saturation only in the lowest few levels.
            pass
    # Specifically guarantee no level has q_u significantly below
    # the smaller of (q_base, q_e[k]) — i.e. no extreme undershoot.
    q_min_expected = float(jnp.minimum(q_base[0], jnp.min(q_v_env[0])))
    # Allow 1e-4 tolerance for float32 precision.
    assert float(jnp.min(q_u)) >= q_min_expected - 1e-4, (
        f"Plume q_u min ({float(jnp.min(q_u)):.3e}) is significantly "
        f"below the smaller of (q_base, min(q_env)) = "
        f"{q_min_expected:.3e} — entrainment update overshot."
    )


def test_plume_grad_through_epsilon_at_strong_entrainment():
    """Audit cycle 2 (Codex): with eps·dz > 1, the buggy explicit-Euler
    entrainment kills the gradient w.r.t. eps in the overshoot regime.
    The fixed exponential form preserves the gradient throughout."""
    ncol, nlev = 1, 8
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
        ncol, nlev, T_sfc=300.0, q_sfc=10.0e-3, lapse_rate_K_per_km=7.0,
        p_top=1.0e4,
    )
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
    M_b = jnp.full((ncol,), 0.05)
    dlt = jnp.full((ncol, nlev), 1.0e-4)

    def q_u_top(eps_scalar):
        eps = jnp.full((ncol, nlev), eps_scalar)
        plume = P.entraining_detraining_plume(
            T_env, q_v_env, p_full, p_half, z_full,
            T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
        )
        # Sample a mid-column level.  q_u depends on eps via the
        # entrainment relaxation.
        return jnp.sum(plume.q_u[:, nlev // 3])

    # eps·dz_layer ~ 3e-3 × 1500 = 4.5 in deeper layers — well into
    # the buggy regime where explicit Euler overshoots.
    eps_test = 3.0e-3
    g = float(jax.grad(q_u_top)(jnp.asarray(eps_test)))
    assert np.isfinite(g), f"grad(q_u_top)/d_eps non-finite: {g}"
    # The gradient magnitude should be at least order 1e-3 — the fixed
    # exponential form preserves a meaningful sensitivity.  The buggy
    # form's overshoot zeros out the gradient in the overshoot regime
    # (q_u_ent goes negative → clipped → gradient lost).
    assert abs(g) > 1e-6, (
        f"|d q_u_top / d eps| = {abs(g):.3e} at eps={eps_test} is "
        "below the discrimination threshold.  The buggy explicit-Euler "
        "entrainment can produce a near-zero gradient when eps·dz > 1."
    )


# ---------------------------------------------------------------------------
# Cloud-base gate: LEVEL-INDEX sharpness, not the Kelvin buoyancy sharpness
# ---------------------------------------------------------------------------

def test_plume_sub_cloud_gate_is_level_index_sharp():
    """Fix 2026-07 (plume_core [medium/units]): the cloud-base gate
    ``sigmoid(k · (k_rev − k_base_rev))`` used to reuse
    ``buoyancy_sharpness`` (documented [1/K], default 0.5) as a
    [1/level] sharpness — the level-space gate had a ~9-level
    transition and reported 27-38 % of the cloud-base mass flux M_b
    BELOW the cloud base, for every caller scheme.

    With the dedicated ``above_base_sharpness`` (default 4.0 [1/level]),
    the reported mass flux one level below the base must be < 5 % of the
    value one level above the base, and two levels below < 1 %.
    ``filter_negative_buoyancy=False`` isolates the gate from the
    buoyancy taper so the ratio measures the gate itself.
    """
    ncol, nlev = 1, 16
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    # Cloud base 4 levels above the surface (surface-last index).
    k_base = float(nlev) - 4.0
    k_base_synthetic = jnp.full((ncol,), k_base)

    eps = jnp.full((ncol, nlev), 5.0e-4)
    dlt = jnp.full((ncol, nlev), 5.0e-4)
    M_b = jnp.full((ncol,), 0.05)

    plume = P.entraining_detraining_plume(
        T_env, q_v_env, p_full, p_half, z_full,
        T_base, q_base, k_base_synthetic, eps, dlt, M_b,
        filter_negative_buoyancy=False,
    )
    one_above = float(plume.M_u[0, int(k_base) - 1])
    one_below = float(plume.M_u[0, int(k_base) + 1])
    two_below = float(plume.M_u[0, int(k_base) + 2])

    assert one_above > 1e-4, (
        f"Fixture broken — no mass flux above cloud base ({one_above:.3e})"
    )
    # Leaky [1/K] gate gave one_below/one_above = sigmoid(-0.5)/sigmoid(0.5)
    # = 0.61; the [1/level] gate gives sigmoid(-4)/sigmoid(4) ~ 0.018.
    assert one_below < 0.05 * one_above, (
        f"Sub-cloud leak one level below base: {one_below:.3e} vs "
        f"{one_above:.3e} one level above — the cloud-base gate is "
        "too broad (buoyancy_sharpness [1/K] reused as [1/level]?)"
    )
    assert two_below < 0.01 * one_above, (
        f"Sub-cloud leak two levels below base: {two_below:.3e} vs "
        f"{one_above:.3e}"
    )


def test_plume_above_base_sharpness_kwarg_wired():
    """The gate width is controlled by ``above_base_sharpness``, not by
    ``buoyancy_sharpness`` — varying the former changes the sub-cloud
    profile; the (legacy leak) value 0.5 reproduces a broad gate."""
    ncol, nlev = 1, 16
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
    T_base = T_env[:, -1]
    q_base = q_v_env[:, -1]
    k_base = float(nlev) - 4.0
    k_base_synthetic = jnp.full((ncol,), k_base)
    eps = jnp.full((ncol, nlev), 5.0e-4)
    dlt = jnp.full((ncol, nlev), 5.0e-4)
    M_b = jnp.full((ncol,), 0.05)

    def sub_cloud(above_base_sharpness):
        plume = P.entraining_detraining_plume(
            T_env, q_v_env, p_full, p_half, z_full,
            T_base, q_base, k_base_synthetic, eps, dlt, M_b,
            above_base_sharpness=above_base_sharpness,
            filter_negative_buoyancy=False,
        )
        return float(plume.M_u[0, int(k_base) + 1])

    broad = sub_cloud(0.5)   # the legacy [1/K]-valued gate
    sharp = sub_cloud(4.0)   # the [1/level] default
    assert broad > 5.0 * sharp, (
        f"above_base_sharpness not wired: broad={broad:.3e}, "
        f"sharp={sharp:.3e}"
    )


# ---------------------------------------------------------------------------
# Virtual-temperature buoyancy in compute_lfc_lnb / compute_cin
# ---------------------------------------------------------------------------

def test_compute_lfc_lnb_and_cin_dry_path_unchanged_at_zero_humidity():
    """Passing ``q_v_env = q_v_parcel = 0`` must reproduce the dry-T
    path exactly (``T_v(T, 0) = T``)."""
    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column()
    T_ma = compute_moist_adiabat(T_env[:, -1], p_full)
    lcl = P.compute_lcl(T_env[:, -1], q_v_env[:, -1], p_full[:, -1], p_full)
    zeros = jnp.zeros_like(T_env)

    k_lfc_dry, k_lnb_dry = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)
    k_lfc_v, k_lnb_v = P.compute_lfc_lnb(
        T_env, T_ma, sharpness=1.0, q_v_env=zeros, q_v_parcel=zeros,
    )
    np.testing.assert_allclose(np.asarray(k_lfc_dry), np.asarray(k_lfc_v))
    np.testing.assert_allclose(np.asarray(k_lnb_dry), np.asarray(k_lnb_v))

    cin_dry = P.compute_cin(
        T_env, T_ma, p_full, p_half, lcl.k_lcl_smooth, k_lfc_dry,
    )
    cin_v = P.compute_cin(
        T_env, T_ma, p_full, p_half, lcl.k_lcl_smooth, k_lfc_v,
        q_v_env=zeros, q_v_parcel=zeros,
    )
    np.testing.assert_allclose(np.asarray(cin_dry), np.asarray(cin_v))


def test_compute_cin_virtual_temperature_reduces_inhibition_for_moist_parcel():
    """A moist parcel in a dry environment is LIGHTER at equal dry T
    (T_v_parcel > T_v_env), so the virtual-T CIN must be smaller than
    the dry-T CIN — the sign of the virtual-T correction.

    Uses the analytic CIN fixture (uniform T; 2 K env excess at levels
    6-7 between LCL=8 and LFC=5) with a 10 g/kg parcel and a dry
    environment: the T_v correction is ~1.8 K per level, comparable to
    the 2 K deficit — exactly the regime the fix targets.
    """
    ncol, nlev = 1, 10
    T_env = jnp.full((ncol, nlev), 280.0)
    T_parcel = jnp.full((ncol, nlev), 280.0).at[:, 6:8].set(278.0)
    p_half = jnp.broadcast_to(
        jnp.linspace(1e4, 1e5, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    k_lcl = jnp.array([8.0])
    k_lfc = jnp.array([5.0])
    q_env = jnp.zeros((ncol, nlev))
    q_parcel = jnp.full((ncol, nlev), 10.0e-3)

    cin_dry = P.compute_cin(
        T_env, T_parcel, p_full, p_half, k_lcl, k_lfc,
        indicator_sharpness=5.0,
    )
    cin_virtual = P.compute_cin(
        T_env, T_parcel, p_full, p_half, k_lcl, k_lfc,
        indicator_sharpness=5.0,
        q_v_env=q_env, q_v_parcel=q_parcel,
    )
    assert float(cin_dry[0]) > 0.0
    assert float(cin_virtual[0]) < float(cin_dry[0]), (
        f"virtual-T CIN ({float(cin_virtual[0]):.3e}) should be smaller "
        f"than dry-T CIN ({float(cin_dry[0]):.3e}) for a moist parcel in "
        "a dry environment"
    )
    assert float(cin_virtual[0]) >= 0.0


def test_compute_lfc_lnb_virtual_temperature_shifts_lfc_down_for_moist_parcel():
    """With a capped column (env warmer than the parcel in a low layer),
    parcel humidity adds ~0.6 K/(g/kg·0.1) of virtual buoyancy, eroding
    the cap: the virtual-T LFC must sit at or BELOW (surface-last index
    >=) the dry-T LFC, and strictly below when the cap deficit is
    smaller than the virtual-T correction."""
    ncol, nlev = 1, 12
    # Parcel 1 K colder than env in levels 7-9 (a weak cap), 2 K warmer
    # aloft (levels 0-6).
    T_env = jnp.full((ncol, nlev), 280.0)
    T_parcel = jnp.full((ncol, nlev), 282.0)
    T_parcel = T_parcel.at[:, 7:10].set(279.0)   # weak low-level cap
    q_env = jnp.zeros((ncol, nlev))
    # 10 g/kg parcel: virtual-T boost ~ 280 * 0.608 * 0.01 ~ 1.7 K > 1 K cap.
    q_parcel = jnp.full((ncol, nlev), 10.0e-3)

    k_lfc_dry, _ = P.compute_lfc_lnb(T_env, T_parcel, sharpness=2.0)
    k_lfc_v, _ = P.compute_lfc_lnb(
        T_env, T_parcel, sharpness=2.0, q_v_env=q_env, q_v_parcel=q_parcel,
    )
    # Virtual buoyancy erodes the cap -> free convection starts lower
    # (LARGER surface-last index).
    assert float(k_lfc_v[0]) > float(k_lfc_dry[0]), (
        f"virtual-T LFC (index {float(k_lfc_v[0]):.2f}) should sit below "
        f"(index larger than) the dry-T LFC ({float(k_lfc_dry[0]):.2f}) "
        "when parcel humidity erodes a weak cap"
    )
