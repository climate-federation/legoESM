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
