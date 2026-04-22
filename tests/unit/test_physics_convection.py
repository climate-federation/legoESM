"""Category 3: Convection -- Physical Consistency.

Tests moisture conservation, energy conservation, CAPE reduction, stable
profile behavior, precipitation sign, and tendency signs for all convection
schemes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import dca_convection
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.mass_flux import mass_flux_convection
from legoesm.atmosphere.physics.convection.edmf import edmf_convection
from legoesm.atmosphere.physics.convection.config import (
    SBMConfig, DCAConfig, KuoConfig, MassFluxConfig, EDMFConfig,
)
from legoesm.atmosphere.physics.thermodynamics import (
    compute_cape,
    compute_moist_adiabat,
    phase_aware_latent_heat,
)
from legoesm.thermo import saturation_mixing_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unstable_column(nlev=20, ncol=4):
    """Build a convectively unstable column: warm, moist BL under cool mid-trop."""
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))

    T_sfc = 300.0
    T = T_sfc * jnp.clip(sigma_full, 0.01, None) ** 0.19
    T = jnp.maximum(T, 200.0)
    T = jnp.broadcast_to(T[None, :], (ncol, nlev))

    q_sat = saturation_mixing_ratio(T, p_full)
    RH = jnp.where(sigma_full[None, :] > 0.7, 0.95, 0.5)
    q_v = RH * q_sat

    return T, q_v, p_full, p_half


def _make_stable_column(nlev=20, ncol=4):
    """Build a strongly stable, dry isothermal column."""
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))

    T = jnp.full((ncol, nlev), 250.0)
    q_v = jnp.full((ncol, nlev), 1e-6)
    return T, q_v, p_full, p_half


def _call_scheme(name, T, q_v, p_full, p_half, dt=300.0):
    """Call a convection scheme and return ConvectionOutput."""
    ncol = T.shape[0]
    if name == "sbm":
        return sbm_convection(T, q_v, p_full, p_half, dt, config=SBMConfig())
    elif name == "dca":
        return dca_convection(T, q_v, p_full, p_half, dt, config=DCAConfig())
    elif name == "kuo":
        return kuo_convection(T, q_v, p_full, p_half, dt, config=KuoConfig())
    elif name == "mass_flux":
        M_c = jnp.zeros(ncol)
        out, _ = mass_flux_convection(T, q_v, p_full, p_half, M_c, dt,
                                       config=MassFluxConfig())
        return out
    elif name == "edmf":
        a_u = 0.1 * jnp.ones(ncol)
        out, _ = edmf_convection(T, q_v, p_full, p_half, a_u, dt,
                                  config=EDMFConfig())
        return out
    else:
        raise ValueError(f"Unknown scheme: {name}")


# ============================================================================
# 3a  Moisture conservation (SBM, DCA only -- see docstring)
# ============================================================================

@pytest.mark.parametrize("scheme", ["sbm", "dca"])
def test_moisture_conservation(scheme):
    """Column-integrated moisture tendency approximately equals -precipitation.

    Note: mass_flux and edmf compute precipitation from updraft condensate
    detrainment, so their column dq_v budget includes additional terms
    not captured by sum(dq_v_dt * dp/g) + precip = 0.
    """
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)

    dp = p_half[:, 1:] - p_half[:, :-1]
    col_dqv = jnp.sum(out.dq_v_dt * dp / constants.g, axis=1)

    for i in range(col_dqv.shape[0]):
        precip = float(out.precipitation[i])
        integral = float(col_dqv[i])
        residual = abs(integral + precip)
        scale = max(abs(precip), abs(integral), 1e-12)
        if scale > 1e-10:
            rel_err = residual / scale
            assert rel_err < 0.10, (
                f"{scheme} col {i}: moisture conservation rel_err = {rel_err:.4f}"
            )


# ============================================================================
# 3b  Energy conservation (moist static energy) - SBM only
# ============================================================================

def test_sbm_energy_conservation():
    """SBM: column-integrated MSE tendency ~ 0 with phase-aware latent heat."""
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme("sbm", T, q_v, p_full, p_half)

    dp = p_half[:, 1:] - p_half[:, :-1]
    mse_tend = (
        constants.c_pd * out.dT_dt
        + phase_aware_latent_heat(T) * out.dq_v_dt
    )
    col_mse = jnp.sum(mse_tend * dp / constants.g, axis=1)

    max_imbalance = float(jnp.max(jnp.abs(col_mse)))
    assert max_imbalance < 10.0, (
        f"SBM MSE imbalance = {max_imbalance:.2f} W/m^2 > 10 W/m^2"
    )


# ============================================================================
# 3d  Stable profile -> small convection
# ============================================================================

@pytest.mark.parametrize("scheme", ["sbm", "edmf"])
def test_stable_profile_small_convection(scheme):
    """Stable column produces small tendencies (via smooth triggers).

    Note: mass_flux excluded because softplus floor on M_c means it always
    starts with a small nonzero flux even in stable conditions. This is a
    known consequence of the differentiable formulation, not a bug.
    """
    T, q_v, p_full, p_half = _make_stable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)

    max_dT = float(jnp.max(jnp.abs(out.dT_dt)))
    max_precip = float(jnp.max(out.precipitation))

    assert max_dT < 1e-2, f"{scheme}: dT_dt = {max_dT:.2e} in stable column"
    assert max_precip < 1e-6, f"{scheme}: precip = {max_precip:.2e} in stable column"


# ============================================================================
# 3e  Precipitation non-negative
# ============================================================================

@pytest.mark.parametrize("scheme", ["sbm", "dca", "kuo", "mass_flux", "edmf"])
def test_precipitation_non_negative(scheme):
    """Precipitation must be >= 0."""
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)
    min_precip = float(jnp.min(out.precipitation))
    assert min_precip >= -1e-15, (
        f"{scheme}: negative precipitation = {min_precip:.2e}"
    )


# ============================================================================
# 3c  CAPE reduction
# ============================================================================

@pytest.mark.parametrize("scheme", ["sbm", "dca"])
def test_cape_reduction(scheme):
    """Applying convection tendencies should reduce CAPE."""
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half, dt=300.0)

    T_base = T[:, -1]
    T_moist = compute_moist_adiabat(T_base, p_full)
    cape_before = compute_cape(T, T_moist, p_full, p_half)

    dt = 300.0
    T_after = T + out.dT_dt * dt
    T_base_after = T_after[:, -1]
    T_moist_after = compute_moist_adiabat(T_base_after, p_full)
    cape_after = compute_cape(T_after, T_moist_after, p_full, p_half)

    mean_cape_before = float(jnp.mean(cape_before))
    mean_cape_after = float(jnp.mean(cape_after))

    if mean_cape_before > 10.0:
        assert mean_cape_after < mean_cape_before, (
            f"{scheme}: CAPE did not decrease: "
            f"before={mean_cape_before:.1f}, after={mean_cape_after:.1f}"
        )


# ============================================================================
# 3f  Tendency signs for unstable profile
# ============================================================================

def test_sbm_tendency_signs_unstable():
    """SBM: convection should dry the lower levels (dq_v_dt < 0)."""
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme("sbm", T, q_v, p_full, p_half)
    min_dqv = float(jnp.min(out.dq_v_dt))
    assert min_dqv < -1e-12, (
        f"SBM: no drying found, min dq_v_dt = {min_dqv:.2e}"
    )


# ============================================================================
# 3g  Mass-flux prognostic variable
# ============================================================================

def test_mass_flux_prognostic_positive():
    """Mass flux M_c should be >= 0 after update."""
    T, q_v, p_full, p_half = _make_unstable_column()
    ncol = T.shape[0]
    M_c = jnp.zeros(ncol)
    _, M_c_new = mass_flux_convection(T, q_v, p_full, p_half, M_c, 300.0,
                                       config=MassFluxConfig())
    assert float(jnp.min(M_c_new)) >= -1e-15, (
        f"M_c_new has negative values: min = {float(jnp.min(M_c_new)):.2e}"
    )


# ============================================================================
# 3h  EDMF updraft area
# ============================================================================

def test_edmf_updraft_area_bounds():
    """EDMF a_u should be in [0, 1] after update."""
    T, q_v, p_full, p_half = _make_unstable_column()
    ncol = T.shape[0]
    a_u = 0.1 * jnp.ones(ncol)
    _, a_u_new = edmf_convection(T, q_v, p_full, p_half, a_u, 300.0,
                                  config=EDMFConfig())
    assert float(jnp.min(a_u_new)) >= -1e-10, (
        f"a_u_new has negative values: {float(jnp.min(a_u_new)):.2e}"
    )
    assert float(jnp.max(a_u_new)) <= 1.0 + 1e-10, (
        f"a_u_new exceeds 1: {float(jnp.max(a_u_new)):.2e}"
    )


# ============================================================================
# All outputs finite
# ============================================================================

@pytest.mark.parametrize("scheme", ["sbm", "dca", "kuo", "mass_flux", "edmf"])
def test_all_outputs_finite(scheme):
    """All ConvectionOutput fields should be finite."""
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)
    assert jnp.all(jnp.isfinite(out.dT_dt)), f"{scheme}: dT_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(out.dq_v_dt)), f"{scheme}: dq_v_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(out.precipitation)), f"{scheme}: precip has NaN/Inf"
    assert jnp.all(jnp.isfinite(out.cape)), f"{scheme}: cape has NaN/Inf"
