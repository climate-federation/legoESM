"""LY09 (Large & Yeager 2009) compliance tests for OMIP bulk fluxes.

Verifies the three OMIP-compliance fixes:

1. C_DN drag formula includes the LY09 high-wind quintic correction
   (−3.14807e-10·U⁶), not just the LY04 linear form.
2. Sea-surface saturation specific humidity is reduced by 0.98 for
   typical seawater salinity (~35 PSU) at the air-sea interface.
3. ``compute_most_fluxes`` accepts separate reference heights z_ref
   (wind), z_t (air temp), z_q (humidity) — JRA55-do delivers winds
   at 10 m and T,q at 2 m.

References
----------
- Large, W. G., & Yeager, S. G. (2009). The global climatology of an
  interannually varying air-sea flux data set. Climate Dynamics, 33,
  341-364. doi:10.1007/s00382-008-0441-3
- Griffies et al. (2016). OMIP contribution to CMIP6: experimental
  and diagnostic protocol for the physical component of OMIP.
  GMD, 9, 3231-3296.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.coupler.bulk_flux import compute_most_fluxes
from legoesm.coupler.config import CouplerConfig
from legoesm.coupler.coupler import _Q_SAT_SALINE_FACTOR, ocean_tile_response
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.thermo import saturation_mixing_ratio


def _ly09_C_DN_reference(U):
    """LY09 Eq. 6 in pure numpy for cross-checking."""
    return (
        2.7 / U
        + 0.142
        + U / 13.09
        - 3.14807e-10 * U ** 6
    ) * 1e-3


def _ly04_C_DN_reference(U):
    """LY04 (legacy) — linear in U, no high-wind correction."""
    return (2.7 / U + 0.142 + 0.0764 * U) * 1e-3


def _extract_C_DN_at_U(U_target, n_iter=20):
    """Drive ``compute_most_fluxes`` to near-neutral conditions at a
    chosen wind speed and back out the implied C_DN from the returned
    stress.

    Uses a tiny ΔT and Δq so the boundary layer is essentially neutral,
    then ``τ = ρ C_D U² ⇒ C_D = |τ| / (ρ U²)``.  At neutral, the LY09
    coefficient-space iteration converges to C_D == C_DN.
    """
    shape = ()
    rho = constants.rho_air
    u_rel = jnp.array(U_target)
    v_rel = jnp.array(0.0)
    T_atm = jnp.array(288.15)
    q_atm = jnp.array(0.008)
    T_sfc = jnp.array(288.20)   # tiny ΔT → near-neutral
    q_sfc = jnp.array(0.00805)  # tiny Δq

    tau_x, tau_y, _, _, _ = compute_most_fluxes(
        u_rel, v_rel, T_atm, q_atm, T_sfc, q_sfc,
        jnp.array(rho),
        z_ref=10.0,
        scheme="large_yeager",
        n_iter=n_iter,
    )
    tau_mag = float(jnp.sqrt(tau_x ** 2 + tau_y ** 2))
    return tau_mag / (rho * U_target ** 2)


# ============================================================================
# Test 1: LY09 drag formula
# ============================================================================

@pytest.mark.parametrize("U", [5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
def test_ly09_drag_matches_published_formula(U):
    """At neutral, the implementation's effective C_D must match LY09 Eq. 6.

    The LY09 paper publishes C_DN as a function of U_10N. We drive the
    iteration to near-neutral conditions and back out the C_D from the
    returned stress.
    """
    C_D_implementation = _extract_C_DN_at_U(U)
    C_D_ly09 = _ly09_C_DN_reference(U)
    rel_err = abs(C_D_implementation - C_D_ly09) / C_D_ly09
    assert rel_err < 0.05, (
        f"At U={U} m/s: implementation gives C_D={C_D_implementation:.4e}, "
        f"LY09 expects {C_D_ly09:.4e}, rel error {rel_err:.2%}"
    )


def test_ly09_high_wind_correction_active():
    """At U=30 m/s, LY09 deviates from LY04 by >5 %.

    This is the regime where the new −3.14807e-10·U⁶ term matters. If
    this test fails, the U⁶ term has been silently dropped (regression).
    """
    U = 30.0
    C_D_ly09 = _ly09_C_DN_reference(U)
    C_D_ly04 = _ly04_C_DN_reference(U)
    # LY09 should be measurably smaller than LY04 at high winds.
    assert C_D_ly09 < C_D_ly04, "LY09 must reduce C_DN vs LY04 at high winds"
    assert (C_D_ly04 - C_D_ly09) / C_D_ly04 > 0.05, (
        "U⁶ term should change C_DN by >5 % at U=30 m/s"
    )

    # And the implementation must agree with LY09, not LY04.
    C_D_impl = _extract_C_DN_at_U(U)
    err_vs_ly09 = abs(C_D_impl - C_D_ly09) / C_D_ly09
    err_vs_ly04 = abs(C_D_impl - C_D_ly04) / C_D_ly04
    assert err_vs_ly09 < err_vs_ly04, (
        f"Implementation closer to LY04 ({err_vs_ly04:.2%}) than to "
        f"LY09 ({err_vs_ly09:.2%}). The U⁶ correction is missing."
    )


# ============================================================================
# Test 2: 0.98 sea-surface saturation correction
# ============================================================================

def test_qsfc_includes_98_percent_saline_factor():
    """ocean_tile_response must reduce q_sfc by 0.98 for seawater salinity.

    This is verified indirectly by comparing the latent heat flux
    against a freshly-computed flux using saturation_mixing_ratio
    without the saline correction. The implementation's q_sfc must
    equal 0.98× the textbook value.
    """
    assert _Q_SAT_SALINE_FACTOR == 0.98, (
        "Saline factor constant must equal 0.98 per LY09 §3"
    )

    # Build a single-cell forcing
    shape = (1,)
    forcing = AtmToSurface(
        sw_down=jnp.full(shape, 200.0),
        lw_down=jnp.full(shape, 350.0),
        precip_total=jnp.zeros(shape),
        precip_snow=jnp.zeros(shape),
        T_lowest=jnp.full(shape, 295.0),
        q_lowest=jnp.full(shape, 0.012),
        u_lowest=jnp.full(shape, 8.0),
        v_lowest=jnp.zeros(shape),
        p_lowest=jnp.full(shape, 100000.0),
        p_surface=jnp.full(shape, 101325.0),
        rho_lowest=jnp.full(shape, 1.2),
        cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )
    sst = jnp.full(shape, 300.0)
    config = CouplerConfig(bulk_scheme="large_yeager")

    response = ocean_tile_response(
        forcing, sst, jnp.zeros(shape), jnp.zeros(shape), config,
    )
    q_sfc_computed = float(response.q_surface[0])

    # Reference: textbook q_sat at SST and surface pressure
    q_sat_textbook = float(saturation_mixing_ratio(
        sst[0], forcing.p_surface[0],
    ))

    # The implementation must apply the 0.98 reduction.
    expected = 0.98 * q_sat_textbook
    rel_err = abs(q_sfc_computed - expected) / expected
    assert rel_err < 1e-6, (
        f"q_sfc = {q_sfc_computed:.6e} does not match 0.98 × q_sat = "
        f"{expected:.6e} (raw q_sat = {q_sat_textbook:.6e}). "
        f"The saline reduction is missing or wrong."
    )


# ============================================================================
# Test 3: Separate reference heights z_ref / z_t / z_q
# ============================================================================

def test_separate_heights_default_is_bit_identical():
    """When z_t and z_q are omitted, results must match the legacy
    single-height path bit-identically.

    This guards against accidental behaviour change for existing
    callers (lake, idealized adapter, AMIP) that pass only z_ref.
    """
    args = dict(
        u_rel=jnp.full((4, 4), 7.0),
        v_rel=jnp.full((4, 4), 3.0),
        T_atm=jnp.full((4, 4), 285.0),
        q_atm=jnp.full((4, 4), 0.006),
        T_sfc=jnp.full((4, 4), 295.0),
        q_sfc=jnp.full((4, 4), 0.018),
        rho=jnp.full((4, 4), 1.2),
        z_ref=10.0,
        scheme="large_yeager",
        n_iter=5,
    )
    # Without explicit z_t, z_q (legacy)
    out_legacy = compute_most_fluxes(**args)
    # With z_t == z_q == z_ref (explicit single height)
    out_explicit = compute_most_fluxes(**args, z_t=10.0, z_q=10.0)

    for a, b in zip(out_legacy, out_explicit):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=0.0)


def test_separate_heights_omip_path_changes_fluxes():
    """For OMIP (z_u=10, z_t=z_q=2), fluxes must differ from the
    single-height (z_ref=10) path under non-neutral stratification.

    JRA55-do supplies winds at 10 m and T,q at 2 m. Treating them all
    as 10 m introduces a known bias; the fix is to specify the correct
    heights. This test verifies the difference is non-trivial under
    moderate stratification.
    """
    args = dict(
        u_rel=jnp.full((4, 4), 8.0),
        v_rel=jnp.zeros((4, 4)),
        T_atm=jnp.full((4, 4), 280.0),  # cold air
        q_atm=jnp.full((4, 4), 0.005),
        T_sfc=jnp.full((4, 4), 295.0),  # warm SST → unstable BL
        q_sfc=jnp.full((4, 4), 0.018),
        rho=jnp.full((4, 4), 1.2),
        z_ref=10.0,
        scheme="large_yeager",
        n_iter=5,
    )
    out_single = compute_most_fluxes(**args)
    out_omip = compute_most_fluxes(**args, z_t=2.0, z_q=2.0)

    # Sensible / latent heat must change by a measurable amount when
    # z_t, z_q drop from 10 m to 2 m under non-neutral stratification.
    # (Momentum stress also changes very slightly due to the Obukhov-
    # length feedback — physically correct, but small. We assert on
    # the heat flux which carries the dominant z_t/z_q dependence.)
    sh_single = float(jnp.mean(out_single[2]))
    sh_omip = float(jnp.mean(out_omip[2]))
    rel_diff = abs(sh_omip - sh_single) / abs(sh_single)
    assert rel_diff > 0.02, (
        f"Sensible heat fluxes are essentially unchanged "
        f"({sh_single:.4f} vs {sh_omip:.4f}, rel diff {rel_diff:.4%}). "
        "The z_t/z_q parameters are likely being ignored."
    )


def test_omip_couplerconfig_propagates_heights():
    """Setting CouplerConfig.z_t_atm/z_q_atm must change ocean fluxes."""
    shape = (1,)
    forcing = AtmToSurface(
        sw_down=jnp.full(shape, 200.0),
        lw_down=jnp.full(shape, 350.0),
        precip_total=jnp.zeros(shape),
        precip_snow=jnp.zeros(shape),
        T_lowest=jnp.full(shape, 280.0),  # cold air
        q_lowest=jnp.full(shape, 0.005),
        u_lowest=jnp.full(shape, 8.0),
        v_lowest=jnp.zeros(shape),
        p_lowest=jnp.full(shape, 100000.0),
        p_surface=jnp.full(shape, 101325.0),
        rho_lowest=jnp.full(shape, 1.2),
        cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )
    sst = jnp.full(shape, 295.0)  # warm SST → unstable

    cfg_legacy = CouplerConfig(bulk_scheme="large_yeager")  # 10/10/10
    cfg_omip = CouplerConfig(
        bulk_scheme="large_yeager", z_t_atm=2.0, z_q_atm=2.0,
    )

    r_legacy = ocean_tile_response(
        forcing, sst, jnp.zeros(shape), jnp.zeros(shape), cfg_legacy,
    )
    r_omip = ocean_tile_response(
        forcing, sst, jnp.zeros(shape), jnp.zeros(shape), cfg_omip,
    )

    # SH flux should differ between configurations under non-neutral BL.
    rel_diff = abs(float(r_omip.shflx[0] - r_legacy.shflx[0])) / abs(float(r_legacy.shflx[0]))
    assert rel_diff > 0.02, (
        f"CouplerConfig.z_t_atm/z_q_atm changes are not flowing through "
        f"to ocean_tile_response (rel diff {rel_diff:.4%})."
    )


# ============================================================================
# AD compatibility — make sure new code path is still differentiable
# ============================================================================

def test_compute_most_fluxes_omip_path_differentiable():
    """jax.grad must produce finite gradients through the OMIP path."""
    def loss(sst):
        u = jnp.full_like(sst, 8.0)
        v = jnp.zeros_like(sst)
        T_a = jnp.full_like(sst, 280.0)
        q_a = jnp.full_like(sst, 0.005)
        q_s = 0.98 * saturation_mixing_ratio(sst, jnp.full_like(sst, 101325.0))
        rho = jnp.full_like(sst, 1.2)
        tau_x, tau_y, sh, lh, _ = compute_most_fluxes(
            u, v, T_a, q_a, sst, q_s, rho,
            z_ref=10.0, z_t=2.0, z_q=2.0,
            scheme="large_yeager", n_iter=5,
        )
        return jnp.mean(sh ** 2 + lh ** 2 + tau_x ** 2)

    sst = jnp.full((4, 4), 295.0)
    grads = jax.grad(loss)(sst)
    assert bool(jnp.all(jnp.isfinite(grads))), "Non-finite gradients"
    assert not bool(jnp.allclose(grads, 0.0)), "Zero gradients (broken AD)"
