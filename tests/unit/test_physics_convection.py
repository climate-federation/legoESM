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
from legoesm.atmosphere.physics.convection.mass_flux import (
    edmf_convection,
    mass_flux_convection,
    stratosphere_mass_flux_gate,
)
from legoesm.atmosphere.physics.convection.config import (
    SBMConfig, DCAConfig, KuoConfig, MassFluxConfig, EDMFConfig,
)
from legoesm.atmosphere.physics.thermodynamics import compute_cape, compute_moist_adiabat
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
    """Column water budget under the post-Option-C semantics.

    Convection emits a 3D ``dq_c_conv_dt`` (rate of cloud-water creation
    at each level) instead of a scalar surface ``precipitation``;
    microphysics owns the surface-flux diagnostic. The intended
    column-integrated balance for the SBM and DCA relaxation schemes
    is:

        ∫ dq_v_dt dp/g  +  ∫ dq_c_conv_dt dp/g  ≈  0

    Both schemes rescale their per-level condensation candidate so
    the column integral exactly equals the column-net drying
    (= the legacy ``precipitation`` formula). This guarantees
    machine-precision column water conservation regardless of the
    sign pattern of ``dq_v_dt`` — without per-level negative cloud
    water source. (A naive per-level ``max(-dq_v_dt, 0)`` would
    create water column-wide whenever drying and moistening layers
    coexist; the rescaling removes that bug.)

    ``kuo`` is intentionally excluded: by design Kuo's column water
    is non-conservative (it imports a ``(1 - alpha_heat) * MC /
    tau_relax`` moistening source from outside the column —
    surface evaporation / large-scale moisture convergence). Under
    Option C, Kuo exposes the full design-intent condensation rate
    to microphysics rather than the underreporting legacy
    formula; see ``kuo.py`` module docstring for the water budget.

    ``mass_flux`` and ``edmf`` are also excluded: their kernels
    include vertical subsidence/detrainment transport so the column
    ``dq_v_dt`` is not the negative of the column condensation by
    design — water conservation is not the right invariant to test
    at the column level for those schemes either.
    """
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)

    dp = p_half[:, 1:] - p_half[:, :-1]
    col_dqv = jnp.sum(out.dq_v_dt * dp / constants.g, axis=1)
    col_dqc = jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1)

    # With column-scaling, conservation is exact in floating point.
    # 1e-6 leaves margin for x64-vs-x32 mode switches without admitting
    # any of the bug patterns this test is meant to catch.
    bound = 1e-6
    for i in range(col_dqv.shape[0]):
        condensation = float(col_dqc[i])
        vapor_loss = float(col_dqv[i])
        residual = abs(vapor_loss + condensation)
        scale = max(abs(condensation), abs(vapor_loss), 1e-12)
        if scale > 1e-10:
            rel_err = residual / scale
            assert rel_err < bound, (
                f"{scheme} col {i}: moisture conservation rel_err = "
                f"{rel_err:.6e} (bound {bound:.0e})"
            )


# ============================================================================
# 3b  Energy conservation (moist static energy) - SBM only
# ============================================================================

def test_sbm_energy_conservation():
    """SBM: column-integrated MSE tendency ~ 0 (cp*dT + Lv*dq_v)."""
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme("sbm", T, q_v, p_full, p_half)

    dp = p_half[:, 1:] - p_half[:, :-1]
    mse_tend = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
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
    # Convective cloud-water source is per-level [kg/kg/s].  Threshold
    # 1e-7 kg/kg/s × 1000 s = 1e-4 kg/kg of cloud water — physically
    # negligible.  The earlier 1e-9 threshold was tight against the
    # (broken) mass-flux kernel that returned ``dq_c_conv_dt ≈ 0`` for
    # entraining-diluted plumes; the corrected kernel yields a small
    # residual ~6e-9 even in stable columns from the always-positive
    # ``dilution × (q_sat_base − q_sat(T_moist))`` term, which is real.
    max_dq_c = float(jnp.max(out.dq_c_conv_dt))

    assert max_dT < 1e-2, f"{scheme}: dT_dt = {max_dT:.2e} in stable column"
    assert max_dq_c < 1e-7, (
        f"{scheme}: dq_c_conv_dt = {max_dq_c:.2e} kg/kg/s in stable column"
    )


# ============================================================================
# 3e  Precipitation non-negative
# ============================================================================

@pytest.mark.parametrize("scheme", ["sbm", "dca", "kuo", "mass_flux", "edmf"])
def test_precipitation_non_negative(scheme):
    """Convective cloud-water source must be >= 0 at every level."""
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)
    min_dq_c = float(jnp.min(out.dq_c_conv_dt))
    assert min_dq_c >= -1e-15, (
        f"{scheme}: negative dq_c_conv_dt = {min_dq_c:.2e}"
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

def test_stratosphere_mass_flux_gate_actually_closes():
    """The gate must vanish (not merely attenuate) in the deep stratosphere.

    Codex caught a regression where the default sharpness equalled the
    cutoff, leaving the gate at sigmoid(-1) ≈ 0.27 at the model top —
    only halving M_u rather than zeroing it.  This test pins the
    behaviour at canonical pressure levels so a future tuning that
    relaxes the cutoff cannot silently weaken the protection.
    """
    p_full = jnp.array([3_470.0, 5_000.0, 8_430.0, 10_000.0,
                        13_370.0, 20_000.0, 50_000.0, 100_000.0])
    gate = stratosphere_mass_flux_gate(p_full)
    # Deep stratosphere: gate must be < 5% (not just < 50%)
    assert float(gate[0]) < 0.05, (
        f"gate at model top (p=3470 Pa) = {float(gate[0]):.3f}; "
        "must be < 0.05 to actually close the convection path"
    )
    assert float(gate[1]) < 0.10, (
        f"gate at 50 hPa = {float(gate[1]):.3f}; must be < 0.10"
    )
    # Tropopause: half-open
    assert 0.4 < float(gate[3]) < 0.6, (
        f"gate at 100 hPa (tropopause) = {float(gate[3]):.3f}; "
        "must be ~0.5 (transition midpoint)"
    )
    # Upper troposphere: nearly fully open
    assert float(gate[4]) > 0.80, (
        f"gate at 130 hPa = {float(gate[4]):.3f}; must be > 0.80 "
        "to leave deep tropical convection unaffected"
    )
    # Lower troposphere: fully open (sigmoid saturates to 1.0 in
    # float32 well before 500 hPa, so use >= 0.999 not strict >)
    assert float(gate[6]) >= 0.999, (
        f"gate at 500 hPa = {float(gate[6]):.6f}; must be ≈ 1.0"
    )
    # Monotonic in pressure (non-decreasing — sigmoid saturates
    # exactly to 1.0 in float32 above ~30 kPa so strict > would
    # fail at the saturated tail).
    assert jnp.all(jnp.diff(gate) >= 0), "gate must be monotonic in pressure"
    # Strict monotonic in the transition region (below saturation).
    transition = gate[:5]
    assert jnp.all(jnp.diff(transition) > 0), (
        "gate must be strictly monotonic across the transition region "
        f"(35–134 hPa); got {[float(v) for v in transition]}"
    )


def test_cmt_gregory_1997_applies_stratospheric_gate():
    """``cmt_gregory_1997`` must apply the same stratospheric gate the
    kernel uses, so the default-enabled CMT path in ZM/Tiedtke/Bechtold
    cannot dump convective momentum into the model top.

    Codex caught that the kernel gate covered T/q_v but the CMT call
    used the ungated ``plume.M_u``, leaving wind-driven dycore
    instabilities (e.g. KF blowup at day 10) on the table.

    Contract: with uniform ``M_u`` the function's output must equal
    the same formula evaluated at ``M_u * gate(p_full)`` — that is the
    operational definition of "the gate is applied inside the function".
    """
    from legoesm.atmosphere.physics.convection._plume import cmt_gregory_1997

    ncol, nlev = 1, 8
    p_full = jnp.array([[3_470.0, 5_000.0, 8_430.0, 10_000.0,
                         13_370.0, 20_000.0, 50_000.0, 100_000.0]])
    p_half = jnp.concatenate([
        jnp.array([[0.0]]),
        0.5 * (p_full[:, :-1] + p_full[:, 1:]),
        jnp.array([[101_300.0]]),
    ], axis=-1)
    # Non-uniform shear so the ungated calculation has a non-zero
    # divergence at every level (uniform du_layer + uniform M_u ⇒
    # uniform flux ⇒ zero divergence in the interior).
    u_env = jnp.array([[60.0, 50.0, 40.0, 25.0, 12.0, 5.0, 2.0, 0.0]])
    v_env = jnp.zeros((ncol, nlev))
    M_u = jnp.full((ncol, nlev), 0.05)
    rho = p_full / (287.0 * 250.0)
    c_u = 0.55

    du_dt_func, _ = cmt_gregory_1997(
        u_env, v_env, M_u, None, p_full, p_half, rho, c_u=c_u, c_d=c_u,
    )

    # Reference: replicate the formula exactly with the gate applied
    # to M_u once.  If the function gates internally, this matches.
    gate = stratosphere_mass_flux_gate(p_full)
    dp = p_half[:, 1:] - p_half[:, :-1]
    du_layer = jnp.diff(u_env, axis=-1, prepend=u_env[:, :1])
    flux_ref = -c_u * (M_u * gate) * du_layer
    dflux_ref = jnp.diff(flux_ref, axis=-1, append=flux_ref[:, -1:])
    g = 9.80616
    du_dt_ref = -g * dflux_ref / dp

    # The function output must equal the gated reference exactly.
    rel_err = float(jnp.max(jnp.abs(du_dt_func - du_dt_ref)
                            / (jnp.abs(du_dt_ref) + 1e-30)))
    assert rel_err < 1e-5, (
        f"cmt_gregory_1997 output disagrees with gated reference "
        f"by {rel_err:.3e}; the function must apply "
        "stratosphere_mass_flux_gate to M_u (and M_d when present)"
    )

    # Sanity contrast with the *ungated* calculation — at the model
    # top the function output must be much smaller than the ungated
    # value (gate ≈ 0.013 at 35 hPa).
    flux_ungated = -c_u * M_u * du_layer
    dflux_ungated = jnp.diff(
        flux_ungated, axis=-1, append=flux_ungated[:, -1:],
    )
    du_dt_ungated = -g * dflux_ungated / dp
    ratio_top = float(jnp.abs(du_dt_func[0, 0])
                      / (jnp.abs(du_dt_ungated[0, 0]) + 1e-30))
    assert ratio_top < 0.10, (
        f"At p=3470 Pa, gated CMT |du/dt| / ungated |du/dt| = "
        f"{ratio_top:.4f}; expected < 0.10 (gate factor at TOA ≈ 0.013)"
    )


@pytest.mark.parametrize("scheme", ["sbm", "dca", "kuo", "mass_flux", "edmf"])
def test_all_outputs_finite(scheme):
    """All ConvectionOutput fields should be finite."""
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)
    assert jnp.all(jnp.isfinite(out.dT_dt)), f"{scheme}: dT_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(out.dq_v_dt)), f"{scheme}: dq_v_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(out.dq_c_conv_dt)), (
        f"{scheme}: dq_c_conv_dt has NaN/Inf"
    )
    assert jnp.all(jnp.isfinite(out.cape)), f"{scheme}: cape has NaN/Inf"
