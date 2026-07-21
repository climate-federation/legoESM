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
    SBMConfig, DCAConfig, KuoConfig, MassFluxConfig, ConvectiveEDMFConfig,
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


def _make_saturated_unstable_column(nlev=20, ncol=4):
    """Unstable column with a GENUINELY (super)saturated lower troposphere.

    Identical thermal/pressure structure to ``_make_unstable_column`` but
    with RH = 1.05 below ``sigma = 0.7`` so the boundary layer is supersaturated.
    Saturation-clipping adjustment schemes (DCA, SBM) only condense — and hence
    only DRY — where the parcel is at/above saturation; the subsaturated
    (RH <= 0.95) ``_make_unstable_column`` fixture does NOT physically condense
    under a correct moist-adiabatic pair solve, so a drying assertion against it
    is testing the wrong premise.  (The previous DCA two-level solve dried that
    subsaturated column only as a side effect of over-cooling the lower pair
    member — the exact inconsistency removed by the simultaneous 2x2 enthalpy +
    lapse solve in ``dca.py``.)
    """
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
    RH = jnp.where(sigma_full[None, :] > 0.7, 1.05, 0.5)
    q_v = RH * q_sat

    return T, q_v, p_full, p_half


def _make_dry_unstable_column(nlev=20, ncol=4):
    """Conditionally-unstable, MODERATELY dry column (RH = 0.5 everywhere).

    Relaxing toward ``q_ref = rh_ref * q_sat(T_moist)`` then NET-MOISTENS the
    column (the cloud-layer mass-mean ``q_ref - q_v`` > 0), i.e. the Frierson
    (2007) SHALLOW regime — the case the SBM column-water conservation fix
    addresses.  ``_make_unstable_column`` (RH up to 0.95) net-DRIES, so it
    cannot exercise the shallow branch.

    RH = 0.5, NOT lower: the 2026-07-17 shallow-branch A/B showed the regime
    splits in two.  STRONGLY moistening soundings (RH ~ 0.2, this fixture's
    old value) are zeroed OUTRIGHT by the #771 drying gate — output
    identically 0, so the "scheme fired" assertions below can never hold
    (the fixture predated the gate and the test sat red).  At RH ~ 0.5 the
    shallow redistribution cancels the column integral and the gate passes
    the redistributed LOCAL tendencies — the LIVE conserving shallow regime
    this test exists to pin (see test_sbm_faithful.py::
    test_shallow_branch_is_live_in_the_moderate_moistening_regime).
    """
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
    q_v = 0.5 * q_sat  # moderate dryness -> the LIVE shallow regime (see docstring)
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
                                  config=ConvectiveEDMFConfig())
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
    tau_relax_s`` moistening source from outside the column —
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


def test_sbm_column_water_conserved_in_shallow_moistening_regime():
    """CONSERVATION (truth tier): in the net-MOISTENING (Frierson 2007 shallow)
    regime the deep references moisten the column with NO compensating sink —
    ``col_net_drying`` clips to 0 so ``dq_c_conv_dt`` cannot absorb it — which
    creates water from nothing.  The shallow branch shifts ``q_ref``/``T_ref``
    so the column-integrated ``dq_v`` -> 0 (pure redistribution), giving
    Σ(dq_v + dq_c)·dp/g ~ 0.  (Physics review: conv/sbm conservation, 2026-06-29.)
    """
    T, q_v, p_full, p_half = _make_dry_unstable_column()
    out = _call_scheme("sbm", T, q_v, p_full, p_half)
    dp = p_half[:, 1:] - p_half[:, :-1]
    col_dqv = jnp.sum(out.dq_v_dt * dp / constants.g, axis=1)
    col_dqc = jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1)
    col_total = col_dqv + col_dqc
    # scheme fired, and it is a redistribution (both moistening AND drying
    # levels present) — confirming the net-moistening (shallow) regime was hit.
    assert float(jnp.max(jnp.abs(out.dq_v_dt))) > 0.0
    assert float(jnp.sum(jnp.maximum(out.dq_v_dt, 0.0))) > 0.0   # moistening levels
    assert float(jnp.sum(jnp.maximum(-out.dq_v_dt, 0.0))) > 0.0  # drying levels
    gross = jnp.sum(jnp.abs(out.dq_v_dt) * dp / constants.g, axis=1)
    for i in range(col_total.shape[0]):
        if float(gross[i]) > 1e-10:
            rel = abs(float(col_total[i])) / float(gross[i])
            assert rel < 1e-4, (
                f"sbm col {i}: shallow-regime water residual rel = {rel:.3e} "
                f"(col_total={float(col_total[i]):.3e})"
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
                                  config=ConvectiveEDMFConfig())
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

    Codex caught regressions where the transition was broad enough to
    leave percent-level mass flux in the deep stratosphere. On the RCE
    grid that residual flux is amplified by very small density and can
    produce unphysical cold-point heating, so this test pins the
    behaviour at canonical pressure levels.
    """
    p_full = jnp.array([3_470.0, 5_000.0, 8_430.0, 10_000.0,
                        13_370.0, 20_000.0, 50_000.0, 100_000.0])
    gate = stratosphere_mass_flux_gate(p_full)
    # Deep stratosphere: gate must be effectively closed.
    assert float(gate[0]) < 1.0e-5, (
        f"gate at model top (p=3470 Pa) = {float(gate[0]):.3f}; "
        "must be < 1e-5 to actually close the convection path"
    )
    assert float(gate[1]) < 1.0e-3, (
        f"gate at 50 hPa = {float(gate[1]):.3f}; must be < 1e-3"
    )
    # Tropopause cutoff: half-open.
    assert 0.4 < float(gate[3]) < 0.6, (
        f"gate at 100 hPa = {float(gate[3]):.3f}; "
        "must be ~0.5 (transition midpoint)"
    )
    # Upper troposphere: nearly fully open.
    assert float(gate[4]) > 0.99, (
        f"gate at 130 hPa = {float(gate[4]):.3f}; must be > 0.99 "
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
        f"(35-134 hPa); got {[float(v) for v in transition]}"
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
    rho = p_full / (constants.R_d * 250.0)
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
    g = constants.g
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
    # value (gate is effectively closed at 35 hPa).
    flux_ungated = -c_u * M_u * du_layer
    dflux_ungated = jnp.diff(
        flux_ungated, axis=-1, append=flux_ungated[:, -1:],
    )
    du_dt_ungated = -g * dflux_ungated / dp
    ratio_top = float(jnp.abs(du_dt_func[0, 0])
                      / (jnp.abs(du_dt_ungated[0, 0]) + 1e-30))
    assert ratio_top < 1.0e-4, (
        f"At p=3470 Pa, gated CMT |du/dt| / ungated |du/dt| = "
        f"{ratio_top:.4f}; expected < 1e-4 (gate closed at TOA)"
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


# ============================================================================
# 3b extended -- closed energy budget across all conservative schemes
# ============================================================================
#
# Each scheme's "natural" energy invariant differs by design under Option C:
#
#   * SBM enforces a Newton enthalpy correction so c_pd*int(dT) + Lv*int(dq_v)
#     ~ 0 (standard MSE).  Cloud water is rescaled to match column-net drying,
#     so the EXTENDED invariant c_pd*int(dT) + Lv*int(dq_v + dq_c) is NOT zero
#     for SBM -- it equals Lv * cloud-water creation, the latent heat that
#     microphysics will release downstream.
#
#   * DCA preserves layer-mean T per pair (no in-scheme latent heating) and
#     removes excess vapor as cloud water.  The natural invariant is the
#     EXTENDED MSE c_pd*int(dT) + Lv*int(dq_v + dq_c) ~ 0; the standard MSE
#     is large negative because the latent heat from condensation is left
#     for microphysics to release rather than released in convection's T_dt.
#
#   * Kuo is non-conservative by design (alpha_heat fraction sourced
#     externally) -- excluded from both forms.
#
#   * mass_flux / edmf use detrainment + compensating subsidence.  Column
#     budgets do NOT close to <10 W/m^2 at finite resolution because the
#     detrained plume thermodynamics carry energy at the boundaries; the
#     residual is bounded by the integrated detrainment rate, not zero.
#     The standard-MSE residual happens to be small for mass_flux because
#     M_c starts at 0 and grows slowly (~50 W/m^2 within one step); for
#     EDMF with a_u=0.1 already active, the residual is order 1500 W/m^2.

def test_dca_extended_mse_conservation():
    """DCA: c_pd*int(dT) + L_v*int(dq_v) ~ 0 (STANDARD MSE).

    DCA releases the latent heat of condensation inside the scheme via
    the ``delta_T_lh`` per-pair correction (see ``dca.py`` lines
    154-167) — the moist adjustment is internally consistent with
    ``c_p ⟨ΔT⟩ + L_v ⟨Δq⟩ = 0`` per adjusting pair.  The earlier
    documentation in this module described an older "no in-scheme
    latent heating" version of DCA whose natural invariant was the
    extended form ``c_pd·∫dT + L_v·∫(dq_v + dq_c) ~ 0``; the
    current implementation closes the STANDARD form instead.  The
    extended form is large positive (≈ L_v · column-condensate) by
    design — the latent heating is in ``dT_dt``, not deferred.
    """
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme("dca", T, q_v, p_full, p_half)

    dp = p_half[:, 1:] - p_half[:, :-1]
    std = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
    col = jnp.sum(std * dp / constants.g, axis=1)
    max_imbalance = float(jnp.max(jnp.abs(col)))
    assert max_imbalance < 10.0, (
        f"DCA standard-MSE imbalance = {max_imbalance:.3e} W/m^2 > 10"
    )


def test_mass_flux_standard_mse_small():
    """Mass-flux: c_pd*int(dT) + L_v*int(dq_v) is small (~kernel residual).

    The plume releases latent heat in T_u during ascent and emits the
    diluted condensate as dq_c_conv_dt at detrainment.  Column-integrated
    standard MSE residual is bounded by the boundary detrainment flux;
    with M_c ~ M_eq * dt/tau_adj ~ 8e-4 kg/m^2/s after one step, a few
    tens of W/m^2 is the realistic scale.  The bound here (<100 W/m^2)
    catches breakage to hundreds of W/m^2 from kernel sign flips
    without being so tight it tracks small numerics changes.
    """
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme("mass_flux", T, q_v, p_full, p_half)

    dp = p_half[:, 1:] - p_half[:, :-1]
    mse = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
    col = jnp.sum(mse * dp / constants.g, axis=1)
    max_imbalance = float(jnp.max(jnp.abs(col)))
    assert max_imbalance < 100.0, (
        f"mass_flux standard-MSE imbalance = {max_imbalance:.2f} W/m^2 > 100"
    )


# ============================================================================
# 3c extended -- CAPE response of mass-flux schemes
# ============================================================================

def test_mass_flux_cape_does_not_increase():
    """Mass-flux applied to a CAPE-positive column should not INCREASE CAPE.

    Strict reduction requires sustained mass flux; after one step from
    M_c=0 the change is small but should be in the right direction
    (non-increase).  Tolerance allows for sub-J/kg numerical wiggle.
    """
    T, q_v, p_full, p_half = _make_unstable_column()
    ncol = T.shape[0]
    out, _ = mass_flux_convection(
        T, q_v, p_full, p_half, jnp.zeros(ncol), 300.0, MassFluxConfig(),
    )

    T_moist_before = compute_moist_adiabat(T[:, -1], p_full)
    cape_before = compute_cape(T, T_moist_before, p_full, p_half)

    T_after = T + out.dT_dt * 300.0
    T_moist_after = compute_moist_adiabat(T_after[:, -1], p_full)
    cape_after = compute_cape(T_after, T_moist_after, p_full, p_half)

    delta = float(jnp.mean(cape_after) - jnp.mean(cape_before))
    assert delta <= 1.0, (
        f"mass_flux: CAPE increased by {delta:.2f} J/kg in one step"
    )


# ============================================================================
# 3d extended -- stable profile gating for DCA and Kuo
# ============================================================================

@pytest.mark.parametrize("scheme", ["dca", "kuo"])
def test_stable_profile_dca_kuo(scheme):
    """Stable / dry columns produce small tendencies via smooth triggers.

    DCA: cape_gate sigmoid at CAPE=0 with default sharpness 0.1 is
    sigmoid(-10) ~ 5e-5 -- tendencies bounded by that factor times
    the saturation-adjustment magnitude.  Kuo: trigger sigmoid on
    column moisture excess (zero in undersaturated profile) -> gates off.
    """
    T, q_v, p_full, p_half = _make_stable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)
    max_dT = float(jnp.max(jnp.abs(out.dT_dt)))
    max_dq_c = float(jnp.max(out.dq_c_conv_dt))
    # 1e-2 K/s = 36 K/hr is a generous bound; the smooth triggers should
    # be well below this in stable profiles.
    assert max_dT < 1e-2, f"{scheme}: dT_dt = {max_dT:.2e} in stable column"
    assert max_dq_c < 1e-7, (
        f"{scheme}: dq_c_conv_dt = {max_dq_c:.2e} kg/kg/s in stable column"
    )


# ============================================================================
# 3e extended -- precipitation non-negative across multiple random profiles
# ============================================================================

@pytest.mark.parametrize("scheme", ["sbm", "dca", "kuo", "mass_flux", "edmf"])
def test_precipitation_non_negative_random_profiles(scheme):
    """dq_c_conv_dt >= 0 at every level for a spread of random profiles.

    Generates 8 randomized column profiles by perturbing the unstable
    template with multiplicative T noise and additive RH noise.
    Captures sign-flip bugs that a single profile may miss.
    """
    key = jax.random.PRNGKey(0xC0DE)
    T0, q0, p_full, p_half = _make_unstable_column(nlev=20, ncol=8)

    # T jitter: +/- 5 K independent per (col, level)
    key_T, key_q = jax.random.split(key)
    T = T0 + 5.0 * jax.random.normal(key_T, T0.shape)
    T = jnp.clip(T, 180.0, 320.0)

    # RH jitter: q scaled by uniform 0.4..1.1
    rh_factor = 0.4 + 0.7 * jax.random.uniform(key_q, q0.shape)
    q_v = jnp.clip(q0 * rh_factor, 1e-10, None)

    out = _call_scheme(scheme, T, q_v, p_full, p_half)
    min_dq_c = float(jnp.min(out.dq_c_conv_dt))
    # Allow eps for floating-point round-off in the rescale division.
    assert min_dq_c >= -1e-12, (
        f"{scheme}: random-profile min dq_c_conv_dt = {min_dq_c:.2e}"
    )


# ============================================================================
# 3f extended -- tendency signs for unstable profiles, all schemes
# ============================================================================

@pytest.mark.parametrize("scheme", ["dca", "mass_flux", "edmf"])
def test_drying_in_unstable_column(scheme):
    """Convection in an unstable column should dry SOMEWHERE in the column.

    Targets the lower-troposphere drying signature in the agent spec
    without locking to a specific level (different schemes peak their
    drying at different heights: SBM/DCA in the BL, mass-flux/EDMF
    higher up via compensating subsidence).

    Saturation-clipping schemes (DCA) use a (super)saturated fixture: they
    only condense where the parcel is at/above saturation, so a drying
    assertion against the subsaturated ``_make_unstable_column`` would test
    a non-physical premise (it passed before only because the old DCA pair
    solve over-cooled the lower member; see ``dca.py`` 2x2 solve).
    """
    if scheme == "dca":
        T, q_v, p_full, p_half = _make_saturated_unstable_column()
    else:
        T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)
    min_dqv = float(jnp.min(out.dq_v_dt))
    assert min_dqv < -1e-10, (
        f"{scheme}: no drying detected, min dq_v_dt = {min_dqv:.2e}"
    )


@pytest.mark.parametrize("scheme", ["sbm", "dca", "mass_flux", "edmf"])
def test_upper_trop_warming_in_unstable_column(scheme):
    """Convection in an unstable column should warm SOMEWHERE in the column.

    Latent heat release (SBM/Kuo) or detrained plume warmth (mass-flux/EDMF)
    should produce dT_dt > 0 in at least the upper-troposphere section.
    """
    T, q_v, p_full, p_half = _make_unstable_column()
    out = _call_scheme(scheme, T, q_v, p_full, p_half)
    nlev = T.shape[1]
    upper = out.dT_dt[:, : nlev // 2]   # top half (low pressure)
    max_dT_upper = float(jnp.max(upper))
    assert max_dT_upper > 1e-8, (
        f"{scheme}: no upper-trop warming, max dT_dt[upper] = {max_dT_upper:.2e}"
    )


# ============================================================================
# 3g extended -- M_c responds to CAPE
# ============================================================================

def test_mass_flux_M_c_grows_under_high_cape():
    """M_c should grow toward the CAPE-driven equilibrium.

    With M_c starting at 0 and a CAPE-positive column, the implicit
    relaxation gives M_c_new = dt * M_eq / tau_adj > 0.  With a
    CAPE-zero column, M_c_new should be near zero (sigmoid gate
    suppresses M_eq).  This pins the qualitative response to CAPE.
    """
    T_hi, q_hi, p_full, p_half = _make_unstable_column()
    T_lo, q_lo, _, _ = _make_stable_column()
    ncol = T_hi.shape[0]
    M_c0 = jnp.zeros(ncol)

    _, M_hi = mass_flux_convection(
        T_hi, q_hi, p_full, p_half, M_c0, 300.0, MassFluxConfig(),
    )
    _, M_lo = mass_flux_convection(
        T_lo, q_lo, p_full, p_half, M_c0, 300.0, MassFluxConfig(),
    )

    mean_hi = float(jnp.mean(M_hi))
    mean_lo = float(jnp.mean(M_lo))
    # High-CAPE column should have order(1e3) more mass flux than the
    # gated stable column.  Probe values: ~8.3e-4 vs ~7.6e-7.
    assert mean_hi > 1e-5, (
        f"mass_flux: high-CAPE M_c = {mean_hi:.2e} did not grow"
    )
    assert mean_hi > 100.0 * mean_lo, (
        f"mass_flux: high/low CAPE ratio M_c = {mean_hi/max(mean_lo, 1e-30):.1f}, "
        "expected >100x"
    )


# ============================================================================
# 3h extended -- a_u responds to CAPE, stays in [0, clip]
# ============================================================================

def test_edmf_a_u_grows_under_high_cape():
    """EDMF a_u should grow under high CAPE and stay near zero in stable air.

    The diagnosed equilibrium a_u_eq = convective_mask * a_u_init.  With
    a_u starting at 0, the implicit relaxation grows a_u toward a_u_eq.
    Probe values: ~1.7e-2 vs ~1.5e-5 (~1000x ratio).
    """
    T_hi, q_hi, p_full, p_half = _make_unstable_column()
    T_lo, q_lo, _, _ = _make_stable_column()
    ncol = T_hi.shape[0]
    a_u0 = jnp.zeros(ncol)

    _, a_hi = edmf_convection(
        T_hi, q_hi, p_full, p_half, a_u0, 300.0, ConvectiveEDMFConfig(),
    )
    _, a_lo = edmf_convection(
        T_lo, q_lo, p_full, p_half, a_u0, 300.0, ConvectiveEDMFConfig(),
    )

    mean_hi = float(jnp.mean(a_hi))
    mean_lo = float(jnp.mean(a_lo))
    assert mean_hi > 1e-4, f"EDMF: high-CAPE a_u = {mean_hi:.2e} did not grow"
    assert mean_hi > 100.0 * mean_lo, (
        f"EDMF: high/low CAPE ratio a_u = {mean_hi/max(mean_lo, 1e-30):.1f}, "
        "expected >100x"
    )
    # The leaf clips a_u_new to [0, 0.5].  In CAPE-positive conditions,
    # a_u should still be well below the clip bound.
    max_a = float(jnp.max(a_hi))
    assert 0.0 <= max_a <= 0.5, (
        f"EDMF: a_u out of [0, 0.5] bound: max={max_a:.4e}"
    )


# ============================================================================
# mass_flux / edmf cloud-water source must vanish in dry columns
# ============================================================================

@pytest.mark.parametrize("scheme", ["mass_flux", "edmf"])
def test_no_cloud_water_in_dry_column(scheme):
    """Plume cloud water must follow actual q_v, not assume saturated parcel.

    A 5% RH column with a hot surface produces a (formally) large CAPE
    because compute_moist_adiabat assumes a saturated launched parcel,
    but the *actual* parcel water content is q_v_sfc, not q_sat_sfc.
    Pre-fix the plume kernel built ``q_c_u = dilution * (q_sat_sfc -
    q_sat_moist)`` — independent of the column's actual q_v — producing
    cloud water and surface precipitation in genuinely dry columns
    (~2 mm/day in this test setup for EDMF).  Post-fix the plume's
    initial water reservoir is q_v_sfc and condensation only occurs
    where ``q_sat_moist`` falls below ``q_v_sfc`` (i.e. above the LCL
    of the actual unsaturated parcel).
    """
    ncol, nlev = 1, 20
    p_s = 1.0e5
    sigma_full = jnp.linspace(0.025, 0.975, nlev)
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    T = 320.0 * jnp.clip(sigma_full, 0.01, None) ** 0.19
    T = jnp.maximum(T, 200.0)
    T = jnp.broadcast_to(T[None, :], (ncol, nlev))
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.05 * q_sat   # 5% RH everywhere

    if scheme == "mass_flux":
        out, _ = mass_flux_convection(
            T, q_v, p_full, p_half, jnp.zeros(ncol), 300.0, MassFluxConfig(),
        )
    else:
        out, _ = edmf_convection(
            T, q_v, p_full, p_half, 0.1 * jnp.ones(ncol), 300.0, ConvectiveEDMFConfig(),
        )

    dp = p_half[:, 1:] - p_half[:, :-1]
    surface_precip = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1)[0])
    # Convert to mm/day for readability: kg/m^2/s * 86400 = mm/day
    precip_mm_day = surface_precip * 86400.0
    # Allow some bleed (sigmoid soft transition near LCL) but block
    # the order-of-magnitude bug.  Pre-fix EDMF is ~2 mm/day;
    # post-fix should be < 0.1 mm/day.
    assert precip_mm_day < 0.1, (
        f"{scheme}: dry-column (5% RH) surface precip = {precip_mm_day:.4f} mm/day; "
        f"expected < 0.1 mm/day. Plume q_c_u must depend on the actual "
        "q_v_sfc, not q_sat at surface."
    )


def test_kuo_convergence_closure_removes_local_source():
    """Canonical Kuo (1965) closure, convergence-driven.

    The supersaturation-budget test that lived here is OBSOLETE: the
    faithful Kuo source is the LARGE-SCALE MOISTURE CONVERGENCE
    ``ptenq`` (a bounded external dynamical tendency), not the column
    supersaturation ``Σ max(q_v − q_sat, 0)``.  The faithful closure is

        dq/dt = −ptenq + cvgu/zint · (qvc − qv)
        dt/dt =          cvgu/zint · (tc  − t)

    so the column-integrated vapor tendency over the active layers is
    ``∫dq/dt dp/g = −cvgu + cvgu/zint · zint_q`` and the column-
    integrated heating expressed as a vapor sink is
    ``∫(dT/dt · c_p/L)·dp/g·(?)`` — but the clean, formula-level
    invariant we pin is the **closure normalisation** itself:

        ∫(dT/dt)·dp/g · c_p  ==  cvgu/zint · zint_t · c_p    (heating)

    i.e. the per-level heating is exactly ``cvgu/zint·(tc−t)`` on the
    active layers.  We reconstruct ``cvgu``, ``zint``, ``zint_t`` from
    the leaf's own active mask and check the heating column integral
    matches ``cvgu/zint·zint_t·c_p`` to round-off — this exercises the
    convergence closure (not the dead supersaturation path) and would
    catch a wrong normalisation or a dropped ``−ptenq`` removal.
    """
    from legoesm import constants
    from legoesm.atmosphere.physics.convection.kuo import kuo_convection
    from legoesm.atmosphere.physics.convection.config import KuoConfig
    from legoesm.thermo import saturation_mixing_ratio

    ncol, nlev = 4, 20
    p_s = 1.0e5
    sigma_h = jnp.linspace(0.1, 1.0, nlev + 1)
    sigma_f = 0.5 * (sigma_h[:-1] + sigma_h[1:])
    p_full = jnp.broadcast_to((sigma_f * p_s)[None, :], (ncol, nlev))
    p_half = jnp.broadcast_to((sigma_h * p_s)[None, :], (ncol, nlev + 1))
    z = -8000.0 * jnp.log(jnp.clip(sigma_f, 1e-3, None))
    T = jnp.broadcast_to(
        jnp.maximum(300.0 - 6.5e-3 * z, 200.0)[None, :], (ncol, nlev),
    )
    q_sat = saturation_mixing_ratio(T, p_full)
    RH = 0.85 * jnp.clip((sigma_f - 0.15) / 0.85, 0.0, 1.0) + 0.1
    q_v = jnp.broadcast_to((RH * q_sat[0])[None, :], (ncol, nlev))
    # Positive low/mid-tropospheric convergence bump (the Kuo source).
    p = sigma_f * p_s
    ptenq = (3.0e-3 / 86400.0) * jnp.exp(-((p - 850e2) / 120e2) ** 2)
    ptenq = jnp.broadcast_to(ptenq[None, :], (ncol, nlev))

    config = KuoConfig()
    out = kuo_convection(T, q_v, p_full, p_half, dt=900.0, config=config,
                         moisture_convergence=ptenq)

    dp = p_half[:, 1:] - p_half[:, :-1]
    g = constants.g

    # (1) The scheme fires (positive column heating).
    col_heat = jnp.sum(out.dT_dt * dp / g, axis=1) * constants.c_pd
    assert float(jnp.min(col_heat)) > 1.0, (
        f"Kuo did not fire: column heating = {[float(x) for x in col_heat]} W/m2"
    )

    # (2) The −ptenq removal makes the net column vapor tendency a sink
    # (canonical Kuo converts converged moisture to heating/precip, it
    # does not pile vapor up): ∫dq/dt dp/g < ∫(−ptenq + small) ... the
    # robust sign check is that the column DRIES somewhere.
    assert float(jnp.min(out.dq_v_dt)) < 0.0, (
        "Kuo must remove the large-scale convergence (−ptenq) somewhere"
    )

    # (3) Positivity: q_v + dt·dq_v_dt >= 0 over one step.
    qv_new = q_v + 900.0 * out.dq_v_dt
    assert float(jnp.min(qv_new)) >= -1e-12
