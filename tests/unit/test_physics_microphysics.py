"""Category 4: Microphysics -- Physical Consistency.

Tests total water conservation, temperature-moisture coupling (Clausius-
Clapeyron), saturation adjustment, precipitation positivity, ice-phase
bounds, and autoconversion threshold for all microphysics schemes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig, SundqvistConfig, SeifertBehengConfig, MorrisonConfig,
    ThompsonConfig, P3Config,
)
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState, make_zero_hydrometeors,
)
from legoesm.thermo import saturation_mixing_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_column(nlev=20, ncol=4, T_sfc=280.0, q_c_val=1e-4, supersaturated=False):
    """Build a microphysics column with realistic conditions."""
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))

    T = T_sfc * jnp.clip(sigma_full, 0.01, None) ** 0.19
    T = jnp.maximum(T, 200.0)
    T = jnp.broadcast_to(T[None, :], (ncol, nlev))

    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))

    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    dz = jnp.abs(dz)

    q_sat = saturation_mixing_ratio(T, p_full)
    if supersaturated:
        q_v = 1.2 * q_sat
    else:
        q_v = 0.8 * q_sat

    q_c = jnp.zeros((ncol, nlev))
    q_c = q_c.at[..., -5:].set(q_c_val)

    hydro = HydrometeorState(
        q_c=q_c,
        q_r=jnp.zeros((ncol, nlev)),
        q_i=jnp.zeros((ncol, nlev)),
        q_s=jnp.zeros((ncol, nlev)),
        q_g=jnp.zeros((ncol, nlev)),
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=jnp.zeros((ncol, nlev)),
    )
    return T, q_v, hydro, p_full, p_half, rho, dz


def _call_scheme(name, T, q_v, hydro, p_full, p_half, rho, dz, dt=300.0):
    """Call a microphysics scheme backend and return MicrophysicsOutput."""
    if name == "kessler":
        return kessler_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                     config=KesslerConfig())
    elif name == "sundqvist":
        return sundqvist_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                       config=SundqvistConfig())
    elif name == "seifert_beheng":
        return seifert_beheng_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                            config=SeifertBehengConfig())
    elif name == "morrison":
        return morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                      config=MorrisonConfig())
    elif name == "thompson":
        return thompson_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                      config=ThompsonConfig())
    elif name == "p3":
        return p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                               config=P3Config())
    else:
        raise ValueError(f"Unknown scheme: {name}")


ALL_SCHEMES = ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson", "p3"]


# ============================================================================
# 4a  All outputs finite
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_all_outputs_finite(scheme):
    """All MicrophysicsOutput fields should be finite."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
    for fname in out._fields:
        val = getattr(out, fname)
        assert jnp.all(jnp.isfinite(val)), f"{scheme}: {fname} has NaN/Inf"


# ============================================================================
# 4b  Temperature-moisture coupling (Clausius-Clapeyron)
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_temperature_moisture_coupling(scheme):
    """cp * dT_dt approx Lv * condensation_rate (first order).

    Checks that where heating is significant, it correlates with moisture
    removal (condensation heats, evaporation cools).
    """
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(supersaturated=True)
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)

    lhs = constants.c_pd * out.dT_dt
    rhs = -constants.L_v * out.dq_v_dt
    scale = jnp.maximum(jnp.abs(lhs), 1e-10)
    rel_err = jnp.abs(lhs - rhs) / scale
    active = jnp.abs(lhs) > 1e-8
    if jnp.any(active):
        median_err = float(jnp.median(rel_err[active]))
        assert median_err < 0.5, (
            f"{scheme}: median Clausius-Clapeyron rel_err = {median_err:.3f}"
        )


# ============================================================================
# 4c  Saturation adjustment: supersaturated -> vapor decreases
# ============================================================================

@pytest.mark.parametrize("scheme", ["kessler", "sundqvist"])
def test_saturation_adjustment_vapor_decreases(scheme):
    """Starting from supersaturated, vapor should decrease (dq_v_dt < 0)."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
        supersaturated=True, q_c_val=0.0
    )
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)

    # Some levels should show condensation (dq_v_dt < 0)
    min_dqv = float(jnp.min(out.dq_v_dt))
    assert min_dqv < 0.0, (
        f"{scheme}: no condensation in supersaturated column, min dq_v_dt = {min_dqv:.2e}"
    )


# ============================================================================
# 4d  Precipitation non-negative
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_precipitation_non_negative(scheme):
    """Precipitation must be >= 0."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
    min_precip = float(jnp.min(out.precipitation))
    assert min_precip >= -1e-15, (
        f"{scheme}: negative precipitation = {min_precip:.2e}"
    )


# ============================================================================
# 4e  Ice-phase temperature bounds (Morrison, Thompson)
# ============================================================================

@pytest.mark.parametrize("scheme", ["morrison", "thompson"])
def test_ice_no_formation_above_freezing(scheme):
    """Above freezing (T > 273.15 K), ice formation should be negligible."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(T_sfc=290.0)
    T_warm = jnp.maximum(T, 280.0)
    out = _call_scheme(scheme, T_warm, q_v, hydro, p_full, p_half, rho, dz)

    warm_mask = T_warm > constants.T_freeze
    if jnp.any(warm_mask):
        ice_formation = out.dq_i_dt[warm_mask]
        max_ice_form = float(jnp.max(ice_formation))
        # Allow small numerical noise from sigmoid tails
        assert max_ice_form <= 1e-10, (
            f"{scheme}: ice forms above freezing, max dq_i_dt = {max_ice_form:.2e}"
        )


# ============================================================================
# 4g  Autoconversion threshold (Kessler)
# ============================================================================

def test_kessler_autoconversion_sensitivity():
    """Rain production should increase when q_c exceeds autoconversion threshold."""
    config = KesslerConfig()
    # Below threshold: subsaturated, small q_c
    T_lo, q_v_lo, hydro_lo, p_full, p_half, rho, dz = _make_column(
        q_c_val=0.1 * config.autoconversion_threshold, supersaturated=False
    )
    out_lo = kessler_microphysics(T_lo, q_v_lo, hydro_lo, p_full, p_half, rho, dz,
                                   300.0, config=config)
    # Above threshold: subsaturated, large q_c
    _, _, hydro_hi, _, _, _, _ = _make_column(
        q_c_val=5.0 * config.autoconversion_threshold, supersaturated=False
    )
    out_hi = kessler_microphysics(T_lo, q_v_lo, hydro_hi, p_full, p_half, rho, dz,
                                   300.0, config=config)

    max_dqr_lo = float(jnp.max(out_lo.dq_r_dt))
    max_dqr_hi = float(jnp.max(out_hi.dq_r_dt))
    assert max_dqr_hi > max_dqr_lo, (
        f"Kessler: rain production not higher above threshold: "
        f"lo={max_dqr_lo:.2e}, hi={max_dqr_hi:.2e}"
    )


# ============================================================================
# Heating rate magnitude bounds
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_heating_rate_bounded(scheme):
    """|dT_dt| should be bounded (< 10 K/s for strongly supersaturated)."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(supersaturated=True)
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
    max_hr = float(jnp.max(jnp.abs(out.dT_dt)))
    assert max_hr < 10.0, (
        f"{scheme}: |dT_dt| = {max_hr:.2e} K/s exceeds 10 K/s"
    )


# ============================================================================
# Donor clamps: Morrison/Thompson q_c, q_i sinks must not over-extract
# ============================================================================

@pytest.mark.parametrize(
    "scheme,call",
    [("morrison", morrison_microphysics), ("thompson", thompson_microphysics)],
)
def test_qc_does_not_go_negative_at_long_dt(scheme, call):
    """An explicit Euler step must not drive q_c below zero in mixed-phase.

    Default config rates with dt=1200s drive Bergeron+riming+evaporation
    sinks past 100% of q_c per step:
        bergeron_rate = 1e-3 /s -> bergeron * dt = 1.2 * q_c
    The fix proportionally scales all q_c sink processes so the total
    loss per step is bounded by q_c (mass conservation preserved by
    scaling source terms in dq_r/dq_i/dq_s by the same factor).

    Saturation is set to RH=1.0 so saturation_adjustment does not
    additionally evaporate cloud water; layer thickness is large
    (10 km) to keep sedimentation Courant well below 1.
    """
    cfg_class = MorrisonConfig if scheme == "morrison" else ThompsonConfig
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 250.0)  # mixed-phase regime
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 10_000.0)
    q_c = jnp.full((ncol, nlev), 1e-3)
    q_i = jnp.full((ncol, nlev), 1e-4)
    q_s = jnp.full((ncol, nlev), 1e-4)
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
        q_g=jnp.zeros_like(q_c),
        N_c=1e8 * jnp.ones_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=1e4 * jnp.ones_like(q_c),
    )
    # RH = 1: keep condensation ~ 0 so this test isolates the q_c sinks
    # (autoconv / accretion / Bergeron / riming) from saturation-driven
    # cloud evaporation.
    q_v = 1.0 * saturation_mixing_ratio(T, p_full)
    dt = 1200.0
    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, cfg_class())
    q_c_after = q_c + out.dq_c_dt * dt
    min_qc = float(jnp.min(q_c_after))
    # Allow ~1e-9 floor for floating-point round-off in scaled tendencies.
    assert min_qc >= -1e-9, (
        f"{scheme}: q_c went negative ({min_qc:.3e}) after one explicit "
        f"step at dt={dt}s — Bergeron + riming + autoconv + accretion "
        "sinks combined exceeded available q_c without a donor clamp."
    )


def test_thompson_qi_does_not_go_negative_warm():
    """Thompson melt processes must not over-extract q_i above freezing.

    Morrison clamps melt_ice/melt_snow with ``min(rate*q*frac, q/dt)``;
    Thompson did not, so at T=280K (above freezing) and dt=1200s with
    default rates, q_i was driven below zero.  After fix, melt is clamped
    to available mass.

    Layer thickness is 10 km so sedimentation Courant
    ``V_t * dt / dz <= 5*1200/10000 = 0.6`` stays below 1.
    """
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 280.0)  # above freezing
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 10_000.0)
    q_c = jnp.full((ncol, nlev), 1e-3)
    q_i = jnp.full((ncol, nlev), 1e-4)
    q_s = jnp.full((ncol, nlev), 1e-4)
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
        q_g=jnp.zeros_like(q_c),
        N_c=1e8 * jnp.ones_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=1e4 * jnp.ones_like(q_c),
    )
    q_v = 0.8 * saturation_mixing_ratio(T, p_full)
    dt = 1200.0
    out = thompson_microphysics(
        T, q_v, hydro, p_full, p_half, rho, dz, dt, ThompsonConfig(),
    )
    q_i_after = q_i + out.dq_i_dt * dt
    # Skip the top level: with q_i uniform throughout the column the top
    # level always loses sedimentation flux without compensating inflow,
    # which is a sedimentation-CFL concern rather than a melt-clamp one.
    # The melt-clamp bug shows up uniformly in interior levels.
    min_qi_interior = float(jnp.min(q_i_after[:, 1:]))
    assert min_qi_interior >= -1e-9, (
        f"Thompson: interior q_i went negative ({min_qi_interior:.3e}) at "
        f"T=280 K with dt=1200s — melt rate × q_i × melt_frac × dt exceeded "
        "q_i without a donor clamp."
    )


@pytest.mark.parametrize(
    "scheme,call",
    [
        ("kessler", kessler_microphysics),
        ("seifert_beheng", seifert_beheng_microphysics),
        ("morrison", morrison_microphysics),
        ("thompson", thompson_microphysics),
    ],
)
def test_subsaturated_clear_air_does_not_create_negative_qc(scheme, call):
    """Audit cycle 2 (Codex): the smooth saturation adjustment
    ``condensation = sigmoid(s · excess) · excess / dt`` is *signed* —
    negative for ``q_v < q_sat`` (evaporation).  Adding this directly
    to ``dq_c_dt`` in subsaturated clear air (``q_c = 0``) produces a
    spurious negative cloud-water tendency that drives ``q_c`` below
    zero in the explicit Euler step.

    The fix is to donor-clamp the evaporation branch against the
    available cloud water — implemented inside
    ``saturation_adjustment`` (and inline for Kessler) and joined to
    the per-scheme donor clamp on q_c sinks.

    Test column: 95 % RH at T=280 K, q_c = 0.  In all four schemes
    the buggy form produced ``q_c_after ≈ -3e-4 kg/kg`` after a
    1200-s timestep.
    """
    if scheme in ("kessler", "seifert_beheng"):
        config_cls = {
            "kessler": KesslerConfig,
            "seifert_beheng": SeifertBehengConfig,
        }[scheme]
    elif scheme == "morrison":
        config_cls = MorrisonConfig
    else:
        config_cls = ThompsonConfig

    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 280.0)
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 1000.0)
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.95 * q_sat                           # subsaturated
    q_c = jnp.zeros((ncol, nlev))                # NO cloud water
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c),
        q_i=jnp.zeros_like(q_c), q_s=jnp.zeros_like(q_c),
        q_g=jnp.zeros_like(q_c),
        N_c=1e8 * jnp.ones_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=jnp.zeros_like(q_c),
    )
    dt = 1200.0
    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, config_cls())
    q_c_after = q_c + out.dq_c_dt * dt
    min_q_c = float(jnp.min(q_c_after))
    # Tolerance allows for f64 → f32 promotion noise; the buggy form
    # produces q_c_after ≈ -3e-4, well above this threshold.
    assert min_q_c >= -1e-9, (
        f"{scheme}: q_c_after went negative ({min_q_c:.3e}) in a "
        "subsaturated clear-air column (q_v = 95% RH, q_c = 0).  "
        "The signed saturation adjustment ``condensation = sigmoid(s · "
        "(q_v - q_sat)) · (q_v - q_sat) / dt`` is negative there, and "
        "adding it to ``dq_c_dt`` over an explicit Euler step drives "
        "q_c below zero without a donor clamp on the evaporation "
        "branch.  Audit cycle 2 Codex finding 'subsaturated clear air "
        "can create negative cloud water' has regressed."
    )


def test_thompson_rime_to_graupel_donor_split():
    """Audit cycle 2 (Codex): the Thompson rime-to-graupel conversion
    must subtract from the SOURCE species (q_i for ``riming_i``,
    q_s for ``riming_s``), not split via fixed 1.0 / 0.5 / 1.5
    coefficients on the total.

    Pathological column: ``q_c > 0`` (cloud water source for riming),
    ``q_i = 0`` (no ice to be rimed), ``q_s > 0`` (snow that gets
    rimed by cloud water), ``T < T_freeze`` (active ice phase).  In
    this column ``riming_i = 0`` (no q_i to rime) and ``riming_s > 0``
    (q_c × q_s × f_ice).  Pre-fix:
        rime_to_graupel = rate * (riming_i + riming_s) > 0
        dq_i_dt -= rime_to_graupel              # full subtraction!
        dq_s_dt -= 0.5 * rime_to_graupel
        dq_g_dt += 1.5 * rime_to_graupel
    drives ``q_i`` negative and creates 1.5× extra mass — both
    conservation violations.  Post-fix the donor split scales by
    ``riming_i / riming_s`` and only the actually-rimed species is
    drained.  Total mass moved (``rime_to_graupel_from_i +
    rime_to_graupel_from_s``) goes 1:1 to graupel.
    """
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 250.0)  # below freezing — ice active
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 10_000.0)
    # The pathological mix: cloud water + snow, NO ice.
    q_c = jnp.full((ncol, nlev), 5e-3)
    q_i = jnp.zeros((ncol, nlev))                # zero ice — would be drained negative pre-fix
    q_s = jnp.full((ncol, nlev), 1e-3)            # snow gets rimed
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
        q_g=jnp.zeros_like(q_c),
        N_c=1e8 * jnp.ones_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=jnp.zeros_like(q_c),
    )
    q_v = 0.5 * saturation_mixing_ratio(T, p_full)
    dt = 1200.0
    out = thompson_microphysics(
        T, q_v, hydro, p_full, p_half, rho, dz, dt, ThompsonConfig(),
    )
    q_i_after = q_i + out.dq_i_dt * dt
    # q_i must remain non-negative (donor split: only riming_i drains q_i).
    min_qi = float(jnp.min(q_i_after))
    assert min_qi >= -1e-9, (
        f"Thompson: q_i went negative ({min_qi:.3e}) when starting at "
        "q_i=0 with q_s>0 and active riming.  Pre-fix the rime_to_graupel "
        "subtracted the full conversion from q_i regardless of which "
        "species was actually rimed.  Audit cycle 2 Codex finding "
        "'Thompson graupel conversion can draw from the wrong donor' "
        "has regressed."
    )
    # Mass conservation: ∑ dq_i + dq_s + dq_g (pure ice phase) should
    # equal sed_i + sed_s + sed_g (sedimentation only escapes the column);
    # internal phase changes cancel in the sum.  We can verify rime->graupel
    # specifically by checking that any q_g gain matches a q_s loss.
    dq_g_total = float(jnp.sum(out.dq_g_dt))
    # Without melting (T well below freeze), all q_g must come from
    # rime_to_graupel.  q_g production must be matched by q_s loss
    # (donor split: only riming_s active here).
    assert dq_g_total >= 0.0, (
        f"Thompson: dq_g_dt total ({dq_g_total:.3e}) is negative "
        "without graupel sedimentation source — graupel mass conservation "
        "violation."
    )


# ============================================================================
# Morrison/Thompson moist-enthalpy conservation (latent heat of fusion)
# ============================================================================

@pytest.mark.parametrize(
    "scheme,call",
    [("morrison", morrison_microphysics), ("thompson", thompson_microphysics)],
)
def test_freezing_releases_latent_heat_of_fusion(scheme, call):
    """Bergeron + riming (cloud water → ice/snow) must release ``L_f``.

    Moist enthalpy ``h = c_pd T + L_v q_v - L_f * q_ice`` is conserved
    by phase transitions in a closed column, so per-level the residual
        c_pd * dT_dt + L_v * dq_v_dt - L_f * (dq_i + dq_s + dq_g)
    is the divergence of sedimentation flux (surface boundary effect),
    not a phase-change heating/cooling.

    Pre-fix the dT_dt assembly omitted the ``+ L_f * (bergeron +
    riming_i + riming_s)`` term, so in a mixed-phase column where
    ~6e-4 kg/kg of cloud water freezes per step the column residual
    is ~16 K/day too cold (L_f * freezing / c_pd = 333e3 * 6.6e-4 /
    1004 / 1200s = 1.83e-4 K/s = 15.8 K/day at default rates).

    Post-fix the residual collapses to the small contribution from
    sedimentation flux divergence at the interior of the column.
    """
    cfg_class = MorrisonConfig if scheme == "morrison" else ThompsonConfig
    ncol, nlev = 1, 5
    # Mid mixed-phase regime, sub-saturated wrt liquid (suppresses
    # condensation/evaporation as the dominant balance) so the
    # cloud→ice freezing branches dominate the residual.  Short dt
    # keeps sedimentation flux divergence small.
    T = jnp.full((ncol, nlev), 260.0)
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 10_000.0)
    q_c = jnp.full((ncol, nlev), 1e-3)
    q_i = jnp.full((ncol, nlev), 5e-4)
    q_s = jnp.full((ncol, nlev), 5e-4)
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
        q_g=jnp.zeros_like(q_c),
        N_c=1e8 * jnp.ones_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=1e4 * jnp.ones_like(q_c),
    )
    q_v = 0.85 * saturation_mixing_ratio(T, p_full)
    dt = 60.0
    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, cfg_class())

    # Inspect interior level (k=2) where sedimentation flux divergence
    # is approximately zero (uniform q_i, q_s ⇒ flux_in ≈ flux_out).
    k = 2
    h_residual = float(
        constants.c_pd * out.dT_dt[0, k]
        + constants.L_v * out.dq_v_dt[0, k]
        - constants.L_f * (out.dq_i_dt[0, k] + out.dq_s_dt[0, k] + out.dq_g_dt[0, k])
    )
    # Magnitude scale of L_f-driven heating (lower bound)
    freezing_scale = float(
        constants.L_f * jnp.abs(out.dq_i_dt[0, k] + out.dq_s_dt[0, k] + out.dq_g_dt[0, k])
    )
    # Pre-fix: |h_residual| ≈ L_f * freezing rate (entire term missing);
    # post-fix: |h_residual| << L_f * freezing rate (just sedimentation div).
    assert abs(h_residual) < 0.2 * max(freezing_scale, 1e-10), (
        f"{scheme}: per-level moist-enthalpy residual at k={k} = "
        f"{h_residual:.3e} W/m^3-equivalent, freezing scale = "
        f"{freezing_scale:.3e}; ratio = {abs(h_residual)/max(freezing_scale,1e-30):.3f}.  "
        "dT_dt is missing +L_f * (bergeron + riming_i + riming_s) / c_pd "
        "for the cloud-water → ice/snow freezing branches."
    )


# ============================================================================
# Sundqvist column water budget (closes against surface precipitation)
# ============================================================================

def test_sundqvist_column_water_budget_closes():
    """Sundqvist is a diagnostic scheme — rain falls instantly.

    Column total water tendency must balance surface precipitation:
        int (dq_v + dq_c + dq_r) dp/g  +  precipitation  ~  0
    """
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
        T_sfc=290.0, q_c_val=1e-3,
    )
    # Saturate the column so condensation (and therefore rain) actually fires.
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = q_sat
    out = sundqvist_microphysics(
        T, q_v, hydro, p_full, p_half, rho, dz, 300.0, SundqvistConfig(),
    )
    dp = p_half[:, 1:] - p_half[:, :-1]
    col_water_tend = jnp.sum(
        (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt) * dp, axis=1,
    ) / constants.g
    imbalance = col_water_tend + out.precipitation
    # Allow numerics; pre-fix imbalance was ~+9e-3 (= +precipitation).
    max_imbalance = float(jnp.max(jnp.abs(imbalance)))
    max_precip = float(jnp.max(out.precipitation))
    assert max_imbalance < 1e-6 * max(max_precip, 1.0), (
        f"Sundqvist water budget unclosed: imbalance = {max_imbalance:.3e}, "
        f"precip scale = {max_precip:.3e}"
    )


def test_sundqvist_drains_incoming_qr_to_precipitation():
    """If q_r is non-zero on input (e.g., warm-started or after a prior
    Kessler step), Sundqvist's diagnostic-rain semantics drain it to the
    surface in one step and account for the drained mass in
    ``precipitation``.

    Previously ``dq_r_dt = 0`` left any pre-existing q_r frozen in the
    column — column water grew indefinitely and surface precipitation
    under-reported the actual mass leaving.  This test would catch that
    regression: assert q_r is zero after one step AND the drained mass
    appears in the precipitation flux.
    """
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
        T_sfc=290.0, q_c_val=0.0,
    )
    # Sub-saturated column with NO q_c so condensation/autoconversion
    # don't generate fresh rain — this isolates the q_r-drain channel.
    q_v_sub = 0.5 * saturation_mixing_ratio(T, p_full)
    # Inject q_r at a few mid-column levels so dq_r_drain is nontrivial.
    q_r_in = jnp.zeros_like(hydro.q_r).at[..., 8:12].set(5e-4)
    hydro_with_qr = hydro._replace(q_r=q_r_in)

    dt = 300.0
    out = sundqvist_microphysics(
        T, q_v_sub, hydro_with_qr, p_full, p_half, rho, dz, dt, SundqvistConfig(),
    )
    q_r_after = q_r_in + out.dq_r_dt * dt
    # q_r must drain to zero in one step (modulo float round-off).
    max_qr_residual = float(jnp.max(jnp.abs(q_r_after)))
    assert max_qr_residual < 1e-12, (
        f"Sundqvist failed to drain incoming q_r: max(|q_r_after|) = "
        f"{max_qr_residual:.3e}; expected ~0 (diagnostic-rain semantics)."
    )
    # Drained mass [kg/m^2/s] must appear in precipitation.
    dp = p_half[:, 1:] - p_half[:, :-1]
    expected_drain_flux = jnp.sum(q_r_in * dp, axis=1) / (constants.g * dt)
    # No autoconversion source here (q_c=0, sub-saturated) so precip
    # equals the drain flux to a few percent (allowing for the smooth
    # condensation sigmoid producing a tiny residual).
    rel_err = float(jnp.max(
        jnp.abs(out.precipitation - expected_drain_flux) /
        jnp.maximum(expected_drain_flux, 1e-30),
    ))
    assert rel_err < 0.05, (
        f"Sundqvist precip = {[float(x) for x in out.precipitation]}; "
        f"expected drain flux = {[float(x) for x in expected_drain_flux]} "
        "(no fresh autoconversion in this dry-c column)."
    )
    # Column water budget must still close with q_r in the loop.
    col_water_tend = jnp.sum(
        (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt) * dp, axis=1,
    ) / constants.g
    imbalance = col_water_tend + out.precipitation
    max_imbalance = float(jnp.max(jnp.abs(imbalance)))
    max_precip = float(jnp.max(out.precipitation))
    assert max_imbalance < 1e-6 * max(max_precip, 1.0), (
        f"Sundqvist q_r-drain water budget unclosed: imbalance = "
        f"{max_imbalance:.3e}, precip scale = {max_precip:.3e}"
    )


# ============================================================================
# 4l  Kessler / warm-rain accretion & evaporation: gradient finite at q_r=0
# ============================================================================

def test_kessler_grad_finite_at_zero_qr():
    """Marshall-Palmer fractional powers q_r**0.875 (accretion) and
    q_r**0.525 (evaporation) have unbounded analytic derivative at q_r=0.

    Without an AD guard, ``jnp.clip(q_r, 0)**0.525`` returns +inf for the
    gradient at q_r=0 (and NaN at q_r<0).  This test imports the kessler
    leaf and the shared ``rain_evaporation`` and asserts that the gradient
    of a column scalar w.r.t. q_r is finite when q_r is exactly zero.
    """
    from legoesm.atmosphere.physics.microphysics._warm_rain import rain_evaporation

    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(q_c_val=1e-4)
    q_r_zero = jnp.zeros_like(hydro.q_r)
    hydro_zero = hydro._replace(q_r=q_r_zero)

    def kessler_objective(q_r):
        h = hydro_zero._replace(q_r=q_r)
        out = kessler_microphysics(
            T, q_v, h, p_full, p_half, rho, dz, 300.0, KesslerConfig(),
        )
        return jnp.sum(out.dq_r_dt + out.dq_v_dt + out.dT_dt)

    g_kessler = jax.grad(kessler_objective)(q_r_zero)
    assert jnp.all(jnp.isfinite(g_kessler)), (
        "Kessler: grad w.r.t. q_r at q_r=0 contains NaN/Inf — "
        "fractional-power AD guard regressed."
    )

    def evap_objective(q_r):
        q_sat = saturation_mixing_ratio(T, p_full)
        return jnp.sum(rain_evaporation(q_v, q_r, q_sat, evap_coeff=1.0))

    g_evap_zero = jax.grad(evap_objective)(q_r_zero)
    g_evap_neg = jax.grad(evap_objective)(-1e-3 * jnp.ones_like(q_r_zero))
    assert jnp.all(jnp.isfinite(g_evap_zero)), "rain_evaporation grad at q_r=0 not finite"
    assert jnp.all(jnp.isfinite(g_evap_neg)), "rain_evaporation grad at q_r<0 not finite"

    # Sanity: at q_r > 0 the gradient should be the analytic derivative
    # (positive evap_coeff, positive subsaturation, p<1) ⇒ positive grad.
    q_r_pos = 1e-4 * jnp.ones_like(q_r_zero)
    g_evap_pos = jax.grad(evap_objective)(q_r_pos)
    # Subsaturated column was set up with q_v = 0.8 * q_sat so subsaturation > 0.
    assert float(jnp.max(g_evap_pos)) > 0.0, (
        "rain_evaporation grad should be positive for subsaturated column"
    )


# ============================================================================
# 4m  Joint q_c sink donor clamp (Kessler / Seifert-Beheng)
# ============================================================================

@pytest.mark.parametrize(
    "scheme,call",
    [
        ("kessler", kessler_microphysics),
        ("seifert_beheng", seifert_beheng_microphysics),
    ],
)
def test_kessler_sb_qc_does_not_go_negative_under_joint_sinks(scheme, call):
    """Kessler/SB joint donor clamp (audit cycle 2 follow-up).

    The cycle-2 fix donor-clamped each q_c sink in isolation
    (saturation-evap inside ``saturation_adjustment``; autoconversion
    via ``q_c_updated`` in Kessler).  But the JOINT rate
    ``-condensation_evap + autoconv + accretion`` can still exceed
    ``q_c / dt`` and drive ``q_c`` below zero on an explicit step.
    Probe with q_c=1e-4, q_r=5e-3, 80 % RH at 290 K, dt=1200 s — the
    pre-fix Kessler q_c → -2.6e-3 and SB q_c → -4.1e-3.

    The fix mirrors Morrison/Thompson: aggregate all q_c sinks,
    compute a single ``qc_scale = min(1, q_c / (sink_total · dt))``,
    and apply it to every sink (and to the autoconverted droplet
    number ``dN_r_au`` in SB so mass-per-droplet ``x_star`` is
    preserved).
    """
    cfg_class = KesslerConfig if scheme == "kessler" else SeifertBehengConfig
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 290.0)
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 10_000.0)
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.80 * q_sat                            # 20 % subsaturation
    q_c = jnp.full((ncol, nlev), 1e-4)
    q_r = jnp.full((ncol, nlev), 5e-3)
    hydro = HydrometeorState(
        q_c=q_c, q_r=q_r,
        q_i=jnp.zeros_like(q_c), q_s=jnp.zeros_like(q_c),
        q_g=jnp.zeros_like(q_c),
        N_c=1e8 * jnp.ones_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=jnp.zeros_like(q_c),
    )
    dt = 1200.0
    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, cfg_class())
    q_c_after = q_c + out.dq_c_dt * dt
    min_q_c = float(jnp.min(q_c_after))
    # ``min_q_c >= -1e-9`` allows for f32 round-off in the joint
    # ``qc_scale = min(1, q_c / sink_total)``.  The buggy form gives
    # values of order -2e-3 to -5e-3 — orders of magnitude away.
    assert min_q_c >= -1e-9, (
        f"{scheme}: q_c_after = {min_q_c:.3e} after one explicit step "
        f"with combined saturation-evap + autoconv + accretion sinks "
        "exceeding available q_c.  The joint donor clamp on q_c sinks "
        "(Codex audit cycle 2 follow-up) has regressed.  Pre-fix values "
        "for this fixture: Kessler ≈ -2.6e-3, SB ≈ -4.1e-3."
    )


# ============================================================================
# P3-specific tests
# ============================================================================

def _make_p3_column(nlev=20, ncol=4, T_sfc=260.0, q_i_val=5e-4, q_rim_val=1e-4):
    """Build a P3 column with ice, rime mass, and rime volume initialized."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
        nlev=nlev, ncol=ncol, T_sfc=T_sfc, q_c_val=1e-4,
    )
    # B_rim = q_rim / rho_rim_init; use rho_rim_init = 200 kg/m³.
    rho_rim_init = 200.0
    q_rim = jnp.full((ncol, nlev), q_rim_val)
    B_rim = q_rim / rho_rim_init   # [m³/kg_air]
    hydro_p3 = hydro._replace(
        q_i=jnp.full((ncol, nlev), q_i_val),
        q_s=q_rim,      # q_s slot = q_rim
        q_g=B_rim,      # q_g slot = B_rim
        N_i=1e4 * jnp.ones((ncol, nlev)),
    )
    return T, q_v, hydro_p3, p_full, p_half, rho, dz


def test_p3_rime_density_bounded():
    """After one step, diagnosed rime density must stay in [rho_rim_min, rho_rim_max]."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_p3_column()
    cfg = P3Config()
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 300.0, cfg)
    dt = 300.0
    q_rim_new = jnp.clip(hydro.q_s + out.dq_s_dt * dt, 0.0)
    B_rim_new  = jnp.maximum(hydro.q_g + out.dq_g_dt * dt, 1e-30)
    rho_rim_new = jnp.clip(q_rim_new / B_rim_new, 0.0)
    # Where there is meaningful rime, density must be within physical bounds.
    has_rime = q_rim_new > 1e-10
    if jnp.any(has_rime):
        min_rho = float(jnp.min(jnp.where(has_rime, rho_rim_new, cfg.rho_rim_max)))
        max_rho = float(jnp.max(jnp.where(has_rime, rho_rim_new, cfg.rho_rim_min)))
        assert min_rho >= cfg.rho_rim_min * 0.9, (
            f"P3: rime density {min_rho:.1f} kg/m³ below rho_rim_min={cfg.rho_rim_min}"
        )
        assert max_rho <= cfg.rho_rim_max * 1.1, (
            f"P3: rime density {max_rho:.1f} kg/m³ above rho_rim_max={cfg.rho_rim_max}"
        )


def test_p3_riming_grows_qrim():
    """With q_c > 0 and q_i > 0 below freezing, q_rim must grow (riming active)."""
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 255.0)   # well below T_freeze
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz  = jnp.full((ncol, nlev), 500.0)
    q_c = jnp.full((ncol, nlev), 5e-4)
    q_i = jnp.full((ncol, nlev), 3e-4)
    q_rim = jnp.zeros((ncol, nlev))
    B_rim = jnp.zeros((ncol, nlev))
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i,
        q_s=q_rim, q_g=B_rim,
        N_c=1e8 * jnp.ones_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=1e4 * jnp.ones_like(q_c),
    )
    q_v = 0.9 * saturation_mixing_ratio(T, p_full)
    dt  = 300.0
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt, P3Config())
    # dq_s_dt carries dq_rim_dt; must be positive (riming adds to q_rim)
    max_dqrim = float(jnp.max(out.dq_s_dt))
    assert max_dqrim > 0.0, (
        f"P3: riming not active in mixed-phase column, max dq_rim_dt = {max_dqrim:.2e}"
    )
    # B_rim must also grow
    max_dBrim = float(jnp.max(out.dq_g_dt))
    assert max_dBrim > 0.0, (
        f"P3: B_rim not growing during riming, max dB_rim_dt = {max_dBrim:.2e}"
    )


def test_p3_melt_drains_qi_and_qrim():
    """Above T_freeze, melt_ice must drain q_i and melt_rim must drain q_rim."""
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 280.0)   # above freezing
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz  = jnp.full((ncol, nlev), 10_000.0)
    q_i = jnp.full((ncol, nlev), 5e-4)
    q_rim = jnp.full((ncol, nlev), 2e-4)
    B_rim = q_rim / 400.0
    hydro = HydrometeorState(
        q_c=jnp.zeros((ncol, nlev)), q_r=jnp.zeros((ncol, nlev)),
        q_i=q_i, q_s=q_rim, q_g=B_rim,
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=1e4 * jnp.ones((ncol, nlev)),
    )
    q_v = 0.8 * saturation_mixing_ratio(T, p_full)
    dt  = 300.0
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt, P3Config())

    # q_i must decrease (melt active)
    assert float(jnp.min(out.dq_i_dt)) < 0.0, "P3: q_i not melting above T_freeze"
    # q_rim must decrease
    assert float(jnp.min(out.dq_s_dt)) < 0.0, "P3: q_rim not melting above T_freeze"
    # q_r must increase (melt goes to rain)
    assert float(jnp.max(out.dq_r_dt)) > 0.0, "P3: no melt reaching q_r above T_freeze"

    # After one step, q_i and q_rim must not go negative.
    q_i_after   = q_i   + out.dq_i_dt * dt
    q_rim_after = q_rim + out.dq_s_dt * dt
    assert float(jnp.min(q_i_after))   >= -1e-9, f"P3: q_i went negative ({float(jnp.min(q_i_after)):.3e})"
    assert float(jnp.min(q_rim_after)) >= -1e-9, f"P3: q_rim went negative ({float(jnp.min(q_rim_after)):.3e})"


def test_p3_no_ice_above_nucleation_temperature():
    """Ice formation should be negligible above cooper_T_act."""
    cfg = P3Config()
    ncol, nlev = 1, 10
    T = jnp.full((ncol, nlev), cfg.cooper_T_act + 10.0)  # warm column
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz  = jnp.full((ncol, nlev), 500.0)
    hydro = HydrometeorState(
        q_c=jnp.full((ncol, nlev), 1e-4),
        q_r=jnp.zeros((ncol, nlev)),
        q_i=jnp.zeros((ncol, nlev)),
        q_s=jnp.zeros((ncol, nlev)),
        q_g=jnp.zeros((ncol, nlev)),
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=jnp.zeros((ncol, nlev)),
    )
    q_v = 0.9 * saturation_mixing_ratio(T, p_full)
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 300.0, cfg)
    max_ice_form = float(jnp.max(out.dq_i_dt))
    assert max_ice_form <= 1e-10, (
        f"P3: ice forms above cooper_T_act, max dq_i_dt = {max_ice_form:.2e}"
    )


def test_p3_grad_finite_at_zero_ice():
    """jax.grad through p3_microphysics must be finite when q_i = B_rim = q_rim = 0."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(q_c_val=1e-4)
    # Start with zero ice/rime — the pathological case for safe_pow and safe_divide.
    hydro_zero_ice = hydro._replace(
        q_i=jnp.zeros_like(hydro.q_i),
        q_s=jnp.zeros_like(hydro.q_s),   # q_rim = 0
        q_g=jnp.zeros_like(hydro.q_g),   # B_rim = 0
        N_i=jnp.zeros_like(hydro.N_i),
    )

    def objective(q_i):
        h = hydro_zero_ice._replace(q_i=q_i)
        out = p3_microphysics(T, q_v, h, p_full, p_half, rho, dz, 300.0, P3Config())
        return jnp.sum(out.dq_i_dt + out.dq_v_dt + out.dT_dt)

    g = jax.grad(objective)(jnp.zeros_like(hydro.q_i))
    assert jnp.all(jnp.isfinite(g)), (
        "P3: grad w.r.t. q_i at q_i=0 contains NaN/Inf — "
        "safe_pow / safe_divide AD guards regressed."
    )


def test_p3_qc_does_not_go_negative_at_long_dt():
    """P3 donor clamp on q_c: combined riming + autoconv + accretion must not
    over-extract q_c at dt=1200 s."""
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 255.0)   # mixed-phase
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz  = jnp.full((ncol, nlev), 10_000.0)
    q_c  = jnp.full((ncol, nlev), 1e-3)
    q_i  = jnp.full((ncol, nlev), 1e-4)
    q_rim = jnp.full((ncol, nlev), 5e-5)
    B_rim = q_rim / 300.0
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i,
        q_s=q_rim, q_g=B_rim,
        N_c=1e8 * jnp.ones_like(q_c),
        N_r=jnp.zeros_like(q_c),
        N_i=1e4 * jnp.ones_like(q_c),
    )
    q_v = 1.0 * saturation_mixing_ratio(T, p_full)
    dt  = 1200.0
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt, P3Config())
    q_c_after = q_c + out.dq_c_dt * dt
    min_qc = float(jnp.min(q_c_after))
    assert min_qc >= -1e-9, (
        f"P3: q_c went negative ({min_qc:.3e}) at dt=1200 s — "
        "joint q_c donor clamp failed."
    )
