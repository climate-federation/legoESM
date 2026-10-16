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
from legoesm.thermo import saturation_specific_humidity


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

    q_sat = saturation_specific_humidity(T, p_full)
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


# ============================================================================
# Dispatch hardening — Thompson snow_scheme must raise on an unknown value
# ============================================================================

def test_thompson_unknown_snow_scheme_raises():
    """A typo in ``snow_scheme`` must raise, not silently fall back to the
    bulk power-law fall speed (which also leaves Thompson-2008 snow
    deposition disabled — a silent physics change)."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    with pytest.raises(ValueError, match="Unknown snow_scheme"):
        thompson_microphysics(
            T, q_v, hydro, p_full, p_half, rho, dz, 300.0,
            config=ThompsonConfig(snow_scheme="thompson_2008"),  # typo
        )


@pytest.mark.parametrize("snow_scheme", ["thompson2008", "bulk_qpower"])
def test_thompson_valid_snow_scheme_runs(snow_scheme):
    """Both supported snow schemes produce finite output."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    out = thompson_microphysics(
        T, q_v, hydro, p_full, p_half, rho, dz, 300.0,
        config=ThompsonConfig(snow_scheme=snow_scheme),
    )
    assert jnp.all(jnp.isfinite(out.dq_s_dt))


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
        if val is None:        # optional field (e.g. dN_s_dt for single-moment)
            continue
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
    """Above freezing (T > 273.15 K), ALL ice tendencies must be negligible.

    Pre-fix bug (caught here as a regression): the Cooper (1986) ice
    nucleation target ``N_i0 · exp(cooper_a · max(T_freeze − T, 0))``
    floors the temperature dependence at T = T_freeze.  Above
    freezing the exponential collapsed to 1 and the bare
    ``N_i0/rho`` target survived — so ``dN_i_nuc`` nucleated ~30
    crystals / kg / s at T = 280 K (≥ 170 / kg / s at low rho
    near the model top) even though every other ice source
    (deposition, riming, aggregation) was already gated by
    ``f_ice = sigmoid(s · (cooper_T_act − T))``.  The legacy
    assertion only checked ``dq_i_dt`` (≤ 1e-10) and missed
    ``dN_i_dt`` entirely.

    Bergeron also leaked ~1e-13 in ``dq_i_dt`` at 280 K because
    ``melt_sharpness=2`` left a non-trivial sigmoid tail above
    freezing.  Multiplying ``bergeron`` by ``f_ice`` collapses the
    tail.
    """
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(T_sfc=290.0)
    T_warm = jnp.maximum(T, 280.0)
    out = _call_scheme(scheme, T_warm, q_v, hydro, p_full, p_half, rho, dz)

    warm_mask = T_warm > constants.T_freeze
    if jnp.any(warm_mask):
        # Mass-bearing ice tendencies — must be ~0 (Bergeron fix tightens
        # tolerance from 1e-10 to 1e-15 at 280 K once f_ice gates it).
        max_ice_form = float(jnp.max(out.dq_i_dt[warm_mask]))
        assert max_ice_form <= 1e-12, (
            f"{scheme}: dq_i_dt = {max_ice_form:.2e} above freezing — "
            "Bergeron f_ice gate regressed."
        )
        max_snow_form = float(jnp.max(out.dq_s_dt[warm_mask]))
        assert max_snow_form <= 1e-12, (
            f"{scheme}: dq_s_dt = {max_snow_form:.2e} above freezing."
        )
        # NUMBER nucleation — pre-fix produced ~30-170 / kg / s; post-fix
        # f_ice gate collapses dN_i_nuc to ~0 in warm columns.  Set
        # tolerance well below any plausible signal (cold-cloud
        # nucleation runs ~1e3-1e5 / kg / s, so 1e-3 is six orders below).
        max_N_i_nuc = float(jnp.max(out.dN_i_dt[warm_mask]))
        assert max_N_i_nuc <= 1e-3, (
            f"{scheme}: dN_i_dt = {max_N_i_nuc:.2e} /kg/s above "
            "freezing — Cooper nucleation needs the f_ice gate (no "
            "natural T-cutoff in the Cooper target).  Pre-fix typical "
            "values were 30-170 /kg/s at 280-290 K."
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
    q_v = 1.0 * saturation_specific_humidity(T, p_full)
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
    q_v = 0.8 * saturation_specific_humidity(T, p_full)
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
    q_sat = saturation_specific_humidity(T, p_full)
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
    q_v = 0.5 * saturation_specific_humidity(T, p_full)
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
    q_v = 0.85 * saturation_specific_humidity(T, p_full)
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
# Sundqvist condensation magnitude: target is q_sat, not RH_crit * q_sat
# ============================================================================

def test_sundqvist_condensation_target_is_qsat_not_rhcrit_qsat():
    """Sundqvist must drive q_v toward q_sat, NOT toward RH_crit * q_sat.

    Pre-fix bug (line 63 of sundqvist.py): the condensation formula was
        condensation = f * max(q_v - RH_crit * q_sat, 0.0) / dt
    which removed any vapor above 0.8*q_sat in one step.  At RH=1.0
    this dropped RH to ~0.80 in a single 300-s call — physically wrong
    (should drop AT MOST to 1.0, since there is no supersaturation to
    remove); at RH=0.95 it still over-condensed to ~0.81.

    Post-fix: target is ``q_sat`` and only the supersaturation
    (``q_v - q_sat``) is removed.  The smooth ``f = sigmoid(s * (RH -
    RH_crit))`` gate is preserved as the *onset* control for partial-
    cloud-fraction subgrid variance, but the *target* is the
    thermodynamic equilibrium ``q_sat``.

    Cross-scheme contract: at RH=1.0 (saturated, not super) and a
    realistic q_c ~ 1e-4, Sundqvist must remove at most ``(q_v -
    q_sat) * dt`` of vapor — i.e., zero.  Kessler at the same point
    removes zero (saturation adjustment is a no-op when q_v == q_sat).
    """
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 290.0)
    p_full = jnp.full((ncol, nlev), 8e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(7e4, 9e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 1000.0)
    q_sat = saturation_specific_humidity(T, p_full)
    q_c = jnp.full((ncol, nlev), 1e-4)
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c),
        q_i=jnp.zeros_like(q_c), q_s=jnp.zeros_like(q_c),
        q_g=jnp.zeros_like(q_c),
        N_c=1e8 * jnp.ones_like(q_c), N_r=jnp.zeros_like(q_c),
        N_i=jnp.zeros_like(q_c),
    )
    dt = 300.0
    cfg = SundqvistConfig()
    k_cfg = KesslerConfig()

    # Case 1: RH = 1.0 exactly (saturated, no supersat to remove).
    q_v_sat = q_sat.copy()
    out_s = sundqvist_microphysics(
        T, q_v_sat, hydro, p_full, p_half, rho, dz, dt, cfg,
    )
    out_k = kessler_microphysics(
        T, q_v_sat, hydro, p_full, p_half, rho, dz, dt, k_cfg,
    )
    # Vapor removed must be at most ``q_v - q_sat`` * f_smooth.
    # At RH=1, q_v - q_sat = 0, so the only allowed removal is
    # float-roundoff from ``saturation_specific_humidity``.  Pre-fix this
    # assertion fired with dq_v_dt ≈ -1e-5 → 3e-3 kg/kg per step (RH
    # dropped 1.00 → 0.80); post-fix it's ~1e-9 kg/kg (f32 rounding).
    # The 1e-7 tolerance is six orders of magnitude below the pre-fix
    # bug magnitude and well above f32 ULP.
    dqv_per_step = float(jnp.max(jnp.abs(out_s.dq_v_dt))) * dt
    assert dqv_per_step < 1e-7, (
        f"Sundqvist over-condenses at RH=1.0 with q_c=1e-4: "
        f"|dq_v_dt| * dt = {dqv_per_step:.3e} kg/kg.  "
        f"Pre-fix bug: condensation target was RH_crit*q_sat=0.8*q_sat "
        f"(legoesm #316-followup); post-fix target is q_sat so "
        f"saturated columns are left alone."
    )
    # Cross-scheme: Kessler removes ~0 at this point (saturation
    # adjustment + no supersat).  Sundqvist should agree to within
    # rounding — same physical situation, same answer.  Use a 1e-7
    # / dt tendency floor (≡ 3e-10 kg/kg/s at dt=300 s) so f32
    # roundoff in saturation_specific_humidity doesn't trip the test
    # while the pre-fix bug (|dq_v_dt|·dt ≈ 3e-3) is caught with
    # six orders of magnitude of margin.
    atol_tend = 1e-7 / dt
    assert bool(jnp.all(jnp.abs(out_s.dq_v_dt) < atol_tend)), (
        f"Sundqvist removes vapor at exactly RH=1.0 (no supersat): "
        f"max(|dq_v_dt|)={float(jnp.max(jnp.abs(out_s.dq_v_dt))):.3e}"
    )
    assert bool(jnp.all(jnp.abs(out_k.dq_v_dt) < atol_tend)), (
        f"Kessler removes vapor at exactly RH=1.0 (no supersat): "
        f"max(|dq_v_dt|)={float(jnp.max(jnp.abs(out_k.dq_v_dt))):.3e}"
    )

    # Case 2: RH = 0.95 (sub-saturated but above RH_crit=0.8).  No
    # supersat → no condensation.  Pre-fix Sundqvist would still
    # over-condense and drop RH to ~0.81.
    q_v_sub = 0.95 * q_sat
    out_s_sub = sundqvist_microphysics(
        T, q_v_sub, hydro, p_full, p_half, rho, dz, dt, cfg,
    )
    rh_after = float(
        (q_v_sub[0, 0] + out_s_sub.dq_v_dt[0, 0] * dt) / q_sat[0, 0]
    )
    assert rh_after > 0.94, (
        f"Sundqvist over-condensed in subsaturated column (RH=0.95 → "
        f"RH_after={rh_after:.3f}).  No supersat to remove — q_v must "
        f"stay close to 0.95*q_sat."
    )

    # Case 3: RH = 1.20 (supersaturated).  Supersat removal must
    # bring RH back to ~1.0 (within sigmoid smoothing).
    q_v_super = 1.20 * q_sat
    out_s_super = sundqvist_microphysics(
        T, q_v_super, hydro, p_full, p_half, rho, dz, dt, cfg,
    )
    rh_after_super = float(
        (q_v_super[0, 0] + out_s_super.dq_v_dt[0, 0] * dt) / q_sat[0, 0]
    )
    assert 0.99 < rh_after_super < 1.01, (
        f"Sundqvist failed to remove supersaturation at RH=1.20 → "
        f"RH_after={rh_after_super:.3f} (expected ~1.0)."
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
    q_sat = saturation_specific_humidity(T, p_full)
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
    q_v_sub = 0.5 * saturation_specific_humidity(T, p_full)
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
        q_sat = saturation_specific_humidity(T, p_full)
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
    q_sat = saturation_specific_humidity(T, p_full)
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
    q_v = 0.9 * saturation_specific_humidity(T, p_full)
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
    q_v = 0.8 * saturation_specific_humidity(T, p_full)
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
    q_v = 0.9 * saturation_specific_humidity(T, p_full)
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 300.0, cfg)
    # Pre-fix: Cooper nucleation in P3 had no f_ice gate (same bug as
    # morrison.py / thompson.py).  dN_i_dt reached ~28 / kg / s at
    # T = cooper_T_act + 25 K even though dq_i_dt stayed ~0 because no
    # ice mass had nucleated yet (the bug only shows up in dN_i_dt
    # until the ice number sources downstream physics).
    max_ice_form = float(jnp.max(out.dq_i_dt))
    assert max_ice_form <= 1e-12, (
        f"P3: ice forms above cooper_T_act, max dq_i_dt = {max_ice_form:.2e}"
    )
    max_N_i_nuc = float(jnp.max(out.dN_i_dt))
    assert max_N_i_nuc <= 1e-3, (
        f"P3: dN_i_dt = {max_N_i_nuc:.2e} /kg/s above cooper_T_act — "
        "Cooper nucleation needs the f_ice gate."
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
    q_v = 1.0 * saturation_specific_humidity(T, p_full)
    dt  = 1200.0
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt, P3Config())
    q_c_after = q_c + out.dq_c_dt * dt
    min_qc = float(jnp.min(q_c_after))
    assert min_qc >= -1e-9, (
        f"P3: q_c went negative ({min_qc:.3e}) at dt=1200 s — "
        "joint q_c donor clamp failed."
    )


def test_p3_qr_does_not_go_negative_under_joint_rain_rime_and_sedimentation():
    """P3 q_r positivity under joint rain_rime + sedimentation drain.

    Pathological column: heavy rain (q_r=5e-3) + heavy ice (q_i=1e-3)
    below freezing in thin layers (dz=500 m) at long dt (1200 s).
    Sinks of q_r in this step are evaporation, rain_rime (q_r is
    collected by q_i and frozen into rime), and sedimentation flux.

    Pre-fix: ``sedimentation_tendency`` for q_r received only
    ``extra_sink=evaporation``.  The sed cap became
    ``(q_r − evap·dt) · ρ·dz / dt`` — sedimentation removed the
    remaining ``q_r − evap·dt`` per step while rain_rime drained
    its own share on top, summing to ``q_r + rain_rime·dt`` and
    driving q_r to ≈ −rain_rime·dt < 0.  Probe (PR #321 adversarial
    review): q_r ended at −5.4e-4 kg/kg.

    Post-fix: ``extra_sink`` is ``evaporation + rain_rime``, so the
    joint per-step drain is bounded by available q_r and the
    explicit Euler step never drives q_r below zero.
    """
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 260.0)
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz  = jnp.full((ncol, nlev), 500.0)
    q_r = jnp.full((ncol, nlev), 5e-3)
    q_i = jnp.full((ncol, nlev), 1e-3)
    q_rim = jnp.full((ncol, nlev), 5e-4)
    B_rim = q_rim / 400.0
    hydro = HydrometeorState(
        q_c=jnp.zeros((ncol, nlev)),
        q_r=q_r, q_i=q_i,
        q_s=q_rim, q_g=B_rim,
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=1e4 * jnp.ones((ncol, nlev)),
    )
    q_v = 0.99 * saturation_specific_humidity(T, p_full)
    dt  = 1200.0
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt, P3Config())
    q_r_after = q_r + out.dq_r_dt * dt
    min_qr = float(jnp.min(q_r_after))
    assert min_qr >= -1e-9, (
        f"P3: q_r went negative ({min_qr:.3e}) at dt=1200 s under "
        "joint rain_rime + sedimentation drain.  The sedimentation "
        "extra_sink must include rain_rime (PR #321 adversarial review)."
    )


def test_p3_column_water_budget_closes():
    """Column-integrated mass-bearing tendencies + precipitation ≈ 0.

    Mass-bearing tendencies in P3: ``dq_v + dq_c + dq_r + dq_i +
    dq_rim``.  ``dB_rim_dt`` excluded — B_rim is volume per kg air,
    not a mass mixing ratio; summing it into the water budget is a
    dimensional error.  Mass closure in a closed column:

        ∫ (dq_v + dq_c + dq_r + dq_i + dq_rim) dp/g + precip = 0
    """
    ncol, nlev = 4, 20
    T_sfc, p_s_val = 260.0, 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to(
        (sigma_half * p_s_val)[None, :], (ncol, nlev + 1),
    )
    p_full = jnp.broadcast_to(
        (sigma_full * p_s_val)[None, :], (ncol, nlev),
    )
    T = jnp.broadcast_to(
        jnp.maximum(T_sfc * jnp.clip(sigma_full, 0.01) ** 0.19, 200.0)[None, :],
        (ncol, nlev),
    )
    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * p_mid))
    q_sat = saturation_specific_humidity(T, p_full)
    q_v = 0.9 * q_sat
    q_c = jnp.zeros((ncol, nlev)).at[..., -8:-3].set(5e-4)
    q_r = jnp.zeros((ncol, nlev)).at[..., -3:].set(5e-4)
    q_i = jnp.zeros((ncol, nlev)).at[..., :10].set(3e-4)
    q_rim = q_i * 0.3
    B_rim = q_rim / 300.0
    hydro = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=q_i, q_s=q_rim, q_g=B_rim,
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=1e4 * jnp.ones((ncol, nlev)),
    )
    dt = 300.0
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt, P3Config())

    # ``out.dq_g_dt`` carries dB_rim_dt [m³/kg_air/s] — not a
    # water-mass tendency; excluded from the budget per the
    # moist_mass_fixer.py slot-reuse warning at lines 117-128.
    total_dq = (
        out.dq_v_dt + out.dq_c_dt + out.dq_r_dt
        + out.dq_i_dt + out.dq_s_dt
    )
    col_tend = jnp.sum(total_dq * dp, axis=1) / constants.g
    imbalance = col_tend + out.precipitation
    max_imbalance = float(jnp.max(jnp.abs(imbalance)))
    max_precip = float(jnp.max(jnp.abs(out.precipitation)))
    # Absolute tolerance: column water tendency scale ~q/dt·column_mass
    # ~5e-4/300·1 ≈ 1.7e-6 kg/m²/s; 1e-9 floor is six orders below
    # per-step activity while still trapping unit-error bugs.
    assert max_imbalance < 1e-9, (
        f"P3: column water budget unclosed.  max |imbalance| = "
        f"{max_imbalance:.3e} kg/m²/s, max precip = {max_precip:.3e}."
    )


def test_p3_melting_transfers_number_to_rain():
    """Melting must transfer NUMBER as well as mass (Morrison & Milbrandt
    2015): dN_i sink and dN_r source in proportion to the melted mass
    fraction.  Pre-fix, q_i melted away while N_i stayed fixed (mean crystal
    mass q_i/N_i collapsed) and the melted crystals never appeared in N_r.
    """
    ncol, nlev = 1, 5
    T = jnp.full((ncol, nlev), 280.0)   # above freezing: melt-only ice path
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz  = jnp.full((ncol, nlev), 10_000.0)
    q_i = jnp.full((ncol, nlev), 5e-4)
    hydro = HydrometeorState(
        q_c=jnp.zeros((ncol, nlev)), q_r=jnp.zeros((ncol, nlev)),
        q_i=q_i, q_s=jnp.zeros((ncol, nlev)), q_g=jnp.zeros((ncol, nlev)),
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=1e4 * jnp.ones((ncol, nlev)),   # [1/kg]
    )
    q_v = 0.8 * saturation_specific_humidity(T, p_full)
    dt  = 300.0
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt, P3Config())
    # Melting is active (mass moves ice -> rain) ...
    assert float(jnp.min(out.dq_i_dt)) < 0.0
    # ... and so does NUMBER: N_i sink, N_r source.
    assert float(jnp.max(out.dN_i_dt)) < 0.0, (
        "P3: melting removed ice mass but not ice number (dN_i melt sink missing)."
    )
    assert float(jnp.min(out.dN_r_dt)) > 0.0, (
        "P3: melted crystals never appear as rain drops (dN_r melt source missing)."
    )
    # In this warm setup melting is the ONLY number pathway (nucleation and
    # aggregation are f_ice-gated off; q_c = N_r = 0 kills the warm-rain
    # number terms), so the rain-number source must equal the ice-number
    # sink converted per-mass [1/kg] -> per-volume [1/m^3] with rho.
    assert bool(jnp.allclose(out.dN_r_dt, -out.dN_i_dt * rho, rtol=1e-3)), (
        "P3: melt number transfer ice->rain is not conservative "
        "(dN_r_dt != -dN_i_dt * rho)."
    )


def test_p3_melt_number_bounded_and_conservative_for_tiny_crystals():
    """The melt-number transfer is bounded to the available N_i and the SAME
    count is removed from ice / added to rain even when the unbounded melt rate
    (melt_ice*N_i/q_i) would exceed N_i/dt (tiny mean crystal mass).  Guards the
    codex fix: the net-N_i floor must not shrink the ice sink without shrinking
    the rain source (which would mint spurious rain drops) and N_i must stay
    >= 0 after the step.
    """
    ncol, nlev = 1, 4
    T = jnp.full((ncol, nlev), 282.0)            # above freezing: melt-only
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 10_000.0)
    q_i = jnp.full((ncol, nlev), 5e-4)
    hydro = HydrometeorState(
        q_c=jnp.zeros((ncol, nlev)), q_r=jnp.zeros((ncol, nlev)),
        q_i=q_i, q_s=jnp.zeros((ncol, nlev)), q_g=jnp.zeros((ncol, nlev)),
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=1e9 * jnp.ones((ncol, nlev)),        # huge N_i, tiny mean mass
    )
    q_v = 0.8 * saturation_specific_humidity(T, p_full)
    # Cover a normal step AND a subsecond LES step (dt < 1): the mass<->number
    # cap must key off the PHYSICAL dt, not clip(dt, 1.0), else subsecond steps
    # melt all the mass but transfer only dt*N_i crystals (stale ice number).
    for dt in (300.0, 0.5):
        out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                              P3Config())
        # Ice-number sink and rain-number source stay exactly equal (per-mass ->
        # per-volume with rho) even though the raw melt rate is capped.
        assert bool(jnp.allclose(out.dN_r_dt, -out.dN_i_dt * rho, rtol=1e-3)), (
            f"P3: bounded melt transfer broke ice->rain number equality (dt={dt})."
        )
        # Post-step ice number is non-negative (bound respected).
        N_i_next = hydro.N_i + out.dN_i_dt * dt
        assert float(jnp.min(N_i_next)) >= -1e-6, (
            f"P3: melt bound let N_i go negative (dt={dt})."
        )
        # Subsecond step must still melt a MEANINGFUL fraction of the crystals
        # when all the mass melts (not throttled to ~dt*N_i by a clip-at-1 cap).
        if dt < 1.0:
            frac_removed = float(-jnp.max(out.dN_i_dt) * dt / 1e9)
            assert frac_removed > 1e-3, (
                "P3: subsecond melt transferred negligible ice number "
                "(dt_floor over-throttled the cap)."
            )


def test_p3_rain_riming_sinks_rain_number():
    """Rain riming (rain mass collected onto ice) must remove rain NUMBER in
    proportion to the rimed rain-mass fraction (whole drops leave the rain
    category).  Controlled comparison: adding ice (riming on) vs no ice
    (riming off) at fixed rain leaves every warm-rain number term identical, so
    the delta in dN_r_dt must equal -dN_r_rime and track the delta in dq_r_dt at
    the mean rain mass q_r/N_r.
    """
    ncol, nlev = 1, 4
    T = jnp.full((ncol, nlev), 262.0)            # cold: no melt, riming active
    p_full = jnp.full((ncol, nlev), 6e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(5e4, 7e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 8_000.0)
    q_r = jnp.full((ncol, nlev), 3e-4)
    N_r = jnp.full((ncol, nlev), 5e3)            # [1/kg]
    # At liquid saturation rain evaporation ~ 0, so the rain-mass sink budget is
    # not donor-saturated and rain_rime shows up directly in dq_r_dt.
    q_v = saturation_specific_humidity(T, p_full)
    dt = 300.0
    base_kw = dict(
        q_c=jnp.zeros((ncol, nlev)), q_r=q_r, q_s=jnp.zeros((ncol, nlev)),
        q_g=jnp.zeros((ncol, nlev)), N_c=jnp.zeros((ncol, nlev)), N_r=N_r,
    )
    no_ice = HydrometeorState(q_i=jnp.zeros((ncol, nlev)),
                              N_i=jnp.zeros((ncol, nlev)), **base_kw)
    with_ice = HydrometeorState(q_i=jnp.full((ncol, nlev), 5e-4),
                                N_i=jnp.full((ncol, nlev), 1e4), **base_kw)
    cfg = P3Config()
    o0 = p3_microphysics(T, q_v, no_ice, p_full, p_half, rho, dz, dt, cfg)
    o1 = p3_microphysics(T, q_v, with_ice, p_full, p_half, rho, dz, dt, cfg)
    d_qr = o1.dq_r_dt - o0.dq_r_dt          # extra rain-mass sink from riming
    d_Nr = o1.dN_r_dt - o0.dN_r_dt          # must be a matching number sink
    assert float(jnp.min(-d_qr)) > 0.0, "no rain riming fired in the with-ice run"
    assert float(jnp.max(d_Nr)) < 0.0, (
        "P3: rain riming removed rain mass but not rain number (stale N_r)."
    )
    # Whole-drop removal: dN_r_rime / rain_rime == N_r / q_r.
    assert bool(jnp.allclose(d_Nr / d_qr, N_r / q_r, rtol=1e-3)), (
        "P3: rain-riming number sink is not at the mean rain mass q_r/N_r."
    )


def test_p3_nucleation_seeds_ice_mass_from_vapor():
    """Ice nucleation must add the seed mass m_i0 per crystal to q_i, taken
    from q_v with L_s heating (mirrors morrison.py MNUCCD = NNUCCD*MI0).
    Pre-fix nucleation was number-only: q_i/N_i -> 0 right after nucleation.
    """
    from legoesm.atmosphere.physics.microphysics import p3 as p3_mod
    from legoesm.thermo import saturation_specific_humidity_ice

    cfg = P3Config()
    ncol, nlev = 1, 4
    T = jnp.full((ncol, nlev), 230.0)   # cold: f_ice ~ 1
    p_full = jnp.full((ncol, nlev), 3e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(2e4, 4e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz  = jnp.full((ncol, nlev), 500.0)
    # Ice-supersaturated but liquid-subsaturated (e_sw/e_si ~ 1.5 at 230 K),
    # with NO pre-existing ice: deposition needs N_i > 0 so it is exactly 0,
    # leaving nucleation as the only vapour->ice pathway.
    q_sat_i = saturation_specific_humidity_ice(T, p_full)
    q_v = 1.2 * q_sat_i
    hydro = HydrometeorState(
        q_c=jnp.zeros((ncol, nlev)), q_r=jnp.zeros((ncol, nlev)),
        q_i=jnp.zeros((ncol, nlev)), q_s=jnp.zeros((ncol, nlev)),
        q_g=jnp.zeros((ncol, nlev)),
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=jnp.zeros((ncol, nlev)),
    )
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 300.0, cfg)
    assert float(jnp.min(out.dN_i_dt)) > 0.0, "no Cooper nucleation fired"
    # Mass source = number source x seed mass m_i0 (only active ice term).
    assert bool(jnp.allclose(out.dq_i_dt, out.dN_i_dt * p3_mod._M_I0,
                             rtol=1e-6)), (
        "P3: nucleated crystals carry no seed mass (dq_i_nuc = dN_i_nuc*m_i0 "
        "missing)."
    )
    assert float(jnp.min(out.dq_i_dt)) > 0.0
    # Seed mass comes FROM vapour and heats with L_s (budget closes).
    assert bool(jnp.allclose(out.dq_v_dt, -out.dq_i_dt, rtol=1e-6)), (
        "P3: nucleation seed mass not removed from vapour."
    )
    assert bool(jnp.allclose(
        out.dT_dt, constants.L_s * out.dq_i_dt / constants.c_pd, rtol=1e-6,
    )), "P3: nucleation seed mass missing its L_s heating."


def test_p3_riming_and_accretion_sink_cloud_number():
    """Riming (and accretion) sweep up WHOLE cloud droplets: N_c must sink in
    proportion to the cloud mass consumed (rate * N_c/q_c), not just via
    autoconversion (SAM NPSACWS/NPRA; mirrors morrison.py dN_c_riming).
    """
    cfg = P3Config()
    ncol, nlev = 1, 4
    T_val = 255.0
    T = jnp.full((ncol, nlev), T_val)   # mixed phase: riming active
    p_full = jnp.full((ncol, nlev), 5e4)
    p_half = jnp.broadcast_to(
        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (constants.R_d * T)
    dz  = jnp.full((ncol, nlev), 500.0)
    q_c = jnp.full((ncol, nlev), 1e-4)
    q_i = jnp.full((ncol, nlev), 3e-4)
    # HIGH droplet number => mean droplet mass x_c << x_star => the SB
    # autoconversion onset sigmoid ~ 0, isolating the riming N_c sink.
    N_c = 1e9 * jnp.ones((ncol, nlev))
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros((ncol, nlev)), q_i=q_i,
        q_s=jnp.zeros((ncol, nlev)), q_g=jnp.zeros((ncol, nlev)),
        N_c=N_c, N_r=jnp.zeros((ncol, nlev)),
        N_i=1e4 * jnp.ones((ncol, nlev)),
    )
    q_v = saturation_specific_humidity(T, p_full)   # saturated: cond ~ 0
    out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, 300.0, cfg)
    # Closed form: riming = rime_coeff*q_i*q_c*f_ice, N_c sink = riming*N_c/q_c.
    f_ice = jax.nn.sigmoid(cfg.ice_sigmoid_sharpness * (cfg.cooper_T_act - T_val))
    expected = -cfg.rime_coeff * q_i * f_ice * N_c
    assert float(jnp.max(out.dN_c_dt)) < 0.0, (
        "P3: riming consumes cloud mass without any N_c sink."
    )
    assert bool(jnp.allclose(out.dN_c_dt, expected, rtol=0.05)), (
        f"P3: dN_c riming sink {float(out.dN_c_dt[0, 0]):.3e} != expected "
        f"{float(expected[0, 0]):.3e} (riming * N_c/q_c)."
    )
    # Adding rain switches on accretion, which must deepen the N_c sink.
    hydro_qr = hydro._replace(q_r=jnp.full((ncol, nlev), 5e-4))
    out_qr = p3_microphysics(T, q_v, hydro_qr, p_full, p_half, rho, dz, 300.0, cfg)
    assert float(jnp.max(out_qr.dN_c_dt - out.dN_c_dt)) < 0.0, (
        "P3: accretion of cloud water by rain does not sink N_c."
    )


# ============================================================================
# Kessler saturation adjustment routes through the shared fp64 helper (#618)
# ============================================================================

def test_kessler_saturation_adjustment_uses_shared_fp64_helper():
    """Kessler must route saturation adjustment through
    ``_warm_rain.saturation_adjustment`` (which computes q_sat AND the
    supersaturation residual in fp64 when x64 is available — issue #618)
    plus its extra positive q_v donor clamp, NOT an inline fp32 copy.

    The fixture is a marginal supersaturation (5e-6 relative), where the
    fp32-computed q_sat carries an absolute error comparable to the residual
    itself — exactly the regime where the pre-fix inline fp32 copy diverges
    from the shared helper.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires JAX_ENABLE_X64=1 for the #618 fp64 q_sat path")
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        saturation_adjustment,
    )

    ncol, nlev = 1, 4
    T32 = jnp.full((ncol, nlev), 285.0, dtype=jnp.float32)
    p32 = jnp.full((ncol, nlev), 9e4, dtype=jnp.float32)
    p_half32 = jnp.broadcast_to(
        jnp.linspace(8.5e4, 9.5e4, nlev + 1, dtype=jnp.float32)[None, :],
        (ncol, nlev + 1),
    )
    rho32 = (p32 / (constants.R_d * T32)).astype(jnp.float32)
    dz32 = jnp.full((ncol, nlev), 500.0, dtype=jnp.float32)
    # Marginal supersaturation relative to the fp64 q_sat of the STORED
    # fp32 (T, p) — the #618 regime.
    q_sat64 = saturation_specific_humidity(
        T32.astype(jnp.float64), p32.astype(jnp.float64),
    )
    q_v32 = (q_sat64 * (1.0 + 5.0e-6)).astype(jnp.float32)
    q_c32 = jnp.full((ncol, nlev), 1e-4, dtype=jnp.float32)
    z32 = jnp.zeros((ncol, nlev), dtype=jnp.float32)
    hydro = HydrometeorState(
        q_c=q_c32, q_r=z32, q_i=z32, q_s=z32, q_g=z32,
        N_c=z32, N_r=z32, N_i=z32,
    )
    dt = 300.0
    cfg = KesslerConfig()

    cond_ref, _ = saturation_adjustment(
        T32, q_v32, p32, jnp.maximum(dt, 1e-10),
        cfg.saturation_sharpness, q_c=q_c32,
    )
    cond_ref = jnp.minimum(cond_ref, jnp.clip(q_v32, 0.0) / dt)

    out = kessler_microphysics(
        T32, q_v32, hydro, p32, p_half32, rho32, dz32, dt, cfg,
    )
    # The marginal supersaturation actually condenses under the fp64 path.
    assert float(jnp.min(cond_ref)) > 0.0
    # Positive saturation-adjustment condensation is exported unscaled as
    # dq_v_to_qc_dt — it must be the shared helper's value exactly.
    assert bool(jnp.allclose(
        out.dq_v_to_qc_dt, jnp.maximum(cond_ref, 0.0), rtol=1e-6, atol=0.0,
    )), (
        "Kessler saturation adjustment diverges from the shared "
        "_warm_rain.saturation_adjustment helper — the #618 fp64 q_sat "
        "routing regressed."
    )
    # Nothing fp64 leaks into the fp32 state tendencies.
    assert out.dq_c_dt.dtype == jnp.float32


# ============================================================================
# Sundqvist SBK89 coalescence + Bergeron precipitation-release enhancement
# ============================================================================

def _sundqvist_two_layer(T_val, q_c_col, q_v_frac=0.5):
    """Minimal 2-layer sundqvist fixture (level 0 = top, level 1 = bottom)."""
    ncol, nlev = 1, 2
    T = jnp.full((ncol, nlev), T_val)
    p_full = jnp.broadcast_to(jnp.array([[5e4, 7e4]]), (ncol, nlev))
    p_half = jnp.broadcast_to(jnp.array([[4e4, 6e4, 8e4]]), (ncol, nlev + 1))
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 1000.0)
    q_sat = saturation_specific_humidity(T, p_full)
    q_v = q_v_frac * q_sat   # below rh_crit: no condensation feeds q_c
    q_c = jnp.asarray(q_c_col)[None, :]
    hydro = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(q_c),
        q_i=jnp.zeros_like(q_c), q_s=jnp.zeros_like(q_c),
        q_g=jnp.zeros_like(q_c),
        N_c=jnp.zeros_like(q_c), N_r=jnp.zeros_like(q_c),
        N_i=jnp.zeros_like(q_c),
    )
    return T, q_v, hydro, p_full, p_half, rho, dz


def test_sundqvist_sbk89_coalescence_enhances_release_below_precip():
    """SBK89 F1: precipitation falling in from above accelerates the
    precipitation release in the layer below (coalescence).  Two columns
    with IDENTICAL q_c in the bottom layer, one with a precipitating layer
    above: the bottom-layer autoconversion must be larger under the falling
    precipitation, and must reduce to the base Sundqvist form when nothing
    falls in from above.
    """
    from legoesm.atmosphere.physics.microphysics.sundqvist import (
        diagnose_sundqvist_process_rates,
    )
    cfg = SundqvistConfig()
    dt = 300.0
    q_c_low = 2e-4
    # Warm column: Bergeron F2 ~ 1, isolating the coalescence term.
    args_precip = _sundqvist_two_layer(290.0, [1e-3, q_c_low])
    args_clear = _sundqvist_two_layer(290.0, [0.0, q_c_low])
    r_precip = diagnose_sundqvist_process_rates(*args_precip, dt=dt, config=cfg)
    r_clear = diagnose_sundqvist_process_rates(*args_clear, dt=dt, config=cfg)
    auto_below_precip = float(r_precip.autoconversion[0, 1])
    auto_below_clear = float(r_clear.autoconversion[0, 1])
    assert auto_below_precip > 1.5 * auto_below_clear, (
        f"SBK89 coalescence enhancement missing: autoconversion below a "
        f"precipitating layer ({auto_below_precip:.3e}) should exceed the "
        f"no-precip value ({auto_below_clear:.3e})."
    )
    # No precip from above => F1 = 1 => base Sundqvist release exactly.
    base = float(jnp.minimum(
        cfg.auto_rate * q_c_low
        * (1.0 - jnp.exp(-(q_c_low / cfg.qc_crit) ** 2)),
        q_c_low / dt,
    ))
    assert auto_below_clear == pytest.approx(base, rel=1e-6), (
        "top-of-precip layer must reduce to the base Sundqvist release "
        "(F1 = 1 with zero incoming flux; warm F2 ~ 1)."
    )


def test_sundqvist_sbk89_bergeron_enhances_mixed_phase_release():
    """SBK89 F2: the same cloud water releases precipitation faster in the
    mixed-phase Bergeron window (~-15 C) than in warm cloud, following
    F2 = 1 + c2*exp(-((T - T_peak)/T_width)^2) applied to both the rate and
    the threshold.
    """
    from legoesm.atmosphere.physics.microphysics.sundqvist import (
        diagnose_sundqvist_process_rates,
    )
    cfg = SundqvistConfig()
    dt = 300.0
    q_c_val = 2e-4
    T_cold = float(cfg.bergeron_T_peak_K)
    r_cold = diagnose_sundqvist_process_rates(
        *_sundqvist_two_layer(T_cold, [0.0, q_c_val]), dt=dt, config=cfg,
    )
    r_warm = diagnose_sundqvist_process_rates(
        *_sundqvist_two_layer(295.0, [0.0, q_c_val]), dt=dt, config=cfg,
    )
    auto_cold = float(r_cold.autoconversion[0, 1])
    auto_warm = float(r_warm.autoconversion[0, 1])
    assert auto_cold > 2.0 * auto_warm, (
        f"SBK89 Bergeron enhancement missing: mixed-phase autoconversion "
        f"({auto_cold:.3e}) should exceed warm-cloud ({auto_warm:.3e})."
    )
    # Closed form at the Gaussian peak: enh = 1 + c2 (F1 = 1, no precip above).
    enh = 1.0 + cfg.bergeron_enh_coeff
    expected_cold = float(jnp.minimum(
        cfg.auto_rate * enh * q_c_val
        * (1.0 - jnp.exp(-(q_c_val * enh / cfg.qc_crit) ** 2)),
        q_c_val / dt,
    ))
    assert auto_cold == pytest.approx(expected_cold, rel=1e-6)


def test_sundqvist_sbk89_enhancements_off_recover_base_scheme():
    """coalescence_enh_coeff = bergeron_enh_coeff = 0 must reproduce the
    plain Sundqvist (1989) base autoconversion everywhere (F1 = F2 = 1),
    including under falling precipitation and in cold layers.
    """
    from legoesm.atmosphere.physics.microphysics.sundqvist import (
        diagnose_sundqvist_process_rates,
    )
    cfg_off = SundqvistConfig(
        coalescence_enh_coeff=0.0, bergeron_enh_coeff=0.0,
    )
    dt = 300.0
    args = _sundqvist_two_layer(258.15, [1e-3, 2e-4])
    rates = diagnose_sundqvist_process_rates(*args, dt=dt, config=cfg_off)
    _, _, hydro, _, _, _, _ = args
    qc_avail = jnp.maximum(hydro.q_c + rates.condensation * dt, 0.0)
    base = jnp.minimum(
        cfg_off.auto_rate * qc_avail
        * (1.0 - jnp.exp(-(qc_avail / cfg_off.qc_crit) ** 2)),
        qc_avail / dt,
    )
    assert bool(jnp.allclose(rates.autoconversion, base, rtol=1e-12)), (
        "with both SBK89 coefficients zeroed the release must equal the "
        "base Sundqvist form exactly."
    )


def test_sundqvist_sbk89_grad_finite_at_zero_precip():
    """The F1 = 1 + c1*sqrt(P_above) coalescence term must be AD-safe at
    P_above = 0 (clear columns): safe_pow guards the sqrt(0) trap.
    """
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(q_c_val=0.0)

    def objective(q_c):
        h = hydro._replace(q_c=q_c)
        out = sundqvist_microphysics(
            T, q_v, h, p_full, p_half, rho, dz, 300.0, SundqvistConfig(),
        )
        return jnp.sum(out.precipitation) + jnp.sum(out.dq_c_dt)

    g = jax.grad(objective)(jnp.zeros_like(hydro.q_c))
    assert bool(jnp.all(jnp.isfinite(g))), (
        "Sundqvist SBK89 coalescence sqrt(P) gradient not finite at "
        "zero precipitation."
    )
