"""Morrison vs P3 comparison tests.

Both schemes share the same Seifert-Beheng warm-rain helpers and Cooper
ice nucleation, so they should agree closely on liquid-phase tendencies
and produce physically consistent (if not identical) ice-phase behaviour.

Tests are deliberately loose on tolerance where the two schemes are
expected to differ (e.g. Morrison has separate snow/aggregation Bergeron
conversion; P3 uses continuous rime density), and tight where they must
agree by construction (e.g. identical liquid-phase helpers).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig, P3Config
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


# ---------------------------------------------------------------------------
# Shared column builder
# ---------------------------------------------------------------------------

def _make_column(
    nlev: int = 20,
    ncol: int = 4,
    T_sfc: float = 280.0,
    q_c_val: float = 1e-4,
    q_i_val: float = 0.0,
    q_r_val: float = 0.0,
    q_rim_val: float = 0.0,
    B_rim_val: float = 0.0,
    supersaturated: bool = False,
    dt: float = 300.0,
):
    """Build a realistic sigma-coordinate column for both schemes."""
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))

    T = T_sfc * jnp.clip(sigma_full, 0.01) ** 0.19
    T = jnp.maximum(T, 200.0)
    T = jnp.broadcast_to(T[None, :], (ncol, nlev))

    rho = p_full / (constants.R_d * jnp.clip(T, 1.0))

    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0)))

    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = (1.2 if supersaturated else 0.8) * q_sat

    q_c = jnp.zeros((ncol, nlev)).at[..., -5:].set(q_c_val)
    q_r = jnp.zeros((ncol, nlev)).at[..., -5:].set(q_r_val)
    q_i = jnp.zeros((ncol, nlev)).at[..., :10].set(q_i_val)

    # Morrison: q_s = snow, q_g = 0
    hydro_morrison = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=q_i,
        q_s=jnp.zeros((ncol, nlev)),
        q_g=jnp.zeros((ncol, nlev)),
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=jnp.zeros((ncol, nlev)),
    )
    # P3: q_s slot = q_rim, q_g slot = B_rim
    hydro_p3 = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=q_i,
        q_s=jnp.zeros((ncol, nlev)).at[..., :10].set(q_rim_val),
        q_g=jnp.zeros((ncol, nlev)).at[..., :10].set(B_rim_val),
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=jnp.zeros((ncol, nlev)),
    )
    return T, q_v, hydro_morrison, hydro_p3, p_full, p_half, rho, dz


def _run_both(T, q_v, hydro_morrison, hydro_p3, p_full, p_half, rho, dz, dt=300.0,
              morrison_config=None, p3_config=None):
    out_m = morrison_microphysics(
        T, q_v, hydro_morrison, p_full, p_half, rho, dz, dt,
        morrison_config if morrison_config is not None else MorrisonConfig(),
    )
    out_p = p3_microphysics(
        T, q_v, hydro_p3, p_full, p_half, rho, dz, dt,
        p3_config if p3_config is not None else P3Config(),
    )
    return out_m, out_p


# ---------------------------------------------------------------------------
# 1. Liquid-phase tendencies agree (identical SB helpers, no ice)
# ---------------------------------------------------------------------------

def test_warm_rain_tendencies_agree():
    """With no ice present and BOTH schemes on the Seifert-Beheng warm-rain
    helpers, liquid-phase tendencies (dq_c_dt, dq_r_dt, dq_v_dt) should be
    nearly identical. Morrison now DEFAULTS to KK2000 (SAM M2005), so this
    cross-check explicitly selects ``warm_rain_scheme='seifert_beheng'`` —
    the KK2000 path is covered by tests/unit/test_kk2000_warm_rain.py."""
    T, q_v, hm, hp, p_full, p_half, rho, dz = _make_column(
        T_sfc=295.0, q_c_val=5e-4, q_i_val=0.0,
    )
    out_m, out_p = _run_both(
        T, q_v, hm, hp, p_full, p_half, rho, dz,
        # N_i0=0 disables Cooper nucleation ON BOTH SCHEMES so the vapour
        # budget reflects only the (shared SB) warm-rain liquid processes,
        # not the ice-nucleation seed-mass vapour sink at cold upper levels
        # (iter-9; P3 now carries the same dq_i_nuc = dN_i_nuc*m_i0 vapour
        # sink as Morrison, so it must be zeroed symmetrically).
        morrison_config=MorrisonConfig(
            warm_rain_scheme="seifert_beheng", N_i0=0.0,
        ),
        p3_config=P3Config(N_i0=0.0),
    )

    for field in ("dq_c_dt", "dq_r_dt", "dq_v_dt"):
        vm = getattr(out_m, field)
        vp = getattr(out_p, field)
        scale = jnp.maximum(jnp.abs(vm), 1e-12)
        rel_err = jnp.max(jnp.abs(vm - vp) / scale)
        assert float(rel_err) < 0.01, (
            f"{field}: Morrison vs P3 rel_err = {float(rel_err):.2e} "
            f"(expected < 1% — both use identical SB warm-rain helpers)"
        )


# ---------------------------------------------------------------------------
# 2. Both conserve total water mass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_sfc,q_i_val", [
    (290.0, 0.0),     # warm column, no ice
    (260.0, 5e-5),    # cold column with ice
])
def test_total_water_conservation(T_sfc, q_i_val):
    """Sum of water tendencies (v+c+r+i) must balance precipitation loss."""
    T, q_v, hm, hp, p_full, p_half, rho, dz = _make_column(
        T_sfc=T_sfc, q_c_val=1e-4, q_i_val=q_i_val,
    )

    for label, out in [("morrison", _run_both(T, q_v, hm, hp, p_full, p_half, rho, dz)[0]),
                       ("p3",       _run_both(T, q_v, hm, hp, p_full, p_half, rho, dz)[1])]:
        # Include dq_s_dt: Morrison uses it for snow; P3 uses it for q_rim.
        # B_rim (dq_g_dt for P3) is [m³/kg_air/s] — excluded from mass sum.
        # Sum over levels: sedimentation redistributes mass vertically so
        # per-level values can be positive (flux from above); only the
        # column integral must be ≤ 0 (precipitation removes mass).
        col_sum = (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt
                   + out.dq_i_dt + out.dq_s_dt)
        col_integrated = jnp.sum(col_sum, axis=-1)  # sum over levels
        max_source = float(jnp.max(col_integrated))
        assert max_source < 1e-8, (
            f"{label} (T_sfc={T_sfc}): spurious column-integrated water source "
            f"= {max_source:.2e} kg/kg/s"
        )


# ---------------------------------------------------------------------------
# 3. Ice grows in cold supersaturated columns
# ---------------------------------------------------------------------------

def test_ice_grows_when_supersaturated_wrt_ice():
    """Both schemes should produce net positive dq_i_dt in a cold
    supersaturated column (vapour deposition + possibly riming)."""
    T, q_v, hm, hp, p_full, p_half, rho, dz = _make_column(
        T_sfc=255.0, q_c_val=1e-4, q_i_val=1e-5, supersaturated=True,
    )
    out_m, out_p = _run_both(T, q_v, hm, hp, p_full, p_half, rho, dz)

    for label, out in [("morrison", out_m), ("p3", out_p)]:
        cold_mask = T[0] < constants.T_freeze
        if jnp.any(cold_mask):
            max_dqi = float(jnp.max(out.dq_i_dt[0][cold_mask]))
            assert max_dqi > 0.0, (
                f"{label}: no ice growth in cold supersaturated column, "
                f"max dq_i_dt = {max_dqi:.2e}"
            )


# ---------------------------------------------------------------------------
# 4. Ice melts above freezing
# ---------------------------------------------------------------------------

def test_ice_melts_above_freezing():
    """Both schemes should produce positive dq_r_dt (melt → rain) and
    negative dq_i_dt where T > T_freeze when ice is present.

    Ice is seeded across ALL levels so some of it sits in the warm
    (T > T_freeze) lower layers where melting is active.
    """
    T, q_v, hm, hp, p_full, p_half, rho, dz = _make_column(
        T_sfc=285.0, q_c_val=0.0, q_i_val=0.0,
    )
    # Seed ice uniformly across the full column so warm levels have ice.
    q_i_uniform = 1e-4 * jnp.ones_like(T)
    hm = hm._replace(q_i=q_i_uniform)
    hp = hp._replace(q_i=q_i_uniform)
    out_m, out_p = _run_both(T, q_v, hm, hp, p_full, p_half, rho, dz)

    for label, out in [("morrison", out_m), ("p3", out_p)]:
        warm_mask = T[0] > constants.T_freeze
        if jnp.any(warm_mask):
            # Ice should be draining
            max_dqi = float(jnp.max(out.dq_i_dt[0][warm_mask]))
            assert max_dqi <= 1e-10, (
                f"{label}: ice growing above freezing, max dq_i_dt = {max_dqi:.2e}"
            )
            # Rain should be gaining from melt
            max_dqr = float(jnp.max(out.dq_r_dt[0][warm_mask]))
            assert max_dqr > 0.0, (
                f"{label}: no melt-to-rain above freezing, max dq_r_dt = {max_dqr:.2e}"
            )


# ---------------------------------------------------------------------------
# 5. Precipitation order-of-magnitude comparable
# ---------------------------------------------------------------------------

def test_precipitation_comparable():
    """P3 and Morrison should produce precipitation within a factor of 10
    of each other for the same initial conditions. A larger difference
    signals a tuning or unit error rather than a scheme difference."""
    T, q_v, hm, hp, p_full, p_half, rho, dz = _make_column(
        T_sfc=270.0, q_c_val=2e-4, q_i_val=5e-5, q_r_val=1e-5,
    )
    out_m, out_p = _run_both(T, q_v, hm, hp, p_full, p_half, rho, dz)

    total_m = float(jnp.sum(out_m.precipitation))
    total_p = float(jnp.sum(out_p.precipitation))

    # Both should be positive
    assert total_m >= 0.0, f"Morrison negative total precipitation: {total_m:.2e}"
    assert total_p >= 0.0, f"P3 negative total precipitation: {total_p:.2e}"

    # Within a factor of 10 of each other (if both are non-negligible)
    if total_m > 1e-12 and total_p > 1e-12:
        ratio = max(total_m, total_p) / min(total_m, total_p)
        assert ratio < 10.0, (
            f"Precipitation ratio Morrison/P3 = {total_m:.2e}/{total_p:.2e} "
            f"= {ratio:.1f}x — exceeds factor-of-10 sanity check"
        )


# ---------------------------------------------------------------------------
# 6. Latent heating sign consistent
# ---------------------------------------------------------------------------

def test_latent_heating_sign_consistent():
    """Both schemes must cool where net condensate is lost (evaporation)
    and warm where condensate is gained (condensation, riming, deposition).
    Check sign of column-mean dT_dt matches sign of column-mean dq_loss_dt."""
    T, q_v, hm, hp, p_full, p_half, rho, dz = _make_column(
        T_sfc=265.0, q_c_val=2e-4, q_i_val=1e-5, supersaturated=True,
    )
    out_m, out_p = _run_both(T, q_v, hm, hp, p_full, p_half, rho, dz)

    for label, out in [("morrison", out_m), ("p3", out_p)]:
        mean_dT = float(jnp.mean(out.dT_dt))
        mean_dqv = float(jnp.mean(out.dq_v_dt))
        # Condensation (dq_v < 0) should warm (dT > 0), and vice versa.
        if abs(mean_dT) > 1e-8 and abs(mean_dqv) > 1e-12:
            assert (mean_dT > 0) == (mean_dqv < 0), (
                f"{label}: latent heating sign inconsistent — "
                f"mean dT_dt={mean_dT:.2e}, mean dq_v_dt={mean_dqv:.2e}"
            )


# ---------------------------------------------------------------------------
# 7. P3-specific: rime density physically bounded
# ---------------------------------------------------------------------------

def test_p3_rime_density_in_bounds():
    """After one step with active riming, rho_rim = q_rim/B_rim must stay
    within [rho_rim_min, rho_rim_max]."""
    cfg = P3Config()
    T, q_v, _, hp, p_full, p_half, rho, dz = _make_column(
        T_sfc=255.0, q_c_val=5e-4, q_i_val=1e-4,
        q_rim_val=1e-5, B_rim_val=1e-5 / 400.0,  # start at 400 kg/m³
    )
    out_p = p3_microphysics(T, q_v, hp, p_full, p_half, rho, dz, 300.0, cfg)

    q_rim_new = jnp.clip(hp.q_s + out_p.dq_s_dt * 300.0, 0.0)
    B_rim_new  = jnp.clip(hp.q_g + out_p.dq_g_dt * 300.0, 0.0)
    rho_rim = jnp.where(
        B_rim_new > 1e-30,
        q_rim_new / jnp.maximum(B_rim_new, 1e-30),
        cfg.rho_rim_min,
    )
    active = B_rim_new > 1e-30
    if jnp.any(active):
        min_rho = float(jnp.min(jnp.where(active, rho_rim, cfg.rho_rim_max)))
        max_rho = float(jnp.max(jnp.where(active, rho_rim, cfg.rho_rim_min)))
        assert min_rho >= cfg.rho_rim_min * 0.99, (
            f"rho_rim below floor: {min_rho:.1f} < {cfg.rho_rim_min} kg/m³"
        )
        assert max_rho <= cfg.rho_rim_max * 1.01, (
            f"rho_rim above ceiling: {max_rho:.1f} > {cfg.rho_rim_max} kg/m³"
        )


# ---------------------------------------------------------------------------
# 8. P3 fall speed increases with rime density
# ---------------------------------------------------------------------------

def test_p3_denser_rime_falls_faster():
    """A column with dense rime (rho_rim ~ 800 kg/m³) should produce more
    precipitation than one with light rime (rho_rim ~ 100 kg/m³) because
    the density enhancement factor increases V_t."""
    T, q_v, _, _, p_full, p_half, rho, dz = _make_column(
        T_sfc=260.0, q_c_val=0.0, q_i_val=2e-4,
    )
    cfg = P3Config()
    dt = 300.0
    q_i_col = jnp.zeros_like(T).at[..., :10].set(2e-4)

    # Light rime: rho_rim ~ 100 kg/m³
    q_rim_light = 1e-5 * q_i_col
    B_rim_light = q_rim_light / 100.0
    hydro_light = HydrometeorState(
        q_c=jnp.zeros_like(T), q_r=jnp.zeros_like(T), q_i=q_i_col,
        q_s=q_rim_light, q_g=B_rim_light,
        N_c=1e8 * jnp.ones_like(T), N_r=jnp.zeros_like(T), N_i=jnp.zeros_like(T),
    )

    # Dense rime: rho_rim ~ 800 kg/m³
    q_rim_dense = 1e-5 * q_i_col
    B_rim_dense = q_rim_dense / 800.0
    hydro_dense = HydrometeorState(
        q_c=jnp.zeros_like(T), q_r=jnp.zeros_like(T), q_i=q_i_col,
        q_s=q_rim_dense, q_g=B_rim_dense,
        N_c=1e8 * jnp.ones_like(T), N_r=jnp.zeros_like(T), N_i=jnp.zeros_like(T),
    )

    out_light = p3_microphysics(T, q_v, hydro_light, p_full, p_half, rho, dz, dt, cfg)
    out_dense = p3_microphysics(T, q_v, hydro_dense, p_full, p_half, rho, dz, dt, cfg)

    precip_light = float(jnp.sum(out_light.precipitation))
    precip_dense = float(jnp.sum(out_dense.precipitation))

    assert precip_dense >= precip_light, (
        f"Dense rime should fall faster: precip_dense={precip_dense:.2e} "
        f"< precip_light={precip_light:.2e}"
    )
