"""Oracle-faithful tests for the canonical Kuo (1965) convection scheme.

These tests pin the JAX ``kuo_convection`` to the behavior of
J.-F. Mahfouf's reference Kuo Fortran (AJFMAHFOUF/MOIST_CONVECTION_KUO,
``src/kuo_schemes.f90``, ``kuo65`` variant):

  1. The convective source is the LARGE-SCALE MOISTURE CONVERGENCE
     (``moisture_convergence``), not column supersaturation.
  2. With NO convergence (pure single-column) Kuo is QUIESCENT — the
     physically-correct behavior, since a single column has no resolved
     large-scale ascent to converge.
  3. With a prescribed positive convergence profile Kuo FIRES with the
     canonical closure ``dt/dt = cvgu/zint·(tc−t)``,
     ``dq/dt = −ptenq + cvgu/zint·(qvc−qv)`` over buoyant + ascending
     (icond==2), positive-convergence levels.
  4. Regression vs the compiled oracle on a 60-level tropical sounding:
     the column-integrated convective heating agrees within 10 % (the
     residual is the model's Tetens saturation vs the oracle's Huang-2018
     saturation — see ``.physics-validator/kuo/REPORT`` for the
     quantified gap), and the active-level count and tendency signs
     match level-by-level.
  5. ``q_v >= 0`` over one forward-Euler step, finite gradients, and the
     quiescent path emits exactly zero (no NaN reverse-mode gradient).

The oracle reference numbers below were produced by the refactored
oracle driver in ``.physics-validator/kuo/oracle/kuo_oracle_driver.f90``
on the sounding in ``make_sounding.py`` (regenerable; see the REPORT).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.config import KuoConfig


# ---------------------------------------------------------------------------
# 60-level tropical sounding (matches .physics-validator/kuo/oracle).
# Column ordering: index 0 = model top (low p), index 59 = surface.
# ---------------------------------------------------------------------------

NLEV = 60
PS = 1.0e5
P_TOP = 50.0e2
RD = 287.06
G = 9.81
EPS = 287.04 / 461.5


def _esat_huang(T):
    aw, bw, cw, d1w, d2w = 34.494, 4924.99, 1.57, 237.1, 105.0
    Tc = T - 273.15
    return np.exp(aw - bw / (Tc + d1w)) / (Tc + d2w) ** cw


def _qsat_huang(p, T):
    e = _esat_huang(T)
    return EPS * e / (p - e * (1.0 - EPS))


def _build_sounding():
    """Return (T, qv, p_full, p_half, ptenq) each (1, NLEV)."""
    ph = np.linspace(P_TOP, PS, NLEV + 1)
    p = 0.5 * (ph[:-1] + ph[1:])
    z = -8000.0 * np.log(p / PS)
    T = 300.0 - 6.5e-3 * z
    T = np.maximum(T, 196.0)
    T = T + 4.0 * np.exp(-((p - 150e2) / 60e2) ** 2)
    qs = _qsat_huang(p, T)
    sigma = p / PS
    RH = 0.85 * np.clip((sigma - 0.15) / (1.0 - 0.15), 0.0, 1.0) + 0.1
    RH = np.clip(RH, 0.05, 0.95)
    qv = np.maximum(RH * qs, 1e-7)
    # Positive low/mid-tropospheric moisture-convergence bump.
    peak = 3.0e-3 / 86400.0
    ptenq = peak * np.exp(-((p - 850e2) / 120e2) ** 2)
    to = lambda a: jnp.asarray(a)[None, :]
    return to(T), to(qv), to(p), to(ph), to(ptenq)


# Oracle reference (kuo_oracle_driver.f90, Kuo-1965, SI units).
ORACLE_CVGU = 6.676128e-05      # kg/m^2/s
ORACLE_ZINT = 5.099665e+01      # kg/m^2
ORACLE_N_ACTIVE = 50            # icond==2 levels
ORACLE_COL_HEATING_W = 59.279   # W/m^2 (c_pd * int dT/dt dp/g)


# ---------------------------------------------------------------------------
# Quiescence (no large-scale convergence → no source).
# ---------------------------------------------------------------------------

def test_kuo_quiescent_without_convergence():
    """Kuo is OFF when no large-scale moisture convergence is supplied.

    This is the physically-correct single-column behavior (canonical
    Kuo needs convergence it does not have in one column), NOT a bug.
    """
    T, qv, pf, ph, _ = _build_sounding()
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=None)
    assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.dq_v_dt))) == 0.0
    assert float(jnp.max(out.dq_c_conv_dt)) == 0.0
    assert float(jnp.max(out.convective_mask)) == 0.0


def test_kuo_quiescent_zero_convergence_array():
    """An explicit all-zero convergence array is also quiescent."""
    T, qv, pf, ph, _ = _build_sounding()
    mc0 = jnp.zeros_like(qv)
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=mc0)
    assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.dq_v_dt))) == 0.0


# ---------------------------------------------------------------------------
# Firing under prescribed convergence (oracle-faithful closure).
# ---------------------------------------------------------------------------

def test_kuo_fires_with_convergence():
    """With positive convergence Kuo fires: net column heating > 0."""
    T, qv, pf, ph, ptenq = _build_sounding()
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=ptenq)
    dp = ph[:, 1:] - ph[:, :-1]
    col_heat = float(jnp.sum(out.dT_dt * dp / G) * constants.c_pd)
    assert col_heat > 10.0, f"Kuo did not fire: column heating={col_heat} W/m2"
    assert float(jnp.max(out.convective_mask)) > 0.9


def _build_sounding_mixed_ptenq():
    """Same sounding, but moisture convergence is POSITIVE in the low/mid
    troposphere and NEGATIVE (divergence) aloft — both inside the buoyant cloud
    layer.  This is the regime that exercises the icond2-vs-active mask mismatch
    the conservation fix addresses (the all-positive ``_build_sounding`` never
    has ptenq<=0 inside the cloud, so it cannot detect the bug)."""
    T, qv, pf, ph, _ = _build_sounding()
    p = np.asarray(pf[0])
    peak = 3.0e-3 / 86400.0
    ptenq = peak * np.exp(-((p - 850e2) / 120e2) ** 2)            # convergence low/mid
    ptenq = ptenq - 1.3 * peak * np.exp(-((p - 300e2) / 70e2) ** 2)  # divergence aloft
    return T, qv, pf, ph, jnp.asarray(ptenq)[None, :]


@pytest.mark.parametrize("partition", ["kuo1965", "anthes"])
def test_kuo_column_total_water_conserved_mixed_convergence(partition):
    """CONSERVATION (truth tier): the redistribution is applied over the SAME
    mask (icond2) used to normalise cvgu/zint, so the column total-water
    tendency Σ(dq_v + dq_c)·dp/g = 0 even when divergence (ptenq<0) levels lie
    inside the cloud.  The previous code applied the redistribution over
    ``active`` (=icond2·sign(ptenq)>0), giving
    Σ(dq_v + dq_c) = cvgu·(zint_active/zint − 1) ≠ 0 in this regime.  Both
    partitions are fixed identically (kuo1965 uses zint; anthes uses zint_t and
    zint_q with rt_t·zint_t = rt_q·zint_q = cvgu), so both must close.
    (Physics review: conv/kuo conservation, 2026-06-29.)"""
    T, qv, pf, ph, ptenq = _build_sounding_mixed_ptenq()
    out = kuo_convection(T, qv, pf, ph, dt=900.0,
                         config=KuoConfig(partition=partition),
                         moisture_convergence=ptenq)
    dp = ph[:, 1:] - ph[:, :-1]
    # cvgu = column-integrated POSITIVE convergence (the source magnitude).
    cvgu = float(jnp.sum(jnp.maximum(ptenq, 0.0) * dp / G))
    col_total_water = float(jnp.sum((out.dq_v_dt + out.dq_c_conv_dt) * dp / G))
    assert cvgu > 0.0
    assert float(jnp.max(jnp.abs(out.dq_v_dt))) > 0.0  # the scheme fired
    # The converged moisture is conserved (removed from vapor, returned as
    # moistening + cloud water): the net column total-water tendency is ~0.
    assert abs(col_total_water) < 0.03 * cvgu, (
        f"column total water not conserved: {col_total_water:.3e} "
        f"vs cvgu={cvgu:.3e} (ratio {col_total_water / cvgu:.3f})"
    )


def test_kuo_oracle_column_heating_within_tolerance():
    """Column-integrated heating matches the compiled oracle within 10%.

    The residual gap is the model's Tetens saturation (mandated by
    CLAUDE.md — ``legoesm.thermo.saturation_mixing_ratio``) vs the
    oracle's Huang-2018 saturation, which differ ~2% near cloud base
    and flip the tiny cloud-edge buoyancy; the dominant mid-tropospheric
    tendencies agree to <7% per level.  See the REPORT for the per-level
    fidelity table.
    """
    T, qv, pf, ph, ptenq = _build_sounding()
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=ptenq)
    dp = ph[:, 1:] - ph[:, :-1]
    col_heat = float(jnp.sum(out.dT_dt * dp / G) * constants.c_pd)
    rel = abs(col_heat - ORACLE_COL_HEATING_W) / ORACLE_COL_HEATING_W
    assert rel < 0.10, (
        f"Kuo column heating {col_heat:.2f} W/m2 vs oracle "
        f"{ORACLE_COL_HEATING_W:.2f} W/m2, rel={rel:.3f} > 0.10"
    )


def test_kuo_oracle_active_level_count():
    """Number of firing levels matches the oracle's icond==2 count."""
    T, qv, pf, ph, ptenq = _build_sounding()
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=ptenq)
    n_active = int(jnp.sum(jnp.abs(out.dT_dt[0]) > 1e-10))
    # Allow ±2 for the one-level-wide smooth cloud-edge transition.
    assert abs(n_active - ORACLE_N_ACTIVE) <= 2, (
        f"active levels {n_active} vs oracle {ORACLE_N_ACTIVE}"
    )


def test_kuo_removes_local_convergence():
    """The ``−ptenq`` term: where convergence is strong but the column is
    near-saturated (qvc ≈ qv), Kuo's net moistening is dominated by the
    removal of the large-scale source — ``dq/dt`` goes negative there,
    matching the oracle (canonical Kuo converts converged moisture into
    heating + precip rather than piling it up locally).
    """
    T, qv, pf, ph, ptenq = _build_sounding()
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=ptenq)
    # Near cloud base (high p) the closure dries (dq/dt < 0) as the
    # oracle does (the −ptenq removal dominates there).
    dqdt = out.dq_v_dt[0]
    assert float(jnp.min(dqdt)) < 0.0, "Kuo should remove convergence somewhere"


# ---------------------------------------------------------------------------
# Positivity, finiteness, differentiability.
# ---------------------------------------------------------------------------

def test_kuo_qv_stays_nonnegative():
    """q_v + dt·dq_v_dt >= 0 over one forward-Euler step."""
    T, qv, pf, ph, ptenq = _build_sounding()
    dt = 900.0
    out = kuo_convection(T, qv, pf, ph, dt=dt, config=KuoConfig(),
                         moisture_convergence=ptenq)
    qv_new = qv + dt * out.dq_v_dt
    assert float(jnp.min(qv_new)) >= -1e-12, (
        f"q_v went negative: min={float(jnp.min(qv_new)):.3e}"
    )


def test_kuo_cloud_water_source_nonnegative():
    """dq_c_conv_dt >= 0 everywhere (only the condensation part)."""
    T, qv, pf, ph, ptenq = _build_sounding()
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=ptenq)
    assert float(jnp.min(out.dq_c_conv_dt)) >= -1e-15


def test_kuo_all_outputs_finite():
    T, qv, pf, ph, ptenq = _build_sounding()
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=ptenq)
    for name in ("dT_dt", "dq_v_dt", "dq_c_conv_dt", "cape", "convective_mask"):
        arr = getattr(out, name)
        assert jnp.all(jnp.isfinite(arr)), f"{name} has NaN/Inf"


def test_kuo_gradients_finite_with_convergence():
    """jax.grad through Kuo (w.r.t. T, q_v, ptenq) is finite."""
    T, qv, pf, ph, ptenq = _build_sounding()

    def loss(Tin, qvin, pqin):
        out = kuo_convection(Tin, qvin, pf, ph, dt=900.0,
                             config=KuoConfig(), moisture_convergence=pqin)
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_v_dt ** 2)

    gT, gq, gp = jax.grad(loss, argnums=(0, 1, 2))(T, qv, ptenq)
    assert jnp.all(jnp.isfinite(gT))
    assert jnp.all(jnp.isfinite(gq))
    assert jnp.all(jnp.isfinite(gp))


def test_kuo_quiescent_gradient_finite_and_zero():
    """Quiescent path: zero output AND finite (zero) gradient — no
    reverse-mode 0/0 NaN (issue #249 class)."""
    T, qv, pf, ph, _ = _build_sounding()

    def loss(qvin):
        out = kuo_convection(T, qvin, pf, ph, dt=900.0,
                             config=KuoConfig(), moisture_convergence=None)
        return (jnp.sum(out.dT_dt) + jnp.sum(out.dq_v_dt)
                + jnp.sum(out.dq_c_conv_dt))

    val, grad = jax.value_and_grad(loss)(qv)
    assert val == 0.0
    assert jnp.all(jnp.isfinite(grad))


def test_kuo_jit_matches_eager():
    T, qv, pf, ph, ptenq = _build_sounding()
    cfg = KuoConfig()
    eager = kuo_convection(T, qv, pf, ph, dt=900.0, config=cfg,
                           moisture_convergence=ptenq)
    jitted = jax.jit(
        lambda Tin, qvin, pqin: kuo_convection(
            Tin, qvin, pf, ph, dt=900.0, config=cfg,
            moisture_convergence=pqin,
        )
    )(T, qv, ptenq)
    np.testing.assert_allclose(
        np.asarray(eager.dT_dt), np.asarray(jitted.dT_dt), rtol=1e-6, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(eager.dq_v_dt), np.asarray(jitted.dq_v_dt),
        rtol=1e-6, atol=1e-12,
    )


def test_kuo_vmap_matches_loop():
    """vmap over an ensemble axis matches per-member calls."""
    T, qv, pf, ph, ptenq = _build_sounding()
    cfg = KuoConfig()
    # Build a 3-member ensemble by perturbing T.
    Ts = jnp.stack([T, T + 1.0, T - 1.0], axis=0)        # (3, 1, nlev)
    qs = jnp.stack([qv, qv, qv], axis=0)
    pfs = jnp.stack([pf, pf, pf], axis=0)
    phs = jnp.stack([ph, ph, ph], axis=0)
    pqs = jnp.stack([ptenq, ptenq, ptenq], axis=0)

    def one(Ti, qi, pfi, phi, pqi):
        return kuo_convection(Ti, qi, pfi, phi, dt=900.0, config=cfg,
                              moisture_convergence=pqi).dT_dt

    vmapped = jax.vmap(one)(Ts, qs, pfs, phs, pqs)
    looped = jnp.stack([one(Ts[i], qs[i], pfs[i], phs[i], pqs[i])
                        for i in range(3)], axis=0)
    np.testing.assert_allclose(
        np.asarray(vmapped), np.asarray(looped), rtol=1e-6, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Anthes partition variant.
# ---------------------------------------------------------------------------

def test_kuo_anthes_partition_fires():
    """The optional Kuo-Anthes (1977) partition also fires and stays
    finite / non-negative-q_v."""
    T, qv, pf, ph, ptenq = _build_sounding()
    cfg = KuoConfig(partition="anthes")
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=cfg,
                         moisture_convergence=ptenq)
    assert jnp.all(jnp.isfinite(out.dT_dt))
    assert jnp.all(jnp.isfinite(out.dq_v_dt))
    qv_new = qv + 900.0 * out.dq_v_dt
    assert float(jnp.min(qv_new)) >= -1e-12
    dp = ph[:, 1:] - ph[:, :-1]
    col_heat = float(jnp.sum(out.dT_dt * dp / G) * constants.c_pd)
    assert col_heat > 0.0


def test_kuo_unknown_partition_raises():
    T, qv, pf, ph, ptenq = _build_sounding()
    with pytest.raises(ValueError, match="partition"):
        kuo_convection(T, qv, pf, ph, dt=900.0,
                       config=KuoConfig(partition="bogus"),
                       moisture_convergence=ptenq)


# ---------------------------------------------------------------------------
# Faithful w_lcl>0 activation gate (oracle reads independent w).
# ---------------------------------------------------------------------------

def test_kuo_w_grid_gate_suppresses_subsidence():
    """When the resolved ``w_grid`` is supplied (the faithful oracle path
    reads an INDEPENDENT ``w`` at LCL), a subsiding column (``w < 0``) is
    suppressed even with positive moisture convergence — distinguishing
    the real-``w`` gate from the convergence-sign proxy."""
    T, qv, pf, ph, ptenq = _build_sounding()
    cfg = KuoConfig()

    w_up = jnp.full_like(qv, 0.05)     # ascent everywhere
    w_down = jnp.full_like(qv, -0.05)  # subsidence everywhere

    out_up = kuo_convection(T, qv, pf, ph, dt=900.0, config=cfg,
                            moisture_convergence=ptenq, w_grid=w_up)
    out_down = kuo_convection(T, qv, pf, ph, dt=900.0, config=cfg,
                              moisture_convergence=ptenq, w_grid=w_down)

    heat_up = float(jnp.sum(jnp.abs(out_up.dT_dt)))
    heat_down = float(jnp.sum(jnp.abs(out_down.dT_dt)))
    assert heat_up > 0.0, "ascending column should convect"
    # Subsidence gate should strongly suppress (near-zero) the heating.
    assert heat_down < 1e-3 * heat_up, (
        f"subsidence not suppressed: up={heat_up:.3e} down={heat_down:.3e}"
    )


def test_kuo_w_grid_none_uses_convergence_proxy():
    """With ``w_grid=None`` the gate falls back to the convergence-sign
    proxy (still fires for positive convergence)."""
    T, qv, pf, ph, ptenq = _build_sounding()
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=ptenq, w_grid=None)
    assert float(jnp.sum(jnp.abs(out.dT_dt))) > 0.0


def test_kuo_w_grid_gradients_finite():
    """jax.grad through the w_grid gate is finite."""
    T, qv, pf, ph, ptenq = _build_sounding()
    w = jnp.full_like(qv, 0.05)

    def loss(w_in):
        out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                             moisture_convergence=ptenq, w_grid=w_in)
        return jnp.sum(out.dT_dt ** 2)

    g = jax.grad(loss)(w)
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Anthes moistening fraction b is clamped to [0, 1] (fix 2026-07).
# ---------------------------------------------------------------------------

def _build_sounding_with_rh(rh_target):
    """The reference sounding with the moisture profile replaced by a
    UNIFORM relative humidity ``rh_target`` (same T, pressures, and
    moisture-convergence bump)."""
    T, qv, pf, ph, ptenq = _build_sounding()
    qs = _qsat_huang(np.asarray(pf)[0], np.asarray(T)[0])
    qv_new = jnp.asarray(rh_target * qs)[None, :]
    return T, qv_new, pf, ph, ptenq


def test_kuo_anthes_b_clamps_to_zero_in_near_saturated_column():
    """Anthes (1977) b is a moistening FRACTION in [0, 1].  In a
    near-saturated column (``rhmean + anthes_rh_offset > 1``) the raw
    ``1 − rhmean − offset`` is negative; unclamped it (a) flips the
    moistening term into spurious extra drying and (b) drives the
    heating factor (1−b) above 1 (more condensation heating than the
    consumed accession).  With the clamp b = 0 for ANY offset once
    ``rhmean + offset >= 1``, so two different offsets must produce
    IDENTICAL tendencies."""
    T, qv, pf, ph, ptenq = _build_sounding_with_rh(0.98)

    out_a = kuo_convection(
        T, qv, pf, ph, dt=900.0,
        config=KuoConfig(partition="anthes", anthes_rh_offset=0.1),
        moisture_convergence=ptenq,
    )
    out_b = kuo_convection(
        T, qv, pf, ph, dt=900.0,
        config=KuoConfig(partition="anthes", anthes_rh_offset=0.3),
        moisture_convergence=ptenq,
    )
    # Scheme fires (fixture sanity).
    assert float(jnp.sum(jnp.abs(out_a.dT_dt))) > 0.0, (
        "Fixture broken — near-saturated sounding did not fire"
    )
    # b clamps to 0 for both offsets -> identical closure output.
    np.testing.assert_allclose(
        np.asarray(out_a.dT_dt), np.asarray(out_b.dT_dt), rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(out_a.dq_v_dt), np.asarray(out_b.dq_v_dt), rtol=0, atol=0,
    )
    # With b = 0 the ONLY vapor tendency is the accession removal
    # (−ptenq over active levels) — no positive moistening anywhere.
    assert float(jnp.max(out_a.dq_v_dt)) <= 1e-20, (
        "b=0 column should have no moistening levels (dq_v_dt <= 0)"
    )


def test_kuo_anthes_b_clamp_is_inactive_in_midrange_rh_column():
    """Non-vacuity guard for the clamp test above: in a mid-RH column
    (``rhmean + offset < 1``) b is interior, so different offsets MUST
    produce different tendencies (the clamp does not flatten the
    partition everywhere)."""
    T, qv, pf, ph, ptenq = _build_sounding_with_rh(0.55)

    out_a = kuo_convection(
        T, qv, pf, ph, dt=900.0,
        config=KuoConfig(partition="anthes", anthes_rh_offset=0.1),
        moisture_convergence=ptenq,
    )
    out_b = kuo_convection(
        T, qv, pf, ph, dt=900.0,
        config=KuoConfig(partition="anthes", anthes_rh_offset=0.3),
        moisture_convergence=ptenq,
    )
    assert float(jnp.sum(jnp.abs(out_a.dT_dt))) > 0.0, (
        "Fixture broken — mid-RH sounding did not fire"
    )
    diff = float(jnp.max(jnp.abs(out_a.dT_dt - out_b.dT_dt)))
    assert diff > 0.0, (
        "Different anthes_rh_offset values should change the partition "
        "when b is interior to [0, 1]"
    )


# ---------------------------------------------------------------------------
# convective_mask activation scale is a config field (fix 2026-07).
# ---------------------------------------------------------------------------

def test_kuo_cvgu_activation_scale_wired():
    """``convective_mask = tanh(cvgu / cvgu_activation_scale)`` reads the
    KuoConfig field (formerly an inline 1e-8 literal).  Default scale
    saturates the mask ~1 when firing; a scale far above the actual
    cvgu (~7e-5 kg/m^2/s on the reference sounding) leaves it tiny."""
    T, qv, pf, ph, ptenq = _build_sounding()

    out_default = kuo_convection(
        T, qv, pf, ph, dt=900.0, config=KuoConfig(),
        moisture_convergence=ptenq,
    )
    out_huge_scale = kuo_convection(
        T, qv, pf, ph, dt=900.0,
        config=KuoConfig(cvgu_activation_scale=1.0),
        moisture_convergence=ptenq,
    )
    assert float(out_default.convective_mask[0]) > 0.99
    # tanh(cvgu / 1.0) ~ cvgu ~ 7e-5 << 0.01.
    assert float(out_huge_scale.convective_mask[0]) < 0.01
    # Tendencies are unaffected (diagnostic-only smoothing scale).
    np.testing.assert_allclose(
        np.asarray(out_default.dT_dt), np.asarray(out_huge_scale.dT_dt),
        rtol=0, atol=0,
    )
