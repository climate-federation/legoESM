"""SAM rain-number (N_r) consistency limiter — morrison.py.

Locks the fix for the cldF_fsd day-540 NaN death (diagnosis_cell9576.md):
the transport-side naive positivity clip inflated N_r to 1e15-1e25 /m^3,
pinning LAMR at lamr_max => sub-drizzle rain fall speed => precipitation
trapped in the lowest layer => period-2 graupel<->rain melt/freeze flip-flop
(+-115 K/step) => NaN.  Rain was the ONE two-moment species without the
SAM "adjust var check" post-step number repair (N_i/N_s/N_g have it).

Contract under test (per-volume N_r [1/m^3], HydrometeorState conventions):
  N_r_post = clip(N_r, 0) + dN_r_dt*dt  is CAPPED at
  lamr_max^3 * rho * q_r_post / (pi*rho_w)   (ceiling-only — the lamr_min
  floor of SAM's check is deliberately NOT enforced: 66% of the campaign
  state sits below the window and raising it is a separate physics change),
  cleared to 0 where q_r_post <= 1e-14, and BITWISE untouched otherwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
    resolve_morrison_flavor,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

NLEV = 8
NCOL = 1


def _column(T_val, q_r_val, N_r_val, q_g_val=0.0, q_v_frac=0.5):
    """Small hydrostatic-ish column with prescribed rain/graupel at the
    lowest level."""
    from legoesm.thermo import saturation_mixing_ratio

    sigma_half = jnp.linspace(0.05, 1.0, NLEV + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_s = 1.0e5
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (NCOL, NLEV + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (NCOL, NLEV))
    T = jnp.full((NCOL, NLEV), 240.0).at[:, -1].set(T_val)
    T = T.at[:, -2].set(0.5 * (240.0 + T_val))
    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = constants.R_d * T * dp / (constants.g * p_full)
    q_v = q_v_frac * saturation_mixing_ratio(T, p_full)
    zeros = jnp.zeros((NCOL, NLEV))
    hydro = HydrometeorState(
        q_c=zeros, q_r=zeros.at[:, -1].set(q_r_val),
        q_i=zeros, q_s=zeros, q_g=zeros.at[:, -1].set(q_g_val),
        N_c=jnp.full((NCOL, NLEV), 1e8),
        N_r=zeros.at[:, -1].set(N_r_val),
        N_i=zeros,
    )
    return T, q_v, hydro, p_full, p_half, rho, dz


def _post_state(hydro, out, dt):
    q_r_new = jnp.maximum(hydro.q_r + out.dq_r_dt * dt, 0.0)
    n_r_new = jnp.clip(hydro.N_r, 0.0) + out.dN_r_dt * dt
    return q_r_new, n_r_new


def _window(cfg, rho, q_r_new):
    cfg = resolve_morrison_flavor(cfg)
    c = jnp.pi * constants.rho_water
    return (cfg.lamr_min ** 3 * rho * q_r_new / c,
            cfg.lamr_max ** 3 * rho * q_r_new / c)


DT = 75.0


def test_pathological_nr_bounded_in_one_call():
    """The forensic case: N_r ~ 5e15 /m^3 with heavy rain must come out of
    ONE morrison call inside the PSD window (=> physical fall speed next
    step)."""
    cfg = MorrisonConfig()
    T, q_v, hydro, p_full, p_half, rho, dz = _column(
        225.0, q_r_val=0.18, N_r_val=4.7e15)
    out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, DT,
                                config=cfg)
    q_r_new, n_r_new = _post_state(hydro, out, DT)
    lo, hi = _window(cfg, rho, q_r_new)
    k = -1
    assert np.isfinite(float(n_r_new[0, k]))
    assert float(n_r_new[0, k]) <= float(hi[0, k]) * (1 + 1e-12), (
        f"post-step N_r {float(n_r_new[0, k]):.3e} above PSD window top "
        f"{float(hi[0, k]):.3e}")
    # And the bound is a REAL reduction (the input was 9+ orders above it).
    assert float(n_r_new[0, k]) < 1e-6 * 4.7e15


def test_orphan_number_cleared():
    """q_r ~ 0 with huge N_r (the re-mothered orphan population) must clear
    to exactly zero post-step.  The fixture has NO rain source (q_c = 0, no
    ice, subsaturated), so q_r_new must stay at/below QSMALL — asserted, not
    assumed (codex fix-review hardening note); the exact ``<=`` boundary is
    not engineered through the full scheme (fp-fragile), the contract is
    locked at q_r_new ~ 0."""
    cfg = MorrisonConfig()
    T, q_v, hydro, p_full, p_half, rho, dz = _column(
        290.0, q_r_val=0.0, N_r_val=1.0e12, q_v_frac=0.3)
    out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, DT,
                                config=cfg)
    q_r_new, n_r_new = _post_state(hydro, out, DT)
    k = -1
    assert float(q_r_new[0, k]) <= 1.0e-14, (
        "fixture drifted: a rain source appeared — the orphan-clearing "
        "branch is no longer exercised")
    assert float(n_r_new[0, k]) == 0.0


def test_in_window_number_untouched():
    """A healthy PSD (N_r consistent with q_r) must pass the ceiling as a
    no-op (post-step number strictly inside the window; the jnp.where guard
    then keeps the pre-limiter dN_r_dt bitwise — locked against the real
    pre-fix module by the caller-level probe _probe_cell9576_i_healthy)."""
    cfg = MorrisonConfig()
    # q_r = 0.5 g/kg with N_r = 3e3 /m^3 -> lamr ~ 2.4e3, well inside
    # [357, 5e4].
    T, q_v, hydro, p_full, p_half, rho, dz = _column(
        285.0, q_r_val=5.0e-4, N_r_val=3.0e3)
    out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, DT,
                                config=cfg)
    q_r_new, n_r_new = _post_state(hydro, out, DT)
    lo, hi = _window(cfg, rho, q_r_new)
    k = -1
    assert float(n_r_new[0, k]) < float(hi[0, k]), (
        "test setup drifted: post-step N_r not below the ceiling")


def test_below_window_not_forced_to_floor():
    """Ceiling-only semantics: a big-drop state (N_r far below the lamr_min
    floor) must NOT be snapped to the floor value by the limiter.  The
    physics itself may add number here (SB2001 breakup raises N for
    over-large drops — that is pre-existing behavior, verified bitwise
    against the pre-fix module by _probe_cell9576_i_healthy on real
    states); the limiter contract is only that nothing is forced UP to
    n_r_lo and the ceiling holds."""
    cfg = MorrisonConfig()
    # q_r = 1 g/kg with N_r = 5 /m^3 -> mean drop far above 2.8 mm ->
    # below-window.
    T, q_v, hydro, p_full, p_half, rho, dz = _column(
        285.0, q_r_val=1.0e-3, N_r_val=5.0)
    out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, DT,
                                config=cfg)
    q_r_new, n_r_new = _post_state(hydro, out, DT)
    lo, hi = _window(cfg, rho, q_r_new)
    k = -1
    assert float(q_r_new[0, k]) > 1.0e-14
    # a floor-enforcing limiter would land EXACTLY on lo; physics-driven
    # number (breakup) is far smaller than the ceiling and unrelated to lo
    assert float(n_r_new[0, k]) != float(lo[0, k])
    assert float(n_r_new[0, k]) < float(hi[0, k])


def test_no_ratchet_over_freeze_melt_cycles():
    """Ten sequential morrison steps from a supercooled heavy-rain state
    (the engaged-cycle regime, T flipping via the scheme's own L_f release)
    must keep N_r bounded by the PSD window at every step — no orphan-number
    ratchet."""
    cfg = MorrisonConfig()
    T, q_v, hydro, p_full, p_half, rho, dz = _column(
        225.0, q_r_val=0.15, N_r_val=1.0e10)
    max_ratio = 0.0
    for _ in range(10):
        out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz,
                                    DT, config=cfg)
        q_r_new, n_r_new = _post_state(hydro, out, DT)
        lo, hi = _window(cfg, rho, q_r_new)
        ratio = float(jnp.max(jnp.where(hi > 0, n_r_new / jnp.maximum(hi, 1e-30),
                                        0.0)))
        max_ratio = max(max_ratio, ratio)
        # advance the state with the scheme's own tendencies
        T = T + out.dT_dt * DT
        q_v = jnp.maximum(q_v + out.dq_v_dt * DT, 0.0)
        hydro = HydrometeorState(
            q_c=jnp.maximum(hydro.q_c + out.dq_c_dt * DT, 0.0),
            q_r=q_r_new,
            q_i=jnp.maximum(hydro.q_i + out.dq_i_dt * DT, 0.0),
            q_s=jnp.maximum(hydro.q_s + out.dq_s_dt * DT, 0.0),
            q_g=jnp.maximum(hydro.q_g + out.dq_g_dt * DT, 0.0),
            N_c=hydro.N_c,
            N_r=jnp.maximum(n_r_new, 0.0),
            N_i=jnp.maximum(hydro.N_i + out.dN_i_dt * DT, 0.0),
        )
        assert bool(jnp.all(jnp.isfinite(T)))
    assert max_ratio <= 1.0 + 1e-9, (
        f"N_r escaped the PSD window over cycles (max N_r/hi = {max_ratio})")


def test_limiter_fails_without_fix_synthetic():
    """Non-vacuity: with a pathological input the post-step N_r WOULD be
    ~unchanged at 4.7e15 without the limiter (the pre-fix behavior measured
    in the forensics); assert the fixed module reduces it by >6 orders, so
    this test FAILS on the pre-fix code (tripwire self-test)."""
    cfg = MorrisonConfig()
    T, q_v, hydro, p_full, p_half, rho, dz = _column(
        225.0, q_r_val=0.18, N_r_val=4.7e15)
    out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, DT,
                                config=cfg)
    _, n_r_new = _post_state(hydro, out, DT)
    assert float(n_r_new[0, -1]) < 4.7e15 * 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
