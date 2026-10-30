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

The cap is applied exactly, but it is EXPORTED as a tendency and the caller
reconstructs N_r + dN_r_dt*dt.  For N_r >= ~1e22 that reconstruction is a
catastrophic cancellation bounded by ulp(N_r), so the achieved contract is
"within a factor ~2 of the ceiling", not a strict bound — see
test_ceiling_holds_across_dynamic_range and the FP LIMIT note in morrison.py.
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


def _column(T_val, q_r_val, N_r_val, q_g_val=0.0, q_v_frac=0.5,
            uniform_rain=False):
    """Small hydrostatic-ish column with prescribed rain/graupel at the
    lowest level.

    ``uniform_rain`` puts q_r in EVERY level so the lowest level is re-supplied
    by sedimentation influx from above.  Without it a bottom-level-only rain
    load sediments ENTIRELY to the surface within one 75 s step (morrison
    applies sedimentation internally), leaving q_r_post = 0 — whereupon the
    PSD ceiling is 0, the pre-existing dN_r_dt >= -N_r/dt clamp zeroes the
    number by itself, and the ceiling tests below become vacuous (measured:
    they passed against the pre-fix module).  Pair it with a supersaturated
    q_v_frac so evaporation does not drain q_r instead.
    """
    from legoesm.thermo import saturation_specific_humidity

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
    q_v = q_v_frac * saturation_specific_humidity(T, p_full)
    zeros = jnp.zeros((NCOL, NLEV))
    q_r_field = (jnp.full((NCOL, NLEV), q_r_val) if uniform_rain
                 else zeros.at[:, -1].set(q_r_val))
    hydro = HydrometeorState(
        q_c=zeros, q_r=q_r_field,
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
    """The forensic case: N_r ~ 5e15 /m^3 with sustained rain must come out of
    ONE morrison call inside the PSD window (=> physical fall speed next
    step).

    Non-vacuity MEASURED against the pre-fix module (63aae5c56): post-step
    N_r = 2.354e15 vs ceiling 7.105e7, i.e. 3.3e7x over the PSD bound.
    """
    cfg = MorrisonConfig()
    T, q_v, hydro, p_full, p_half, rho, dz = _column(
        290.0, q_r_val=1.0e-3, N_r_val=4.7e15, q_v_frac=1.05,
        uniform_rain=True)
    out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, DT,
                                config=cfg)
    q_r_new, n_r_new = _post_state(hydro, out, DT)
    lo, hi = _window(cfg, rho, q_r_new)
    k = -1
    assert np.isfinite(float(n_r_new[0, k]))
    # 1e-6 relative, not bitwise: the limiter clamps n_r_new exactly, but it
    # is EXPORTED as a tendency (dN_r_dt = (n_r_new - N_r)/dt) and this test
    # reconstructs N_r + dN_r_dt*dt — a lossy fp round-trip worth ~5e-9
    # relative once the ceiling actually binds (it did not on the old
    # bottom-level-only fixture, where both sides were 0).
    assert float(n_r_new[0, k]) <= float(hi[0, k]) * (1 + 1e-6), (
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
        268.0, q_r_val=1.0e-3, N_r_val=1.0e10, q_v_frac=1.05,
        uniform_rain=True)
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
    assert max_ratio <= 1.0 + 1e-6, (  # dN_r_dt round-trip, see above
        f"N_r escaped the PSD window over cycles (max N_r/hi = {max_ratio})")


def test_ceiling_holds_across_dynamic_range():
    """The ceiling must hold for EVERY fp mantissa the campaign could reach,
    not just for round decades.

    The cap is applied to a number but exported as a RATE, and the caller
    stores max(N_r + dt*dN_r_dt, 0).  Collapsing N_r ~ 1e23 to ~7e7 in one
    step resolves only ulp(N_r), so a target placed exactly AT the ceiling
    reconstructs above it (1.0036x at 1e22, 1.89x at 1e24 without the 4-ulp
    margin).  Sampling round decades hides this — an adversarial mantissa
    does not.  This sweeps random mantissas per binade over the whole range
    the failing campaign produced (observed global max 2.25e23 = 2^77.6) and
    asserts the STRICT bound the margin earns there.

    Beyond the crossover N_r > ceiling/eps the lattice at N_r is coarser than
    the ceiling itself and no rate export can hold the bound; that regime is
    only reachable from an already-corrupt restart (it takes the very ratchet
    this limiter removes to get there) and is covered by the weaker
    finiteness assertion.  The crossover moves with the ceiling, so it is a
    function of q_r/rho/dt, not a fixed N_r — this fixture pins q_r = 1 g/kg
    and dt = 75 s, where it sits just above 2^78.  See the margin note in
    morrison.py.
    """
    cfg = MorrisonConfig()
    rng = np.random.default_rng(20260804)
    for exp2 in range(40, 78):  # 1.1e12 .. 2.3e23, 8 mantissas per binade
        n_ins = [float(np.ldexp(1.0 + rng.random(), exp2)) for _ in range(8)]
        for n_in in n_ins:
            T, q_v, hydro, p_full, p_half, rho, dz = _column(
                290.0, q_r_val=1.0e-3, N_r_val=n_in, q_v_frac=1.05,
                uniform_rain=True)
            out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho,
                                        dz, DT, config=cfg)
            q_r_new, n_r_new = _post_state(hydro, out, DT)
            _, hi = _window(cfg, rho, q_r_new)
            # exactly what every production caller stores
            n = max(float(n_r_new[0, -1]), 0.0)
            h = float(hi[0, -1])
            assert np.isfinite(n), f"N_r={n_in:.6e} -> {n}"
            assert h > 0.0, "fixture drifted: rain did not survive the step"
            # n <= h alone proves the ceiling engaged: the input was n_in >>
            # h, so this IS the reduction assertion (a separate ">= 1e6x"
            # check would be redundant, and wrong at the low binades where
            # n_in/h is only ~2e4).
            assert n <= h, (
                f"N_r={n_in:.6e} (2^{exp2}): caller-stored {n:.6e} exceeds "
                f"the PSD ceiling {h:.6e} (ratio {n / h:.6f})")

    # Above the representable range: no strict bound is possible, but the
    # scheme must still not produce NaN/Inf.
    for exp2 in (80, 85, 90):
        n_in = float(np.ldexp(1.5, exp2))
        T, q_v, hydro, p_full, p_half, rho, dz = _column(
            290.0, q_r_val=1.0e-3, N_r_val=n_in, q_v_frac=1.05,
            uniform_rain=True)
        out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz,
                                    DT, config=cfg)
        _, n_r_new = _post_state(hydro, out, DT)
        assert np.isfinite(float(n_r_new[0, -1])), f"N_r={n_in:.3e} -> non-finite"


def test_limiter_fails_without_fix_synthetic():
    """Non-vacuity tripwire: on the pre-fix module this fixture leaves the
    post-step N_r at 2.354e15 /m^3 (MEASURED against 63aae5c56); the fixed
    module pins it at the PSD ceiling 7.105e7, a 7.5-order reduction.  The
    assertion below therefore FAILS if the limiter is removed."""
    cfg = MorrisonConfig()
    T, q_v, hydro, p_full, p_half, rho, dz = _column(
        290.0, q_r_val=1.0e-3, N_r_val=4.7e15, q_v_frac=1.05,
        uniform_rain=True)
    out = morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, DT,
                                config=cfg)
    _, n_r_new = _post_state(hydro, out, DT)
    assert float(n_r_new[0, -1]) < 4.7e15 * 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
