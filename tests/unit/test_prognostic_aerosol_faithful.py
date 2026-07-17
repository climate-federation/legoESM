"""Faithfulness pins for the prognostic bulk-aerosol number budget.

Target: ``_sources_and_sink_rate`` / ``aerosol_number_tendency`` /
``step_prognostic_aerosol`` in
``legoesm.atmosphere.physics.microphysics.prognostic_aerosol`` — the optional
single-mode accumulation aerosol NUMBER budget ``dN/dt = S - L*N``.

Most-trustful source
--------------------
This is a bespoke simplified bulk scheme (a prescribed-AOD-proxy augmentation),
NOT a published solver, so an oracle re-deriving the same accounting would be
CIRCULAR.  Instead each process is pinned against its FIRST-PRINCIPLES closed
form at discriminating points where the expected value is a known constant.  The
backward-Euler step is pinned two ways: (i) an INDEPENDENT first-principles
expected value (test_backward_euler_step_exact), and (ii) an implicit<->explicit
SELF-CONSISTENCY identity relating step and tendency (test_budget_closure_...);
the scheme is:

  SOURCES (S, +):  surface emission E/dz_surface (BOTTOM level only);
                   SO2 oxidation so2/tau_ox (ALL levels).
  SINKS  (L, +):   wet scavenging k_scav*P (ALL levels);
                   dry deposition v_dep/dz_surface (BOTTOM level only).
  Integrator:      N_new = (N + dt*S)/(1 + dt*L)   (backward Euler, >= 0).

The existing test_prognostic_aerosol.py checks SIGNS / directions / positivity
only (0 exact-magnitude assertions); this adds the exact closed forms, the
level-selectivity discrimination, the budget-closure identity, config plumbing,
and the safety floors (dz, tau_ox, dt).

Certification (test-only):
1. EXACT per-process closed forms at discriminating points (E/dz bottom-only,
   so2/tau_ox all-levels, k_scav*P all-levels, v_dep/dz bottom-only), each a
   canary against a wrong coefficient, wrong divisor, or wrong level.
2. Backward-Euler step: independent first-principles value + an implicit<->explicit
   self-consistency identity N_new*(1+dt*L) == N + dt*S (S,L from the module's own
   tendency -- a self-consistency pin, NOT reimplementation-independent).
3. Safety floors (dz>=1, tau_ox>=1e-3, dt>=1e-6, incl. NEGATIVE inputs), and the
   AGGREGATE non-negativity clamps clip(S_total)/clip(L_total) (distinguished from
   component-wise clamping), plus so2/precip/N clamps, disabled no-op, full config
   plumbing (all four coefficients), differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.microphysics.prognostic_aerosol import (
    PrognosticAerosolConfig,
    aerosol_number_tendency,
    step_prognostic_aerosol,
)

jax.config.update("jax_enable_x64", True)

_DZ_FLOOR = 1.0
_TAU_OX_FLOOR = 1e-3
_DT_FLOOR = 1e-6


def _cfg(enabled=True, e=1e8, v=1e-3, tau=8.64e4, k=1.0):
    return PrognosticAerosolConfig(
        enabled=enabled, emission_number_flux_m2_s=e, dry_dep_velocity_m_s=v,
        so2_oxidation_timescale_s=tau, wet_scavenging_coeff_m2_kg=k,
    )


def _a(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _tend(N, dz, precip, cfg, e_flux=None, so2=None):
    return np.asarray(aerosol_number_tendency(
        _a(N), _a(dz), _a(precip), cfg,
        emission_flux=None if e_flux is None else _a(e_flux),
        so2_precursor=None if so2 is None else _a(so2)))


def _step(N, dt, dz, precip, cfg, e_flux=None, so2=None):
    return np.asarray(step_prognostic_aerosol(
        _a(N), dt, _a(dz), _a(precip), cfg,
        emission_flux=None if e_flux is None else _a(e_flux),
        so2_precursor=None if so2 is None else _a(so2)))


# ---------------------------------------------------------------------------
# 1. SOURCES — exact closed forms + level selectivity.
# ---------------------------------------------------------------------------
def test_emission_source_bottom_layer_only_exact():
    # N=0 -> tendency == S (the L*N term vanishes).  Only emission active (no
    # so2, no precip), so S = E/dz_surface at the BOTTOM level, 0 elsewhere.
    dz = [200.0, 150.0, 100.0]                 # dz_surface = 100
    cfg = _cfg(e=2e8, v=0.0, k=0.0)            # kill sinks so N=0 isolates emission
    t = _tend([0.0, 0.0, 0.0], dz, 0.0, cfg)
    np.testing.assert_allclose(t[-1], 2e8 / 100.0, rtol=1e-12)   # 2e6, bottom only
    np.testing.assert_allclose(t[:-1], 0.0, atol=0.0)            # interior exactly 0


def test_oxidation_source_all_levels_exact():
    # E=0 -> only SO2 oxidation: S = so2/tau_ox at EVERY level (not just bottom).
    so2 = [4.0e5, 8.0e5, 1.2e6]
    cfg = _cfg(e=0.0, v=0.0, k=0.0, tau=8.64e4)
    t = _tend([0.0, 0.0, 0.0], [100.0, 100.0, 100.0], 0.0, cfg, so2=so2)
    np.testing.assert_allclose(t, np.array(so2) / 8.64e4, rtol=1e-12)


def test_oxidation_timescale_floored():
    # tau_ox = 0 must floor to 1e-3 (finite S = so2/1e-3), not divide by zero.
    cfg = _cfg(e=0.0, v=0.0, k=0.0, tau=0.0)
    t = _tend([0.0], [100.0], 0.0, cfg, so2=[5.0])
    np.testing.assert_allclose(t[-1], 5.0 / _TAU_OX_FLOOR, rtol=1e-12)   # 5000
    assert np.isfinite(t[-1])


def test_negative_so2_clamped_to_zero():
    cfg = _cfg(e=0.0, v=0.0, k=0.0)
    t = _tend([0.0], [100.0], 0.0, cfg, so2=[-1e6])
    assert t[-1] == 0.0                        # negative precursor -> no source


def test_emission_uses_dz_floor():
    # dz_surface = 0.5 < 1 must floor to 1.0, so S = E/1.0 (not E/0.5 = 2E).
    cfg = _cfg(e=1e8, v=0.0, k=0.0)
    t = _tend([0.0, 0.0], [10.0, 0.5], 0.0, cfg)
    np.testing.assert_allclose(t[-1], 1e8 / _DZ_FLOOR, rtol=1e-12)   # 1e8, not 2e8
    # NEGATIVE dz floors via max(dz, 1) -> 1 (NOT abs(dz)): dz=-3 gives E/1, while
    # an abs(dz) bug would give E/3.
    t_neg = _tend([0.0, 0.0], [10.0, -3.0], 0.0, cfg)
    np.testing.assert_allclose(t_neg[-1], 1e8 / _DZ_FLOOR, rtol=1e-12)   # E/1, not E/3


# ---------------------------------------------------------------------------
# 2. SINKS — exact closed forms + level selectivity.
# ---------------------------------------------------------------------------
def test_wet_scavenging_all_levels_exact():
    # E=0, v_dep=0 -> only wet scavenging L = k_scav*P at EVERY level.
    # N=1 uniform -> tendency = -L*N = -k_scav*P everywhere.
    cfg = _cfg(e=0.0, v=0.0, k=2.0)
    t = _tend([1.0, 1.0, 1.0], [100.0, 100.0, 100.0], 3.0, cfg)
    np.testing.assert_allclose(t, -2.0 * 3.0, rtol=1e-12)           # -6 all levels


def test_dry_deposition_bottom_layer_only_exact():
    # E=0, k_scav=0 -> only dry deposition L = v_dep/dz_surface at the BOTTOM.
    # N=1 -> tendency = -v_dep/dz_surface at bottom, 0 in the interior.
    cfg = _cfg(e=0.0, v=0.5, k=0.0)
    t = _tend([1.0, 1.0, 1.0], [100.0, 100.0, 100.0], 0.0, cfg)     # dz_surface=100
    np.testing.assert_allclose(t[-1], -0.5 / 100.0, rtol=1e-12)     # -5e-3
    np.testing.assert_allclose(t[:-1], 0.0, atol=0.0)               # interior exactly 0


def test_negative_precip_clamped_to_zero():
    cfg = _cfg(e=0.0, v=0.0, k=5.0)
    t = _tend([1.0], [100.0], -3.0, cfg)       # negative precip -> no wet sink
    assert t[-1] == 0.0


def test_dry_deposition_uses_dz_floor():
    # dz_surface 0.5 -> floored to 1.0, so L_dry = v_dep/1.0 (not /0.5).
    cfg = _cfg(e=0.0, v=0.2, k=0.0)
    t = _tend([1.0, 1.0], [10.0, 0.5], 0.0, cfg)
    np.testing.assert_allclose(t[-1], -0.2 / _DZ_FLOOR, rtol=1e-12)  # -0.2, not -0.4


def test_source_clamp_is_aggregate_not_componentwise():
    # The non-negativity clamp is on the TOTAL source (emission + oxidation),
    # applied once after summing.  A NEGATIVE emission plus a larger POSITIVE
    # oxidation at the bottom gives S = clip(E/dz + so2/tau, 0) = E/dz + so2/tau
    # (still > 0, but BELOW so2/tau alone).  Component-wise clamping (clip emission
    # to 0 first) would give so2/tau -- higher -- so this distinguishes them.
    cfg = _cfg(e=-1000.0, v=0.0, k=0.0, tau=1e5)     # E/dz = -1000/100 = -10
    so2 = 3e6                                          # so2/tau = 30
    t = _tend([0.0, 0.0], [100.0, 100.0], 0.0, cfg, so2=[so2, so2])
    np.testing.assert_allclose(t[-1], -10.0 + so2 / 1e5, rtol=1e-12)  # 20 (aggregate)
    np.testing.assert_allclose(t[0], so2 / 1e5, rtol=1e-12)           # interior 30 (no emission)


def test_sink_clamp_is_aggregate_not_componentwise():
    # The sink clamp is on the TOTAL rate (wet + dry) once after summing.  A
    # NEGATIVE wet coeff plus a larger POSITIVE dry deposition at the bottom gives
    # L = clip(k*P + v/dz, 0) = k*P + v/dz (> 0, below v/dz alone); component-wise
    # (clip wet to 0 first) would give v/dz -- higher.  N=1 -> tendency = -L.
    cfg = _cfg(e=0.0, v=16.0, k=-0.1)                 # dz_surface=2 -> v/dz=8 ; k*P=-0.1*50=-5
    t = _tend([1.0, 1.0], [100.0, 2.0], 50.0, cfg)
    np.testing.assert_allclose(t[-1], -(-5.0 + 8.0), rtol=1e-12)      # -3 (aggregate)
    assert t[0] == 0.0                                                 # interior: clip(-5,0)=0


def test_negative_tau_ox_and_dt_floored():
    # NEGATIVE oxidation timescale floors to 1e-3 (finite source, right sign).
    t = _tend([0.0], [100.0], 0.0, _cfg(e=0.0, v=0.0, k=0.0, tau=-42.0), so2=[5.0])
    np.testing.assert_allclose(t[-1], 5.0 / _TAU_OX_FLOOR, rtol=1e-12)
    # NEGATIVE dt floors to 1e-6 (a step, not a backward jump).
    got = _step([0.0, 0.0], -100.0, [100.0, 100.0], 0.0, _cfg(e=2e8, v=0.0, k=0.0))
    np.testing.assert_allclose(got[-1], _DT_FLOOR * 2e8 / 100.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# 3. Explicit tendency = S - L*N (full, discriminating).
# ---------------------------------------------------------------------------
def test_explicit_tendency_source_minus_sink_times_n():
    # All processes active; verify dN/dt = S - L*N at the bottom level, where
    # S = E/dz + so2/tau, L = k*P + v/dz.
    dz_s = 250.0
    cfg = _cfg(e=3e8, v=0.4, k=2.0, tau=1e5)
    N_bot, so2_bot, P = 7.0, 6e5, 3.0
    t = _tend([0.0, N_bot], [400.0, dz_s], P, cfg, so2=[so2_bot, so2_bot])
    s_bot = 3e8 / dz_s + so2_bot / 1e5
    l_bot = 2.0 * P + 0.4 / dz_s
    np.testing.assert_allclose(t[-1], s_bot - l_bot * N_bot, rtol=1e-12)


def test_tendency_clamps_negative_number():
    # N < 0 is clamped to 0 before the sink term -> tendency = S (no removal).
    cfg = _cfg(e=1e8, v=0.0, k=5.0)
    t = _tend([-100.0], [100.0], 3.0, cfg)
    np.testing.assert_allclose(t[-1], 1e8 / 100.0, rtol=1e-12)      # S only, sink*0


# ---------------------------------------------------------------------------
# 4. Backward-Euler step + budget-closure truth tier.
# ---------------------------------------------------------------------------
def test_backward_euler_step_exact():
    cfg = _cfg(e=2e8, v=0.3, k=1.5, tau=5e4)
    dz = [300.0, 120.0]
    N = [10.0, 40.0]
    so2 = [5e5, 9e5]
    dt = 100.0
    got = _step(N, dt, dz, 2.0, cfg, so2=so2)
    # Independent first-principles S, L (typed from the documented budget).
    dz_s = 120.0
    S = np.array([5e5 / 5e4, 2e8 / dz_s + 9e5 / 5e4])
    L = np.array([1.5 * 2.0, 1.5 * 2.0 + 0.3 / dz_s])
    exp = (np.array(N) + dt * S) / (1.0 + dt * L)
    np.testing.assert_allclose(got, exp, rtol=1e-12)


def test_budget_closure_implicit_explicit_self_consistency():
    # SELF-CONSISTENCY (not reimplementation-independent -- the independent value
    # pin is test_backward_euler_step_exact above): the implicit step must satisfy
    # its OWN budget exactly:  N_new - N == dt*(S - L*N_new), where S and L are
    # recovered from the module's OWN explicit tendency
    # (S = tendency at N=0; L = S - tendency at N=1, since dN/dt = S - L*N).  This
    # catches step and tendency drifting apart (a shared error passes -- that is
    # covered by the independent value pin).
    cfg = _cfg(e=2e8, v=0.3, k=1.5, tau=5e4)
    dz, so2, P, dt = [300.0, 120.0], [5e5, 9e5], 2.0, 137.0
    N = np.array([10.0, 40.0])
    s = _tend([0.0, 0.0], dz, P, cfg, so2=so2)               # S
    lam = s - _tend([1.0, 1.0], dz, P, cfg, so2=so2)         # L (slope of -L*N)
    n_new = _step(N, dt, dz, P, cfg, so2=so2)
    # Non-cancelling form of  N_new - N == dt*(S - L*N_new):  the backward-Euler
    # equation  N_new*(1 + dt*L) == N + dt*S  (S - L*N_new is a difference of two
    # ~1.7e6 terms, so the subtractive form loses precision; this form does not).
    np.testing.assert_allclose(n_new * (1.0 + dt * lam), N + dt * s, rtol=1e-10)


def test_step_dt_floored():
    # dt = 0 floors to 1e-6, so the state moves by ~1e-6 of the tendency rather
    # than staying frozen.
    cfg = _cfg(e=2e8, v=0.0, k=0.0)
    dz = [100.0, 100.0]
    got = _step([0.0, 0.0], 0.0, dz, 0.0, cfg)
    exp = (0.0 + _DT_FLOOR * 2e8 / 100.0) / (1.0 + _DT_FLOOR * 0.0)
    np.testing.assert_allclose(got[-1], exp, rtol=1e-9)


def test_disabled_returns_state_unchanged():
    cfg = _cfg(enabled=False, e=9e9, v=9.0, k=9.0)
    N = [3.0, 7.0, 11.0]
    got = _step(N, 500.0, [100.0, 100.0, 100.0], 5.0, cfg, so2=[1e6, 1e6, 1e6])
    # gate returns the input array unchanged (enabled processing would move these
    # finite values substantially, so this discriminates the feature gate).
    np.testing.assert_array_equal(got, np.array(N, dtype=np.float64))


def test_implicit_matches_explicit_as_dt_small():
    # (N_new - N)/dt -> tendency as dt -> 0.
    cfg = _cfg(e=2e8, v=0.3, k=1.5, tau=5e4)
    dz, so2, P = [300.0, 120.0], [5e5, 9e5], 2.0
    N = [10.0, 40.0]
    dt = 1e-4                                       # small: the O(dt) implicit lag ~ L*dt
    n_new = _step(N, dt, dz, P, cfg, so2=so2)
    approx = (n_new - np.array(N)) / dt
    exact = _tend(N, dz, P, cfg, so2=so2)
    np.testing.assert_allclose(approx, exact, rtol=1e-3)   # first-order in dt


# ---------------------------------------------------------------------------
# 5. Config plumbing, keyword overrides, differentiability, defaults.
# ---------------------------------------------------------------------------
def test_config_plumbing_each_param():
    # Each of the FOUR coefficients moves the tendency per its own closed form.
    # SOURCES isolated with N=0 (no sink term); SINKS isolated with e=0 and no so2
    # (S=0) -- so each difference is a difference of small comparable numbers, not
    # a cancellation against a ~1e6 source.
    dz, so2, p_rate = [100.0, 100.0], [1e6, 1e6], 2.0
    src = _tend([0.0, 0.0], dz, p_rate, _cfg(e=1e8, v=0.1, k=1.0, tau=1e5), so2=so2)
    # +1e8 emission -> bottom source grows by E/dz.
    hi_e = _tend([0.0, 0.0], dz, p_rate, _cfg(e=2e8, v=0.1, k=1.0, tau=1e5), so2=so2)
    np.testing.assert_allclose(hi_e[-1] - src[-1], 1e8 / 100.0, rtol=1e-12)
    # 2x oxidation timescale -> source halves its oxidation term so2/tau (all levels).
    hi_tau = _tend([0.0, 0.0], dz, p_rate, _cfg(e=1e8, v=0.1, k=1.0, tau=2e5), so2=so2)
    np.testing.assert_allclose(src[0] - hi_tau[0], 1e6 / 1e5 - 1e6 / 2e5, rtol=1e-12)

    snk = _tend([1.0, 1.0], dz, p_rate, _cfg(e=0.0, v=0.1, k=1.0, tau=1e5))   # S=0
    # +1.0 wet coeff -> sink grows by k*P*N at every level (N=1, P=2).
    hi_k = _tend([1.0, 1.0], dz, p_rate, _cfg(e=0.0, v=0.1, k=2.0, tau=1e5))
    np.testing.assert_allclose(snk[0] - hi_k[0], 1.0 * 2.0 * 1.0, rtol=1e-12)
    # +0.1 dry-dep velocity -> bottom sink grows by (dv/dz)*N (bottom only).
    hi_v = _tend([1.0, 1.0], dz, p_rate, _cfg(e=0.0, v=0.2, k=1.0, tau=1e5))
    np.testing.assert_allclose(snk[-1] - hi_v[-1], (0.1 / 100.0) * 1.0, rtol=1e-12)


def test_keyword_overrides_beat_config():
    # emission_flux / so2_precursor kwargs override the config values.
    cfg = _cfg(e=1e8, v=0.0, k=0.0, tau=1e5)
    # emission_flux is a per-column scalar (broadcasts to dz_surface); so2 is a
    # per-level field.
    t = _tend([0.0], [100.0], 0.0, cfg, e_flux=5e8, so2=[2e6])
    np.testing.assert_allclose(t[-1], 5e8 / 100.0 + 2e6 / 1e5, rtol=1e-12)


def test_step_is_differentiable():
    cfg = _cfg(e=2e8, v=0.3, k=1.5, tau=5e4)

    def loss(N):
        out = step_prognostic_aerosol(
            N, 100.0, _a([300.0, 120.0]), _a(2.0), cfg,
            so2_precursor=_a([5e5, 9e5]))
        return jnp.sum(out)
    g = jax.grad(loss)(_a([10.0, 40.0]))
    assert jnp.all(jnp.isfinite(g))


def test_config_defaults():
    c = PrognosticAerosolConfig()
    assert c.enabled is False
    assert c.emission_number_flux_m2_s == 1.0e8
    assert c.dry_dep_velocity_m_s == 1.0e-3
    assert c.so2_oxidation_timescale_s == 8.64e4      # ~1 day
    assert c.wet_scavenging_coeff_m2_kg == 1.0
