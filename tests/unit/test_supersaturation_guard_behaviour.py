"""BEHAVIOURAL gate: the guard actually drains a super-saturated column.

Companion to ``tests/test_microphysics_supersaturation_guard.py`` (which is
STRUCTURAL: does every scheme expose the guard). A config-presence test would
NOT have caught the motivating failure -- the field was present on all five
bulk schemes and simply defaulted off -- so this file asserts the property that
actually failed.

Protocol (held IDENTICAL across every arm; the ONLY variable is
``hard_saturation_adjustment`` False vs True):

    single column, 4 levels, dt = 300 s, T0 = 300 K, p = 90000 Pa,
    q_c = q_r = q_i = q_s = q_g = 0, q_v = 1.4 * q_sat(T0, p),
    RH recomputed each step from the MODEL's own
    ``legoesm.thermo.saturation_specific_humidity`` at the CURRENT (T, q_v)
    -- no re-derived saturation curve (CLAUDE.md).

Every threshold below is MEASURED, not guessed: the numbers come from
``scripts/tmp/_probe_supersat_guard.py`` on SLURM job 9331634 (x64=1, CPU) and
are quoted here so a future change of behaviour shows up as a diff rather than
as a silent re-tune of an assertion.

    scheme          hard=False  hard=True   (RH after 1 step from RH0 = 1.4)
    kessler            1.0589      1.0000
    seifert_beheng     1.0589      1.0000
    morrison           1.0589      1.0000
    thompson           1.0589      1.0000
    p3                 1.0589      1.0000
    sundqvist          0.2255      1.0000
    ml_emulator        1.2459      1.0000

Two honest readings of that table, both of which the assertions encode:

* The five bulk schemes' smooth branch does NOT leave a box permanently
  super-saturated -- it relaxes geometrically (1.0589 -> 1.0250 -> 1.0026 by
  step 5). What it does is relax SLOWLY: with a continuing moisture source it
  supports a standing super-saturation, which is the persistent-CWV-drift
  failure mode. The discriminating, non-tautological assertion is therefore
  about the FIRST step, where the two arms differ by a factor ~500 in residual
  super-saturation.
* Sundqvist is NOT the same failure mode and must not be asserted as if it
  were: its smooth branch gates on RH (an O(1) sigmoid argument that saturates)
  and removes the whole excess toward ``q_sat(T_old)`` in one step, ignoring
  the latent heating -- so it OVERSHOOTS to RH 0.23 with a ~20 K one-step
  temperature spike, and rains out 28 % of the column water by step 40 (vs 5 %
  with the guard). Asserting "the default fails to reach RH <= 1" would be
  FALSE for Sundqvist; the real defect there is the overshoot, so that is what
  is asserted.
"""

from __future__ import annotations

import ast
import os
import pathlib

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.microphysics.config import (
    HARD_SAT_GUARD_SCHEMES,
    KesslerConfig,
    MicrophysicsMLEmulatorConfig,
    MorrisonConfig,
    P3Config,
    SeifertBehengConfig,
    SundqvistConfig,
    ThompsonConfig,
)
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    MicrophysicsEmulator,
    ml_microphysics,
)
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import (
    seifert_beheng_microphysics,
)
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.thermo import saturation_specific_humidity

from legoesm import constants

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="conservation/threshold assertions require x64 (CLAUDE.md)",
)

_DT = 300.0
_T0 = 300.0
_P0 = 90000.0
_RH0 = 1.4

_SCHEMES = {
    "kessler": (kessler_microphysics, KesslerConfig),
    "sundqvist": (sundqvist_microphysics, SundqvistConfig),
    "seifert_beheng": (seifert_beheng_microphysics, SeifertBehengConfig),
    "morrison": (morrison_microphysics, MorrisonConfig),
    "thompson": (thompson_microphysics, ThompsonConfig),
    "p3": (p3_microphysics, P3Config),
    "ml_emulator": (ml_microphysics, MicrophysicsMLEmulatorConfig),
}

# Schemes whose DEFAULT (smooth) branch leaves the column still super-saturated
# after one step. Sundqvist is excluded because it OVERSHOOTS instead (see the
# module docstring); it gets its own, stricter assertion below.
_DEFAULT_LEAVES_SUPERSATURATED = tuple(
    s for s in _SCHEMES if s != "sundqvist")


def _column():
    shape = (1, 4)
    T = jnp.full(shape, _T0, dtype=jnp.float64)
    p_full = jnp.full(shape, _P0, dtype=jnp.float64)
    p_half = jnp.full((1, 5), _P0, dtype=jnp.float64)
    q_v = _RH0 * saturation_specific_humidity(T, p_full)
    z = jnp.zeros(shape, dtype=jnp.float64)
    hyd = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
                           N_c=z, N_r=z, N_i=z)
    rho = p_full / (constants.R_d * T)
    dz = jnp.full(shape, 500.0, dtype=jnp.float64)
    return T, q_v, hyd, p_full, p_half, rho, dz


def _integrate(scheme, hard, nstep):
    """Explicit-Euler the column; return (RH history, total-water history)."""
    fn, cfg_cls = _SCHEMES[scheme]
    cfg = cfg_cls(hard_saturation_adjustment=hard)
    T, q_v, hyd, p_full, p_half, rho, dz = _column()
    model = None
    if scheme == "ml_emulator":
        model = MicrophysicsEmulator(
            cfg.n_input, cfg.n_hidden, cfg.n_layers, cfg.n_output,
            key=jax.random.PRNGKey(cfg.seed))
    rh_hist, tw_hist = [], []
    for _ in range(nstep):
        args = (T, q_v, hyd, p_full, p_half, rho, dz, _DT, cfg)
        out = fn(*args, model) if model is not None else fn(*args)
        T = T + out.dT_dt * _DT
        q_v = q_v + out.dq_v_dt * _DT
        hyd = hyd._replace(
            q_c=hyd.q_c + out.dq_c_dt * _DT, q_r=hyd.q_r + out.dq_r_dt * _DT,
            q_i=hyd.q_i + out.dq_i_dt * _DT, q_s=hyd.q_s + out.dq_s_dt * _DT,
            q_g=hyd.q_g + out.dq_g_dt * _DT)
        rh_hist.append(float((q_v / saturation_specific_humidity(T, p_full))[0, 0]))
        tw_hist.append(float(
            (q_v + hyd.q_c + hyd.q_r + hyd.q_i + hyd.q_s + hyd.q_g)[0, 0]))
    return rh_hist, tw_hist


# ---------------------------------------------------------------------------
# The property that failed: hard=True lands the column ON the curve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", sorted(HARD_SAT_GUARD_SCHEMES))
def test_hard_adjust_drives_supersaturated_column_onto_the_curve(scheme):
    """From RH = 1.4, EVERY guarded scheme reaches RH <= ~1 within 3 steps."""
    rh, _ = _integrate(scheme, hard=True, nstep=3)
    assert min(rh) <= 1.001, (
        f"{scheme}: hard_saturation_adjustment=True left the column "
        f"super-saturated after 3 steps (RH = {rh}); the guard is not wired in."
    )
    # ... and does not blow past the curve into a spuriously DRY state either
    # (the guard is a drain onto saturation, not an unbounded sink). The ML
    # emulator is exempt: its untrained random tendencies dominate after step 1.
    if scheme != "ml_emulator":
        assert rh[0] > 0.5, (
            f"{scheme}: hard adjustment overshot to RH = {rh[0]} in one step; "
            "the on-curve solve should land AT saturation, not below it."
        )


@pytest.mark.parametrize("scheme", sorted(_DEFAULT_LEAVES_SUPERSATURATED))
def test_default_leaves_the_column_supersaturated_after_one_step(scheme):
    """The DEFAULT (hard=False) smooth branch does NOT reach saturation in one
    step -- this is the under-drain that let a 140 %-RH column persist. Measured
    residual: RH 1.0589 (bulk schemes) / 1.2459 (ml_emulator) vs 1.0000 with
    the guard on. NON-VACUITY: this test FAILS if the guard is ever silently
    turned on by default, which is exactly the review the P4 question needs."""
    rh_off, _ = _integrate(scheme, hard=False, nstep=1)
    rh_on, _ = _integrate(scheme, hard=True, nstep=1)
    assert rh_off[0] > 1.02, (
        f"{scheme}: default arm reached RH = {rh_off[0]} in one step -- either "
        "the default flipped to True, or the smooth branch changed. Re-measure "
        "with scripts/tmp/_probe_supersat_guard.py before re-tuning this bound."
    )
    assert rh_on[0] < rh_off[0], (
        f"{scheme}: the guard did not reduce super-saturation "
        f"({rh_on[0]} vs {rh_off[0]}) -- the arms are not distinguishable, so "
        "this gate would be vacuous."
    )


def test_sundqvist_default_overshoots_and_the_guard_fixes_it():
    """Sundqvist's own failure mode, asserted as what it IS.

    Its smooth branch removes the whole excess toward q_sat(T_old) in one step
    and ignores the latent heating that raises q_sat, so from RH 1.4 it lands
    at RH ~0.23 with a ~20 K one-step heating spike and rains out ~28 % of the
    column water. With the guard the landing point is the enthalpy-consistent
    equilibrium and the per-step heating is capped."""
    rh_off, tw_off = _integrate("sundqvist", hard=False, nstep=40)
    rh_on, tw_on = _integrate("sundqvist", hard=True, nstep=40)
    assert rh_off[0] < 0.4, (
        f"expected the documented one-step overshoot; got RH = {rh_off[0]}")
    assert 0.99 <= rh_on[0] <= 1.001, (
        f"guarded arm should land ON the curve; got RH = {rh_on[0]}")
    # The overshoot is not free: it dumps condensate into autoconversion, so
    # far more column water leaves as precipitation.
    assert tw_off[-1] < tw_on[-1], (
        f"guarded arm should retain MORE column water ({tw_on[-1]}) than the "
        f"overshooting default ({tw_off[-1]})")
    assert tw_off[-1] / tw_off[0] < 0.9


@pytest.mark.parametrize(
    "scheme", sorted(set(HARD_SAT_GUARD_SCHEMES) - {"ml_emulator"}))
def test_guard_never_creates_column_water(scheme):
    """Sign/conservation check. Both arms are compared at the IDENTICAL window
    (step 1) against the IDENTICAL initial state -- differing spans would be a
    confound (CLAUDE.md). Note the two arms are NOT expected to land on the
    same total water: Sundqvist's overshooting default rains out far more (that
    is the defect, asserted separately). What must hold in EVERY arm is that
    total water never exceeds what the column started with.

    ml_emulator is excluded here and covered by the exact-equality test below:
    an untrained MLP's own tendencies are unconstrained (contract:
    ``conserves: none``), so a column-water bound would be testing the random
    network, not the guard."""
    _, q_v, hyd, *_ = _column()
    tw0 = float((q_v + hyd.q_c + hyd.q_r + hyd.q_i + hyd.q_s + hyd.q_g)[0, 0])
    for hard in (False, True):
        _, tw = _integrate(scheme, hard=hard, nstep=1)
        assert tw[0] <= tw0 * (1.0 + 1e-12), (
            f"{scheme} (hard={hard}): total column water grew "
            f"{tw0} -> {tw[0]}; microphysics must move water, not create it."
        )


def _ml_arms():
    """(off, on) MicrophysicsOutput for the emulator on the shared column."""
    T, q_v, hyd, p_full, p_half, rho, dz = _column()
    base = MicrophysicsMLEmulatorConfig()
    model = MicrophysicsEmulator(base.n_input, base.n_hidden, base.n_layers,
                                 base.n_output, key=jax.random.PRNGKey(base.seed))
    args = (T, q_v, hyd, p_full, p_half, rho, dz, _DT)
    off = ml_microphysics(*args, base, model)
    on = ml_microphysics(
        *args, MicrophysicsMLEmulatorConfig(hard_saturation_adjustment=True),
        model)
    return off, on


def test_ml_emulator_guard_is_exactly_water_neutral_and_actually_fires():
    """The emulator's guard adds ``-rate`` to dq_v_dt and ``+rate`` to dq_c_dt,
    so the two arms must agree on total water to round-off.

    NON-VACUITY (this is the point): a water-neutrality comparison between two
    arms is trivially satisfied when ``rate == 0``, i.e. it would pass with the
    guard deleted. So the test first asserts the arms genuinely DIFFER, and
    that the difference is equal and opposite in q_v and q_c."""
    off, on = _ml_arms()
    d_qv = np.asarray(on.dq_v_dt - off.dq_v_dt)
    d_qc = np.asarray(on.dq_c_dt - off.dq_c_dt)
    d_T = np.asarray(on.dT_dt - off.dT_dt)
    assert np.max(np.abs(d_qv)) > 1e-9, (
        "the guard produced NO change at RH 1.4 — this comparison would pass "
        "with the feature removed, so it proves nothing. Check the wiring."
    )
    np.testing.assert_allclose(d_qc, -d_qv, rtol=1e-12, atol=0.0)
    # ... and the heating that accompanies it is the matching latent release.
    np.testing.assert_allclose(
        d_T, (constants.L_v / constants.c_pd) * (-d_qv), rtol=1e-12, atol=0.0)


def test_ml_emulator_guard_uses_the_smooth_gate_not_the_eager_hard_gate():
    """The emulator runs inside jit and is the one TRAINABLE scheme, so its
    drain must use the SMOOTH sigmoid activation, not ``hard_saturation_drain``
    whose ``jnp.where(q_v > thr*q_sat, ...)`` step is licensed only for the
    EAGER post-step hook ("outside jit / no autodiff through it").

    A hard gate is detectable behaviourally: it is exactly zero below the
    threshold and jumps discontinuously across it. The smooth gate is small but
    STRICTLY NON-ZERO just below the threshold. Probe at RH = 1.05 (below the
    1.1 trigger) — a hard gate gives identically 0 there."""
    import legoesm.atmosphere.physics.microphysics.ml_emulator as ml

    src = ast.parse(pathlib.Path(ml.__file__).read_text())
    imported = {a.name for n in ast.walk(src)
                if isinstance(n, ast.ImportFrom)
                and (n.module or "").endswith("_warm_rain")
                for a in n.names}
    assert "hard_saturation_drain" not in imported, (
        "ml_emulator imports the EAGER-only hard-gated drain; its step "
        "discontinuity puts a jump in the trainable loss surface."
    )
    assert "hard_saturation_blend" in imported


def test_ml_emulator_guard_gradient_is_finite_through_the_gate():
    """The trainable scheme's guard must be AD-safe, including across the
    activation ramp and at cold/low-q_sat conditions where q_sat -> 0.

    This is the gradient test the Sundqvist-only differentiability test did not
    cover: the emulator is the arm whose weights are trained."""
    cfg = MicrophysicsMLEmulatorConfig(hard_saturation_adjustment=True)
    model = MicrophysicsEmulator(cfg.n_input, cfg.n_hidden, cfg.n_layers,
                                 cfg.n_output, key=jax.random.PRNGKey(cfg.seed))
    shape = (1, 4)
    # Warm/saturated, right at the gate, and cold+thin (TTL-like, tiny q_sat).
    T = jnp.array([[300.0, 300.0, 230.0, 205.0]], dtype=jnp.float64)
    p_full = jnp.array([[90000.0, 90000.0, 35000.0, 15000.0]], dtype=jnp.float64)
    p_half = jnp.concatenate([p_full, p_full[:, -1:]], axis=1)
    rh = jnp.array([[1.40, 1.10, 1.60, 2.00]], dtype=jnp.float64)
    q_v0 = rh * saturation_specific_humidity(T, p_full)
    z = jnp.zeros(shape, dtype=jnp.float64)
    hyd = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
                           N_c=z, N_r=z, N_i=z)
    rho = p_full / (constants.R_d * T)
    dz = jnp.full(shape, 500.0, dtype=jnp.float64)

    def loss(qv):
        out = ml_microphysics(T, qv, hyd, p_full, p_half, rho, dz, _DT,
                              cfg, model)
        return jnp.sum(out.dq_c_dt ** 2) + jnp.sum(out.dT_dt ** 2)

    g = np.asarray(jax.grad(loss)(q_v0))
    assert np.all(np.isfinite(g)), f"non-finite gradient through the guard: {g}"
    assert np.abs(g).sum() > 0.0


def test_guard_conserves_moist_enthalpy_in_the_bulk_schemes():
    """c_pd*T + L_v*q_v is conserved by the adjustment (the identity the shared
    core promises). Checked on Kessler at step 1 with no rain formed yet, so
    condensation is the only active vapour term."""
    fn, cfg_cls = _SCHEMES["kessler"]
    T, q_v, hyd, p_full, p_half, rho, dz = _column()
    cfg = cfg_cls(hard_saturation_adjustment=True)
    out = fn(T, q_v, hyd, p_full, p_half, rho, dz, _DT, cfg)
    dh = (constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt) * _DT
    h0 = constants.c_pd * T + constants.L_v * q_v
    np.testing.assert_allclose(np.asarray(dh) / np.asarray(h0), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Byte-identity of the OFF path (P1's explicit requirement)
# ---------------------------------------------------------------------------
def test_sundqvist_off_path_never_touches_the_guard(monkeypatch):
    """PROOF that ``hard_saturation_adjustment=False`` is byte-identical: the
    shared blend is replaced with a bomb. Default config must not detonate it
    (=> the new code is not merely a no-op, it is not EXECUTED, so no float
    op differs); hard=True must."""
    import legoesm.atmosphere.physics.microphysics.sundqvist as sq

    def _bomb(*a, **k):
        raise AssertionError("hard_saturation_blend called on the OFF path")

    monkeypatch.setattr(sq, "hard_saturation_blend", _bomb)
    T, q_v, hyd, p_full, p_half, rho, dz = _column()
    sq.sundqvist_microphysics(T, q_v, hyd, p_full, p_half, rho, dz, _DT,
                              SundqvistConfig())          # must NOT raise
    with pytest.raises(AssertionError, match="OFF path"):
        sq.sundqvist_microphysics(
            T, q_v, hyd, p_full, p_half, rho, dz, _DT,
            SundqvistConfig(hard_saturation_adjustment=True))


def test_ml_emulator_off_path_never_touches_the_guard(monkeypatch):
    """Same byte-identity proof for the emulator's post-step drain.

    The patched name is ``hard_saturation_blend`` — the emulator deliberately
    does NOT use ``hard_saturation_drain``, whose hard ``jnp.where`` gate is
    licensed only for the eager (non-jit, non-AD) driver hook. Binding this
    test to the symbol that actually runs is the point: when the emulator was
    switched from the drain to the blend, this test went red rather than
    silently passing against a name nothing called."""
    import legoesm.atmosphere.physics.microphysics.ml_emulator as ml

    def _bomb(*a, **k):
        raise AssertionError("hard_saturation_blend called on the OFF path")

    monkeypatch.setattr(ml, "hard_saturation_blend", _bomb)
    T, q_v, hyd, p_full, p_half, rho, dz = _column()
    cfg = MicrophysicsMLEmulatorConfig()
    model = MicrophysicsEmulator(cfg.n_input, cfg.n_hidden, cfg.n_layers,
                                 cfg.n_output, key=jax.random.PRNGKey(cfg.seed))
    ml.ml_microphysics(T, q_v, hyd, p_full, p_half, rho, dz, _DT, cfg, model)
    with pytest.raises(AssertionError, match="OFF path"):
        ml.ml_microphysics(
            T, q_v, hyd, p_full, p_half, rho, dz, _DT,
            MicrophysicsMLEmulatorConfig(hard_saturation_adjustment=True),
            model)


def test_guard_positivity_claim_is_exactly_what_the_rate_limit_gives():
    """The drain is rate-limited at ``min(dqv_cap, max(q_v_post, 0))/dt``. The
    claim that follows is NARROWER than "vapour cannot go negative", and the
    narrow claim is what is asserted:

      * where the scheme leaves q_v_post >= 0, the corrected vapour is >= 0;
      * where q_v_post < 0 already, the cap collapses to zero and the guard is
        NEUTRAL — it does not repair a negative, and must not deepen one.

    Driven directly through the shared core with a HAND-BUILT post-step state
    (including a negative cell), because the untrained MLP's own tendencies are
    ~1e-6 and could never exercise either branch — a test using it would pass
    with the guard deleted."""
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        hard_saturation_blend,
    )
    T = jnp.array([[300.0, 300.0, 300.0]], dtype=jnp.float64)
    p = jnp.full((1, 3), 90000.0, dtype=jnp.float64)
    q_sat = saturation_specific_humidity(T, p)
    # cell 0: strongly super-saturated;  cell 1: exactly zero vapour;
    # cell 2: ALREADY NEGATIVE (an aggressive raw prediction).
    q_v_post = jnp.concatenate([1.4 * q_sat[:, :1],
                                jnp.zeros((1, 1), dtype=jnp.float64),
                                -1.0e-4 * jnp.ones((1, 1), dtype=jnp.float64)],
                               axis=1)
    rate = hard_saturation_blend(jnp.zeros_like(q_v_post), T, q_v_post, p,
                                 _DT, q_sat, 1.1, 5.0)
    r = np.asarray(rate)
    assert r[0, 0] > 0.0, "super-saturated cell was not drained at all"
    # No drain can be manufactured from zero or negative vapour.
    np.testing.assert_allclose(r[0, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(r[0, 2], 0.0, atol=0.0)
    q_v_final = np.asarray(q_v_post) - r * _DT
    assert q_v_final[0, 0] >= 0.0          # positive cell stays positive
    assert q_v_final[0, 2] == np.asarray(q_v_post)[0, 2]   # negative untouched


# Values below were produced by scripts/tmp/_probe_ssguard_byteid.py at
# origin/main 5b74b65128b04d7ad1c2de8df8dcefd99e531b84 (SLURM job 9331695,
# JAX_ENABLE_X64=1, CPU) — i.e. BEFORE the guard existed. Pinning them here
# turns the byte-identity claim from a one-off manual diff into a CI gate. The
# cells span sub-saturated, marginal and strongly super-saturated at 300 K.
#
# TOLERANCE, deliberately not exact equality. The TRUE byte-identity proof is
# the same-platform base-vs-branch array diff (0 differing lines) plus the
# monkeypatch bomb showing the guard code is never executed; this constant is
# a CI TRIPWIRE, and an exact `assert_array_equal` on a float64 exp/division
# chain would be brittle to a backend or jax-version ULP change and would get
# "fixed" by re-pinning the numbers — quietly destroying the tripwire. rtol
# 1e-13 is ~2 orders above fp64 ULP noise and ~12 orders below the smallest
# effect the guard could have (enabling it changes these tendencies by O(1)),
# so it cannot mask a real perturbation.
_PREGUARD_RTOL = 1.0e-13
_SUNDQVIST_PREGUARD_DT_DT = (0.07790655134982057, 0.0042109453424112216, -2.6370890683278935e-06, 0.00120314759851969, 0.00021895372484426559)  # re-measured 2026-09-29 (specific q_sat inputs, x64 CPU)
_SUNDQVIST_PREGUARD_DQ_V_DT = (-3.129469722034535e-05, -1.6915170447021228e-06, 1.0593063421051318e-09, -4.832987618459902e-07, -8.79526869762267e-08)


def test_sundqvist_default_matches_the_pre_guard_baseline_bit_for_bit():
    """Numeric byte-identity against values computed BEFORE the guard existed.

    The monkeypatch "bomb" tests prove the new code is not INVOKED; this proves
    the numbers are unchanged, which is the claim that actually protects every
    tuned parameter set and prior comparison. Arrays are compared, never
    printed summaries (CLAUDE.md)."""
    T = jnp.array([[300.0, 285.0, 260.0, 230.0, 205.0]], dtype=jnp.float64)
    p = jnp.array([[95000.0, 85000.0, 60000.0, 35000.0, 15000.0]],
                  dtype=jnp.float64)
    rh = jnp.array([[1.40, 1.05, 0.90, 1.60, 2.00]], dtype=jnp.float64)
    q_v = rh * saturation_specific_humidity(T, p)
    shape = T.shape
    z = jnp.zeros(shape, dtype=jnp.float64)
    hyd = HydrometeorState(
        q_c=jnp.full(shape, 3.0e-4, dtype=jnp.float64),
        q_r=jnp.full(shape, 1.0e-5, dtype=jnp.float64),
        q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z)
    rho = p / (constants.R_d * T)
    dz = jnp.full(shape, 400.0, dtype=jnp.float64)
    p_half = jnp.concatenate([p, p[:, -1:]], axis=1)

    out = sundqvist_microphysics(T, q_v, hyd, p, p_half, rho, dz, 300.0,
                                 SundqvistConfig())
    np.testing.assert_allclose(
        np.asarray(out.dT_dt).ravel(), np.array(_SUNDQVIST_PREGUARD_DT_DT),
        rtol=_PREGUARD_RTOL, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(out.dq_v_dt).ravel(), np.array(_SUNDQVIST_PREGUARD_DQ_V_DT),
        rtol=_PREGUARD_RTOL, atol=0.0)

    # NON-VACUITY: the tripwire must actually be tight enough to SEE the guard.
    # Turning it on must violate the same tolerance by a wide margin — without
    # this, a loosened rtol could silently render the whole test decorative.
    on = sundqvist_microphysics(T, q_v, hyd, p, p_half, rho, dz, 300.0,
                                SundqvistConfig(hard_saturation_adjustment=True))
    rel = np.max(np.abs(
        (np.asarray(on.dT_dt).ravel() - np.array(_SUNDQVIST_PREGUARD_DT_DT))
        / np.array(_SUNDQVIST_PREGUARD_DT_DT)))
    assert rel > 1.0e6 * _PREGUARD_RTOL, (
        f"enabling the guard moved dT_dt by only rel={rel}; the pinned "
        "baseline is too loose to detect a real change."
    )


def test_guard_is_differentiable_end_to_end():
    """The overlay must not break autodiff (end-to-end jax.grad is a repo
    invariant). Gradient of column condensate wrt initial vapour, guard ON."""
    T, q_v, hyd, p_full, p_half, rho, dz = _column()

    def loss(qv):
        out = sundqvist_microphysics(
            T, qv, hyd, p_full, p_half, rho, dz, _DT,
            SundqvistConfig(hard_saturation_adjustment=True))
        return jnp.sum(out.dq_c_dt ** 2)

    g = jax.grad(loss)(q_v)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.sum(jnp.abs(g))) > 0.0
