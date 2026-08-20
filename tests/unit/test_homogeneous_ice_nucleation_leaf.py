"""Homogeneous (Koop/Ren-MacKenzie) cirrus ice nucleation: LEAF PHYSICS.

Companion to ``test_homogeneous_ice_nucleation_wiring.py`` (which pins the
config/CLI reachability).  This module calls ``morrison_microphysics`` itself
with the REAL signature ``(T, q_v, HydrometeorState, p_full, p_half, rho, dz,
dt, config)`` -- the wiring module's leaf check used a keyword form that does
not exist and therefore SKIPPED, leaving the physics untested.

What must hold for the flag to be both effective and safe in a century run:

* it fires ONLY above ``S_hom(T)`` and ONLY below ``hom_freeze_T_max``
  (no mixed-phase side effects from the nucleation source itself);
* above the threshold it produces a strictly larger vapour SINK than the
  Cooper-only path -- otherwise the wiring is inert;
* the nucleated NUMBER lands on ``hom_ice_nuc_N`` -- NOT on
  ``hom_ice_nuc_N + N_i_nuc_max`` (the Cooper source fires in the same step
  because its ``gate_ice`` opens at RH_ice >= 1.08, well below S_hom);
* one step never removes MORE than the available ice supersaturation
  (a one-sided cap makes a 2-step deposit/sublimate limit cycle);
* multi-step relaxation is monotone-ish, not a sawtooth;
* water is conserved to machine precision and gradients are finite and
  FD-accurate through the steep (sharpness 200) gate.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio_ice

_DT = 240.0  # the campaign timestep [s]
_DZ = 400.0  # layer thickness [m]


def _s_hom(cfg, T):
    """Ren & MacKenzie (2005) linear fit to the Koop et al. (2000) threshold."""
    return float(np.clip(cfg.koop_s_hom_a - cfg.koop_s_hom_b * T,
                         cfg.koop_s_hom_min, cfg.koop_s_hom_max))


def _run(T, p, rh_ice, flag, *, q_i=0.0, N_i=0.0, q_c=0.0, q_s=0.0,
         dt=_DT, cfg=None):
    """One ``morrison_microphysics`` call on a single (1, 1) column."""
    Ta = jnp.full((1, 1), T)
    pa = jnp.full((1, 1), p)
    q_sat_i = saturation_mixing_ratio_ice(Ta, pa)
    q_v = rh_ice * q_sat_i
    rho = pa / (constants.R_d * Ta)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=jnp.full((1, 1), q_c), q_r=z, q_i=jnp.full((1, 1), q_i),
        q_s=jnp.full((1, 1), q_s), q_g=z,
        N_c=z, N_r=z, N_i=jnp.full((1, 1), N_i),
    )
    config = (cfg or MorrisonConfig())._replace(homogeneous_ice_nucleation=flag)
    out = morrison_microphysics(
        Ta, q_v, hm, pa, jnp.full((1, 2), p), rho, jnp.full((1, 1), _DZ),
        dt, config,
    )
    return out, float(q_v[0, 0]), float(q_sat_i[0, 0])


# ---------------------------------------------------------------------------
# The flag must actually do something -- and only where it should
# ---------------------------------------------------------------------------

def test_fires_above_s_hom_and_consumes_supersaturation():
    """THE non-vacuous check: in ice-free cirrus above S_hom the flag must
    produce a strictly larger vapour sink than the Cooper-only path."""
    T, p = 228.0, 2.22e4                      # the day-150 autopsy cell
    rh = 2.88                                 # the observed runaway
    assert rh > _s_hom(MorrisonConfig(), T)
    off, _, _ = _run(T, p, rh, False)
    on, _, _ = _run(T, p, rh, True)
    dqv_off = float(off.dq_v_dt[0, 0])
    dqv_on = float(on.dq_v_dt[0, 0])
    assert dqv_off < 0.0, "Cooper seed mass should already be a small sink"
    assert dqv_on < 2.0 * dqv_off, (
        f"flag must MATERIALLY strengthen the vapour sink; got "
        f"{dqv_on:.4e} vs {dqv_off:.4e}")


def test_inert_below_s_hom():
    """Below the homogeneous threshold the ON path must be bit-identical to
    OFF -- if the gate leaked, the flag would be a global ice source."""
    T, p = 228.0, 2.22e4
    rh = 1.20                                  # < S_hom(228) ~ 1.47
    assert rh < _s_hom(MorrisonConfig(), T)
    off, _, _ = _run(T, p, rh, False)
    on, _, _ = _run(T, p, rh, True)
    # sigmoid(200*(1.20-1.47)) ~ 4e-24, so the gate is closed to ~1 ULP rather
    # than exactly (a SMOOTH gate is the point -- a hard cut would kill AD).
    assert float(on.dq_v_dt[0, 0]) == pytest.approx(
        float(off.dq_v_dt[0, 0]), rel=1e-12)
    assert float(on.dN_i_dt[0, 0]) == pytest.approx(
        float(off.dN_i_dt[0, 0]), rel=1e-12)


def test_cold_gate_blocks_the_warm_mixed_phase():
    """The nucleation SOURCE must not fire above hom_freeze_T_max even at
    extreme ice supersaturation (homogeneous freezing of aqueous haze needs
    T <~ -38 C)."""
    cfg = MorrisonConfig()
    T = cfg.hom_freeze_T_max + 15.0            # 250 K
    off, _, _ = _run(T, 4.0e4, 1.6, False)
    on, _, _ = _run(T, 4.0e4, 1.6, True)
    # dN_i_dt is dominated by the nucleation source in this ice-free cell.
    assert float(on.dN_i_dt[0, 0]) == pytest.approx(
        float(off.dN_i_dt[0, 0]), rel=1e-3), "cold gate leaked into 250 K"


@pytest.mark.parametrize("T,q_i,N_i", [
    (260.0, 1e-3, 1e6), (255.0, 5e-4, 5e6), (245.0, 1e-3, 1e7)])
def test_documented_warm_mixed_phase_side_effect(T, q_i, N_i):
    """PIN a KNOWN, DELIBERATE side effect so it cannot surprise anyone.

    The ON-path deposition caps are gated on the config flag but NOT on the
    homogeneous RH/T gate, so enabling cirrus nucleation also BOUNDS explicit
    mixed-phase (WBF) deposition at temperatures where no homogeneous freezing
    occurs -- up to a 7.6x rate reduction.  This is a physics change beyond
    cirrus, accepted here because it moves toward boundedness: the DEFAULT path
    removes up to 5.8x the available ice supersaturation in one 240 s Euler
    step (an unphysical overshoot that flips the sign of the next step), while
    the capped path removes <= ~1.1x.

    Making the cap unconditional would fix the default path too, but changes
    every in-flight run, so it is deferred to its own reviewed change.  If that
    lands, THIS test is the one to update.
    """
    off, q_v, q_sat_i = _run(T, 5.0e4, 1.20, False, q_i=q_i, N_i=N_i, q_c=1e-4)
    on, _, _ = _run(T, 5.0e4, 1.20, True, q_i=q_i, N_i=N_i, q_c=1e-4)
    excess = q_v - q_sat_i
    off_frac = -float(off.dq_v_dt[0, 0]) * _DT / excess
    on_frac = -float(on.dq_v_dt[0, 0]) * _DT / excess
    assert off_frac > 1.5, (
        f"default path no longer overshoots ({off_frac:.2f}) -- if the cap was "
        "made unconditional, delete this test")
    assert on_frac < 1.2, f"ON path overshoots too ({on_frac:.2f})"
    assert on_frac < off_frac


# ---------------------------------------------------------------------------
# No double-count against the Cooper source
# ---------------------------------------------------------------------------

def test_nucleated_number_lands_on_the_homogeneous_target():
    """Cooper's ``gate_ice`` (RH_ice >= 1.08) is ALWAYS open when the
    homogeneous gate (RH_ice >= S_hom ~ 1.47) is, so both sources fire in the
    same step.  N_i must top up TO ``hom_ice_nuc_N``, not to
    ``hom_ice_nuc_N + N_i_nuc_max`` (which was 1.5x too many crystals)."""
    cfg = MorrisonConfig()
    T, p = 228.0, 2.22e4
    rho = p / (constants.R_d * T)
    on, _, _ = _run(T, p, 2.88, True)
    got = float(on.dN_i_dt[0, 0]) * _DT
    target = cfg.hom_ice_nuc_N / rho
    cooper_cap = cfg.N_i_nuc_max / rho
    assert got == pytest.approx(target, rel=0.02), (
        f"nucleated {got:.4e}/kg vs homogeneous target {target:.4e}/kg "
        f"(sum-with-Cooper would be {target + cooper_cap:.4e})")
    assert got < target + 0.5 * cooper_cap, "double-counted the Cooper top-up"


def test_number_ceiling_holds_with_concurrent_droplet_freezing():
    """Below ``homogeneous_freeze_T`` (233 K) WITH cloud water present, the
    homogeneous DROPLET-freezing number source (a different process: q_c -> q_i
    + L_f) fires in the same step as cirrus nucleation.  Both relaxed toward
    their target from the OLD N_i, stacking to ~1.5x the ceiling.  Adding
    liquid water must not inflate the ice number."""
    cfg = MorrisonConfig()
    T, p = 228.0, 2.22e4
    assert T < cfg.homogeneous_freeze_T, "regime must activate droplet freezing"
    dry, _, _ = _run(T, p, 2.00, True)
    wet, _, _ = _run(T, p, 2.00, True, q_c=1.0e-4)
    target = cfg.hom_ice_nuc_N / (p / (constants.R_d * T))
    assert float(wet.dN_i_dt[0, 0]) * _DT == pytest.approx(
        float(dry.dN_i_dt[0, 0]) * _DT, rel=0.02), "droplet freezing stacked"
    assert float(wet.dN_i_dt[0, 0]) * _DT <= target * 1.02


# ---------------------------------------------------------------------------
# Boundedness: never remove more vapour than is available above ice saturation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rh", [1.47, 1.60, 2.00, 2.88, 6.0])
def test_one_step_never_overshoots_ice_saturation(rh):
    """The deposition cap must be SYMMETRIC in effect: a single step may not
    drive q_v meaningfully below q_sat_i, or the next step sublimates it all
    back (a 2-step limit cycle with a +/-1 K/step latent-heating sawtooth).

    Bound is ``excess + seed``: the MI0 nucleation seed mass (Cooper +
    homogeneous, ``dN_i*mi0``) is a SEPARATE vapour sink that is not, and
    should not be, inside the deposition cap -- it is mass carried by the new
    crystals themselves.  Liquid condensation (RH_liq > 1 above RH_i ~ 1.54 at
    228 K) is a third legitimate channel targeting q_sat_liq, which is why the
    2 % head-room matters at rh=6.  Pre-fix this ran at 1.13-1.36.
    """
    cfg = MorrisonConfig()
    T, p = 228.0, 2.22e4
    on, q_v, q_sat_i = _run(T, p, rh, True)
    removed = -float(on.dq_v_dt[0, 0]) * _DT
    excess = q_v - q_sat_i
    mi0 = 4.0 / 3.0 * np.pi * cfg.rho_cloud_ice * cfg.ice_nuc_radius ** 3
    seed = float(on.dN_i_dt[0, 0]) * _DT * mi0
    assert 0.0 < removed <= excess + seed + 0.02 * excess, (
        f"removed {removed:.4e} of an available {excess:.4e} "
        f"(+ seed {seed:.4e}); fraction {removed / excess:.4f}")


def _relax(flag, *, q_s0=0.0, n=25, rh0=2.88, T0=228.0, p=2.22e4):
    """RH_ice after each of ``n`` explicit steps, no vapour supply."""
    pa = jnp.full((1, 1), p)
    q_sat_i0 = float(saturation_mixing_ratio_ice(jnp.full((1, 1), T0), pa)[0, 0])
    q_v, q_i, q_c, q_s, N_i, Tc = rh0 * q_sat_i0, 0.0, 0.0, q_s0, 0.0, T0
    traj = []
    cfg = MorrisonConfig()._replace(homogeneous_ice_nucleation=flag)
    for _ in range(n):
        Ta = jnp.full((1, 1), Tc)
        z = jnp.zeros((1, 1))
        hm = HydrometeorState(
            q_c=jnp.full((1, 1), q_c), q_r=z, q_i=jnp.full((1, 1), q_i),
            q_s=jnp.full((1, 1), q_s), q_g=z, N_c=z, N_r=z,
            N_i=jnp.full((1, 1), N_i))
        out = morrison_microphysics(
            Ta, jnp.full((1, 1), q_v), hm, pa, jnp.full((1, 2), p),
            pa / (constants.R_d * Ta), jnp.full((1, 1), _DZ), _DT, cfg)
        q_v = q_v + _DT * float(out.dq_v_dt[0, 0])
        q_i = max(q_i + _DT * float(out.dq_i_dt[0, 0]), 0.0)
        q_c = max(q_c + _DT * float(out.dq_c_dt[0, 0]), 0.0)
        q_s = max(q_s + _DT * float(out.dq_s_dt[0, 0]), 0.0)
        N_i = max(N_i + _DT * float(out.dN_i_dt[0, 0]), 0.0)
        Tc = Tc + _DT * float(out.dT_dt[0, 0])
        traj.append(q_v / float(
            saturation_mixing_ratio_ice(jnp.full((1, 1), Tc), pa)[0, 0]))
    return traj


@pytest.mark.parametrize("q_s0", [0.0, 1e-5, 1e-4, 5e-4])
def test_multi_step_relaxation_does_not_oscillate(q_s0):
    """25 explicit steps from the observed runaway with no vapour supply: the
    ON path must settle, not ring.  Pre-fix this ran
    2.88 -> 0.27 -> 2.22 -> 0.61 -> 2.19 ...

    Parametrised over pre-existing SNOW because snow sublimation (`prds`) has
    only a q_s donor limit, NOT the vapour-deficit cap cloud ice now has -- so
    it is the obvious candidate to re-introduce the sawtooth.  It does not.
    """
    traj = _relax(True, q_s0=q_s0)
    assert min(traj) > 0.9, f"undershoot below ice saturation: {traj[:6]}"
    assert max(traj[1:]) < 1.5, f"rebound above S_hom: {traj[:6]}"
    assert traj[-1] == pytest.approx(1.0, abs=0.05), traj[-6:]


@pytest.mark.parametrize("q_g0", [1e-5, 1e-4, 5e-4, 1e-3])
def test_multi_step_relaxation_with_pre_existing_graupel(q_g0):
    """Graupel counterpart of the snow case.

    Unlike snow, heavy graupel DOES deepen the first-step undershoot on the ON
    path (measured RH_ice 2.88 -> 0.52 at q_g = 1 g/kg, vs 1.03 with no
    graupel).  Instrumented cause -- NOT the ice branch:

      total removed = 4.546e-4 = liquid condensation 1.259e-4
                                + cloud ice (dep+seed) 2.204e-4
                                + graupel deposition  1.083e-4
      ice-phase subtotal 3.287e-4 < available ice supersaturation 3.754e-4

    so the joint ice-phase bound is SATISFIED; the extra draw is LIQUID
    condensation targeting q_sat_l (3.07e-4) while ice deposition targets
    q_sat_i (2.00e-4).  Two independent relaxations toward two different
    equilibria on one vapour reservoir is a PRE-EXISTING structural issue of
    the mixed-phase closure (a joint saturation adjustment would fix it); the
    flag only amplifies it by making ice deposition strong.  About half the
    apparent RH drop is also q_sat_i rising with the L_s heating (+1.24 K).

    So the invariant asserted here is the one that actually matters: bounded,
    and converging FASTER than the default path (which rings for ~8 steps).
    """
    p, T0 = 2.22e4, 228.0
    pa = jnp.full((1, 1), p)
    q_sat_i0 = float(saturation_mixing_ratio_ice(jnp.full((1, 1), T0), pa)[0, 0])
    q_v, q_i, q_c, q_g, N_i, Tc = 2.88 * q_sat_i0, 0.0, 0.0, q_g0, 0.0, T0
    traj = []
    cfg = MorrisonConfig()._replace(homogeneous_ice_nucleation=True)
    for _ in range(25):
        Ta = jnp.full((1, 1), Tc)
        z = jnp.zeros((1, 1))
        hm = HydrometeorState(
            q_c=jnp.full((1, 1), q_c), q_r=z, q_i=jnp.full((1, 1), q_i),
            q_s=z, q_g=jnp.full((1, 1), q_g), N_c=z, N_r=z,
            N_i=jnp.full((1, 1), N_i))
        out = morrison_microphysics(
            Ta, jnp.full((1, 1), q_v), hm, pa, jnp.full((1, 2), p),
            pa / (constants.R_d * Ta), jnp.full((1, 1), _DZ), _DT, cfg)
        q_v = q_v + _DT * float(out.dq_v_dt[0, 0])
        q_i = max(q_i + _DT * float(out.dq_i_dt[0, 0]), 0.0)
        q_c = max(q_c + _DT * float(out.dq_c_dt[0, 0]), 0.0)
        q_g = max(q_g + _DT * float(out.dq_g_dt[0, 0]), 0.0)
        N_i = max(N_i + _DT * float(out.dN_i_dt[0, 0]), 0.0)
        Tc = Tc + _DT * float(out.dT_dt[0, 0])
        traj.append(q_v / float(
            saturation_mixing_ratio_ice(jnp.full((1, 1), Tc), pa)[0, 0]))
    assert min(traj) > 0.4, f"unbounded undershoot: {traj[:6]}"
    assert max(traj[1:]) < 1.5, f"rebound above S_hom: {traj[:6]}"
    # settles within 5 steps and STAYS settled (the default path needs ~8-10)
    assert all(abs(v - 1.0) < 0.02 for v in traj[5:]), traj[:8]


@pytest.mark.parametrize("q_s0,N_s_present", [(0.0, False), (1e-4, True)])
def test_burst_number_is_invariant_to_snow(q_s0, N_s_present):
    """DOCUMENTS AN ACCEPTED DESIGN DECISION (do not "fix" without reading).

    Cooper subtracts the TOTAL frozen number N_i+N_s+N_g because it models an
    ice-NUCLEI budget (morrison.py:347-358).  Homogeneous freezing of aqueous
    haze has no such budget -- every haze droplet freezes once the Koop water
    activity threshold is crossed, however many snowflakes are nearby.  What
    physically suppresses it is pre-existing ice QUENCHING the supersaturation,
    which the `rh_ice >= S_hom` gate captures dynamically.  So the burst number
    is deliberately independent of N_s/N_g -- and bounded at the target either
    way, which is what makes that safe."""
    cfg = MorrisonConfig()
    T, p = 228.0, 2.22e4
    on, _, _ = _run(T, p, 2.88, True, q_s=q_s0)
    target = cfg.hom_ice_nuc_N / (p / (constants.R_d * T))
    assert float(on.dN_i_dt[0, 0]) * _DT == pytest.approx(target, rel=0.02)


def test_on_path_relaxation_is_better_damped_than_off():
    """Non-vacuous framing of the above: the OFF path rings for ~10 steps
    (2.15 1.13 0.90 1.17 0.88 1.14 ...).  ON must be strictly better, or the
    symmetric cap bought nothing."""
    on, off = _relax(True), _relax(False)
    assert min(on) > min(off), (min(on), min(off))
    assert abs(on[3] - 1.0) < abs(off[3] - 1.0), (on[:5], off[:5])


# ---------------------------------------------------------------------------
# Conservation + differentiability with the flag ON
# ---------------------------------------------------------------------------

def _water_residual(out):
    return sum(float(getattr(out, f"dq_{s}_dt")[0, 0])
               for s in ("v", "c", "r", "i", "s", "g"))


def test_total_water_conserved_with_the_flag_on():
    """Sum of all dq/dt = 0 to machine precision when nothing sediments."""
    on, _, _ = _run(228.0, 2.22e4, 2.88, True)
    scale = abs(float(on.dq_v_dt[0, 0]))
    assert abs(_water_residual(on)) < 1e-12 * scale, (
        f"residual {_water_residual(on):.3e} vs sink scale {scale:.3e}")


def test_flag_adds_no_water_imbalance_when_condensate_sediments():
    """With pre-existing q_i/q_s the local dq/dt sum is legitimately non-zero
    (condensate falls OUT of the single layer), so the invariant that matters
    is that the flag changes the residual by NOTHING: same sedimentation, no
    new leak, despite a 2.7x larger vapour sink."""
    kw = dict(q_s=1e-5, q_i=1e-6, N_i=1e4)
    off, _, _ = _run(228.0, 2.22e4, 2.88, False, **kw)
    on, _, _ = _run(228.0, 2.22e4, 2.88, True, **kw)
    assert abs(float(on.dq_v_dt[0, 0])) > 2.0 * abs(float(off.dq_v_dt[0, 0]))
    assert _water_residual(on) == pytest.approx(
        _water_residual(off), rel=1e-10)


def test_gradients_are_finite_and_match_finite_differences():
    """sharpness=200 is a steep gate; AD must still be exact and finite."""
    T, p = 228.0, 2.22e4
    q_sat_i = float(saturation_mixing_ratio_ice(
        jnp.full((1, 1), T), jnp.full((1, 1), p))[0, 0])

    def sink(q_v_scalar, T_scalar):
        Ta = jnp.full((1, 1), T_scalar)
        pa = jnp.full((1, 1), p)
        z = jnp.zeros((1, 1))
        hm = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
                              N_c=z, N_r=z, N_i=z)
        out = morrison_microphysics(
            Ta, jnp.full((1, 1), q_v_scalar), hm, pa, jnp.full((1, 2), p),
            pa / (constants.R_d * Ta), jnp.full((1, 1), _DZ), _DT,
            MorrisonConfig()._replace(homogeneous_ice_nucleation=True))
        return jnp.sum(out.dq_v_dt)

    for rh in (1.40, 1.47, 1.55, 2.88):      # below / at / above the gate
        g_qv, g_T = jax.grad(sink, argnums=(0, 1))(rh * q_sat_i, T)
        assert np.isfinite(float(g_qv)) and np.isfinite(float(g_T))
        h = 1e-7 * q_sat_i
        fd = float((sink(rh * q_sat_i + h, T) - sink(rh * q_sat_i - h, T))
                   / (2.0 * h))
        assert float(g_qv) == pytest.approx(fd, rel=1e-4, abs=1e-12), (
            f"RH_i={rh}: grad {float(g_qv):.6e} vs FD {fd:.6e}")
    # non-vacuous: the gradient is NOT dead at the runaway point
    g_qv, _ = jax.grad(sink, argnums=(0, 1))(2.88 * q_sat_i, T)
    assert abs(float(g_qv)) > 1e-6


def test_jit_matches_eager():
    args = (228.0, 2.22e4, 2.88, True)
    eager, _, _ = _run(*args)
    Ta = jnp.full((1, 1), 228.0)
    pa = jnp.full((1, 1), 2.22e4)
    q_v = 2.88 * saturation_mixing_ratio_ice(Ta, pa)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
                          N_c=z, N_r=z, N_i=z)
    cfg = MorrisonConfig()._replace(homogeneous_ice_nucleation=True)
    jitted = jax.jit(morrison_microphysics, static_argnums=(7, 8))(
        Ta, q_v, hm, pa, jnp.full((1, 2), 2.22e4),
        pa / (constants.R_d * Ta), jnp.full((1, 1), _DZ), _DT, cfg)
    assert float(jitted.dq_v_dt[0, 0]) == pytest.approx(
        float(eager.dq_v_dt[0, 0]), rel=1e-12)


def test_float32_is_finite_through_the_steep_gate():
    """The steep (200) gate and the 1/rho number target must not overflow in
    fp32 -- MPAS production has run both precisions."""
    jax.config.update("jax_enable_x64", False)
    try:
        for T, p, rh in ((228.0, 2.22e4, 2.88), (190.0, 1.0e4, 5.0),
                         (205.0, 8.0e3, 1.6)):
            on, _, _ = _run(np.float32(T), np.float32(p), np.float32(rh), True)
            for name in ("dq_v_dt", "dq_i_dt", "dN_i_dt", "dT_dt"):
                v = float(getattr(on, name)[0, 0])
                assert np.isfinite(v), f"{name} not finite at T={T} in fp32"
    finally:
        jax.config.update("jax_enable_x64", True)
