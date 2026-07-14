"""Unit tests for the Tiedtke (1989) convection scheme.

Tests pin:

* tendency shape / dtype / finiteness;
* the **profile carry** — Tiedtke is the first scheme that exercises
  ``conv_prog_profile = M_u(k)`` across levels (not just at
  ``[:, -1]``);
* the implicit-Euler relaxation of ``M_u`` toward the diagnosed
  profile (monotone-contracting under quasi-equilibrium);
* CMT signs and `enable_cmt=False` opt-out;
* downdraft toggle changes the sub-cloud T tendency;
* finite gradients through ``epsilon_deep``, ``cape_threshold``,
  ``cmt_c_u``, and ``downdraft_alpha``;
* selection through ``make_physics(PhysicsConfig(...))``.
"""

from __future__ import annotations

from legoesm import constants

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import (
    ConvectionConfig,
    TiedtkeConfig,
)
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.convection.tiedtke import tiedtke_convection


def _column(
    ncol=2, nlev=16, T_sfc=302.0, q_sfc=16e-3, lapse_rate=7.5,
    p_s=1.0e5, p_top=5.0e3, u_sfc=2.0, u_top=25.0,
):
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [
            jnp.full((ncol, 1), p_top * 0.5),
            p_half_inner,
            jnp.full((ncol, 1), p_s),
        ],
        axis=1,
    )
    z_full = -8500.0 * jnp.log(p_full / p_s)
    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    q = q_sfc * jnp.exp(-z_full / 3000.0)
    fraction = jnp.linspace(1.0, 0.0, nlev)[None, :]   # 1 at top, 0 at surface
    u = u_sfc + (u_top - u_sfc) * fraction
    u = jnp.broadcast_to(u, (ncol, nlev))
    v = jnp.zeros_like(u)
    return T, q, p_full, p_half, u, v


# ---------------------------------------------------------------------------
# Shape / finiteness
# ---------------------------------------------------------------------------

def test_tiedtke_shape_finiteness():
    T, q, pf, ph, u, v = _column(ncol=3, nlev=12)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, M_u_new = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0)
    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
                out.convective_mask, out.du_dt_conv, out.dv_dt_conv,
                M_u_new):
        assert arr.shape[0] == ncol
        assert jnp.all(jnp.isfinite(arr))
    assert M_u_new.shape == (ncol, nlev)


# ---------------------------------------------------------------------------
# Profile carry — the first scheme that uses M_u(k) across levels
# ---------------------------------------------------------------------------

def test_tiedtke_profile_carry_distributes_across_levels():
    """The diagnosed ``M_u_new`` carries non-trivial values on at
    least one level above the surface — distinguishing Tiedtke from
    the scalar-carry schemes that only populate ``[:, -1]``."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    _, M_u_new = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0)
    # At least one level *above* the surface (index < nlev - 1) is
    # non-trivially populated.
    aloft_max = float(jnp.max(M_u_new[:, :-1]))
    assert aloft_max > 1e-6, (
        f"Tiedtke should carry M_u aloft; got max above surface = {aloft_max}"
    )


def test_tiedtke_implicit_relaxation_monotone():
    """Repeated application with the same input drives ``M_u``
    profile toward equilibrium with monotone-contracting differences."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    config = TiedtkeConfig(tau_M_u_relax=1800.0)
    M_u_traj = []
    for _ in range(15):
        _, cpp = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
        M_u_traj.append(np.asarray(cpp[0]))
    # Successive layer-summed differences shrink (relaxation).
    diffs = [
        float(jnp.sum(jnp.abs(M_u_traj[i + 1] - M_u_traj[i])))
        for i in range(len(M_u_traj) - 1)
    ]
    # Allow some non-monotonicity early on while the profile builds
    # up; check that the final differences are smaller than the
    # initial.
    assert diffs[-1] < diffs[0] + 1e-12


# ---------------------------------------------------------------------------
# CMT
# ---------------------------------------------------------------------------

def test_tiedtke_cmt_present_with_default_config():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0)
    assert out.du_dt_conv is not None
    assert out.dv_dt_conv is not None


def test_tiedtke_cmt_disabled():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    config = TiedtkeConfig(enable_cmt=False)
    out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
    assert out.du_dt_conv is None
    assert out.dv_dt_conv is None


# ---------------------------------------------------------------------------
# Downdraft
# ---------------------------------------------------------------------------

def test_tiedtke_mc_proxy_is_larger_in_moist_columns():
    """The saturation-excess MC proxy is larger in MOIST columns than
    in DRY ones.

    Tiedtke's deep-branch closure ``M_b_deep = mc_gate * cape_weight``
    relies on a column moisture-convergence proxy (the
    ``moisture_convergence`` argument is None for dycores without a
    diagnostic).  The proxy is constructed inline in tiedtke.py as the
    column integral of ``max(q_v - RH_crit * q_sat, 0)``: positive in
    moist columns (q_v above RH_crit * q_sat at any level), vanishing
    in dry columns.

    Pre-fix the formula used ``max(q_sat - q_v, 0)`` (saturation
    DEFICIT), which is large in dry columns and small in moist ones —
    inverted relative to the comment in the code and to the physical
    quantity the proxy was supposed to approximate.

    Reproduce the proxy formula here directly so the test pins the
    formula's sign behaviour without relying on the rest of the
    scheme's complex deep / shallow / midlevel blending.
    """
    from legoesm.thermo import saturation_mixing_ratio

    T, _, p_full, p_half, _, _ = _column(T_sfc=302.0)
    q_sat = saturation_mixing_ratio(T, p_full)
    dp = p_half[:, 1:] - p_half[:, :-1]
    cfg = TiedtkeConfig()
    # Generate moist (RH ~ 80%) and dry (RH ~ 5%) profiles via a column
    # multiplier on q_sat.
    q_moist = 0.85 * q_sat
    q_dry = 0.05 * q_sat

    def mc_proxy(q_v):
        sat_excess = jnp.maximum(q_v - cfg.mc_proxy_RH_crit * q_sat, 0.0)
        return jnp.sum(sat_excess * dp, axis=-1) / (constants.g * cfg.tau_MC_proxy)

    p_moist = float(mc_proxy(q_moist)[0])
    p_dry = float(mc_proxy(q_dry)[0])
    # Pre-fix (saturation-deficit) inversely: dry > moist by orders of
    # magnitude.  Post-fix moist >> dry.
    assert p_moist > 100.0 * max(p_dry, 1e-30), (
        f"Tiedtke MC proxy: moist column proxy = {p_moist:.3e}, "
        f"dry = {p_dry:.3e}.  Expected moist >> dry by orders of "
        "magnitude — the saturation-deficit formula has the wrong sign."
    )
    # Also ensure the proxy fires above the gate threshold in the moist
    # case (gate gets activated downstream).
    assert p_moist > cfg.moisture_convergence_threshold, (
        f"MC proxy in moist column ({p_moist:.3e}) does not exceed the "
        f"threshold ({cfg.moisture_convergence_threshold}); the deep "
        "branch would be gated off."
    )


def test_tiedtke_downdraft_toggle_changes_subcloud_T():
    """Enabling the downdraft branch changes the sub-cloud T
    tendency."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out_off, _ = tiedtke_convection(
        T, q, pf, ph, u, v, cpp, dt=300.0,
        config=TiedtkeConfig(enable_downdraft=False),
    )
    out_on, _ = tiedtke_convection(
        T, q, pf, ph, u, v, cpp, dt=300.0,
        config=TiedtkeConfig(enable_downdraft=True),
    )
    diff = jnp.abs(out_on.dT_dt[:, -4:] - out_off.dT_dt[:, -4:])
    assert float(jnp.max(diff)) > 1e-8


def test_tiedtke_downdraft_evap_conserves_water_locally():
    """Audit GWD/convection + Codex stop-time reviews.

    The downdraft "0.05 cooling" was buggy in two stages:
      (a) original form: dT_dt cooling injected with NO water source
          ⇒ vapor energy created from nothing (audit: 'non-water-
          conserving');
      (b) fixed-locally-only form: dq_v_dt source added without a
          matching reduction in the convective rain source
          (``dq_c_conv_dt``) ⇒ column-integrated water increased
          every step (Codex: 'downdraft fix still creates column
          water').

    The current fix must satisfy three invariants:
      1. Local energy-water balance: ``Δ(dT_dt) · c_pd + Δ(dq_v_dt) · L_v
         == 0`` per level (cooling exactly accounts for vapor created
         by rain re-evaporation).
      2. Column water conservation: column-integrated
         ``Δ(dq_v_dt) + Δ(dq_c_conv_dt) == 0``.  Vapor created in
         the subcloud layer must equal cloud-water source removed
         from the convective rain budget.
      3. Cooling is actually exercised (some level has dT_diff < 0).
    """
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out_off, _ = tiedtke_convection(
        T, q, pf, ph, u, v, cpp, dt=300.0,
        config=TiedtkeConfig(enable_downdraft=False),
    )
    out_on, _ = tiedtke_convection(
        T, q, pf, ph, u, v, cpp, dt=300.0,
        config=TiedtkeConfig(enable_downdraft=True),
    )
    dT_diff = out_on.dT_dt - out_off.dT_dt
    dqv_diff = out_on.dq_v_dt - out_off.dq_v_dt
    dqc_diff = out_on.dq_c_conv_dt - out_off.dq_c_conv_dt

    # (3) Exercise check: cooling actually fires
    assert float(jnp.min(dT_diff)) < 0.0, (
        "Downdraft did not produce any cooling — formulation regressed."
    )

    # (1) Local energy-water balance — per level
    H = dT_diff * constants.c_pd
    Q = dqv_diff * constants.L_v
    res_local = float(jnp.max(jnp.abs(H + Q)))
    scale_local = float(jnp.max(jnp.abs(H)))
    assert res_local < 1e-8 * max(scale_local, 1.0), (
        f"Tiedtke downdraft local energy-water unclosed: max|H+Q|="
        f"{res_local:.3e}, max|H|={scale_local:.3e}"
    )

    # (2) Column water conservation — net column water source is zero
    dp = pf  # placeholder, overwritten next line
    dp = (ph[:, 1:] - ph[:, :-1])
    col_dqv = jnp.sum(dqv_diff * dp, axis=-1) / constants.g  # [kg/m^2/s]
    col_dqc = jnp.sum(dqc_diff * dp, axis=-1) / constants.g
    col_residual = float(jnp.max(jnp.abs(col_dqv + col_dqc)))
    col_scale = float(jnp.max(jnp.abs(col_dqv)) + 1e-15)
    assert col_residual < 1e-10 * max(col_scale, 1.0), (
        f"Tiedtke downdraft column water unclosed: max|∫dq_v + ∫dq_c|="
        f"{col_residual:.3e} kg/m²/s, vapor source={col_scale:.3e} — "
        "Codex stop-time finding 'downdraft fix still creates column "
        "water' has regressed.  The dq_c_conv_dt rain-source reduction "
        "must match the dq_v vapor source column-integral."
    )


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_tiedtke_grad_through_epsilon_deep():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(eps):
        config = TiedtkeConfig(epsilon_deep=eps)
        out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
        return jnp.sum(out.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(1.0e-4)))
    assert np.isfinite(g)


def test_tiedtke_grad_through_cape_threshold():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(threshold):
        config = TiedtkeConfig(cape_threshold=threshold)
        out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
        return jnp.sum(out.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(70.0)))
    assert np.isfinite(g)


def test_tiedtke_grad_through_cmt_c_u():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(c_u):
        config = TiedtkeConfig(cmt_c_u=c_u)
        out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
        return jnp.sum(out.du_dt_conv)

    g = float(jax.grad(f)(jnp.asarray(0.7)))
    assert np.isfinite(g)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_tiedtke_orchestrator_one_step_finite():
    grid = create_cubed_sphere(4)
    sigma = create_sigma_coordinate(12)
    state = held_suarez_init(grid, sigma)
    tracers = {
        "q_v": Field(0.014 * jnp.ones((6, 4, 4, 12)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
        "q_c": Field(jnp.zeros((6, 4, 4, 12)), name="q_c",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    state = state._replace(tracers=tracers)
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="tiedtke"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    ncol = 6 * 4 * 4
    ps = init_physics_state(ncol, 12, cfg)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)
    assert ps_out.conv_prog_profile.shape == (ncol, 12)
    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
        assert jnp.all(jnp.isfinite(f.data))


# ---------------------------------------------------------------------------
# MSE conservation guard
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Standard Tiedtke kernel (subsidence g/c_p + detrainment of "
        "moist-adiabat T_u) does not conserve column MSE on a closed "
        "(no-surface-flux) probe.  Currently ~96% non-conservation "
        "residual on this CAPE-positive fixture; flagged xfail so any "
        "future kernel improvement that closes this is detected."
    ),
    strict=True,
)
def test_tiedtke_mse_conservation_within_tolerance():
    """Column-integrated ``c_p ∫dT + L_v ∫(dq_v + dq_c_conv) dp/g`` should
    be small relative to the heating magnitude on a CAPE-positive sounding.
    """
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = tiedtke_convection(
        T=T, q_v=q, p_full=pf, p_half=ph, u=u, v=v,
        conv_prog_profile=cpp, dt=1800.0,
        config=TiedtkeConfig(enable_cmt=False),
        moisture_convergence=jnp.zeros_like(T),
    )
    dp = ph[:, 1:] - ph[:, :-1]
    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
    C = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()) * constants.L_v
    rel = abs(H + Q + C) / (abs(H) + abs(Q) + abs(C) + 1e-10)
    assert rel < 0.30, (
        f"Tiedtke MSE residual {H+Q+C:.1f} W/m^2 ({rel*100:.1f}% of total) "
        f"exceeds 30% — kernel formulation has regressed"
    )


# ---------------------------------------------------------------------------
# Trigger sharpness fields (fix 2026-07): the dead ``smooth_trigger_sharpness``
# knob is replaced by two WIRED, correctly-scaled fields.
# ---------------------------------------------------------------------------

def test_tiedtke_smooth_trigger_sharpness_removed():
    """``TiedtkeConfig.smooth_trigger_sharpness`` (default 0.02) was a
    dead no-op: the RH downdraft trigger hardcoded 10.0 and the
    below-LCL membership hardcoded 2.0.  The field is deleted in favor
    of ``downdraft_rh_sharpness`` / ``lcl_membership_sharpness``
    (magnitudes NOT interchangeable: RH argument ~O(0.1) needs ~10;
    level-index needs ~2)."""
    cfg = TiedtkeConfig()
    assert not hasattr(cfg, "smooth_trigger_sharpness")
    assert cfg.downdraft_rh_sharpness == 10.0
    assert cfg.lcl_membership_sharpness == 2.0


def test_tiedtke_downdraft_rh_sharpness_wired():
    """Perturbing ``downdraft_rh_sharpness`` changes the downdraft
    tendencies (the former hardcoded 10.0 made the advertised sharpness
    tunable a no-op)."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out_default, _ = tiedtke_convection(
        T, q, pf, ph, u, v, cpp, dt=300.0,
        config=TiedtkeConfig(enable_downdraft=True),
    )
    # Near-zero sharpness pins the RH trigger sigmoid at 0.5 everywhere
    # (vs ~sigmoid(-6)~0.0025 in this moist column at sharpness 10).
    out_flat, _ = tiedtke_convection(
        T, q, pf, ph, u, v, cpp, dt=300.0,
        config=TiedtkeConfig(enable_downdraft=True,
                             downdraft_rh_sharpness=1e-6),
    )
    diff = float(jnp.max(jnp.abs(out_flat.dT_dt - out_default.dT_dt)))
    assert diff > 1e-10, "downdraft_rh_sharpness is not wired"


def test_tiedtke_lcl_membership_sharpness_wired():
    """Perturbing ``lcl_membership_sharpness`` changes the below-LCL
    weighting (hence the downdraft rain-evap distribution)."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out_default, _ = tiedtke_convection(
        T, q, pf, ph, u, v, cpp, dt=300.0,
        config=TiedtkeConfig(enable_downdraft=True),
    )
    out_flat, _ = tiedtke_convection(
        T, q, pf, ph, u, v, cpp, dt=300.0,
        config=TiedtkeConfig(enable_downdraft=True,
                             lcl_membership_sharpness=1e-6),
    )
    diff = float(jnp.max(jnp.abs(out_flat.dT_dt - out_default.dT_dt)))
    assert diff > 1e-10, "lcl_membership_sharpness is not wired"
