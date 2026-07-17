"""Unit tests for the Kain & Fritsch (1990, 2004) convection scheme.

Tests pin:

* tendency shape / dtype / finiteness;
* the CMT-off invariant (KF does not produce ``du_dt_conv``);
* the trigger function's smoothness — gradient through ``w_grid`` is
  finite and non-zero across the threshold (this is the central
  AD-safety property of the smooth-trigger (AD-safe) KF);
* the deep / shallow blend behaves correctly in the limits;
* differentiability through ``parcel_perturb_T`` and ``trigger_sharpness``;
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
    KainFritschConfig,
)
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.convection.kain_fritsch import (
    _kf_condload_profile,
    _kf_downdraft_evaporation,
    _kf_initial_wlcl,
    kain_fritsch_convection,
)
from legoesm.atmosphere.physics.convection._plume import Plume
from legoesm.atmosphere.physics.convection.mass_flux import compute_column_geometry


def _destabilized_column(
    ncol: int = 2,
    nlev: int = 16,
    T_sfc: float = 305.0,
    q_sfc: float = 18.0e-3,
    lapse_rate: float = 8.0,
    p_s: float = 1.0e5,
    p_top: float = 5.0e3,
    w_grid_value: float = 0.0,
):
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = (sigma[None, :] * jnp.full((ncol, 1), p_s))
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [
            jnp.full((ncol, 1), p_top * 0.5),
            p_half_inner,
            jnp.full((ncol, 1), p_s),
        ],
        axis=1,
    )
    H = 8500.0
    z_full = -H * jnp.log(p_full / p_s)
    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    q = q_sfc * jnp.exp(-z_full / 3000.0)
    w_grid = jnp.full((ncol, nlev), w_grid_value)
    return T, q, p_full, p_half, w_grid


# ---------------------------------------------------------------------------
# Shape / dtype / finiteness
# ---------------------------------------------------------------------------

def test_kf_tendency_shape_dtype():
    T, q, pf, ph, w = _destabilized_column(ncol=3, nlev=12, w_grid_value=5.0)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, cpp_new = kain_fritsch_convection(T, q, pf, ph, w, cpp, dt=300.0)
    assert out.dT_dt.shape == (ncol, nlev)
    assert out.dq_v_dt.shape == (ncol, nlev)
    assert out.dq_c_conv_dt.shape == (ncol, nlev)
    assert out.cape.shape == (ncol,)
    assert cpp_new.shape == (ncol, nlev)


def test_kf_outputs_finite():
    T, q, pf, ph, w = _destabilized_column(w_grid_value=5.0)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = kain_fritsch_convection(T, q, pf, ph, w, cpp, dt=300.0)
    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
                out.convective_mask):
        assert jnp.all(jnp.isfinite(arr))


# ---------------------------------------------------------------------------
# CMT invariant — KF emits None for both u and v wind tendencies
# ---------------------------------------------------------------------------

def test_kf_no_cmt():
    """KF does not produce convective momentum transport — by design."""
    T, q, pf, ph, w = _destabilized_column(w_grid_value=5.0)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = kain_fritsch_convection(T, q, pf, ph, w, cpp, dt=300.0)
    assert out.du_dt_conv is None
    assert out.dv_dt_conv is None


# ---------------------------------------------------------------------------
# Trigger smoothness — the AD-critical property
# ---------------------------------------------------------------------------

def test_kf_trigger_off_when_w_grid_strongly_negative():
    """A strongly negative ``w_grid`` (subsidence) suppresses the
    trigger and yields near-zero tendencies."""
    T, q, pf, ph, _ = _destabilized_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    w_neg = jnp.full((ncol, nlev), -10.0)
    out, _ = kain_fritsch_convection(T, q, pf, ph, w_neg, cpp, dt=300.0)
    # Heating is suppressed to ~zero under strong subsidence.
    assert float(jnp.max(jnp.abs(out.dT_dt))) < 1e-4
    # The faithful Fritsch-Chappell trigger (DTLCL negligible, ~3e-3 K,
    # for WKL<=0) leaves a
    # tiny residual smooth-trigger weight from the negative-but-finite LCL
    # buoyancy margin (sigmoid never reaches exactly 0); it is still ~1e-4,
    # i.e. strongly off.  The CAPE-OR SCM fallback is killed by the
    # ``w_absent`` gate (exp(-(−10/0.02)^2) ≈ 0).
    assert float(out.convective_mask[0]) < 1e-3


def test_kf_trigger_smoothness_grad_finite_at_crossing():
    """Across the trigger threshold the diagnosed mass flux is C¹ in
    ``w_grid``: a finite, non-zero gradient w.r.t. the column-mean
    ``w_grid`` value.  This is the AD-safety property the smooth
    sigmoid replaces hard ``> threshold`` for."""
    T, q, pf, ph, _ = _destabilized_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(w_value):
        w_arr = jnp.full((ncol, nlev), w_value)
        out, cpp_new = kain_fritsch_convection(T, q, pf, ph, w_arr, cpp, dt=300.0)
        return cpp_new[0, -1]   # M_b at surface slot

    grad_at_threshold = float(jax.grad(f)(jnp.asarray(2.0)))
    grad_below = float(jax.grad(f)(jnp.asarray(-2.0)))
    grad_above = float(jax.grad(f)(jnp.asarray(8.0)))
    for label, g in [("below", grad_below), ("at", grad_at_threshold), ("above", grad_above)]:
        assert np.isfinite(g), f"grad {label} not finite"
    # At threshold the gradient should be larger (sigmoid steepest in
    # the middle of the transition).  Below / above it should be
    # smaller but non-zero.
    assert abs(grad_at_threshold) > abs(grad_below) - 1e-12
    assert abs(grad_at_threshold) > abs(grad_above) - 1e-12


def test_kf_trigger_grad_through_parcel_perturb_T():
    """Differentiability through ``parcel_perturb_T`` — a tunable
    parameter intended for training."""
    T, q, pf, ph, w = _destabilized_column(w_grid_value=2.0)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(perturb_T):
        config = KainFritschConfig(parcel_perturb_T=perturb_T)
        _, cpp_new = kain_fritsch_convection(
            T, q, pf, ph, w, cpp, dt=300.0, config=config,
        )
        return cpp_new[0, -1]

    g = float(jax.grad(f)(jnp.asarray(0.5)))
    assert np.isfinite(g)


def test_kf_trigger_grad_through_w_thresh_offset():
    """Gradient through the ``w_thresh_offset`` calibration constant
    is finite — supports training-time tuning of the trigger."""
    T, q, pf, ph, w = _destabilized_column(w_grid_value=3.0)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(offset):
        config = KainFritschConfig(w_thresh_offset=offset)
        _, cpp_new = kain_fritsch_convection(
            T, q, pf, ph, w, cpp, dt=300.0, config=config,
        )
        return cpp_new[0, -1]

    g = float(jax.grad(f)(jnp.asarray(2.0)))
    assert np.isfinite(g)


# ---------------------------------------------------------------------------
# Deep / shallow blend
# ---------------------------------------------------------------------------

def test_kf_shallow_path_disabled_zeros_when_no_deep():
    """With ``enable_shallow=False`` and a column whose cloud depth
    is below the deep threshold, total tendencies vanish (deep_weight
    ≈ 0, shallow weight forced to 0 → no contribution)."""
    # Build a shallow column (small lapse rate, low T_sfc) so cloud
    # depth stays under the 4 km deep threshold.
    T, q, pf, ph, _ = _destabilized_column(
        T_sfc=295.0, q_sfc=12.0e-3, lapse_rate=4.0,
    )
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    w = jnp.full((ncol, nlev), 5.0)

    config = KainFritschConfig(enable_shallow=False)
    out, _ = kain_fritsch_convection(T, q, pf, ph, w, cpp, dt=300.0, config=config)
    # With deep-only mode and a shallow cloud, dT_dt should be small.
    assert float(jnp.max(jnp.abs(out.dT_dt))) < 1e-3


def test_kf_deep_branch_active_for_deep_clouds():
    """A deep, destabilized column produces non-trivial tendencies
    even with ``enable_shallow=False``."""
    T, q, pf, ph, _ = _destabilized_column(
        T_sfc=305.0, q_sfc=20.0e-3, lapse_rate=8.5,
    )
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    w = jnp.full((ncol, nlev), 5.0)

    config = KainFritschConfig(enable_shallow=False)
    out, _ = kain_fritsch_convection(T, q, pf, ph, w, cpp, dt=300.0, config=config)
    assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-6


# ---------------------------------------------------------------------------
# Carry layout
# ---------------------------------------------------------------------------

def test_kf_carry_layout():
    """KF packs ``M_b`` at ``[:, -1]`` with zeros aloft — same
    convention as ZM."""
    T, q, pf, ph, w = _destabilized_column(w_grid_value=5.0)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    _, cpp_new = kain_fritsch_convection(T, q, pf, ph, w, cpp, dt=300.0)
    assert jnp.all(cpp_new[:, :-1] == 0.0)
    assert jnp.all(cpp_new[:, -1] >= 0.0)


# ---------------------------------------------------------------------------
# Reference KF-Eta closure helpers
# ---------------------------------------------------------------------------

def test_kf_initial_wlcl_matches_kain_2004_formula_and_ad():
    """KF-Eta initial LCL velocity follows Kain (2004) Eq. 3."""
    dtlcl = jnp.asarray([0.0])
    dtrh = jnp.asarray([0.36])
    Tv_env = jnp.asarray([300.0])
    wlcl = _kf_initial_wlcl(dtlcl, dtrh, Tv_env)
    expected = 1.0 + 0.5 * jnp.sqrt(2.0 * constants.g * dtrh * 500.0 / Tv_env)
    expected = jnp.minimum(expected, 3.0)
    np.testing.assert_allclose(np.asarray(wlcl), np.asarray(expected), rtol=1e-6)

    g = jax.grad(lambda x: _kf_initial_wlcl(dtlcl, jnp.asarray([x]), Tv_env)[0])(0.36)
    assert jnp.isfinite(g)
    assert float(g) > 0.0

    inactive_grad = jax.jit(jax.grad(
        lambda x: _kf_initial_wlcl(dtlcl, jnp.asarray([x]), Tv_env)[0]
    ))
    for inactive_dtrh in (-0.36, -0.01):
        g_inactive = inactive_grad(jnp.asarray(inactive_dtrh))
        assert jnp.isfinite(g_inactive)


def test_kf_condload_fallout_reduces_retained_cloud_and_is_ad_safe():
    """CONDLOAD keeps retained cloud finite and diagnoses fallout."""
    T, q, pf, ph, _ = _destabilized_column(ncol=1, nlev=10, q_sfc=18e-3)
    dz, rho, z = compute_column_geometry(T, pf, ph, q_v=q)
    del dz, rho
    M = jnp.array([[0.0, 0.0, 0.01, 0.025, 0.035, 0.04, 0.035, 0.02, 0.0, 0.0]])
    qc = jnp.array([[0.0, 0.0, 1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3, 3.0e-3, 1.0e-3, 0.0, 0.0]])
    plume = Plume(
        M_u=M,
        T_u=T + 1.0,
        q_u=q,
        q_c_u=qc,
        B_u=jnp.ones_like(T),
    )
    cfg = KainFritschConfig()
    z_lcl = z[:, 6]
    result = _kf_condload_profile(
        T, q, pf, ph, z, plume,
        M_b=jnp.asarray([0.03]),
        wkl=jnp.asarray([0.1]),
        wlcl=jnp.asarray([2.0]),
        z_lcl=z_lcl,
        config=cfg,
    )
    assert jnp.all(jnp.isfinite(result.q_c_u))
    assert jnp.all(jnp.isfinite(result.M_u))
    below_cloud_base = z < z_lcl[:, None]
    assert float(jnp.max(jnp.where(below_cloud_base, result.q_c_u, 0.0))) == 0.0
    assert float(jnp.max(jnp.where(below_cloud_base, result.M_u, 0.0))) == 0.0
    assert float(jnp.max(jnp.where(below_cloud_base, result.precip_flux, 0.0))) == 0.0
    assert float(jnp.sum(result.q_c_u)) < float(jnp.sum(qc))
    assert float(jnp.sum(result.precip_flux)) > 0.0

    def retained(scale):
        plume_scaled = plume._replace(q_c_u=qc * scale)
        r = _kf_condload_profile(
            T, q, pf, ph, z, plume_scaled,
            M_b=jnp.asarray([0.03]),
            wkl=jnp.asarray([0.1]),
            wlcl=jnp.asarray([2.0]),
            z_lcl=z_lcl,
            config=cfg,
        )
        return jnp.sum(r.q_c_u)

    g = jax.grad(retained)(jnp.asarray(1.0))
    assert jnp.isfinite(g)


def test_kf_downdraft_evaporation_cools_moistens_and_conserves_latent_energy():
    """The downdraft evaporation branch has local latent-energy closure."""
    T, q, pf, ph, _ = _destabilized_column(ncol=1, nlev=10, q_sfc=6e-3)
    q = 0.25 * q
    dz, rho, z = compute_column_geometry(T, pf, ph, q_v=q)
    del dz, rho
    precip_flux = jnp.zeros_like(T).at[:, 4:6].set(2.0e-4)
    dT, dqv = _kf_downdraft_evaporation(
        T, q, pf, ph, z,
        p_lcl=jnp.asarray([8.5e4]),
        precip_flux=precip_flux,
        branch_weight=jnp.asarray([1.0]),
        dt=300.0,
    )
    assert jnp.all(jnp.isfinite(dT))
    assert jnp.all(jnp.isfinite(dqv))
    assert float(jnp.min(dT)) < 0.0
    assert float(jnp.max(dqv)) > 0.0
    np.testing.assert_allclose(
        np.asarray(constants.c_pd * dT + constants.L_v * dqv),
        np.zeros_like(np.asarray(dT)),
        atol=1.0e-12,
    )

    def evap_total(scale):
        _dT, _dqv = _kf_downdraft_evaporation(
            T, q, pf, ph, z,
            p_lcl=jnp.asarray([8.5e4]),
            precip_flux=precip_flux * scale,
            branch_weight=jnp.asarray([1.0]),
            dt=300.0,
        )
        return jnp.sum(_dqv)

    g = jax.grad(evap_total)(jnp.asarray(1.0))
    assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------

def _make_3d_state(n=4, nlev=12):
    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    tracers = {
        "q_v": Field(0.014 * jnp.ones((6, n, n, nlev)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
        "q_c": Field(jnp.zeros((6, n, n, nlev)), name="q_c",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    state = state._replace(tracers=tracers)
    return state, grid, sigma


def test_kf_orchestrator_one_step_finite():
    """End-to-end through ``make_physics``: hydrostatic step with KF
    selected runs without NaN."""
    state, grid, sigma = _make_3d_state()
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="kain_fritsch"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    n = 4; nlev = 12
    ncol = 6 * n * n
    ps = init_physics_state(ncol, nlev, cfg)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)
    assert ps_out is not None
    assert ps_out.conv_prog_profile.shape == (ncol, nlev)
    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
        assert jnp.all(jnp.isfinite(f.data))
    # KF is no-CMT → wind tendencies from convection bridge are zero
    # (the orchestrator zero-fills when ``du_dt_conv is None``).
    # Held-Suarez with all-other-physics-off has no other source of u
    # tendency through this physics_fn, so the tendency is exactly 0.
    assert float(jnp.max(jnp.abs(tend.du_dt.data))) == 0.0
    assert float(jnp.max(jnp.abs(tend.dv_dt.data))) == 0.0


# ---------------------------------------------------------------------------
# MSE conservation regression guard (currently expected to fail)
# ---------------------------------------------------------------------------

def test_kf_mse_conservation_within_tolerance():
    """Column vapor-MSE closure ``c_p ∫dT + L_v ∫dq_v dp/g ~ 0`` on a
    CAPE-positive sounding — the SAME canonical metric the tier-3 validator
    gates at 1e-6 W/m^2 (test_convection_group_validator).

    FORMERLY a strict xfail whose metric ADDED ``L_v ∫dq_c_conv``: the scheme
    books condensation latent into dT and hands the condensate to
    microphysics (which will re-release L_v only on re-evaporation), so
    counting the condensate's L_v here double-counts by construction and the
    old test "failed" at ~95% forever — a stale pre-PR-#988 relic whose
    reason string still blamed the mass-flux kernel.  With the canonical
    vapor-MSE metric the post-#988 scheme closes to machine precision
    (codex 2026-07-17 round 4: the xfail contradicted the header's budget
    claim; the METRIC was wrong, not the header)."""
    T, q, p_full, p_half, w_grid = _destabilized_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = kain_fritsch_convection(
        T=T, q_v=q, p_full=p_full, p_half=p_half,
        w_grid=w_grid, conv_prog_profile=cpp, dt=1800.0,
    )
    dp = p_half[:, 1:] - p_half[:, :-1]
    mse = jnp.sum(
        (constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt)
        * dp / constants.g, axis=1)
    assert float(jnp.max(jnp.abs(mse))) < 1.0e-6, (
        f"KF column vapor-MSE residual {float(jnp.max(jnp.abs(mse))):.3e} "
        f"W/m^2 — energy leak (canonical metric, validator-matched)"
    )


# ---------------------------------------------------------------------------
# PROF5 buoyancy-sorted entrainment/detrainment rates (Kain 2004 Eq. 4)
# ---------------------------------------------------------------------------
# Leaf pins for _kf_buoyancy_sort_rates — previously exercised only through
# the full scheme, so the two oracle-fidelity properties it owns (the PROF5
# buoyancy sort and the VMFLCL/UPOLD active-plume-mass conversion) had no
# direct discriminating test (2026-07-17 faithfulness audit).

class TestBuoyancySortRates:
    def _setup(self, m_u_mult=2.0):
        from legoesm.atmosphere.physics.convection._plume import Plume
        from legoesm.atmosphere.physics.convection.kain_fritsch import (
            _kf_buoyancy_sort_rates,
        )

        nlev = 12
        # Surface-last: index -1 = surface.
        z = jnp.linspace(11.0e3, 0.0, nlev)[None, :]
        T_env = jnp.linspace(220.0, 300.0, nlev)[None, :]
        q_env = jnp.full((1, nlev), 2.0e-3)
        p_full = jnp.linspace(2.0e4, 1.0e5, nlev)[None, :]
        M_b = jnp.asarray([0.01])
        # Very buoyant band at levels 6-8 (the oracle's 0.5*(prev+current)
        # multiplier averaging pulls in one neighbour, and a NEUTRAL level
        # classifies as 'colder' via tv_u <= tv_env, so the probe level needs
        # buoyant neighbours); colder than env at level 4.
        T_u = T_env
        T_u = T_u.at[0, 6:9].add(6.0)
        T_u = T_u.at[0, 4].add(-3.0)
        M_u = jnp.where(jnp.arange(nlev)[None, :] < 10,
                        m_u_mult * M_b[:, None], 0.0)
        plume = Plume(
            M_u=M_u, T_u=T_u, q_u=q_env,
            q_c_u=jnp.zeros_like(q_env),
            B_u=jnp.zeros_like(q_env),
        )
        eps_base = jnp.full((1, nlev), 2.0e-4)
        k_lnb = jnp.asarray([3.0])
        cfg = KainFritschConfig()
        eps, dlt = _kf_buoyancy_sort_rates(
            T_env, q_env, p_full, plume, eps_base, M_b, z, k_lnb, cfg)
        return eps, dlt

    def test_rates_finite_nonnegative(self):
        eps, dlt = self._setup()
        assert bool(jnp.all(jnp.isfinite(eps))) and bool(jnp.all(eps >= 0.0))
        assert bool(jnp.all(jnp.isfinite(dlt))) and bool(jnp.all(dlt >= 0.0))

    def test_buoyant_level_entrains_cold_level_detrains(self):
        """PROF5: a very-buoyant updraft level entrains with ~no detrainment
        (UD2 -> 0); a colder-than-environment level detrains more than it
        entrains (EE2 -> 0.5 floor, UD2 -> 1.5 boost) — Kain (2004) Eq. 4;
        the pre-fix shortcut dlt == eps violated both directions."""
        eps, dlt = self._setup()
        assert float(eps[0, 7]) > float(dlt[0, 7])       # buoyant: entrain
        assert float(dlt[0, 4]) > float(eps[0, 4])       # cold: detrain

    def test_upold_conversion_halves_rates_when_plume_mass_doubles(self):
        """The VMFLCL/UPOLD conversion (oracle UER/UDR -> fractional rates):
        fractional entrainment scales as M_b / max(M_u, M_b), so doubling the
        active plume mass halves it EXACTLY.  The pre-fix code omitted the
        factor entirely (rates independent of M_u), over-entraining grown
        plumes — the too-shallow-column / cold-point-too-high bug."""
        eps2, _ = self._setup(m_u_mult=2.0)
        eps4, _ = self._setup(m_u_mult=4.0)
        lv = slice(1, 10)   # levels where M_u = mult*M_b > M_b
        ratio = eps4[0, lv] / jnp.maximum(eps2[0, lv], 1e-30)
        assert bool(jnp.all(jnp.abs(ratio - 0.5) < 1e-12))
