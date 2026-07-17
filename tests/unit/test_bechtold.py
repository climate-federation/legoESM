"""Unit tests for the Bechtold/IFS convection scheme.

Tests pin:

* tendency shape / dtype / finiteness;
* the PBL-CAPE closure (use_pbl_cape toggle changes diagnosed CAPE);
* CMT signs and ``enable_cmt`` opt-out;
* downdraft toggle effect;
* **stochastic perturbation** — when enabled with a given PRNG key
  the AR1 noise state evolves; when disabled the result is
  deterministic and identical across calls;
* the AR1 decorrelation: variance of the AR1 process matches the
  expected stationary variance ``1`` for a sufficiently long run;
* PhysicsState ``conv_stoch_state`` field is round-tripped
  correctly;
* finite gradients through ``epsilon_deep``, ``cape_pbl_depth``,
  ``stochastic_amplitude``, ``cmt_c_u``;
* selection through ``make_physics(PhysicsConfig(...))``.
"""

from __future__ import annotations

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.physics_state import (
    PhysicsState, init_physics_state,
)
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import (
    ConvectionConfig,
    BechtoldConfig,
)
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection


def _column(
    ncol=2, nlev=16, T_sfc=302.0, q_sfc=16e-3, lapse_rate=7.5,
    p_s=1.0e5, p_top=5.0e3,
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
    u = jnp.linspace(0, 25, nlev)[None, :]
    u = jnp.broadcast_to(u, (ncol, nlev))
    v = jnp.zeros_like(u)
    return T, q, p_full, p_half, u, v


# ---------------------------------------------------------------------------
# Shape / finiteness
# ---------------------------------------------------------------------------

def test_bechtold_shape_finiteness():
    T, q, pf, ph, u, v = _column(ncol=3, nlev=12)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    out, M_u_new, stoch_new = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
    )
    assert M_u_new.shape == (ncol, nlev)
    assert stoch_new.shape == (ncol,)
    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
                out.convective_mask, out.du_dt_conv, out.dv_dt_conv,
                M_u_new, stoch_new):
        assert jnp.all(jnp.isfinite(arr))


# ---------------------------------------------------------------------------
# PBL-CAPE closure: switching to surface-parcel CAPE changes M_b
# ---------------------------------------------------------------------------

def test_bechtold_pbl_cape_changes_m_b():
    """Toggling between ``use_pbl_cape=True`` and ``use_pbl_cape=False``
    changes the diagnosed cloud-base mass flux."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    _, M_u_pbl, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(use_pbl_cape=True),
    )
    _, M_u_sfc, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(use_pbl_cape=False),
    )
    # The two diagnoses differ by some non-trivial amount.
    assert float(jnp.sum(jnp.abs(M_u_pbl - M_u_sfc))) > 1e-12


# ---------------------------------------------------------------------------
# CMT
# ---------------------------------------------------------------------------

def test_bechtold_cmt_present():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    out, _, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0)
    assert out.du_dt_conv is not None
    assert out.dv_dt_conv is not None


def test_bechtold_cmt_disabled():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    out, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(enable_cmt=False),
    )
    assert out.du_dt_conv is None
    assert out.dv_dt_conv is None


def test_bechtold_downdraft_evap_conserves_water_locally():
    """Bechtold inherits the same downdraft fix as Tiedtke.

    Three invariants (see ``tests/unit/test_tiedtke.py::
    test_tiedtke_downdraft_evap_conserves_water_locally`` for the
    detailed audit / Codex rationale):
      1. Local energy-water balance per level: ``Δ(dT_dt)·c_pd +
         Δ(dq_v_dt)·L_v == 0``.
      2. Column water conservation: column-integrated
         ``Δ(dq_v_dt) + Δ(dq_c_conv_dt) == 0`` (vapor source matched
         by reduction in convective rain source).
      3. Cooling is actually exercised.
    """
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    out_off, _, _ = bechtold_convection(
        T=T, q_v=q, p_full=pf, p_half=ph, u=u, v=v,
        conv_prog_profile=cpp, conv_stoch_state=stoch, prng_key=None,
        dt=300.0,
        # evap pinned OFF: this test pins the LEGACY downdraft
        # re-evaporation machinery; the default-on IFS Kessler evap replaces
        # it (own conservation tests in the subcloud_evap section).
        config=BechtoldConfig(enable_downdraft=False, enable_stochastic=False,
                              use_ifs_subcloud_evap=False,
                              use_ifs_inplume_precip=False),
        moisture_convergence=jnp.zeros_like(T),
    )
    out_on, _, _ = bechtold_convection(
        T=T, q_v=q, p_full=pf, p_half=ph, u=u, v=v,
        conv_prog_profile=cpp, conv_stoch_state=stoch, prng_key=None,
        dt=300.0,
        config=BechtoldConfig(enable_downdraft=True, enable_stochastic=False,
                              use_ifs_subcloud_evap=False,
                              use_ifs_inplume_precip=False),
        moisture_convergence=jnp.zeros_like(T),
    )
    dT_diff = out_on.dT_dt - out_off.dT_dt
    dqv_diff = out_on.dq_v_dt - out_off.dq_v_dt
    dqc_diff = out_on.dq_c_conv_dt - out_off.dq_c_conv_dt

    assert float(jnp.min(dT_diff)) < 0.0, (
        "Bechtold downdraft did not produce cooling — formulation regressed."
    )

    # (1) Local energy-water balance
    H = dT_diff * constants.c_pd
    Q = dqv_diff * constants.L_v
    res_local = float(jnp.max(jnp.abs(H + Q)))
    scale_local = float(jnp.max(jnp.abs(H)))
    assert res_local < 1e-8 * max(scale_local, 1.0), (
        f"Bechtold downdraft local energy-water budget unclosed: max|H+Q|="
        f"{res_local:.3e}, max|H|={scale_local:.3e}"
    )

    # (2) Column water conservation.  The production default splits detrained
    # condensate into anvil cloud (dq_c_conv_dt) + in-updraft rain
    # (dq_r_conv_dt) at precip_efficiency (#929); the downdraft evaporation
    # acts on the PRE-split condensate, so the column budget closes over the
    # TOTAL convective condensate source dq_c + dq_r.  The rain split is a pure
    # re-partition of a shared positive quantity and cannot move this balance.
    dqr_diff = out_on.dq_r_conv_dt - out_off.dq_r_conv_dt
    dp = ph[:, 1:] - ph[:, :-1]
    col_dqv = jnp.sum(dqv_diff * dp, axis=-1) / constants.g
    col_dqc = jnp.sum((dqc_diff + dqr_diff) * dp, axis=-1) / constants.g
    col_residual = float(jnp.max(jnp.abs(col_dqv + col_dqc)))
    col_scale = float(jnp.max(jnp.abs(col_dqv)) + 1e-15)
    assert col_residual < 1e-10 * max(col_scale, 1.0), (
        f"Bechtold downdraft column water unclosed: max|∫dq_v + ∫(dq_c+dq_r)|="
        f"{col_residual:.3e} kg/m²/s, vapor source={col_scale:.3e}"
    )


# ---------------------------------------------------------------------------
# Stochastic perturbation
# ---------------------------------------------------------------------------

def test_bechtold_deterministic_when_stochastic_off():
    """With ``enable_stochastic=False``, two calls with the same input
    produce identical output."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    out1, M1, s1 = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0)
    out2, M2, s2 = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0)
    assert jnp.allclose(out1.dT_dt, out2.dT_dt)
    assert jnp.allclose(M1, M2)
    assert jnp.allclose(s1, s2)


def test_bechtold_stochastic_changes_with_key():
    """With ``enable_stochastic=True``, two different PRNG keys produce
    different AR1 noise states and different diagnosed mass fluxes.

    The fixture uses a high-CAPE sounding that drives diagnosed M_b
    above the production ``M_b_max=0.05`` cap on both keys; we set
    ``M_b_max=10.0`` here so the cap does not bind and mask the
    stochastic variation.  In production the cap is intentional — it
    bounds single-step shocks from outlier columns — and a no-cap
    setup like this should never appear in a real run.
    """
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    config = BechtoldConfig(
        enable_stochastic=True, stochastic_amplitude=0.5, M_b_max=10.0,
    )
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(7)
    _, M1, s1 = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, key1, dt=300.0, config=config)
    _, M2, s2 = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, key2, dt=300.0, config=config)
    assert not jnp.allclose(s1, s2)
    assert not jnp.allclose(M1, M2)


def test_bechtold_AR1_stationary_variance():
    """Long-run AR1 noise has stationary variance ≈ 1 (per unit
    amplitude).  We integrate 500 steps and check that the empirical
    variance lands near 1."""
    T, q, pf, ph, u, v = _column(ncol=200)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    config = BechtoldConfig(
        enable_stochastic=True, stochastic_amplitude=1.0,
        stochastic_decorrelation=1800.0,
    )
    key = jax.random.PRNGKey(0)
    # 500 steps; sample stoch_new at the end.
    for i in range(500):
        key, subkey = jax.random.split(key)
        _, _, stoch = bechtold_convection(
            T, q, pf, ph, u, v, cpp, stoch, subkey, dt=300.0, config=config,
        )
    var = float(jnp.var(stoch))
    # Stationary variance is theoretically 1; allow generous tolerance.
    assert 0.5 < var < 2.0, f"AR1 stationary variance off-target: {var}"


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_bechtold_grad_through_epsilon_deep():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))

    def f(eps):
        out, _, _ = bechtold_convection(
            T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
            config=BechtoldConfig(epsilon_deep=eps),
        )
        return jnp.sum(out.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(1.75e-3)))
    assert bool(jnp.isfinite(g))


def test_bechtold_grad_through_cape_pbl_depth():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))

    def f(depth):
        out, _, _ = bechtold_convection(
            T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
            config=BechtoldConfig(cape_pbl_depth=depth),
        )
        return jnp.sum(out.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(500.0)))
    assert bool(jnp.isfinite(g))


def test_bechtold_grad_through_stochastic_amplitude_when_off():
    """Even when ``enable_stochastic=False``, gradient through
    ``stochastic_amplitude`` is finite (it's a static config field
    that doesn't enter the computation in the off branch)."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))

    def f(amp):
        out, _, _ = bechtold_convection(
            T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
            config=BechtoldConfig(enable_stochastic=False, stochastic_amplitude=amp),
        )
        return jnp.sum(out.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(0.5)))
    assert bool(jnp.isfinite(g))
    # In the off branch the gradient is 0 (parameter unused) — that's
    # fine, just must be finite.


# ---------------------------------------------------------------------------
# PhysicsState round-trip
# ---------------------------------------------------------------------------

def test_bechtold_physics_state_has_conv_stoch_state():
    """``init_physics_state`` produces a PhysicsState with
    ``conv_stoch_state`` of shape ``(ncol,)``."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="bechtold"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    ps = init_physics_state(64, 12, cfg)
    assert hasattr(ps, "conv_stoch_state"), \
        "PhysicsState should expose conv_stoch_state field"
    assert ps.conv_stoch_state.shape == (64,)
    assert jnp.all(ps.conv_stoch_state == 0.0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_bechtold_orchestrator_one_step_finite():
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
        convection=ConvectionConfig(scheme="bechtold"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    ncol = 6 * 4 * 4
    ps = init_physics_state(ncol, 12, cfg)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)
    assert ps_out.conv_prog_profile.shape == (ncol, 12)
    assert ps_out.conv_stoch_state.shape == (ncol,)
    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
        assert jnp.all(jnp.isfinite(f.data))


# ---------------------------------------------------------------------------
# PRNG threading through the bridge
# ---------------------------------------------------------------------------

def test_bechtold_orchestrator_threads_prng_key():
    """When ``enable_stochastic=True``, two PhysicsStates with different
    master PRNG keys produce different conv_stoch_state outputs after
    one orchestrator step.  Same key → same output (reproducibility)."""
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
        convection=ConvectionConfig(
            scheme="bechtold",
            bechtold=BechtoldConfig(
                enable_stochastic=True, stochastic_amplitude=1.0,
                stochastic_decorrelation=1800.0,
            ),
        ),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    ncol = 6 * 4 * 4
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)

    ps_seed_0 = init_physics_state(ncol, 12, cfg, prng_seed=0)
    ps_seed_1 = init_physics_state(ncol, 12, cfg, prng_seed=1)
    ps_seed_0_again = init_physics_state(ncol, 12, cfg, prng_seed=0)

    _, out_0 = physics_fn(state, grid, sigma, phys_state=ps_seed_0)
    _, out_1 = physics_fn(state, grid, sigma, phys_state=ps_seed_1)
    _, out_0_again = physics_fn(state, grid, sigma, phys_state=ps_seed_0_again)

    # Different seeds → different stochastic state.
    assert not jnp.allclose(out_0.conv_stoch_state, out_1.conv_stoch_state), (
        "Two different PRNG seeds should produce different AR1 noise"
    )
    # Same seed → same state (bit-for-bit reproducibility).
    assert jnp.allclose(out_0.conv_stoch_state, out_0_again.conv_stoch_state), (
        "Same PRNG seed should produce identical AR1 noise"
    )

    # Master key advances after the call (so a subsequent step sees
    # fresh randomness).
    assert not jnp.array_equal(out_0.prng_key, ps_seed_0.prng_key), (
        "Master PRNG key should advance through the orchestrator step"
    )


def test_bechtold_orchestrator_grad_through_phys_state():
    """jax.grad through the orchestrator with stochastic Bechtold
    succeeds — the AR1 perturbation does not break differentiability of
    the deterministic mass flux (the noise enters multiplicatively as a
    fixed factor at the time of differentiation)."""
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
        convection=ConvectionConfig(
            scheme="bechtold",
            bechtold=BechtoldConfig(
                enable_stochastic=True, stochastic_amplitude=0.5,
            ),
        ),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    ncol = 6 * 4 * 4
    ps = init_physics_state(ncol, 12, cfg, prng_seed=42)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)

    def loss(scale):
        scaled_T = state.T.replace(data=scale * state.T.data)
        s2 = state._replace(T=scaled_T)
        tend, _ = physics_fn(s2, grid, sigma, phys_state=ps)
        return jnp.sum(tend.dT_dt.data ** 2)

    g = jax.grad(loss)(jnp.array(1.0))
    assert bool(jnp.isfinite(g))


def test_bechtold_orchestrator_does_not_advance_key_when_stochastic_off():
    """When ``enable_stochastic=False``, the convection bridge must NOT
    consume the master PRNG key — calling the orchestrator a hundred
    times should leave ``ps.prng_key`` byte-identical."""
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
        convection=ConvectionConfig(
            scheme="bechtold",
            bechtold=BechtoldConfig(enable_stochastic=False),
        ),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    ncol = 6 * 4 * 4
    ps = init_physics_state(ncol, 12, cfg, prng_seed=11)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)

    _, ps_out = physics_fn(state, grid, sigma, phys_state=ps)
    # No advance — bit-identical key.
    assert jnp.array_equal(ps_out.prng_key, ps.prng_key), (
        "Master PRNG key must NOT advance when enable_stochastic=False"
    )


def test_bechtold_orchestrator_with_radiation_merges_dict_correctly():
    """Regression for orchestrator dict-merge bug.

    With ``radiation=gray`` + ``convection=bechtold``, Bechtold is the
    SECOND tagged_fn.  The pre-fix orchestrator's ``tagged_fns[1:]``
    loop assigned ``phys_updates[field_name] = field_val`` directly,
    so Bechtold's multi-field dict (``conv_prog_profile`` /
    ``conv_stoch_state`` / ``prng_key``) ended up nested under
    ``conv_prog_profile`` and ``update_physics_state`` then set
    ``ps.conv_prog_profile`` to a *dict*.  This test pins the fix:
    every PhysicsState slot must be the right shape after the orchestrator
    step, even with a non-Bechtold module registered first.
    """
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
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(
            scheme="bechtold",
            bechtold=BechtoldConfig(
                enable_stochastic=True, stochastic_amplitude=0.3,
            ),
        ),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    ncol = 6 * 4 * 4
    ps = init_physics_state(ncol, 12, cfg, prng_seed=7)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)

    # Each PhysicsState slot must be a JAX array of the right shape —
    # NOT a dict (which is what the pre-fix orchestrator produced).
    assert ps_out.conv_prog_profile.shape == (ncol, 12), (
        f"conv_prog_profile got shape {ps_out.conv_prog_profile.shape!r} — "
        "the orchestrator's dict-merge bug stored the entire multi-field "
        "carry under this key."
    )
    assert ps_out.conv_stoch_state.shape == (ncol,)
    assert ps_out.prng_key.shape == (2,)
    # Master key advanced through the dict-merge.
    assert not jnp.array_equal(ps_out.prng_key, ps.prng_key), (
        "Master PRNG key must advance through the orchestrator even "
        "when Bechtold is not the first tagged module."
    )
    # Tendencies are finite.
    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
        assert jnp.all(jnp.isfinite(f.data))


# ---------------------------------------------------------------------------
# MSE conservation regression guard
# ---------------------------------------------------------------------------

def test_bechtold_mse_conservation_within_tolerance():
    """Column moist-static-energy budget closes to within tolerance.

    The DEFAULT advective compensating-subsidence ``(M/ρ)·∂φ/∂z`` does NOT
    telescope on column integration — it leaves a ``(φ/ρ)·dM/dz`` residual
    that leaks ~40 % of the column MSE budget at this coarse nlev=16 (the
    leak shrinks with resolution but never vanishes; the advective
    non-closure is the documented design gap pinned by
    ``test_tier3_massflux_schemes_total_water_NOT_closed_in_scheme_KNOWN``).

    The fix is the IMPLICIT (backward-Euler) CONSERVATIVE flux-form solve
    (``subsidence_solve="implicit_flux"``;
    mass_flux.apply_mass_flux_kernel_implicit_flux), which transports dry
    static energy ``s = c_p T + g z`` and vapor ``q_v`` in flux form so the
    column integrals telescope to the vanishing top/base boundary flux —
    column MSE ``h = s + L_v q_v`` is conserved by the TRANSPORT to machine
    precision, and the detrained condensate (matched by a vapor sink) keeps
    the total-water budget closed.  On this nlev=16 column the residual
    drops from ~40 % (advective) to ~4 % (the remainder is the downdraft's
    condensate→vapor conversion, a separate MSE-neutral process; with the
    downdraft off the transport conserves to ~1e-14).  The 0.10 guard
    catches any regression that re-introduces a transport leak.  See
    tests/unit/test_bechtold_implicit_flux.py for the machine-precision
    kernel-level conservation, multi-step stability, and AD tests.
    """
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    out, _, _ = bechtold_convection(
        T=T, q_v=q, p_full=pf, p_half=ph, u=u, v=v,
        conv_prog_profile=cpp, conv_stoch_state=stoch, prng_key=None,
        dt=1800.0,
        # Sub-cloud evap pinned OFF: this test's object is the mass-flux
        # SOLVE conservation.  The H+Q+C metric is not evap-invariant BY
        # CONSTRUCTION (it books +L_v*rain for never-evaporated rain, so the
        # exactly-conservative form-then-evaporate pair shifts it by -L_v*e);
        # the evap's own water/enthalpy closure is machine-exact tested in
        # the subcloud_evap section.
        config=BechtoldConfig(
            enable_stochastic=False, enable_cmt=False,
            subsidence_solve="implicit_flux",
            use_ifs_subcloud_evap=False,
        ),
        moisture_convergence=jnp.zeros_like(T),
    )
    dp = ph[:, 1:] - ph[:, :-1]
    # The latent-heat sink C is the FULL detrained condensate: the #929 rain
    # split moves precip_efficiency of it from dq_c_conv_dt into dq_r_conv_dt,
    # but the latent heat of ALL of it is already booked in dT_dt (H), so the
    # enthalpy budget must sum dq_c + dq_r (the split re-partitions water
    # downstream; it does not change the scheme's internal energy balance).
    dqr = out.dq_r_conv_dt if out.dq_r_conv_dt is not None else 0.0
    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
    C = float(jnp.sum((out.dq_c_conv_dt + dqr) * dp / constants.g, axis=1).mean()) * constants.L_v
    rel = abs(H + Q + C) / (abs(H) + abs(Q) + abs(C) + 1e-10)
    assert rel < 0.10, (
        f"Bechtold (implicit_flux) MSE residual {H+Q+C:.1f} W/m^2 "
        f"({rel*100:.1f}% of total)"
    )


# ---------------------------------------------------------------------------
# In-updraft precipitation: the shared convective rain split
# ---------------------------------------------------------------------------

def test_bechtold_precip_efficiency_splits_rain_conserving_mass():
    """precip_efficiency>0 emits a rain source (dq_r_conv_dt) that is exactly
    the pe-fraction of the detrained condensate; cloud+rain conserves the
    positive condensate; the default (0) is byte-identical with no rain."""
    T, q, pf, ph, u, v = _column(ncol=3, nlev=16)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))

    base, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        # evap pinned OFF: this test pins the bit-exact pe-fraction split.
        config=BechtoldConfig(precip_efficiency=0.0,
                              use_ifs_subcloud_evap=False,
                              use_ifs_inplume_precip=False),
    )
    assert base.dq_r_conv_dt is None                 # pe=0 => no split

    pe = 0.6
    split, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(precip_efficiency=pe,
                              use_ifs_subcloud_evap=False,
                              use_ifs_inplume_precip=False),
    )
    assert split.dq_r_conv_dt is not None
    # cloud + rain == the ORIGINAL positive condensate (base cloud), so the
    # split introduces no extra source/sink
    total = split.dq_c_conv_dt + split.dq_r_conv_dt
    assert jnp.allclose(total, base.dq_c_conv_dt, atol=1e-20)
    # rain is exactly pe of the original condensate
    assert jnp.allclose(split.dq_r_conv_dt, base.dq_c_conv_dt * pe, rtol=1e-6,
                        atol=1e-20)
    # heating / vapor tendencies are untouched by the diagnostic split
    assert jnp.allclose(split.dT_dt, base.dT_dt, atol=1e-20)
    assert jnp.allclose(split.dq_v_dt, base.dq_v_dt, atol=1e-20)
# Trigger sharpness fields (fix 2026-07) — same defect class as Tiedtke:
# BechtoldConfig.smooth_trigger_sharpness was dead; the downdraft RH trigger
# and below-LCL membership hardcoded 10.0 / 2.0.
# ---------------------------------------------------------------------------

def test_bechtold_smooth_trigger_sharpness_removed():
    cfg = BechtoldConfig()
    assert not hasattr(cfg, "smooth_trigger_sharpness")
    assert cfg.downdraft_rh_sharpness == 10.0
    assert cfg.lcl_membership_sharpness == 2.0


def test_bechtold_downdraft_sharpness_fields_wired():
    """Perturbing either new sharpness field changes the downdraft-branch
    tendencies (both were hardcoded literals before)."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    # evap pinned OFF: the RH-trigger sharpness reaches dT_dt only through
    # the LEGACY evap branch (under IFS evap the trigger feeds CMT momentum
    # only, and lcl sharpness feeds the evap gate separately).
    out_default, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(enable_downdraft=True,
                              use_ifs_subcloud_evap=False,
                              use_ifs_inplume_precip=False),
    )
    out_rh_flat, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(enable_downdraft=True,
                              downdraft_rh_sharpness=1e-6,
                              use_ifs_subcloud_evap=False,
                              use_ifs_inplume_precip=False),
    )
    out_lcl_flat, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(enable_downdraft=True,
                              lcl_membership_sharpness=1e-6,
                              use_ifs_subcloud_evap=False,
                              use_ifs_inplume_precip=False),
    )
    assert float(jnp.max(jnp.abs(out_rh_flat.dT_dt - out_default.dT_dt))) > 1e-10, (
        "downdraft_rh_sharpness is not wired"
    )
    assert float(jnp.max(jnp.abs(out_lcl_flat.dT_dt - out_default.dT_dt))) > 1e-10, (
        "lcl_membership_sharpness is not wired"
    )


# ---------------------------------------------------------------------------
# Penetrative-downdraft thermodynamic transport (marine-BL ventilation)
# ---------------------------------------------------------------------------

from legoesm.atmosphere.physics.convection.bechtold import (  # noqa: E402
    _penetrative_downdraft_transport,
)


def _humid_bl_dry_midtrop_column(ncol=1, nlev=24):
    """Synthetic column: a humid marine BL over a dry mid-troposphere.

    Moist static energy has a mid-tropospheric minimum (the downdraft origin)
    whose ``q_v`` is far below the boundary-layer value, so a penetrative
    downdraft MUST dry the sub-cloud layer.  Surface-last (index -1 =
    surface).
    """
    p_s, p_top = 1.0e5, 5.0e3
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), p_half_inner,
         jnp.full((ncol, 1), p_s)], axis=1)
    dp_full = p_half[:, 1:] - p_half[:, :-1]
    z = -8500.0 * jnp.log(p_full / p_s)              # height [m], 0 at surface
    T = 300.0 - 7.0e-3 * z
    # Humid BL (~16 g/kg near surface) decaying with height + a tiny floor:
    # gives a dry (~0.2 g/kg) mid-/upper troposphere => low-MSE origin aloft.
    q_v = 16.0e-3 * jnp.exp(-z / 1500.0) + 2.0e-4
    levels_arr = jnp.arange(nlev, dtype=p_full.dtype)
    # LCL ~1 km above the surface -> a fractional level index near the surface.
    k_lcl = jnp.interp(jnp.array([-1000.0]), -z[0], levels_arr)
    k_lcl_smooth = jnp.broadcast_to(k_lcl, (ncol,))
    return T, q_v, z, dp_full, k_lcl_smooth, levels_arr


def test_penetrative_downdraft_transport_conserves_column():
    """Flux form => the column integrals of q_v and dry static energy vanish
    (the transport is adiabatic redistribution, not a source/sink)."""
    T, q_v, z, dp_full, k_lcl, lev = _humid_bl_dry_midtrop_column()
    dT_dd, dq_dd = _penetrative_downdraft_transport(
        T, q_v, z, dp_full, k_lcl, lev,
        jnp.array([0.02]), entrain_rate=5.0e-4,
        detrain_scale_m=700.0, dt=150.0,
    )
    g = constants.g
    col_q = jnp.sum(dq_dd * dp_full / g, axis=1)
    col_s = jnp.sum(dT_dd * constants.c_pd * dp_full / g, axis=1)
    q_scale = jnp.sum(jnp.abs(dq_dd) * dp_full / g, axis=1) + 1e-30
    s_scale = jnp.sum(jnp.abs(dT_dd) * constants.c_pd * dp_full / g, axis=1) + 1e-30
    assert float(jnp.abs(col_q)[0] / q_scale[0]) < 1e-9, "q_v not conserved"
    assert float(jnp.abs(col_s)[0] / s_scale[0]) < 1e-9, "dry static energy not conserved"
    assert bool(jnp.all(jnp.isfinite(dT_dd)) & jnp.all(jnp.isfinite(dq_dd)))


def test_penetrative_downdraft_transport_dries_subcloud():
    """The penetrative downdraft NET-dries the sub-cloud layer — the whole
    point of the lever (a re-evaporation-only downdraft would moisten it)."""
    T, q_v, z, dp_full, k_lcl, lev = _humid_bl_dry_midtrop_column()
    _, dq_dd = _penetrative_downdraft_transport(
        T, q_v, z, dp_full, k_lcl, lev,
        jnp.array([0.02]), entrain_rate=5.0e-4,
        detrain_scale_m=700.0, dt=150.0,
    )
    g = constants.g
    below_lcl = lev[None, :] > k_lcl[:, None]
    subcloud_dq = jnp.sum(jnp.where(below_lcl, dq_dd * dp_full / g, 0.0), axis=1)
    assert float(subcloud_dq[0]) < 0.0            # net sub-cloud drying
    assert float(dq_dd[0, -1]) < 0.0              # the surface level dries


def test_penetrative_downdraft_transport_differentiable():
    """Finite gradient of the sub-cloud drying wrt the entrainment rate
    (the tunable knob) — the lever must stay ``jax.grad``-safe."""
    T, q_v, z, dp_full, k_lcl, lev = _humid_bl_dry_midtrop_column()

    def _subcloud_drying(eps):
        _, dq_dd = _penetrative_downdraft_transport(
            T, q_v, z, dp_full, k_lcl, lev, jnp.array([0.02]),
            entrain_rate=eps, detrain_scale_m=700.0, dt=150.0,
        )
        return jnp.sum(dq_dd[:, -6:])

    grad = jax.grad(_subcloud_drying)(5.0e-4)
    assert bool(jnp.isfinite(grad))


def test_bechtold_downdraft_transport_toggle_dries_bl():
    """End-to-end: enabling the transport dries the BL vs the re-evap-only
    default, and the OFF default is byte-identical to the pre-existing path."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=20, q_sfc=18e-3, lapse_rate=8.0)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    base = BechtoldConfig()
    assert base.downdraft_transport is False       # opt-in default OFF
    out_off, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0, config=base,
    )
    out_default, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
    )
    out_on, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=base._replace(downdraft_transport=True),
    )
    # OFF branch leaves the default path untouched (feature-gated on a static
    # bool => byte-identical to before the transport existed).
    assert bool(jnp.array_equal(out_off.dq_v_dt, out_default.dq_v_dt))
    # Turning it ON changes the moisture tendency and NET-dries the BL.
    assert float(jnp.sum(jnp.abs(out_on.dq_v_dt - out_off.dq_v_dt))) > 1e-12, (
        "downdraft_transport had no effect (column may not be convecting)"
    )
    bl = slice(nlev - 4, nlev)                      # lowest 4 levels = the BL
    dq_bl = jnp.sum(out_on.dq_v_dt[:, bl] - out_off.dq_v_dt[:, bl])
    assert float(dq_bl) < 0.0, "transport must dry the boundary layer"
# In-updraft precipitation split (#929) — divert precip_efficiency of the
# detrained condensate to RAIN (dq_r_conv_dt) so microphysics can drain the
# polar-night anvil instead of it radiatively loading the column to runaway.
# ---------------------------------------------------------------------------

def _run_pe(pe):
    """Run bechtold on the default moist column at a given precip_efficiency,
    holding EVERY other config field at its default so PE is the ONLY variable
    (controlled comparison).  Deterministic: enable_stochastic defaults False,
    prng_key=None."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    out, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        # evap pinned OFF: this fixture pins the bit-exact (1-PE)/PE split
        # partition; the default-on IFS sub-cloud evap rescales dq_r
        # downstream of the split (its own conservation is tested in the
        # subcloud_evap section).
        config=BechtoldConfig(precip_efficiency=pe,
                              use_ifs_subcloud_evap=False,
                              use_ifs_inplume_precip=False),
    )
    return out


def test_bechtold_rain_split_conserves_total_condensate():
    """Split re-partitions the detrained condensate; column total water is
    unchanged.  PE=0 returns the legacy suspended-cloud source
    ``max(dq_c_raw, 0)``; at PE=0.7 the anvil remainder + the rain fraction
    are EXACT fractions of that legacy source, so together they hold all of it.

    Asserted via the two PIECES separately (each is a bit-exact fraction of the
    legacy total — same op on the same operand as the scheme), NOT via the
    ``a*(1-pe)+a*pe`` reconstruction SUM, which rounds to ~1 ULP and would fail
    an fp32 rtol.  A real water leak from a wrong split fraction would move a
    piece by O(pe), far outside any rounding."""
    pe = jnp.clip(jnp.asarray(0.7), 0.0, 1.0)
    legacy = _run_pe(0.0).dq_c_conv_dt  # == jnp.maximum(dq_c_raw, 0.0)
    assert float(jnp.max(legacy)) > 0.0, "fixture must fire convection (dq_c>0)"
    out = _run_pe(0.7)
    # dq_r is EXACTLY pe*legacy and the anvil remainder EXACTLY (1-pe)*legacy;
    # their fractions sum to 1, so total condensate is conserved with no water
    # created or destroyed (precision-independent — holds in fp32 and fp64).
    assert jnp.array_equal(out.dq_r_conv_dt, legacy * pe), (
        "rain fraction != PE * legacy condensate (water not conserved)"
    )
    assert jnp.array_equal(out.dq_c_conv_dt, legacy * (1.0 - pe)), (
        "anvil remainder != (1-PE) * legacy condensate (water not conserved)"
    )


def test_bechtold_rain_split_leaves_heat_and_vapor_byte_identical():
    """dT_dt and dq_v_dt are BYTE-UNTOUCHED by the split — it only moves
    already-condensed water between two positive sink species and the latent
    heat is already booked in dT_dt (energy-neutral).  PE=0 vs PE=0.7, every
    other field held equal, so the split is the only difference."""
    out0 = _run_pe(0.0)
    out7 = _run_pe(0.7)
    assert jnp.array_equal(out0.dT_dt, out7.dT_dt), "dT_dt moved with the split"
    assert jnp.array_equal(out0.dq_v_dt, out7.dq_v_dt), "dq_v_dt moved with the split"


def test_bechtold_rain_split_off_is_legacy_none():
    """PE=0 (legacy) emits ``dq_r_conv_dt=None`` so downstream consumers that
    only detrain cloud stay byte-identically unaffected."""
    assert _run_pe(0.0).dq_r_conv_dt is None


def test_bechtold_rain_split_on_partitions_cloud():
    """PE=0.7: ``dq_r_conv_dt`` is finite and >=0 (a rain SOURCE), and exactly
    ``(1-PE)`` of the legacy cloud source remains as anvil (bit-exact
    partition — same op on the same operand as the scheme)."""
    pe_val = jnp.clip(jnp.asarray(0.7), 0.0, 1.0)
    legacy = _run_pe(0.0).dq_c_conv_dt
    out = _run_pe(0.7)
    assert out.dq_r_conv_dt is not None
    assert jnp.all(jnp.isfinite(out.dq_r_conv_dt))
    assert float(jnp.min(out.dq_r_conv_dt)) >= 0.0, "rain source must be >= 0"
    assert float(jnp.max(out.dq_r_conv_dt)) > 0.0, "fixture must produce rain"
    assert jnp.array_equal(out.dq_c_conv_dt, legacy * (1.0 - pe_val)), (
        "anvil remainder != (1-PE) * legacy cloud source"
    )
    assert jnp.array_equal(out.dq_r_conv_dt, legacy * pe_val), (
        "rain fraction != PE * legacy cloud source"
    )


# ---------------------------------------------------------------------------
# IFS-faithfulness fixes (audit vs ecmwf-ifs/openifs): F1/F4/F5 shipped;
# F6 REFUTED by SCM-RCE (mismapped coefficient) and reverted to the tuned value
# ---------------------------------------------------------------------------

from legoesm.atmosphere.physics.convection.bechtold import (  # noqa: E402
    _ifs_updraft_mean_velocity,
    _ifs_deep_turnover_scale,
    _ifs_cloud_base_qsat,
    _BECHTOLD_RH_CAP,
    _BECHTOLD_RH_ENTR,
    _BECHTOLD_RH_DETR,
    _IFS_TAU_MIN,
    _IFS_TAU_MAX,
    _IFS_WMEAN_MAX,
)


def test_bechtold_f6_midlevel_entrainment_stays_tuned_not_entshalp():
    """F6 (documented fidelity gap): IFS applies ENTSHALP*ENTRORG=3.5e-3
    entrainment to KTYPE>=2 — BOTH shallow (KTYPE=2) and elevated mid (KTYPE=3),
    cuascn.F90:500-507 — already reflected in ``epsilon_shallow``.  Our
    ``midlevel_weight`` is NOT an IFS KTYPE trigger; it is the cloud-depth
    transition blend (1 - deep - shallow) that also tags deepening SURFACE-based
    plumes.  Attaching 3.5e-3 to that blend over-entrains growing deep plumes and
    regressed equilibrium SCM-RCE by +19 K (isolated controlled comparison).  The
    coefficient value is IFS-correct but structurally mis-attached, so the blend
    keeps the tuned 1.0e-4 (below the deep 1.75e-3 rate surface plumes carry)
    pending a proper elevated-source classifier."""
    cfg = BechtoldConfig()
    assert cfg.epsilon_midlevel == 1.0e-4
    assert cfg.epsilon_midlevel < cfg.epsilon_deep
    assert cfg.epsilon_shallow == 3.5e-3  # the KTYPE>=2 ENTSHALP*ENTRORG value


def test_bechtold_f5_rh_cap_is_ifs_unity():
    """F5: RH capped at 1.0 (IFS MIN(1,q/qsat)) so the (1.3-RH)/(1.6-RH)
    entrainment/detrainment factors floor at 0.3/0.6 in saturated air
    (cuascn.F90:510,673), never below."""
    assert _BECHTOLD_RH_CAP == 1.0
    # At saturation the factors hit exactly the IFS floors.
    assert abs((_BECHTOLD_RH_ENTR - _BECHTOLD_RH_CAP) - 0.3) < 1e-12
    assert abs((_BECHTOLD_RH_DETR - _BECHTOLD_RH_CAP) - 0.6) < 1e-12


def test_bechtold_f1_turnover_tau_on_by_default():
    """F1: the IFS-structured state-dependent convective-turnover closure is the
    default (resolution-magnitude ZTAURES held at 1.0, a documented approximation);
    the fixed-tau_bl closure is opt-out."""
    assert BechtoldConfig().use_convective_turnover_tau is True


def test_bechtold_f1_updraft_velocity_helper_bounds_and_monotone():
    """The IFS updraught-velocity helper w_mean = sqrt(2*<PKINEU>) is bounded
    to [sqrt(0.02), 15] m/s (cuascn.F90:845-846, cumastrn.F90:773) and increases
    with plume buoyancy (a more buoyant plume rises faster)."""
    import math
    ncol, nlev = 3, 30
    T, q, pf, ph, u, v = _column(ncol=ncol, nlev=nlev)
    dz = jnp.abs(jnp.diff(-8500.0 * jnp.log(pf / 1e5), axis=-1,
                          append=(-8500.0 * jnp.log(pf / 1e5))[:, -1:]))
    dp = ph[:, 1:] - ph[:, :-1]
    eps = jnp.full((ncol, nlev), 1.75e-3)
    dlt = jnp.full((ncol, nlev), 0.75e-4)
    # above cloud base = upper two-thirds; in-cloud = middle third.
    above = jnp.zeros((ncol, nlev)).at[:, : 2 * nlev // 3].set(1.0)
    in_cloud = jnp.zeros((ncol, nlev)).at[:, nlev // 3: 2 * nlev // 3].set(1.0)
    # positive buoyancy in the cloud layer.
    B = jnp.zeros((ncol, nlev)).at[:, nlev // 3: 2 * nlev // 3].set(1.0)
    ke_floor_v = math.sqrt(2.0 * 1e-2)   # IFS mean-KE floor -> ~0.141 m/s
    w1 = _ifs_updraft_mean_velocity(B, T, q, dz, eps, dlt, dp, above, in_cloud)
    w2 = _ifs_updraft_mean_velocity(3.0 * B, T, q, dz, eps, dlt, dp, above, in_cloud)
    w0 = _ifs_updraft_mean_velocity(jnp.zeros_like(B), T, q, dz, eps, dlt, dp, above, in_cloud)
    for w in (w0, w1, w2):
        assert jnp.all(jnp.isfinite(w))
        assert jnp.all(w >= ke_floor_v - 1e-6) and jnp.all(w <= _IFS_WMEAN_MAX)
    # more buoyant -> faster updraught (until the 15 m/s cap).
    assert float(jnp.mean(w2)) > float(jnp.mean(w1))
    assert float(jnp.mean(w1)) >= float(jnp.mean(w0))


def test_bechtold_f1_turnover_tau_clamp_constants():
    """F1: the turnover-time clamp matches the IFS [3600/5, 3*3600] s bounds
    (cumastrn.F90:827)."""
    assert _IFS_TAU_MIN == 720.0
    assert _IFS_TAU_MAX == 10800.0


def test_bechtold_f1_turnover_toggle_changes_result():
    """F1: enabling the turnover closure changes the mass-flux profile vs the
    fixed-tau_bl closure (the rescale is live).  Uses a large M_b_max so the
    tau rescale is observable rather than masked by the M_b_max clip (a deep
    buoyant column would otherwise saturate the cap under both closures — which
    is itself the correct BUG4-fix behaviour, mb_scale->1 at the cap)."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=299.0, q_sfc=13e-3,
                                 lapse_rate=6.5)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.zeros((ncol,))
    on, mu_on, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_convective_turnover_tau=True,
                              use_ifs_cape_closure=False, M_b_max=1.0))
    off, mu_off, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_convective_turnover_tau=False,
                              use_ifs_cape_closure=False, M_b_max=1.0))
    assert jnp.all(jnp.isfinite(mu_on)) and jnp.all(jnp.isfinite(mu_off))
    assert float(jnp.max(jnp.abs(mu_on - mu_off))) > 0.0, "toggle must be live"
    # a deep, buoyant column has tau_conv < tau_bl (fast turnover) => turnover
    # intensifies the mass flux relative to the fixed 3600 s closure.
    assert float(jnp.sum(mu_on)) > float(jnp.sum(mu_off))


def test_bechtold_f1_turnover_stable_column_quiesces():
    """F1: a stable, zero-CAPE column stays quiescent under the turnover
    rescale — the cape_weight**2 * M_b_max launch cap dominates any tau
    correction (spurious heating well under the <1 W/m^2 bar)."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=280.0, q_sfc=2e-3,
                                 lapse_rate=3.0)
    ncol, nlev = T.shape
    out, mu, _ = bechtold_convection(
        T, q, pf, ph, u, v, jnp.zeros((ncol, nlev)), jnp.zeros((ncol,)),
        None, dt=600.0, config=BechtoldConfig(use_convective_turnover_tau=True,
                                              use_ifs_cape_closure=False))
    # crude column heating rate proxy: max|dT/dt| * c_pd * p_s/g  [W/m^2]
    w_m2 = float(jnp.max(jnp.abs(out.dT_dt)) * constants.c_pd * 1e5 / constants.g)
    assert w_m2 < 1.0, f"stable column not quiescent: {w_m2:.3f} W/m^2"


def test_bechtold_f1_turnover_grad_finite():
    """F1: jax.grad flows through the turnover-tau KE budget + rescale."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=30, T_sfc=300.0, q_sfc=14e-3)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.zeros((ncol,))

    def loss(eps_deep):
        cfg = BechtoldConfig(epsilon_deep=eps_deep,
                             use_convective_turnover_tau=True,
                             use_ifs_cape_closure=False)
        o, _, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, st, None,
                                      dt=600.0, config=cfg)
        return jnp.sum(o.dT_dt ** 2)

    g = jax.grad(loss)(1.75e-3)
    assert jnp.isfinite(g)


def test_bechtold_f1_turnover_grad_finite_on_quiescent_column():
    """F1/BUG4 edge: grad must stay finite on a STABLE column where the
    cloud-base flux M_b -> 0 (the tau rescale's division-by-M_b_before edge).
    The AD-safe mb_scale selects the no-division branch when below the M_b_max
    clip, so no 1/M_b_before gradient trap."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=30, T_sfc=280.0, q_sfc=2e-3,
                                 lapse_rate=3.0)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.zeros((ncol,))

    def loss(eps_deep):
        cfg = BechtoldConfig(epsilon_deep=eps_deep,
                             use_convective_turnover_tau=True,
                             use_ifs_cape_closure=False)
        o, _, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, st, None,
                                      dt=600.0, config=cfg)
        return jnp.sum(o.dT_dt ** 2)

    g = jax.grad(loss)(1.75e-3)
    assert jnp.isfinite(g)


def test_bechtold_f1_updraft_velocity_subcloud_insensitive():
    """F1/BUG1: w_mean is a CLOUD-window mean — a PHYSICAL sub-cloud buoyancy
    perturbation (below the cloud base) must barely move it (the launch KE
    propagates through the sub-cloud gate; production is pre-gated to
    at/above base).  Residual smooth-mask leak stays well under 5%."""
    ncol, nlev = 1, 8
    T = jnp.full((ncol, nlev), 300.0)
    q = jnp.full((ncol, nlev), 1e-2)
    dz = jnp.full((ncol, nlev), 100.0)
    eps = jnp.full((ncol, nlev), 1e-3)
    dlt = jnp.full((ncol, nlev), 0.75e-4)
    dp = jnp.full((ncol, nlev), 1000.0)
    lev = jnp.arange(nlev, dtype=T.dtype)[None, :]
    above = jax.nn.sigmoid(4.0 * (5.0 - lev))          # cloud base ~ level 5
    in_cloud = above * jax.nn.sigmoid(4.0 * (lev - 2.0))
    B = jnp.zeros((ncol, nlev)).at[:, 2:6].set(1.0)     # in-cloud buoyancy
    w0 = _ifs_updraft_mean_velocity(B, T, q, dz, eps, dlt, dp, above, in_cloud)[0]
    # perturb ONLY the sub-cloud layers (indices 6,7 below base) by +/-2 K.
    w_pos = _ifs_updraft_mean_velocity(B.at[:, 6:].add(2.0), T, q, dz, eps, dlt, dp, above, in_cloud)[0]
    w_neg = _ifs_updraft_mean_velocity(B.at[:, 6:].add(-2.0), T, q, dz, eps, dlt, dp, above, in_cloud)[0]
    rel = max(float(jnp.abs(w_pos - w0)), float(jnp.abs(w_neg - w0))) / float(w0)
    assert rel < 0.05, f"sub-cloud buoyancy leaks {rel:.1%} into w_mean"


def test_ifs_deep_turnover_scale_caps_after_rescale():
    """F1/codex finding-1: M_b_max is applied AFTER the turnover rescale
    (cumastrn.F90:828-831).  For an uncapped flux above the cap with a lengthening
    turnover time (r<1), the correct result is ``min(uncapped*r, cap)``, NOT the
    cap-first ``min(uncapped, cap)*r`` that under-scales the flux."""
    M_b_uncapped = jnp.array([0.20])
    M_b_capped = jnp.array([0.05])          # = min(0.20, M_b_max)
    M_b_max = 0.05
    r = jnp.array([0.5])                     # turnover lengthens
    scale = _ifs_deep_turnover_scale(M_b_uncapped, M_b_capped, r,
                                     jnp.array([1.0]), M_b_max)     # deep
    M_b_new = float(M_b_capped[0] * scale[0])
    # cap-after (correct): min(0.20*0.5, 0.05) = min(0.10, 0.05) = 0.05
    assert abs(M_b_new - 0.05) < 1e-12
    # cap-first (the bug) would give 0.05*0.5 = 0.025
    assert abs(M_b_new - 0.025) > 1e-3
    # and when r>1 with an already-capped flux, the min binds at the cap.
    scale_hi = _ifs_deep_turnover_scale(M_b_uncapped, M_b_capped, jnp.array([2.0]),
                                        jnp.array([1.0]), M_b_max)
    assert abs(float(M_b_capped[0] * scale_hi[0]) - 0.05) < 1e-12
    # sub-cap (uncapped==capped, below M_b_max): scale reduces to r exactly.
    scale_sub = _ifs_deep_turnover_scale(jnp.array([0.01]), jnp.array([0.01]),
                                         jnp.array([0.7]), jnp.array([1.0]), 0.05)
    assert abs(float(scale_sub[0]) - 0.7) < 1e-12


def test_ifs_deep_turnover_scale_shallow_is_noop():
    """F1/codex finding-2: the turnover CAPE closure is deep-only (KTYPE==1,
    cumastrn.F90:762).  With deep_weight=0 (shallow/mid) the rescale factor is
    exactly 1 regardless of the turnover time; it blends smoothly to the full
    deep scale as deep_weight->1."""
    args = (jnp.array([0.03]), jnp.array([0.02]), jnp.array([3.0]))  # uncapped, capped, r
    assert abs(float(_ifs_deep_turnover_scale(*args, jnp.array([0.0]), 0.05)[0]) - 1.0) < 1e-12
    # half-deep column: scale is the midpoint of 1.0 and the full deep scale.
    full = float(_ifs_deep_turnover_scale(*args, jnp.array([1.0]), 0.05)[0])
    half = float(_ifs_deep_turnover_scale(*args, jnp.array([0.5]), 0.05)[0])
    assert abs(half - 0.5 * (1.0 + full)) < 1e-12


def test_ifs_deep_turnover_scale_quiescent_ad_safe():
    """F1: the M_b_capped->0 quiescent edge is finite in value AND gradient (the
    AD-safe double-where avoids a 1/M_b_capped**2 reverse-mode reciprocal)."""
    def scaled_flux(mb_capped):
        arr = jnp.array([mb_capped])
        s = _ifs_deep_turnover_scale(jnp.array([0.0]), arr, jnp.array([2.0]),
                                     jnp.array([1.0]), 0.05)
        return jnp.sum(arr * s)
    assert jnp.isfinite(scaled_flux(0.0))
    assert jnp.isfinite(jax.grad(scaled_flux)(0.0))


def test_bechtold_f1_ke_drag_keys_on_entrainment_active_not_eps_lt_dlt():
    """F1/codex R2 finding-2: our KE-drag switch keys on the prescribed
    entrainment rate being ACTIVE (``eps > 0``), NOT on ``eps < dlt`` — a SURROGATE
    for the IFS ``IF(ZDMFEN>0) eps ELSE dlt`` switch (cuascn.F90:646-652; IFS keys
    on the dynamically-diagnosed ZDMFEN, a documented deviation).  This asserts OUR
    switch behavior, four upper-cloud regimes over the SAME layers, dlt > eps in all:
      (a) eps small-POSITIVE (5e-6 < dlt) -> selects eps (weak drag) -> high w;
      (b) eps TINY-POSITIVE (1e-10 < dlt) -> STILL eps (exact eps>0 switch, no
          invented floor) -> ~ no drag -> high w;
      (c) eps == 0, dlt > 0 -> substitutes dlt (drag maintained) -> lower w;
      (d) eps == 0, dlt == 0 -> no drag at all -> highest w (KE undamped).
    So w(a),w(b) > w(c) [any positive eps selects eps, not the larger dlt — even
    1e-10, which a floored switch would misroute to dlt] and w(d) > w(c) [dlt
    substitution only where eps is exactly 0].  A wrong ``max(eps,dlt)`` surrogate
    would pick dlt in (a)/(b) and fail these."""
    ncol, nlev = 1, 20
    T = jnp.full((ncol, nlev), 300.0)
    q = jnp.full((ncol, nlev), 1e-2)
    dz = jnp.full((ncol, nlev), 300.0)
    dp = jnp.full((ncol, nlev), 1000.0)
    lev = jnp.arange(nlev, dtype=T.dtype)[None, :]
    above = jax.nn.sigmoid(4.0 * (14.0 - lev))
    in_cloud = above * jax.nn.sigmoid(4.0 * (lev - 3.0))
    B = jnp.zeros((ncol, nlev)).at[:, 3:14].set(1.0)
    dlt = jnp.full((ncol, nlev), 4.5e-5)                 # > every eps below
    up = slice(3, 9)                                     # upper-cloud layers

    def w_of(eps_up, dlt_arr):
        eps = jnp.full((ncol, nlev), 1.75e-3).at[:, up].set(eps_up)
        return float(_ifs_updraft_mean_velocity(B, T, q, dz, eps, dlt_arr, dp, above, in_cloud)[0])

    w_a = w_of(5.0e-6, dlt)      # small positive eps
    w_b = w_of(1.0e-10, dlt)     # tiny positive eps (below any plausible floor)
    w_c = w_of(0.0, dlt)         # eps exactly 0 -> dlt substituted
    w_d = w_of(0.0, jnp.zeros_like(dlt))   # eps 0, dlt 0 -> no drag
    assert w_a > w_c, "small positive eps must select eps (IFS ZDMFEN>0), not the larger dlt"
    assert w_b > w_c, "even a 1e-10 eps is active (exact eps>0 switch, no floor)"
    assert w_d > w_c, "dlt substitution maintains drag only where eps is exactly 0"


def test_bechtold_f1_zero_entrainment_config_reaches_else_branch_finite():
    """F1/codex R4: the drag ``where(eps>0, eps, dlt)`` ELSE branch IS reachable in
    production (not "structurally unreached") — a config with all ``epsilon_* = 0``
    drives a convecting column with ``eps_profile == 0`` everywhere, so every KE
    layer takes the delta substitution.  The turnover closure must stay finite in
    value AND gradient there (the eps==0 drag jump is a finite discontinuity
    inherited from IFS's discrete switch, not a NaN)."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=300.0, q_sfc=15e-3,
                                 lapse_rate=7.5)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev)); st = jnp.zeros((ncol,))
    zero_ent = BechtoldConfig(
        epsilon_deep=0.0, epsilon_shallow=0.0, epsilon_midlevel=0.0,
        use_convective_turnover_tau=True)
    out, mu, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, st, None,
                                     dt=600.0, config=zero_ent)
    assert jnp.all(jnp.isfinite(out.dT_dt)) and jnp.all(jnp.isfinite(mu))

    def loss(delta_deep):
        cfg = zero_ent._replace(delta_deep=delta_deep)
        o, _, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, st, None,
                                      dt=600.0, config=cfg)
        return jnp.sum(o.dT_dt ** 2)

    assert jnp.isfinite(jax.grad(loss)(0.75e-4))   # grad through the delta-drag branch


def test_bechtold_f1_turnover_deep_weighted_integration_live_and_bounded():
    """F1/codex R2 finding-3 (integration): the turnover rescale is DEEP-WEIGHTED
    (smooth ``deep_weight`` blend, the AD analog of IFS's discrete KTYPE==1), not a
    hard gate.  Here we only assert the integration-level SANITY: the toggle is
    live (changes the mass flux) yet bounded/finite on both a shallow-ish and a
    deep column.  The EXACT deep-weight semantics — ``deep_weight=0`` is a perfect
    no-op and a transitional cloud gets the correct convex-combination partial
    rescale — are locked by ``test_ifs_deep_turnover_scale_shallow_is_noop``, which
    tests the extracted helper directly (an integration proxy for deep_weight is
    fragile because a genuinely-shallow column barely convects)."""
    for lapse_rate, q_sfc in ((5.0, 10e-3), (7.5, 16e-3)):
        T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=299.0,
                                     q_sfc=q_sfc, lapse_rate=lapse_rate)
        ncol, nlev = T.shape
        cpp = jnp.zeros((ncol, nlev)); st = jnp.zeros((ncol,))
        _, mu_on, _ = bechtold_convection(
            T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
            config=BechtoldConfig(use_convective_turnover_tau=True,
                                  use_ifs_cape_closure=False, M_b_max=10.0))
        _, mu_off, _ = bechtold_convection(
            T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
            config=BechtoldConfig(use_convective_turnover_tau=False,
                                  use_ifs_cape_closure=False, M_b_max=10.0))
        assert jnp.all(jnp.isfinite(mu_on)) and jnp.all(jnp.isfinite(mu_off))
        denom = float(jnp.maximum(jnp.max(jnp.abs(mu_off)), 1e-12))
        rel = float(jnp.max(jnp.abs(mu_on - mu_off))) / denom
        assert 0.0 < rel < 5.0, f"turnover rescale not live/bounded (rel={rel})"


def test_bechtold_f5_supersaturated_column_finite_and_convecting():
    """F5 (behavioral): a SUPERSATURATED sounding (q_v > q_sat, RH>1) is driven
    through the scheme.  The IFS RH cap (MIN(1,q/qsat), _BECHTOLD_RH_CAP=1.0)
    floors the (1.3-RH) entrainment factor at 0.3 instead of letting it fall to 0
    (or negative) as the former 1.3 cap allowed — so entrainment stays positive,
    the tendencies stay finite, and the column still convects."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=301.0, q_sfc=17e-3,
                                 lapse_rate=7.0)
    ncol, nlev = T.shape
    q_sat = saturation_mixing_ratio(T, pf)
    q_super = jnp.maximum(q, 1.15 * q_sat)      # force RH ~ 1.15 (supersaturated)
    out, mu, _ = bechtold_convection(
        T, q_super, pf, ph, u, v, jnp.zeros((ncol, nlev)), jnp.zeros((ncol,)),
        None, dt=600.0, config=BechtoldConfig(use_convective_turnover_tau=True,
                                              use_ifs_cape_closure=False))
    assert jnp.all(jnp.isfinite(out.dT_dt)) and jnp.all(jnp.isfinite(mu))
    assert float(jnp.max(out.convective_mask)) > 0.0, "supersaturated column must convect"


def test_bechtold_f4_qsat_base_gathered_at_cloud_base_not_surface():
    """F4: the entrainment ``f_scale = (q_sat/q_sat_base)^3`` anchors ``q_sat_base``
    at the smooth cloud base (IFS PQSEN(.,IKB), IKB=KCBOT, cuascn.F90:674), NOT the
    surface.  Test the extracted soft-gather directly: with a monotone q_sat
    profile it returns ~q_sat at the cloud-base index and is negligibly sensitive
    to the (much larger) surface-level q_sat (the Gaussian softmax exponentially
    downweights it — nonzero weight, but tiny)."""
    ncol, nlev = 1, 20
    lev = jnp.arange(nlev, dtype=float)[None, :]           # adaptive dtype (x64/float32)
    # surface-last, q_sat decreasing upward (cold aloft); base at index 12.
    q_sat_env = jnp.linspace(1e-4, 2e-2, nlev)[None, :]     # index -1 (surface) largest
    k_base = jnp.array([12.0])
    q_base = _ifs_cloud_base_qsat(q_sat_env, k_base, lev[0])
    # gathered value tracks q_sat at the cloud-base index, NOT the surface value.
    assert abs(float(q_base[0, 0]) - float(q_sat_env[0, 12])) < 0.15 * float(q_sat_env[0, 12])
    assert float(q_base[0, 0]) < float(q_sat_env[0, -1])   # well below the surface q_sat
    # perturbing ONLY the surface level leaves q_sat_base essentially unchanged.
    q_sat_surf_hot = q_sat_env.at[:, -1].multiply(3.0)
    q_base2 = _ifs_cloud_base_qsat(q_sat_surf_hot, k_base, lev[0])
    assert abs(float(q_base2[0, 0]) - float(q_base[0, 0])) < 1e-3 * float(q_base[0, 0])


def test_bechtold_f4_qsat_base_helper_differentiable():
    """F4: the cloud-base q_sat soft-gather is smooth in the (fractional) base
    index, so jax.grad flows (the entrainment anchor stays differentiable)."""
    ncol, nlev = 1, 16
    lev = jnp.arange(nlev, dtype=float)[None, :]          # adaptive dtype (x64/float32)
    q_sat_env = jnp.linspace(1e-4, 2e-2, nlev)[None, :]

    def base_qsat(kb):
        return jnp.sum(_ifs_cloud_base_qsat(q_sat_env, jnp.array([kb]), lev[0]))

    assert jnp.isfinite(jax.grad(base_qsat)(9.0))


# ---------------------------------------------------------------------------
# IFS deep CAPE closure ZMFUB1 = ZCAPE*ZMFUB/(ZHEAT*ZXTAU) (cumastrn.F90:704-833)
# ---------------------------------------------------------------------------

from legoesm.atmosphere.physics.convection.bechtold import (  # noqa: E402
    _ifs_cape_closure_target,
    _ifs_deep_target_scale,
    _IFS_RETV,
    _IFS_ZCAPE_MAX_PA,
    _IFS_ZHEAT_FLOOR,
    _IFS_MB_DEEP_FLOOR,
)


def test_ifs_cape_closure_constants_match_oracle():
    """The closure constants are the oracle's: RETV = R_v/R_d - 1
    (yomcst.F90:342), ZCAPE cap 5000 Pa (cumastrn.F90:825), ZHEAT floor 1e-4
    (cumastrn.F90:826), deep M_b floor 0.001 (cumastrn.F90:829)."""
    assert abs(_IFS_RETV - (constants.R_v / constants.R_d - 1.0)) < 1e-12
    assert _IFS_ZCAPE_MAX_PA == 5000.0
    assert _IFS_ZHEAT_FLOOR == 1.0e-4
    assert _IFS_MB_DEEP_FLOOR == 1.0e-3


def test_ifs_cape_closure_target_analytic():
    """Hand-computed ZCAPE/ZHEAT on a 4-level column reproduce the helper.

    The expected values are built with explicit Fortran-shaped loops mirroring
    cumastrn.F90:722-733 (k-1 = the level above k; both index downward), so an
    orientation or off-by-one bug in the vectorized helper goes red."""
    import numpy as np
    T = np.array([[250.0, 270.0, 285.0, 295.0]])       # surface-last
    q = np.array([[1e-4, 1e-3, 5e-3, 1e-2]])
    z = np.array([[9000.0, 6000.0, 3000.0, 500.0]])
    # Deliberately STRETCHED full-level pressures: the ZCAPE measure is the
    # oracle's PAP(k)-PAP(k-1) full-level spacing (cumastrn.F90:728), which on
    # this grid differs from any half-level layer thickness — a measure bug
    # goes red here.
    p = np.array([[300e2, 500e2, 800e2, 1000e2]])
    T_u = T + np.array([[0.0, 1.5, 2.0, 0.0]])         # buoyant in-cloud plume
    q_u = q + np.array([[0.0, 5e-4, 1e-3, 0.0]])
    q_c_u = np.array([[0.0, 1e-3, 1.5e-3, 0.0]])
    M_u = np.array([[0.0, 0.08, 0.10, 0.10]])
    M_d = np.array([[0.0, -0.02, -0.03, -0.03]])
    in_cloud = np.array([[0.0, 1.0, 1.0, 0.0]])
    tau = np.array([1500.0])
    M_b_fg = np.array([0.10])
    cape_w = np.array([1.0])

    retv = constants.R_v / constants.R_d - 1.0
    g, cpd = constants.g, constants.c_pd
    zcape = 0.0
    zheat = 0.0
    for k in range(1, 4):                               # k-1 exists
        if in_cloud[0, k] == 1.0:
            zcape += (
                (T_u[0, k] - T[0, k]) / T[0, k]
                + retv * (q_u[0, k] - q[0, k])
                - q_c_u[0, k]
            ) * (p[0, k] - p[0, k - 1])                 # ZDZ = PAP(k)-PAP(k-1)
            stab = (
                (T[0, k - 1] - T[0, k] + g * (z[0, k - 1] - z[0, k]) / cpd)
                / T[0, k]
                + retv * (q[0, k - 1] - q[0, k])
            )
            zheat += max(0.0, stab) * g * (M_u[0, k] + M_d[0, k])
    # cape_weight multiplies the target as the smooth LDCUM membership
    # (IFS runs the closure only where the trigger fired).
    expected = (
        cape_w[0] * min(zcape, 5000.0) * M_b_fg[0]
        / (max(zheat, 1e-4) * tau[0])
    )
    expected = max(expected, 1e-3 * cape_w[0] ** 2)

    got = _ifs_cape_closure_target(
        jnp.asarray(T), jnp.asarray(q), jnp.asarray(z), jnp.asarray(p),
        jnp.asarray(T_u), jnp.asarray(q_u), jnp.asarray(q_c_u),
        jnp.asarray(M_u), jnp.asarray(M_d), jnp.asarray(in_cloud),
        jnp.asarray(tau), jnp.asarray(M_b_fg), jnp.asarray(cape_w),
    )
    assert got.shape == (1,)
    tol = 1e-9 if got.dtype == jnp.float64 else 1e-6
    assert abs(float(got[0]) - expected) < tol * max(abs(expected), 1.0), (
        f"helper {float(got[0]):.6e} != hand-computed {expected:.6e}"
    )
    # The floor is live and quiescence-gated: zero plume buoyancy + zero
    # trigger must NOT be handed the 0.001 deep floor.
    got_quiet = _ifs_cape_closure_target(
        jnp.asarray(T), jnp.asarray(q), jnp.asarray(z), jnp.asarray(p),
        jnp.asarray(T), jnp.asarray(q), jnp.asarray(0.0 * q_c_u),
        jnp.asarray(0.0 * M_u), jnp.asarray(0.0 * M_d), jnp.asarray(in_cloud),
        jnp.asarray(tau), jnp.asarray(0.0 * M_b_fg), jnp.asarray(0.0 * cape_w),
    )
    assert float(got_quiet[0]) == 0.0
    # Model-top exclusion symmetry: perturbing ONLY the top level's plume
    # buoyancy must not move the target (both ZCAPE and ZHEAT start at k=1).
    got_top = _ifs_cape_closure_target(
        jnp.asarray(T), jnp.asarray(q), jnp.asarray(z), jnp.asarray(p),
        jnp.asarray(T_u).at[:, 0].add(5.0), jnp.asarray(q_u),
        jnp.asarray(q_c_u), jnp.asarray(M_u), jnp.asarray(M_d),
        jnp.asarray(in_cloud).at[:, 0].set(0.5), jnp.asarray(tau),
        jnp.asarray(M_b_fg), jnp.asarray(cape_w),
    )
    assert float(jnp.abs(got_top[0] - got[0])) == 0.0
    # Oracle restart semantics (codex R6): a ZERO first guess with a NONZERO
    # launched plume (the floored-launch split) zeroes the numerator, so the
    # target is EXACTLY the 0.001*cape_weight**2 floor — not the shape-only
    # closure value the cancellation would give.
    got_restart = _ifs_cape_closure_target(
        jnp.asarray(T), jnp.asarray(q), jnp.asarray(z), jnp.asarray(p),
        jnp.asarray(T_u), jnp.asarray(q_u), jnp.asarray(q_c_u),
        jnp.asarray(M_u), jnp.asarray(M_d), jnp.asarray(in_cloud),
        jnp.asarray(tau), jnp.asarray(0.0 * M_b_fg), jnp.asarray(cape_w),
    )
    assert float(got_restart[0]) == 1e-3 * float(cape_w[0]) ** 2
    # Trigger membership: a half-triggered column gets half the target
    # (above the floor) — the smooth LDCUM analog (codex R3 #1).
    got_half = _ifs_cape_closure_target(
        jnp.asarray(T), jnp.asarray(q), jnp.asarray(z), jnp.asarray(p),
        jnp.asarray(T_u), jnp.asarray(q_u), jnp.asarray(q_c_u),
        jnp.asarray(M_u), jnp.asarray(M_d), jnp.asarray(in_cloud),
        jnp.asarray(tau), jnp.asarray(M_b_fg), jnp.asarray(0.5 * cape_w),
    )
    assert abs(float(got_half[0]) - 0.5 * float(got[0])) < tol * max(
        abs(0.5 * float(got[0])), 1.0
    )


def test_ifs_cape_closure_first_guess_cancels_above_floor():
    """Above the ZHEAT floor the closure is independent of the first-guess
    magnitude (ZHEAT scales linearly with M_b_fg via M_u, so ZMFUB cancels —
    the closure depends only on the plume SHAPE)."""
    T = jnp.array([[250.0, 270.0, 285.0, 295.0]])
    q = jnp.array([[1e-4, 1e-3, 5e-3, 1e-2]])
    z = jnp.array([[9000.0, 6000.0, 3000.0, 500.0]])
    p = jnp.array([[300e2, 500e2, 800e2, 1000e2]])
    T_u = T + jnp.array([[0.0, 1.5, 2.0, 0.0]])
    q_u = q
    q_c_u = jnp.zeros_like(q)
    in_cloud = jnp.array([[0.0, 1.0, 1.0, 0.0]])
    tau = jnp.array([1500.0])
    cape_w = jnp.array([1.0])
    shape = jnp.array([[0.0, 0.8, 1.0, 1.0]])

    def target(mb_fg):
        return _ifs_cape_closure_target(
            T, q, z, p, T_u, q_u, q_c_u,
            mb_fg * shape, jnp.zeros_like(shape), in_cloud,
            tau, jnp.array([mb_fg]), cape_w,
        )[0]

    t1 = float(target(0.05))
    t2 = float(target(0.5))
    tol = 1e-9 if jnp.asarray(0.0).dtype == jnp.float64 else 1e-5
    assert abs(t1 - t2) < tol * max(t1, 1e-30), (
        "first guess did not cancel above the ZHEAT floor"
    )


def test_ifs_deep_target_scale_caps_after_rescale_and_shallow_noop():
    """The generalized target-scale kernel caps AFTER the rescale
    (min(target, M_b_max), not min-then-scale) and is an exact no-op for the
    non-deep classes (deep_weight -> 0)."""
    M_b_capped = jnp.array([0.05])
    target = jnp.array([0.20])                       # above the 0.05 cap
    tol = 1e-12 if M_b_capped.dtype == jnp.float64 else 1e-8
    s = _ifs_deep_target_scale(target, M_b_capped, jnp.array([1.0]), 0.05)
    assert abs(float(M_b_capped[0] * s[0]) - 0.05) < tol
    s0 = _ifs_deep_target_scale(target, M_b_capped, jnp.array([0.0]), 0.05)
    assert abs(float(s0[0]) - 1.0) < tol
    # quiescent edge: M_b_capped == 0 takes the constant-0 branch.
    sq = _ifs_deep_target_scale(target, jnp.array([0.0]), jnp.array([1.0]), 0.05)
    assert jnp.isfinite(sq[0])
    # tiny-POSITIVE edge (codex R1 #4): the target is not proportional to
    # M_b_capped (trigger-gated floor), so without the divisor floor the ratio
    # overflows in fp32 and deep_weight=0 turns 0*inf into NaN.  Both weights
    # must stay finite, the realized flux bounded by the cap, and the non-deep
    # blend an exact no-op.
    tiny_mb = jnp.array([1e-30])
    st1 = _ifs_deep_target_scale(jnp.array([1e-3]), tiny_mb, jnp.array([1.0]), 0.05,
                                 divisor_floor=1e-20)
    st0 = _ifs_deep_target_scale(jnp.array([1e-3]), tiny_mb, jnp.array([0.0]), 0.05,
                                 divisor_floor=1e-20)
    assert jnp.isfinite(st1[0]) and jnp.isfinite(st0[0])
    assert float(tiny_mb[0] * st1[0]) <= 0.05 * (1.0 + 1e-6)
    assert abs(float(st0[0]) - 1.0) < 1e-6
    # The TURNOVER wrapper must NOT be floored (codex R2 #2): its target is
    # proportional to M_b, so for a tiny-but-positive flux the legacy scale is
    # exactly tau_correction — flooring would collapse it by orders of
    # magnitude vs main on the default-on turnover path.
    r = jnp.array([0.7])
    s_turn = _ifs_deep_turnover_scale(tiny_mb, tiny_mb, r, jnp.array([1.0]), 0.05)
    assert abs(float(s_turn[0]) - 0.7) < 1e-6


def test_ifs_cape_closure_default_on_and_toggle_live():
    """Default True (2026-07-16: codex x11 + gray-RCE A/B + AMIP smoke A/B);
    toggling it changes the mass-flux profile on a deep convecting column
    (the closure is live, False restores the legacy surrogate)."""
    assert BechtoldConfig().use_ifs_cape_closure is True
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=299.0, q_sfc=13e-3,
                                 lapse_rate=6.5)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.zeros((ncol,))
    on, mu_on, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_ifs_cape_closure=True, M_b_max=1.0))
    off, mu_off, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_ifs_cape_closure=False, M_b_max=1.0))
    assert jnp.all(jnp.isfinite(mu_on)) and jnp.all(jnp.isfinite(mu_off))
    assert float(jnp.max(jnp.abs(mu_on - mu_off))) > 0.0, "toggle must be live"
    for o in (on, off):
        assert jnp.all(jnp.isfinite(o.dT_dt))
        assert jnp.all(jnp.isfinite(o.dq_v_dt))


def test_ifs_cape_closure_without_turnover_flag_runs():
    """The closure builds the tau_conv machinery itself even when the F1
    turnover flag is off (the or-gate), and stays finite."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=30, T_sfc=300.0, q_sfc=14e-3)
    ncol, nlev = T.shape
    out, mu, _ = bechtold_convection(
        T, q, pf, ph, u, v, jnp.zeros((ncol, nlev)), jnp.zeros((ncol,)),
        None, dt=600.0,
        config=BechtoldConfig(use_ifs_cape_closure=True,
                              use_convective_turnover_tau=False))
    assert jnp.all(jnp.isfinite(mu))
    assert jnp.all(jnp.isfinite(out.dT_dt))


def test_ifs_cape_closure_stable_column_quiesces():
    """A stable, zero-CAPE column stays quiescent under the full closure —
    the quiescence-gated deep floor must not inject the IFS 0.001 kg/m^2/s
    minimum into an untriggered column (<1 W/m^2 bar, same as F1)."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=280.0, q_sfc=2e-3,
                                 lapse_rate=3.0)
    ncol, nlev = T.shape
    out, mu, _ = bechtold_convection(
        T, q, pf, ph, u, v, jnp.zeros((ncol, nlev)), jnp.zeros((ncol,)),
        None, dt=600.0, config=BechtoldConfig(use_ifs_cape_closure=True))
    w_m2 = float(jnp.max(jnp.abs(out.dT_dt)) * constants.c_pd * 1e5 / constants.g)
    assert w_m2 < 1.0, f"stable column not quiescent: {w_m2:.3f} W/m^2"


def test_ifs_cape_closure_grad_finite_convecting_and_quiescent():
    """jax.grad flows through ZCAPE/ZHEAT/tau and the target rescale on both a
    convecting and a stable column (floored divisions only)."""
    for kwargs in (dict(T_sfc=300.0, q_sfc=14e-3),
                   dict(T_sfc=280.0, q_sfc=2e-3, lapse_rate=3.0)):
        T, q, pf, ph, u, v = _column(ncol=2, nlev=30, **kwargs)
        ncol, nlev = T.shape
        cpp = jnp.zeros((ncol, nlev))
        st = jnp.zeros((ncol,))

        def loss(eps_deep):
            cfg = BechtoldConfig(epsilon_deep=eps_deep,
                                 use_ifs_cape_closure=True)
            o, _, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, st, None,
                                          dt=600.0, config=cfg)
            return jnp.sum(o.dT_dt ** 2)

        g = jax.grad(loss)(1.75e-3)
        assert jnp.isfinite(g), f"non-finite grad on column {kwargs}"


def test_ifs_cape_closure_restarts_stochastic_zeroed_deep_column():
    """IFS floors the triggered deep flux at 0.001 kg/m^2/s (cumastrn.F90:829).
    A convecting deep column whose AR1 stochastic factor clips M_b to hard
    zero must still launch a plume under the closure — the floor acts on the
    first guess BEFORE plume integration, because a pure rescale of a zero
    plume stays zero (codex R5).  The legacy path documents the divergence
    (it stays shut down)."""
    T, q, pf, ph, u, v = _column(ncol=1, nlev=40, T_sfc=299.0, q_sfc=13e-3,
                                 lapse_rate=6.5)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.full((ncol,), -50.0)      # AR1 state so 1 + 0.5*state' <= 0
    key = jax.random.PRNGKey(0)
    mu_on = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, key, dt=600.0,
        config=BechtoldConfig(use_ifs_cape_closure=True,
                              enable_stochastic=True))[1]
    mu_off = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, key, dt=600.0,
        config=BechtoldConfig(use_ifs_cape_closure=False,
                              enable_stochastic=True))[1]
    assert float(jnp.max(mu_on)) > 0.0, (
        "closure floor failed to restart the stochastically-zeroed deep column"
    )
    assert float(jnp.max(mu_off)) == 0.0, (
        "legacy path unexpectedly restarted (fixture no longer isolates the floor)"
    )


def test_ifs_cape_closure_scale_transition_band_convex():
    """The blend realizes d*min(target,max) + (1-d)*M_b_true (codex R7): with a
    hard-zero true M_b and a floored launch, the transition band (d=0.5) must
    give the CONVEX floor*d, not floor*d*(2-d); with the floor inactive it
    reduces exactly to the legacy scale blend."""
    from legoesm.atmosphere.physics.convection.bechtold import (
        _ifs_cape_closure_scale,
    )
    d = jnp.array([0.5])
    floor_f = jnp.array([1e-3])                      # target == restart floor
    launch = floor_f * d                             # floored launch, M_b_true=0
    s = _ifs_cape_closure_scale(floor_f, jnp.array([0.0]), launch, d, 0.05)
    realized = float(launch[0] * s[0])
    assert abs(realized - float(floor_f[0] * d[0])) < 1e-12 * 1e-3 + 1e-15, (
        f"transition restart {realized:.3e} != convex {float(floor_f[0]*d[0]):.3e}"
    )
    # Floor inactive (M_b_true == launch): exact reduction to the legacy blend.
    mb = jnp.array([0.02])
    tgt = jnp.array([0.04])
    s_new = _ifs_cape_closure_scale(tgt, mb, mb, d, 0.05)
    s_old = _ifs_deep_target_scale(tgt, mb, d, 0.05, divisor_floor=1e-20)
    assert abs(float(s_new[0]) - float(s_old[0])) < 1e-12


def test_ifs_profile_scale_limit_matches_oracle_semantics():
    """ZMFS limiter (cumastrn.F90:913-932): one column-uniform scale, reduced
    so no level of the scaled profile exceeds the cap — an s>1 request on a
    cap-touching profile realizes exactly M_b_max at the peak (codex R10);
    s<1 and dead-profile columns pass through untouched."""
    from legoesm.atmosphere.physics.convection.bechtold import (
        _ifs_profile_scale_limit,
    )
    cap = 0.05
    prof = jnp.array([[0.0, 0.02, cap, 0.01],       # touches the cap
                      [0.0, 0.01, 0.02, 0.005],     # headroom 2.5x
                      [0.0, 0.0, 0.0, 0.0]])        # dead plume
    s = jnp.array([2.0, 2.0, 2.0])
    out = _ifs_profile_scale_limit(s, prof, cap)
    assert abs(float(out[0]) - 1.0) < 1e-12          # limited: peak at cap
    assert abs(float(out[1]) - 2.0) < 1e-12          # 2 < 2.5 headroom: kept
    assert abs(float(out[2]) - 2.0) < 1e-12          # dead profile: no-op
    peak_realized = float(jnp.max(prof[0] * out[0]))
    assert abs(peak_realized - cap) < 1e-12
    s_small = jnp.array([0.5, 0.5, 0.5])
    out_small = _ifs_profile_scale_limit(s_small, prof, cap)
    assert jnp.allclose(out_small, s_small)          # s<1 never touched


# ---------------------------------------------------------------------------
# IFS Kessler sub-cloud rain evaporation (cuflxn.F90:436-475)
# ---------------------------------------------------------------------------

from legoesm.atmosphere.physics.convection.bechtold import (  # noqa: E402
    _ifs_subcloud_rain_evaporation,
    _IFS_RCPECONS,
    _IFS_EVAP_EXPONENT,
    _IFS_RCVRFACTOR,
    _IFS_RCUCOV,
    _IFS_RCUCOV_DEEP_FACTOR,
    _IFS_RHEBC_OCEAN,
    _IFS_RHEBC_OCEAN_DEEP,
)


def test_ifs_subcloud_evap_constants_match_oracle():
    """Constants are the oracle's: RCPECONS=5.44e-4/g (sucumf.F90:176),
    exponent 0.5777 (cuflxn.F90:453), RCVRFACTOR=5.09e-3 (sucumf.F90:177),
    RCUCOV=0.05 (sucumf.F90:175) with the 0.6 deep area factor
    (cuflxn.F90:442), RHEBC ocean 0.92 / deep-ocean 0.85
    (sucumf.F90:179, cuflxn.F90:226)."""
    assert abs(_IFS_RCPECONS - 5.44e-4 / constants.g) < 1e-18
    assert _IFS_EVAP_EXPONENT == 0.5777
    assert _IFS_RCVRFACTOR == 5.09e-3
    assert _IFS_RCUCOV == 0.05
    assert _IFS_RCUCOV_DEEP_FACTOR == 0.6
    assert _IFS_RHEBC_OCEAN == 0.92
    assert _IFS_RHEBC_OCEAN_DEEP == 0.85


def test_ifs_subcloud_evap_analytic_fortran_mirror():
    """Hand-computed Fortran-shaped downward recurrence (cuflxn.F90:449-460)
    on a 4-level column reproduces the helper: rain forms in the upper two
    (cloud) layers, evaporates in the dry sub-cloud layers below; a sign,
    orientation, area-blend or RH-break bug goes red."""
    import numpy as np
    g = constants.g
    dt = 600.0
    q = np.array([[1e-4, 2e-3, 4e-3, 6e-3]])          # surface-last, dry BL
    qsat = np.array([[5e-4, 4e-3, 1.2e-2, 1.6e-2]])   # sub-cloud RH = 1/3, 3/8
    p_half = np.array([[200e2, 400e2, 620e2, 830e2, 1000e2]])
    dp = p_half[:, 1:] - p_half[:, :-1]
    dq_r = np.array([[2e-7, 3e-7, 0.0, 0.0]])          # rain source aloft
    below = np.array([[0.0, 0.0, 1.0, 1.0]])           # crisp sub-cloud gate
    # ZRHM = 0.85 > 0.8 so the non-deep case exercises a NONZERO RCUCOV RH
    # enhancement, factor 1 + 0.05/0.025 = 3 (codex R2: 0.8 exactly zeroed it).
    rh_b, rh_t = np.array([0.95]), np.array([0.75])
    dw = np.array([1.0])                                # fully deep

    def mirror(q_np, qsat_np, dq_r_np, below_np, deep: bool):
        # Oracle ordering: evap acts on the flux entering the layer TOP; the
        # layer's own source joins the flux downstream (cuflxn.F90:449,470).
        if deep:
            area = _IFS_RCUCOV * _IFS_RCUCOV_DEEP_FACTOR
            rhebc = _IFS_RHEBC_OCEAN_DEEP
        else:
            zrhm = 0.5 * (rh_b[0] + rh_t[0])
            area = _IFS_RCUCOV * (1.0 + (max(0.8, zrhm) - 0.8) / 0.025)
            rhebc = _IFS_RHEBC_OCEAN
        zcons2 = 1.0 / (g * dt)
        flux = 0.0
        evap_exp = np.zeros(4)
        for k in range(4):                              # downward (surface-last)
            zrfl = flux
            if zrfl > 1e-12:
                zdrfl1 = (
                    _IFS_RCPECONS * max(0.0, qsat_np[0, k] - q_np[0, k]) * area
                    * (np.sqrt(p_half[0, k] / p_half[0, -1]) / _IFS_RCVRFACTOR
                       * zrfl / area) ** _IFS_EVAP_EXPONENT
                    * dp[0, k]
                )
                zrmin = zrfl - area * max(0.0, rhebc * qsat_np[0, k] - q_np[0, k]) \
                    * zcons2 * dp[0, k]
                zrfln = max(max(zrfl - zdrfl1, zrmin), 0.0)
                evap_exp[k] = (zrfl - zrfln) * below_np[0, k]
            flux = zrfl - evap_exp[k] + max(dq_r_np[0, k], 0.0) * dp[0, k] / g
        return evap_exp

    for deep in (True, False):                          # both area branches
        dw_case = np.array([1.0 if deep else 0.0])
        evap_exp = mirror(q, qsat, dq_r, below, deep)
        evap_rate, rain_scale = _ifs_subcloud_rain_evaporation(
            jnp.asarray(q), jnp.asarray(qsat), jnp.asarray(p_half),
            jnp.asarray(dp), jnp.asarray(dq_r), jnp.asarray(below),
            jnp.asarray(rh_b), jnp.asarray(rh_t), jnp.asarray(dw_case), dt,
        )
        got_evap = np.asarray(evap_rate) * dp / g       # back to kg/m2/s
        # tol keyed on the JAX compute dtype (np promotion above would
        # always report float64 even when the scan ran in fp32).
        tol = 1e-9 if evap_rate.dtype == jnp.float64 else 1e-5
        assert np.allclose(got_evap[0], evap_exp, rtol=tol, atol=1e-18), (
            f"deep={deep}: evap {got_evap[0]} != hand-computed {evap_exp}"
        )
        assert float(evap_exp.sum()) > 0.0, "fixture must actually evaporate"
        rain_total = float((np.maximum(dq_r, 0.0) * dp / g).sum())
        scale_exp = 1.0 - evap_exp.sum() / rain_total
        assert abs(float(rain_scale[0, 0]) - scale_exp) < tol


def test_ifs_subcloud_evap_rh_break_and_conservation():
    """A sub-cloud layer already wetter than the RH break evaporates ~nothing;
    a dry layer evaporates; per-level evap never exceeds the through-flux
    (fluxes stay >= 0) and the debit scale lands in [0, 1] with column water
    closing exactly: sum(evap) == (1 - scale) * rain_total."""
    dt = 600.0
    ncol, nlev = 1, 5
    p_half = jnp.linspace(150e2, 1000e2, nlev + 1)[None, :]
    dp = p_half[:, 1:] - p_half[:, :-1]
    qsat = jnp.full((ncol, nlev), 1e-2)
    dq_r = jnp.zeros((ncol, nlev)).at[:, 1].set(5e-7)
    below = jnp.zeros((ncol, nlev)).at[:, 3:].set(1.0)
    rh_b, rh_t = jnp.array([0.9]), jnp.array([0.7])
    dw = jnp.array([0.0])                                # non-deep branch

    def run(q):
        return _ifs_subcloud_rain_evaporation(
            q, qsat, p_half, dp, dq_r, below, rh_b, rh_t, dw, dt)

    # wetter than the 0.92 break everywhere below cloud: ~no evaporation.
    e_wet, s_wet = run(qsat * 0.95)
    assert float(jnp.sum(e_wet)) == 0.0
    assert abs(float(s_wet[0, 0]) - 1.0) < 1e-12
    # dry sub-cloud: evaporates, scale in [0,1), exact closure.
    q_dry = qsat * 0.3
    e_dry, s_dry = run(q_dry)
    evap_total = float(jnp.sum(e_dry * dp / constants.g))
    rain_total = float(jnp.sum(jnp.maximum(dq_r, 0.0) * dp / constants.g))
    assert evap_total > 0.0
    assert 0.0 <= float(s_dry[0, 0]) < 1.0
    assert abs(evap_total - (1.0 - float(s_dry[0, 0])) * rain_total) <= (
        1e-12 * rain_total + 1e-30
    )
    assert evap_total <= rain_total * (1.0 + 1e-9)


def test_ifs_subcloud_evap_toggle_and_leaf_integration():
    """Default ON (2026-07-16 flip); ON re-evaporates sub-cloud rain on a convecting column
    with a dry boundary layer (vapor added below the LCL, surface-reaching
    rain reduced, column vapor gain == rain debit) and stays finite."""
    assert BechtoldConfig().use_ifs_subcloud_evap is True
    # Proven convecting fixture (same as the closure toggle tests); the
    # analytic sub-cloud RH ~0.4 sits far below the 0.85/0.92 break, so the
    # Kessler evap fires whenever rain exists.  Legacy downdraft evap is
    # disabled in BOTH runs (downdraft_evap_efficiency=0) so the toggle
    # isolates the IFS path alone.
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=299.0, q_sfc=13e-3,
                                 lapse_rate=6.5)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.zeros((ncol,))
    dp = ph[:, 1:] - ph[:, :-1]
    common = dict(downdraft_evap_efficiency=0.0)
    out_on, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_ifs_subcloud_evap=True, **common))
    out_off, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_ifs_subcloud_evap=False, **common))
    for o in (out_on, out_off):
        assert jnp.all(jnp.isfinite(o.dT_dt))
        assert jnp.all(jnp.isfinite(o.dq_v_dt))
        assert o.dq_r_conv_dt is not None
        assert float(jnp.min(o.dq_r_conv_dt)) >= 0.0
    rain_on = float(jnp.sum(out_on.dq_r_conv_dt * dp / constants.g))
    rain_off = float(jnp.sum(out_off.dq_r_conv_dt * dp / constants.g))
    assert rain_off > 0.0, "fixture must rain (else the test is vacuous)"
    assert rain_on < rain_off, "IFS evap must reduce surface-reaching rain"
    # Column water closure: the vapor the evap adds equals the rain debit.
    # Relative tolerance keyed on the compute dtype (fp32 accumulates ~1e-7
    # relative over the two 40-level column sums — codex R3).
    rtol = 1e-9 if out_on.dq_v_dt.dtype == jnp.float64 else 3e-5
    dv_on = float(jnp.sum((out_on.dq_v_dt - out_off.dq_v_dt) * dp / constants.g))
    assert abs(dv_on - (rain_off - rain_on)) < 1e-12 + rtol * abs(rain_off), (
        "column vapor gain != rain debit (water leak in the evap block)"
    )
    # Evaporative cooling accompanies the moistening (L_v/c_p ratio).
    dh_on = float(jnp.sum((out_on.dT_dt - out_off.dT_dt) * dp / constants.g))
    assert dh_on < 0.0
    assert abs(dh_on * constants.c_pd + dv_on * constants.L_v) < (
        rtol * abs(dv_on * constants.L_v) + 1e-12
    )


def test_ifs_subcloud_evap_no_rain_noop_and_grad():
    """precip_efficiency=0 + constant split => no rain to evaporate: the flag
    is a no-op (no crash on dq_r=None); jax.grad stays finite through the
    evap scan on a raining column (the **0.5777 zero-flux guard)."""
    T, q, pf, ph, u, v = _column(ncol=1, nlev=24, T_sfc=300.0, q_sfc=10e-3)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.zeros((ncol,))
    out, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_ifs_subcloud_evap=True,
                              use_ifs_inplume_precip=False,
                              precip_efficiency=0.0))
    assert out.dq_r_conv_dt is None
    assert jnp.all(jnp.isfinite(out.dT_dt))

    if jnp.asarray(0.0).dtype != jnp.float64:
        import pytest
        pytest.skip("full-scheme grad NaNs under fp32 on main irrespective of "
                    "this flag (pre-existing); grad coverage runs under x64 "
                    "like the file's other grad tests")

    def loss(eps_deep):
        cfg = BechtoldConfig(epsilon_deep=eps_deep, use_ifs_subcloud_evap=True)
        o, _, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, st, None,
                                      dt=600.0, config=cfg)
        return jnp.sum(o.dq_v_dt ** 2)

    assert jnp.isfinite(jax.grad(loss)(1.75e-3))


def test_ifs_subcloud_evap_fractional_gate_is_convex_blend():
    """A fractional below-LCL membership g realizes the CONVEX BLEND of the
    two oracle branches per level — flux_out = (1-g)*zrfl + g*zrfln, vapor
    deposit g*(zrfl - zrfln) — so g=0.5 evaporation is exactly half the
    hard-gate evaporation of the same single active layer (codex R1 #2)."""
    dt = 600.0
    ncol, nlev = 1, 3
    p_half = jnp.linspace(400e2, 1000e2, nlev + 1)[None, :]
    dp = p_half[:, 1:] - p_half[:, :-1]
    qsat = jnp.full((ncol, nlev), 1e-2)
    q = qsat * 0.3
    dq_r = jnp.zeros((ncol, nlev)).at[:, 0].set(4e-7)   # source in the top layer
    rh_b, rh_t = jnp.array([0.9]), jnp.array([0.7])
    dw = jnp.array([1.0])

    def evap_with_gate(g_mid):
        below = jnp.asarray([[0.0, g_mid, 0.0]])        # only the middle layer
        e, _ = _ifs_subcloud_rain_evaporation(
            q, qsat, p_half, dp, dq_r, below, rh_b, rh_t, dw, dt)
        return float(e[0, 1] * dp[0, 1] / constants.g)

    e_full = evap_with_gate(1.0)
    e_half = evap_with_gate(0.5)
    assert e_full > 0.0
    assert abs(e_half - 0.5 * e_full) < 1e-12 + 1e-9 * e_full


# ---------------------------------------------------------------------------
# IFS in-updraft precipitation formation (cuascn.F90:718-773)
# ---------------------------------------------------------------------------

from legoesm.atmosphere.physics.convection.bechtold import (  # noqa: E402
    _ifs_inplume_precip_conversion,
    _ifs_updraft_ke_profile,
    _ifs_liquid_fraction_cu,
    _IFS_RPRCON,
    _IFS_ZDNOPRC,
    _IFS_Z_CLDMAX,
    _IFS_Z_CPRC2,
    _IFS_RTBERCU_OFFSET_K,
    _IFS_RTICECU_OFFSET_K,
)


def test_ifs_inplume_constants_and_liquid_fraction():
    """Constants are the oracle's (sucumf.F90:164, cuascn.F90:277-280,
    suphec.F90:200-202); FOEALFCU is 1 above freezing, 0 at/below -23C and
    the quadratic in between (fcttre.func.h:130)."""
    assert _IFS_RPRCON == 1.4e-3
    assert _IFS_ZDNOPRC == 3.0e-4
    assert _IFS_Z_CLDMAX == 5.0e-3
    assert _IFS_Z_CPRC2 == 0.5
    assert _IFS_RTBERCU_OFFSET_K == 5.0
    assert _IFS_RTICECU_OFFSET_K == 23.0
    tf = constants.T_freeze
    assert float(_ifs_liquid_fraction_cu(jnp.array(tf + 5.0))) == 1.0
    assert float(_ifs_liquid_fraction_cu(jnp.array(tf - 23.0))) == 0.0
    assert float(_ifs_liquid_fraction_cu(jnp.array(tf - 40.0))) == 0.0
    mid = float(_ifs_liquid_fraction_cu(jnp.array(tf - 11.5)))
    assert abs(mid - 0.25) < 1e-12          # ((23-11.5)/23)^2 = 0.25


def test_ifs_inplume_conversion_analytic_fortran_mirror():
    """Hand-computed Fortran-shaped upward recurrence (cuascn.F90:721-773) on
    a 4-level ascent reproduces the helper: dilution + fresh condensation
    recovered from the plume outputs, ZWU from the level-below KE, Bergeron
    factor, ZDNOPRC threshold IF, Z_CLDMAX clip — a sign/orientation/factor
    bug goes red."""
    import numpy as np
    g = constants.g
    tf = constants.T_freeze
    # surface-last inputs (index 0 = top).
    q_c = np.array([[1.2e-3, 2.0e-3, 8.0e-4, 0.0]])
    T_u = np.array([[tf - 15.0, tf - 2.0, tf + 8.0, tf + 16.0]])
    z = np.array([[9000.0, 6000.0, 3000.0, 500.0]])
    eps = np.array([[2e-4, 4e-4, 8e-4, 1e-3]])
    ke = np.array([[4.0, 6.0, 2.0, 0.3]])

    # upward order (surface-first): reverse.
    q_sf, T_sf, z_sf = q_c[0, ::-1], T_u[0, ::-1], z[0, ::-1]
    eps_sf, ke_sf = eps[0, ::-1], ke[0, ::-1]
    L_prev = 0.0
    L_exp, P_exp = np.zeros(4), np.zeros(4)
    for k in range(4):
        dz = max(z_sf[k] - z_sf[k - 1], 1.0) if k > 0 else 0.0
        decay = np.exp(-eps_sf[k] * dz)
        q_prev = q_sf[k - 1] if k > 0 else 0.0
        cond = max(q_sf[k] - q_prev * decay, 0.0)
        L_pre = L_prev * decay + cond
        ke_below = ke_sf[k - 1] if k > 0 else ke_sf[k]
        zwu = min(15.0, np.sqrt(2.0 * max(0.5, ke_below)))
        alpha = min(1.0, ((min(max(T_sf[k], tf - 23.0), tf) - (tf - 23.0)) / 23.0) ** 2)
        zdt = min(23.0 - 5.0, max((tf - 5.0) - T_sf[k], 0.0))
        zcbf = 1.0 + 0.5 * np.sqrt(zdt)
        zlcrit = 3.0e-4 / zcbf
        zzco = (1.4e-3 / g) / (0.75 * zwu) * (1.0 + 0.3 * alpha) * zcbf
        # Oracle mapping: ZLUOLD = diluted condensate (cuascn.F90:543),
        # ZC = fresh condensation alone (cuascn.F90:761); launch level
        # (dz=0) never converts (ascent loop starts above departure).
        L_old = L_prev * decay
        if L_pre > 3.0e-4 and dz > 0.0:
            zc = cond
            zd = zzco * (1.0 - np.exp(-((L_pre / zlcrit) ** 2))) * g * dz
            zint = np.exp(-zd)
            src = zc / zd * (1.0 - zint) if zd > 1e-8 else zc * (1.0 - 0.5 * zd)
            L_new = min(max(L_old * zint + src, 0.0), min(L_pre, 5.0e-3))
        else:
            L_new = L_pre
        P_exp[k] = max(L_pre - L_new, 0.0)
        L_exp[k] = L_new
        L_prev = L_new

    L_got, P_got = _ifs_inplume_precip_conversion(
        jnp.asarray(q_c), jnp.asarray(T_u), jnp.asarray(z),
        jnp.asarray(eps), jnp.asarray(ke),
    )
    tol = 1e-9 if L_got.dtype == jnp.float64 else 1e-5
    assert np.allclose(np.asarray(L_got)[0, ::-1], L_exp, rtol=tol, atol=1e-18)
    assert np.allclose(np.asarray(P_got)[0, ::-1], P_exp, rtol=tol, atol=1e-18)
    assert P_exp.sum() > 0.0, "fixture must actually convert"
    # invariants: converted L never exceeds the pre-conversion condensate and
    # never the 5e-3 cap; precip non-negative.
    assert np.all(np.asarray(L_got) <= np.maximum(np.asarray(q_c), 5.0e-3) + 1e-15)
    assert np.all(np.asarray(P_got) >= 0.0)


def test_ifs_inplume_toggle_and_leaf_integration():
    """Default ON (2026-07-16 flip); ON produces rain from the
    in-plume formation (dq_r>0 without any precip split), reduces the
    detrained anvil condensate, stays finite, and a stable column stays
    quiescent."""
    assert BechtoldConfig().use_ifs_inplume_precip is True
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=299.0, q_sfc=13e-3,
                                 lapse_rate=6.5)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.zeros((ncol,))
    dp = ph[:, 1:] - ph[:, :-1]
    # precip_efficiency=0: the legacy path would emit NO rain (dq_r None), so
    # any rain under the flag comes from the in-plume formation alone.
    out_on, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_ifs_inplume_precip=True,
                              use_ifs_subcloud_evap=False,
                              precip_efficiency=0.0))
    out_off, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_ifs_inplume_precip=False,
                              use_ifs_subcloud_evap=False,
                              precip_efficiency=0.0))
    assert out_off.dq_r_conv_dt is None
    assert out_on.dq_r_conv_dt is not None
    assert jnp.all(jnp.isfinite(out_on.dq_r_conv_dt))
    assert float(jnp.min(out_on.dq_r_conv_dt)) >= 0.0
    rain_on = float(jnp.sum(out_on.dq_r_conv_dt * dp / constants.g))
    assert rain_on > 0.0, "in-plume formation must rain on a deep column"
    # anvil source shrinks (condensate converted out before detrainment).
    anvil_on = float(jnp.sum(out_on.dq_c_conv_dt * dp / constants.g))
    anvil_off = float(jnp.sum(out_off.dq_c_conv_dt * dp / constants.g))
    assert anvil_on < anvil_off
    for o in (out_on, out_off):
        assert jnp.all(jnp.isfinite(o.dT_dt))
        assert jnp.all(jnp.isfinite(o.dq_v_dt))
    # stable column quiesces under the flag (<1 W/m^2 bar, as elsewhere).
    Ts, qs, pfs, phs, us, vs = _column(ncol=2, nlev=40, T_sfc=280.0,
                                       q_sfc=2e-3, lapse_rate=3.0)
    out_s, _, _ = bechtold_convection(
        Ts, qs, pfs, phs, us, vs, jnp.zeros((2, 40)), jnp.zeros((2,)),
        None, dt=600.0, config=BechtoldConfig(use_ifs_inplume_precip=True))
    w_m2 = float(jnp.max(jnp.abs(out_s.dT_dt)) * constants.c_pd * 1e5 / constants.g)
    assert w_m2 < 1.0


def test_ifs_inplume_chain_with_subcloud_evap_and_grad():
    """The formed rain feeds the sub-cloud evaporation (cuascn->cuflxn chain:
    evap reduces surface-reaching rain vs evap-off, both flags on) and
    jax.grad stays finite through the conversion + KE scans."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=299.0, q_sfc=13e-3,
                                 lapse_rate=6.5)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.zeros((ncol,))
    dp = ph[:, 1:] - ph[:, :-1]
    kw = dict(use_ifs_inplume_precip=True, precip_efficiency=0.0,
              downdraft_evap_efficiency=0.0)
    r_evap = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_ifs_subcloud_evap=True, **kw))[0]
    r_noev = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_ifs_subcloud_evap=False, **kw))[0]
    rain_evap = float(jnp.sum(r_evap.dq_r_conv_dt * dp / constants.g))
    rain_noev = float(jnp.sum(r_noev.dq_r_conv_dt * dp / constants.g))
    # COMPLETE sub-cloud evaporation is permitted by the oracle-style flux
    # limiter (a dry-enough BL under the RH break can consume all the rain
    # — fp32 rounding lands there on this fixture), so the lower bound is
    # inclusive (codex R2).
    assert 0.0 <= rain_evap < rain_noev
    assert rain_noev > 0.0

    if jnp.asarray(0.0).dtype != jnp.float64:
        import pytest
        pytest.skip("full-scheme grad NaNs under fp32 on main (pre-existing); "
                    "grad coverage runs under x64")

    def loss(eps_deep):
        cfg = BechtoldConfig(epsilon_deep=eps_deep,
                             use_ifs_inplume_precip=True)
        o, _, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, st, None,
                                      dt=600.0, config=cfg)
        return jnp.sum(o.dq_v_dt ** 2) + jnp.sum(o.dq_r_conv_dt ** 2)

    assert jnp.isfinite(jax.grad(loss)(1.75e-3))


def test_ifs_inplume_dilution_and_launch_edge_cases():
    """Codex R1 #1/#3 regressions: (a) a strongly-entraining layer with NO
    fresh condensation must not spuriously convert (dilution is not a source
    — ZC is condensation only, so with cond=0 the analytic solution decays
    from the DILUTED state and precip only reflects the conversion sink, not
    the dilution); (b) a supersaturated LAUNCH level (dz=0) sheds no rain."""
    import numpy as np
    tf = constants.T_freeze
    # 3 levels surface-first geometry via surface-last arrays.
    z = jnp.asarray([[7000.0, 3500.0, 500.0]])
    T_u = jnp.full((1, 3), tf + 10.0)
    ke = jnp.full((1, 3), 2.0)
    # (a) plume condensate DECAYS upward exactly as pure dilution would:
    # q_c(k) = q_c(below)*exp(-eps*dz) => recovered cond = 0 everywhere.
    eps = jnp.full((1, 3), 2.5e-3)                    # strong entrainment
    L0 = 2.0e-3
    d1 = float(jnp.exp(-eps[0, 1] * (z[0, 1] - z[0, 2])))
    d2 = float(jnp.exp(-eps[0, 0] * (z[0, 0] - z[0, 1])))
    q_c = jnp.asarray([[L0 * d1 * d2, L0 * d1, L0]])
    L_new, precip = _ifs_inplume_precip_conversion(q_c, T_u, z, eps, ke)
    # zero fresh condensation => the analytic source term vanishes; the
    # converted L is the diluted state damped by the (small) sink only, and
    # the total precip must be FAR below the dilution-driven spurious value
    # (which converted nearly all of L0 under the old ZC = L_pre - L_prev).
    assert float(jnp.sum(precip)) < 0.5 * L0
    assert jnp.all(L_new <= q_c + 1e-15)
    # (b) launch level above Z_CLDMAX: no zero-path-length rain.
    q_c_launch = jnp.asarray([[0.0, 0.0, 8.0e-3]])    # > 5e-3 at launch
    L2, P2 = _ifs_inplume_precip_conversion(
        q_c_launch, T_u, z, jnp.full((1, 3), 1e-4), ke)
    assert float(P2[0, 2]) == 0.0, "launch level must not convert (dz=0)"


def test_ifs_inplume_rain_water_budget_coupled():
    """Codex R1 #2 regression: the in-plume rain carries a COLUMN-exact
    vapor sink (distributed by moisture mass q_v*dp — a per-level debit at
    the formation level overdrew dry upper levels into negative q_v in the
    100-day RCE gate), so switching the flag on changes the column total
    water sum(dq_v + dq_c + dq_r) by EXACTLY zero relative to the flag-off
    run (evap off isolates the pairing)."""
    T, q, pf, ph, u, v = _column(ncol=2, nlev=40, T_sfc=299.0, q_sfc=13e-3,
                                 lapse_rate=6.5)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    st = jnp.zeros((ncol,))
    dp = ph[:, 1:] - ph[:, :-1]

    # Closure OFF in both runs: the default CAPE closure would rescale M_u
    # via the CONVERTED condensate loading (a real feedback, but a CONFOUND
    # here) — with a fixed M_u the level-wise pairings make the total-water
    # sum EXACTLY flag-invariant (dq_c and dq_r are both vapor-sink paired,
    # transport identical).
    # implicit_flux kernel: the budget-closed solve whose dq_c carries the
    # vapor-sink pairing (the default advective kernel's documented
    # non-closure would mask the rain pairing under the L-dependent dq_c
    # change).
    common = dict(use_ifs_subcloud_evap=False, enable_downdraft=False,
                  precip_efficiency=0.0, use_ifs_cape_closure=False,
                  use_convective_turnover_tau=False,
                  subsidence_solve="implicit_flux")

    def run(flag):
        return bechtold_convection(
            T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
            config=BechtoldConfig(use_ifs_inplume_precip=flag, **common))[0]

    o_on, o_off = run(True), run(False)

    def total_water(o):
        dqr = o.dq_r_conv_dt if o.dq_r_conv_dt is not None else 0.0
        return float(jnp.sum((o.dq_v_dt + o.dq_c_conv_dt + dqr) * dp
                             / constants.g))

    rain_total = float(jnp.sum(o_on.dq_r_conv_dt * dp / constants.g))
    assert rain_total > 0.0
    shift = abs(total_water(o_on) - total_water(o_off))
    assert shift < 1e-9 * rain_total + 1e-18, (
        f"budget shift {shift:.3e} vs rain {rain_total:.3e} — "
        "vapor-sink pairing broken"
    )


def test_bechtold_downdraft_entrain_rate_default_is_ifs_entrdd():
    """The penetrative-downdraft entrainment default is the oracle ENTRDD =
    3.0e-4 1/m (sucumf.F90:144), not the earlier unsourced 5.0e-4."""
    assert BechtoldConfig().downdraft_entrain_rate == 3.0e-4


def test_ifs_ztaures_matches_oracle_piecewise():
    """ZTAURES (cumastrn.F90:713,762-768): 0 disables; dx floored at 100 m;
    fine branch 1+ln(8km/dx)^2 below 8 km; coarse 1+1.6*dx/125km above,
    capped at 3 beyond 125 km — pinned at the oracle's own branch points,
    and the turnover tau actually consumes it (integration toggle)."""
    import math
    from legoesm.atmosphere.physics.convection.bechtold import _ifs_ztaures
    assert _ifs_ztaures(0.0) == 1.0                          # legacy sentinel
    assert _ifs_ztaures(-5.0) == 1.0
    assert _ifs_ztaures(50.0) == _ifs_ztaures(100.0)         # 100 m floor
    assert abs(_ifs_ztaures(4.0e3) - (1 + math.log(2.0) ** 2)) < 1e-12
    assert abs(_ifs_ztaures(8.0e3) - (1 + 1.6 * 8e3 / 125e3)) < 1e-12
    assert abs(_ifs_ztaures(50.0e3) - (1 + 1.6 * 50e3 / 125e3)) < 1e-12
    assert _ifs_ztaures(200.0e3) == 3.0                      # coarse cap
    # integration: dx changes the mass flux on a convecting column.  Probed
    # on the tau-only turnover path — under the full CAPE closure this
    # fixture's mb_scale is pinned by the column-uniform profile limiter
    # (peak M_u >> M_b_max) in BOTH runs, masking tau; the turnover rescale
    # tau_bl/tau_conv consumes tau directly.
    T, q, pf, ph, u, v = _column(ncol=1, nlev=40, T_sfc=299.0, q_sfc=13e-3,
                                 lapse_rate=6.5)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev)); st = jnp.zeros((ncol,))
    # This fixture's raw tau is ~213 s, so the coarse x3 factor (640 s) still
    # lands under the oracle's 720 s clamp floor — faithfully equalized.  The
    # fine-branch dx=500 m gives factor 1+ln(16)^2 ~ 8.7 (tau 1854 s), which
    # clears the floor and must show up in the mass flux.
    kw = dict(M_b_max=1.0, use_ifs_cape_closure=False,
              use_convective_turnover_tau=True)
    mu0 = bechtold_convection(T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
                              config=BechtoldConfig(**kw))[1]
    mu_fine = bechtold_convection(T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
                                  config=BechtoldConfig(dx_m=500.0, **kw))[1]
    assert float(jnp.max(jnp.abs(mu0 - mu_fine))) > 0.0, "dx_m must be live"
    assert float(jnp.sum(mu_fine)) < float(jnp.sum(mu0)), (
        "longer turnover => weaker deep flux"
    )
