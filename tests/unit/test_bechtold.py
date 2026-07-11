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

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.held_suarez import held_suarez_init
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
        config=BechtoldConfig(enable_downdraft=False, enable_stochastic=False),
        moisture_convergence=jnp.zeros_like(T),
    )
    out_on, _, _ = bechtold_convection(
        T=T, q_v=q, p_full=pf, p_half=ph, u=u, v=v,
        conv_prog_profile=cpp, conv_stoch_state=stoch, prng_key=None,
        dt=300.0,
        config=BechtoldConfig(enable_downdraft=True, enable_stochastic=False),
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
        config=BechtoldConfig(
            enable_stochastic=False, enable_cmt=False,
            subsidence_solve="implicit_flux",
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
    out_default, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(enable_downdraft=True),
    )
    out_rh_flat, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(enable_downdraft=True,
                              downdraft_rh_sharpness=1e-6),
    )
    out_lcl_flat, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(enable_downdraft=True,
                              lcl_membership_sharpness=1e-6),
    )
    assert float(jnp.max(jnp.abs(out_rh_flat.dT_dt - out_default.dT_dt))) > 1e-10, (
        "downdraft_rh_sharpness is not wired"
    )
    assert float(jnp.max(jnp.abs(out_lcl_flat.dT_dt - out_default.dT_dt))) > 1e-10, (
        "lcl_membership_sharpness is not wired"
    )


# ---------------------------------------------------------------------------
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
        config=BechtoldConfig(precip_efficiency=pe),
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
