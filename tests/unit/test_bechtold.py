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
# Parcel theta cap (#929 polar-night deeper harden)
# ---------------------------------------------------------------------------

def _profile_column(profile, ncol=2, nlev=16, p_s=1.0e5, p_top=5.0e3,
                    q_sfc=4e-4):
    """Column with an arbitrary T(z) profile (z from the 8.5-km scale height)."""
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), inner, jnp.full((ncol, 1), p_s)],
        axis=1,
    )
    z = -8500.0 * jnp.log(p_full / p_s)
    T = profile(z)
    q = q_sfc * jnp.exp(-z / 3000.0)
    return T, q, p_full, p_half


def test_bechtold_parcel_theta_cap_quiesces_polar_inversion():
    """#929 deeper harden: a polar-night surface-inversion column must be
    QUIESCENT.  Without the cap, the theta-warmer PBL-mean parcel manufactures
    O(1000 J/kg) CAPE and tens of K/day heating in a column where no BL air
    can convect (the mid-Feb ~71N latlon24 runaway); with the cap the parcel
    collapses to the surface parcel (the coldest air) and CAPE is exactly 0."""
    # T rises 18 K over the lowest 800 m (strong polar inversion), weak
    # stable lapse aloft; nearly dry.
    inv = lambda z: (245.0 + 18.0 * jnp.minimum(z, 800.0) / 800.0
                     - 6.5e-3 * jnp.maximum(z - 800.0, 0.0))
    T, q, pf, ph = _profile_column(inv, q_sfc=4e-4)
    zeros = jnp.zeros_like(T)
    stoch = jnp.zeros((T.shape[0],))

    out_on, _, _ = bechtold_convection(
        T, q, pf, ph, zeros, zeros, zeros, stoch, None, dt=600.0,
        config=BechtoldConfig(parcel_theta_cap=True),
    )
    assert float(out_on.cape.max()) < 1.0
    assert float(jnp.abs(out_on.dT_dt).max()) * 86400.0 < 1e-3   # K/day
    assert float(out_on.convective_mask.max()) < 5e-3            # gate floor

    # documents the leak the cap removes (empirical: cape ~1683 J/kg,
    # heating ~39 K/day, mask 1.0 on this fixture)
    out_off, _, _ = bechtold_convection(
        T, q, pf, ph, zeros, zeros, zeros, stoch, None, dt=600.0,
        config=BechtoldConfig(parcel_theta_cap=False),
    )
    assert float(out_off.cape.max()) > 100.0
    assert float(out_off.convective_mask.max()) > 0.9


def test_bechtold_parcel_theta_cap_inert_in_well_mixed_bl():
    """The cap must not disturb genuinely convecting columns: in a well-mixed
    (dry-adiabatic) BL the PBL-mean theta equals the surface theta, so the cap
    is inert to within the parcel perturbation (empirical: CAPE differs ~1%,
    heating within ~0.05 K/day on this fixture)."""
    wm = lambda z: (300.0 - 9.8e-3 * jnp.minimum(z, 600.0)
                    - 7.5e-3 * jnp.maximum(z - 600.0, 0.0))
    T, q, pf, ph = _profile_column(wm, q_sfc=14e-3)
    zeros = jnp.zeros_like(T)
    stoch = jnp.zeros((T.shape[0],))
    out_on, _, _ = bechtold_convection(
        T, q, pf, ph, zeros, zeros, zeros, stoch, None, dt=600.0,
        config=BechtoldConfig(parcel_theta_cap=True),
    )
    out_off, _, _ = bechtold_convection(
        T, q, pf, ph, zeros, zeros, zeros, stoch, None, dt=600.0,
        config=BechtoldConfig(parcel_theta_cap=False),
    )
    cape_on, cape_off = float(out_on.cape.max()), float(out_off.cape.max())
    assert cape_off > 1000.0                    # the fixture convects
    assert abs(cape_on - cape_off) / cape_off < 0.05
    dheat = float(jnp.abs(out_on.dT_dt - out_off.dT_dt).max()) * 86400.0
    assert dheat < 0.5                          # K/day


def test_bechtold_parcel_theta_cap_noop_for_surface_parcel():
    """With use_pbl_cape=False the parcel IS the surface parcel, so the cap
    must be an exact no-op (bit-identical tendencies)."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    out_a, Mu_a, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(use_pbl_cape=False, parcel_theta_cap=True),
    )
    out_b, Mu_b, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(use_pbl_cape=False, parcel_theta_cap=False),
    )
    assert jnp.array_equal(out_a.dT_dt, out_b.dT_dt)
    assert jnp.array_equal(out_a.dq_v_dt, out_b.dq_v_dt)
    assert jnp.array_equal(Mu_a, Mu_b)


# ---------------------------------------------------------------------------
# CAPE quasi-equilibrium heating ceiling (cape_relaxation_sink; C12 runaway)
# ---------------------------------------------------------------------------

def _lapse_column(T_sfc, lapse_K_km, q_sfc, ncol=1, nlev=16, p_s=1.0e5,
                  p_top=5.0e3):
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), inner, jnp.full((ncol, 1), p_s)],
        axis=1,
    )
    z = -8500.0 * jnp.log(p_full / p_s)
    T = (T_sfc - lapse_K_km * 1e-3 * jnp.minimum(z, 11000.0)
         - 2e-3 * jnp.maximum(z - 11000.0, 0.0))
    q = q_sfc * jnp.exp(-z / 2500.0)
    return T, q, p_full, p_half


def _column_heating_W_m2(out, p_half):
    dp = p_half[:, 1:] - p_half[:, :-1]
    return (constants.c_pd / constants.g) * jnp.sum(
        jnp.maximum(out.dT_dt, 0.0) * dp, axis=-1)


def test_bechtold_cape_sink_inert_on_vigorous_convection():
    """A vigorous tower (large CAPE => large ceiling) must be BIT-identical
    with the sink on: f = clip(big, 0, 1) == 1.0 exactly."""
    T, q, pf, ph = _lapse_column(300.0, 9.8, 14e-3)  # ~18700 J/kg fixture
    z0 = jnp.zeros_like(T)
    st = jnp.zeros((T.shape[0],))
    on, Mu_on, _ = bechtold_convection(
        T, q, pf, ph, z0, z0, z0, st, None, dt=600.0,
        config=BechtoldConfig(cape_relaxation_sink=True))
    off, Mu_off, _ = bechtold_convection(
        T, q, pf, ph, z0, z0, z0, st, None, dt=600.0,
        config=BechtoldConfig(cape_relaxation_sink=False))
    assert float(off.cape.max()) > 5000.0        # genuinely vigorous
    assert jnp.array_equal(on.dT_dt, off.dT_dt)
    assert jnp.array_equal(on.dq_v_dt, off.dq_v_dt)
    assert jnp.array_equal(Mu_on, Mu_off)


def test_bechtold_cape_sink_throttles_runaway_mode():
    """The C12/RCE runaway mode — large sustained heating over MODEST CAPE
    (pilot autopsy: 132 columns at 50-3943 W/m2 with CAPE 51-444 J/kg) —
    must be throttled to the quasi-equilibrium ceiling eff*M_b*CAPE, while
    the uniform rescale preserves the scheme's water bookkeeping."""
    T, q, pf, ph = _lapse_column(296.0, 7.0, 8e-3)   # CAPE ~263, H ~59 W/m2
    z0 = jnp.zeros_like(T)
    st = jnp.zeros((T.shape[0],))
    off, _, _ = bechtold_convection(
        T, q, pf, ph, z0, z0, z0, st, None, dt=600.0,
        config=BechtoldConfig(cape_relaxation_sink=False))
    on, _, _ = bechtold_convection(
        T, q, pf, ph, z0, z0, z0, st, None, dt=600.0,
        config=BechtoldConfig(cape_relaxation_sink=True))
    H_off = float(_column_heating_W_m2(off, ph)[0])
    H_on = float(_column_heating_W_m2(on, ph)[0])
    cape = float(off.cape[0])
    assert 50.0 < cape < 1000.0                  # the modest-CAPE regime
    assert H_off > 2.0 * H_on                    # sink bit hard
    # throttled heating sits AT the ceiling (f<1 => H_on == eff*M_b*CAPE);
    # M_b is internal, so bound the ceiling by its M_b_max upper limit and
    # a generous positive floor instead of reconstructing M_b exactly.
    cfg = BechtoldConfig()
    assert H_on <= cfg.cape_sink_heating_ratio * cfg.M_b_max * cape * 1.001
    assert H_on > 0.0                            # throttled, not silenced
    # uniform rescale: the ON tendencies are an exact scalar multiple of OFF
    ratio = H_on / H_off
    assert jnp.allclose(on.dT_dt, off.dT_dt * ratio, rtol=1e-6, atol=1e-12)
    assert jnp.allclose(on.dq_v_dt, off.dq_v_dt * ratio, rtol=1e-6, atol=1e-15)
    tot_on = on.dq_c_conv_dt + (0.0 if on.dq_r_conv_dt is None else on.dq_r_conv_dt)
    tot_off = off.dq_c_conv_dt + (0.0 if off.dq_r_conv_dt is None else off.dq_r_conv_dt)
    assert jnp.allclose(tot_on, tot_off * ratio, rtol=1e-6, atol=1e-18)


def test_bechtold_cape_sink_heating_ratio_gradient_finite():
    """The ceiling must stay differentiable through the active bound (the
    sink is a calibration knob for the SCM-RCE loop)."""
    T, q, pf, ph = _lapse_column(296.0, 7.0, 8e-3)
    z0 = jnp.zeros_like(T)
    st = jnp.zeros((T.shape[0],))

    def loss(eff):
        out, _, _ = bechtold_convection(
            T, q, pf, ph, z0, z0, z0, st, None, dt=600.0,
            config=BechtoldConfig(cape_sink_heating_ratio=eff))
        return jnp.sum(jnp.abs(out.dT_dt))

    g = jax.grad(loss)(5.0)
    assert jnp.isfinite(g) and float(g) != 0.0


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
    # With the in-updraft rain split ON by default, the detrained water is
    # divided between dq_c and dq_r; the water budget books BOTH.
    _zero = jnp.zeros_like(out_on.dq_c_conv_dt)
    dqr_on = out_on.dq_r_conv_dt if out_on.dq_r_conv_dt is not None else _zero
    dqr_off = out_off.dq_r_conv_dt if out_off.dq_r_conv_dt is not None else _zero
    dqr_diff = dqr_on - dqr_off

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
    above the production ``M_b_max=0.02`` cap on both keys; we set
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
# In-updraft precipitation split (convective precipitation efficiency)
# ---------------------------------------------------------------------------

def test_bechtold_precip_efficiency_rain_split():
    """Bechtold must be able to rain (the SCM-RCE zero-precip runaway).

    Without a rain split, Bechtold detrains 100% of its condensate as
    SUSPENDED cloud water: convective precip is ~0 (SCM-RCE a-priori gate:
    precip 5.4e-7 mm/day vs reference 3.2, equilibrium T runs to 409 K —
    no precipitating heat-removal path), and AMIP shows the June albedo
    0.57 / precip 0.81 signature. Mirror tiedtke's gated in-updraft split:
    ``precip_efficiency`` of the detrained condensate goes to RAIN
    (``dq_r_conv_dt``, sediments via microphysics, invisible to radiation),
    the rest stays anvil cloud water.

    Contracts: pe=0 (default) is BYTE-IDENTICAL legacy (dq_r_conv_dt None);
    pe>0 splits the same total bit-for-bit (dq_c + dq_r == legacy dq_c).
    """
    T, q, pf, ph, u, v = _column(ncol=3, nlev=12)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))

    out0, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(precip_efficiency=0.0),
    )
    assert out0.dq_r_conv_dt is None            # explicit 0 = legacy detrain-all
    assert jnp.any(out0.dq_c_conv_dt > 0)       # convecting fixture detrains

    # the DEFAULT config rains (PE on by default: PE=0 bechtold is unusable —
    # zero convective precip => RCE 409 K runaway / AMIP day-15 blowup)
    out_d, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(),
    )
    assert out_d.dq_r_conv_dt is not None
    assert jnp.any(out_d.dq_r_conv_dt > 0)

    pe = 0.7
    out1, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
        config=BechtoldConfig(precip_efficiency=pe),
    )
    assert out1.dq_r_conv_dt is not None
    assert jnp.array_equal(out1.dq_r_conv_dt, out0.dq_c_conv_dt * pe)
    assert jnp.array_equal(out1.dq_c_conv_dt, out0.dq_c_conv_dt * (1.0 - pe))
    # split conserves the detrained total bit-for-bit
    assert jnp.allclose(
        out1.dq_c_conv_dt + out1.dq_r_conv_dt, out0.dq_c_conv_dt,
        rtol=0, atol=1e-18)
    # heat/vapour tendencies untouched by the split
    assert jnp.array_equal(out1.dT_dt, out0.dT_dt)
    assert jnp.array_equal(out1.dq_v_dt, out0.dq_v_dt)


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
        config=BechtoldConfig(use_convective_turnover_tau=True, M_b_max=1.0))
    off, mu_off, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
        config=BechtoldConfig(use_convective_turnover_tau=False, M_b_max=1.0))
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
        None, dt=600.0, config=BechtoldConfig(use_convective_turnover_tau=True))
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
                             use_convective_turnover_tau=True)
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
                             use_convective_turnover_tau=True)
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
        # cape_relaxation_sink=False isolates the turnover knob: the QE heating
        # ceiling zeroes this fixture's marginal shallow column (M_u ~ 6e-8) in
        # BOTH branches, which would make the on/off delta vacuously 0.
        _, mu_on, _ = bechtold_convection(
            T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
            config=BechtoldConfig(use_convective_turnover_tau=True, M_b_max=10.0,
                                  cape_relaxation_sink=False))
        _, mu_off, _ = bechtold_convection(
            T, q, pf, ph, u, v, cpp, st, None, dt=600.0,
            config=BechtoldConfig(use_convective_turnover_tau=False, M_b_max=10.0,
                                  cape_relaxation_sink=False))
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
        None, dt=600.0, config=BechtoldConfig(use_convective_turnover_tau=True))
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
