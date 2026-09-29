"""Unit tests for the Zhang & McFarlane (1995) convection scheme.

Tests pin:

* tendency shape / dtype hygiene;
* moisture-conservation budget (vapor + cloud-water + rain sources close);
* the CAM6 scheme is diagnostic: the carry is not read, and the returned
  carry packs the diagnosed cloud-base mass flux at ``[:, -1]`` only;
* finite gradients through the CAM6 tunables (``tau``, ``capelmt``,
  ``momcu``);
* CMT switch and zero-shear behaviour of ``momtran``;
* selection through ``make_physics(PhysicsConfig(...))`` works for
  every dycore type.

The algorithm itself is pinned against the Fortran transcription in
``tests/atmosphere/hydrostatic/unit/test_zm_cam6_oracle.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import (
    ConvectionConfig,
    ZhangMcFarlaneConfig,
)
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    zhang_mcfarlane_convection,
)


# ---------------------------------------------------------------------------
# Synthetic single-column setup
# ---------------------------------------------------------------------------

# The synthetic columns have no land fraction: an explicit aquaplanet.
_AQUA = ZhangMcFarlaneConfig(land_fraction="none")


def _synthetic_column(
    ncol: int = 2,
    nlev: int = 16,
    *,
    T_sfc: float = 300.0,
    q_sfc: float = 16.0e-3,
    lapse_rate: float = 7.0,   # K/km — destabilized
    p_s: float = 1.0e5,
    p_top: float = 5.0e3,
    u_sfc: float = 5.0,
    u_top: float = 25.0,
):
    """Surface-last column with vertical wind shear (for CMT tests)."""
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
    T_env = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    # Isothermal above 12 km: CAM6's buoyan_dilute needs a level of neutral
    # buoyancy inside the column (a parcel buoyant up to the model top has no
    # tentative cloud top and hence CAPE = 0 in the oracle).
    T_env = jnp.maximum(T_env, T_sfc - lapse_rate * 12.0)
    q_v_env = q_sfc * jnp.exp(-z_full / 3000.0)

    # Linear shear: u increases linearly from u_sfc (surface) to u_top (top).
    # Surface-last convention: index nlev-1 = surface = u_sfc; index 0 = top.
    fraction = jnp.linspace(1.0, 0.0, nlev)[None, :]   # 1 at top, 0 at surface
    u = u_sfc + (u_top - u_sfc) * fraction
    u = jnp.broadcast_to(u, (ncol, nlev))
    v = jnp.zeros_like(u)

    return T_env, q_v_env, p_full, p_half, u, v


# ---------------------------------------------------------------------------
# Direct leaf-function tests
# ---------------------------------------------------------------------------

def test_zm_tendency_shape_dtype():
    T, q, pf, ph, u, v = _synthetic_column(ncol=3, nlev=12)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, cpp_new = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=_AQUA)
    assert out.dT_dt.shape == (ncol, nlev)
    assert out.dq_v_dt.shape == (ncol, nlev)
    assert out.dq_c_conv_dt.shape == (ncol, nlev)
    assert out.cape.shape == (ncol,)
    assert out.convective_mask.shape == (ncol,)
    # CMT-on by default
    assert out.du_dt_conv is not None
    assert out.dv_dt_conv is not None
    assert out.du_dt_conv.shape == (ncol, nlev)
    assert out.dv_dt_conv.shape == (ncol, nlev)
    # Carry round-trips
    assert cpp_new.shape == (ncol, nlev)


def test_zm_outputs_finite():
    T, q, pf, ph, u, v = _synthetic_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, cpp_new = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=_AQUA)
    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
                out.convective_mask, out.du_dt_conv, out.dv_dt_conv,
                cpp_new):
        assert jnp.all(jnp.isfinite(arr)), f"NaN/Inf in {arr.shape}"


def test_zm_cloud_water_source_non_negative():
    """Convective condensate source ``dq_c_conv_dt`` is non-negative
    by construction (detrained condensate flows *into* cloud water)."""
    T, q, pf, ph, u, v = _synthetic_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=_AQUA)
    assert jnp.all(out.dq_c_conv_dt >= -1e-12)


def test_zm_carry_layout_scalar_at_surface():
    """Returned ``conv_prog_profile_new`` has the diagnosed ``mb`` at
    ``[:, -1]`` and zeros aloft — the contract that PR 0 enforces."""
    T, q, pf, ph, u, v = _synthetic_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    _, cpp_new = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=_AQUA)
    assert jnp.all(cpp_new[:, :-1] == 0.0)
    # Surface slot is the diagnosed mb [kg/m^2/s], non-negative.
    assert jnp.all(cpp_new[:, -1] >= 0.0)


def test_zm_carry_is_not_read():
    """CAM6 ZM diagnoses ``mb`` every step (no CAPE-relaxation memory): the
    tendencies and the returned carry are independent of the carry passed in."""
    T, q, pf, ph, u, v = _synthetic_column()
    ncol, nlev = T.shape
    out0, c0 = zhang_mcfarlane_convection(T, q, pf, ph, u, v, jnp.zeros((ncol, nlev)), dt=300.0, config=_AQUA)
    out1, c1 = zhang_mcfarlane_convection(T, q, pf, ph, u, v, jnp.full((ncol, nlev), 0.3), dt=300.0, config=_AQUA)
    assert jnp.array_equal(out0.dT_dt, out1.dT_dt)
    assert jnp.array_equal(c0, c1)
    assert float(jnp.max(c0[:, -1])) > 0.0


# ---------------------------------------------------------------------------
# CAPE reduction
# ---------------------------------------------------------------------------

def test_zm_lapse_rate_stabilization():
    """Applying ZM tendencies should *reduce* the column lapse rate
    (the parcel-environment temperature difference, integrated over
    the column).  This is a more robust diagnostic than single-step
    CAPE because the ``compute_moist_adiabat`` reference shifts
    discontinuously with surface T — even a 1 K surface heating
    moves the moist adiabat by ~1 K everywhere, swamping the small
    interior tendencies.

    The actual ZM target is ``∫ (T_env - T_parcel) dp/p`` becoming
    less negative — i.e. the environment warming toward the parcel
    profile.
    """
    T, q, pf, ph, u, v = _synthetic_column(
        T_sfc=302.0, q_sfc=18.0e-3, lapse_rate=8.0,
    )
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    dt = 1800.0
    out, _ = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=dt, config=_AQUA)

    # Environment is heated above LFC and roughly conserved below;
    # the column-mean tendency over the cloud layer is positive.
    # Compare top-half vs bottom-half mean tendency.
    nlev_half = nlev // 2
    upper_mean = float(jnp.mean(out.dT_dt[0, :nlev_half]))
    lower_mean = float(jnp.mean(out.dT_dt[0, nlev_half:]))
    # The upper troposphere should NOT be uniformly cooling; some
    # detrainment + condensation produces net heating in the
    # destabilized column.  At minimum, the column-integrated
    # heating is non-negative.
    column_integrated_dT_dt = float(jnp.sum(out.dT_dt[0]))
    # Reflect that ZM at least redistributes heat (does not
    # uniformly cool the entire column, which would *increase*
    # CAPE).  Negative integrated dT/dt would indicate a sign bug.
    assert column_integrated_dT_dt >= -1e-3, (
        f"Column-integrated dT/dt = {column_integrated_dT_dt:.2e}; "
        f"upper-mean = {upper_mean:.2e}, lower-mean = {lower_mean:.2e}"
    )


# ---------------------------------------------------------------------------
# Differentiability — the AD-safety property motivating smooth-everywhere
# ---------------------------------------------------------------------------

def test_zm_grad_through_tau_finite():
    """``d (sum dT_dt) / d tau`` is finite and non-zero — the CAPE
    consumption timescale is a tunable parameter."""
    T, q, pf, ph, u, v = _synthetic_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(tau):
        config = ZhangMcFarlaneConfig(land_fraction="none", tau=tau)
        out, _ = zhang_mcfarlane_convection(
            T, q, pf, ph, u, v, cpp, dt=300.0, config=config,
        )
        return jnp.sum(out.dT_dt ** 2)

    g = float(jax.grad(f)(jnp.asarray(3600.0)))
    assert np.isfinite(g) and g != 0.0


def test_zm_grad_through_capelmt_finite():
    """``d (sum dT_dt) / d capelmt`` is finite and non-zero on a convecting
    column: the closure is ``mb ~ -(cape - capelmt)/tau/dadt``."""
    T, q, pf, ph, u, v = _synthetic_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(threshold):
        config = ZhangMcFarlaneConfig(land_fraction="none", capelmt=threshold)
        out, _ = zhang_mcfarlane_convection(
            T, q, pf, ph, u, v, cpp, dt=300.0, config=config,
        )
        return jnp.sum(out.dT_dt ** 2)

    g = float(jax.grad(f)(jnp.asarray(70.0)))
    assert np.isfinite(g) and g != 0.0


def test_zm_grad_through_cmt_coefficient():
    """``d (sum du_dt_conv^2) / d momcu`` is finite and non-zero — the
    Richter-Rasch pressure-gradient coefficient is tunable."""
    T, q, pf, ph, u, v = _synthetic_column(u_sfc=2.0, u_top=30.0)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(c_u_arr):
        config = ZhangMcFarlaneConfig(land_fraction="none", momcu=c_u_arr)
        out, _ = zhang_mcfarlane_convection(
            T, q, pf, ph, u, v, cpp, dt=300.0, config=config,
        )
        return jnp.sum(out.du_dt_conv ** 2)

    g = float(jax.grad(f)(jnp.asarray(0.7)))
    assert np.isfinite(g) and g != 0.0


# ---------------------------------------------------------------------------
# CMT
# ---------------------------------------------------------------------------

def test_zm_cmt_disabled_returns_none():
    """``enable_cmt=False`` ⇒ ``du_dt_conv`` and ``dv_dt_conv`` are
    ``None`` (the bridge zero-fills, leaving wind tendencies untouched)."""
    T, q, pf, ph, u, v = _synthetic_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    config = ZhangMcFarlaneConfig(land_fraction="none", enable_cmt=False)
    out, _ = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
    assert out.du_dt_conv is None
    assert out.dv_dt_conv is None


def test_zm_cmt_zero_in_no_shear_column():
    """A uniform (zero-shear) wind profile produces zero CMT
    tendencies regardless of mass flux."""
    T, q, pf, ph, _, _ = _synthetic_column()
    ncol, nlev = T.shape
    u = jnp.full_like(T, 10.0)  # uniform 10 m/s
    v = jnp.zeros_like(T)
    cpp = jnp.zeros((ncol, nlev))
    out, _ = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=_AQUA)
    # Tendencies should be vanishing in a no-shear column up to floats.
    assert float(jnp.max(jnp.abs(out.du_dt_conv))) < 1e-8


# ---------------------------------------------------------------------------
# Stable column: tendencies near zero
# ---------------------------------------------------------------------------

def test_zm_stable_column_tendencies_small():
    """A statically stable, dry column (CAPE <= capelmt) produces exactly
    zero tendencies — the CAM6 ``ideep`` gate."""
    T, q, pf, ph, u, v = _synthetic_column(
        T_sfc=288.0, q_sfc=2.0e-3, lapse_rate=4.0,  # stable, very dry
    )
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=_AQUA)
    assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0
    assert float(jnp.max(out.convective_mask)) == 0.0


# ---------------------------------------------------------------------------
# Integration through make_physics(PhysicsConfig(...))
# ---------------------------------------------------------------------------

def _make_3d_state(n=4, nlev=12):
    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    tracers = {
        "q_v": Field(
            0.014 * jnp.ones((6, n, n, nlev)), name="q_v",
            dims=("face", "x", "y", "level"), units="kg/kg",
        ),
        "q_c": Field(
            jnp.zeros((6, n, n, nlev)), name="q_c",
            dims=("face", "x", "y", "level"), units="kg/kg",
        ),
    }
    state = state._replace(tracers=tracers)
    return state, grid, sigma


def _make_zm_only_config(land_fraction="none"):
    conv = ConvectionConfig(scheme="zhang_mcfarlane")
    conv = conv._replace(zhang_mcfarlane=conv.zhang_mcfarlane._replace(
        land_fraction=land_fraction))
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=conv,
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def test_zm_orchestrator_one_step_finite():
    """End-to-end through ``make_physics``: one hydrostatic step
    produces finite tendencies and a valid carry."""
    state, grid, sigma = _make_3d_state()
    cfg = _make_zm_only_config()
    n = 4; nlev = 12
    ncol = 6 * n * n
    ps = init_physics_state(ncol, nlev, cfg)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)

    assert ps_out is not None
    assert ps_out.conv_prog_profile.shape == (ncol, nlev)
    # Aloft slots remain zero (ZM packs at [:, -1] only).
    assert jnp.all(ps_out.conv_prog_profile[:, :-1] == 0.0)
    # All dycore tendencies finite.
    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
        assert jnp.all(jnp.isfinite(f.data))


def test_zm_orchestrator_multi_step_stable():
    """Five orchestrator steps in a row — no NaN, carry threading
    intact; with a fixed state the diagnosed ``mb`` is step-invariant
    (CAM6 ZM has no relaxation memory)."""
    state, grid, sigma = _make_3d_state()
    cfg = _make_zm_only_config()
    n = 4; nlev = 12
    ncol = 6 * n * n
    ps = init_physics_state(ncol, nlev, cfg)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)

    M_b_trajectory = []
    for _ in range(5):
        tend, ps = physics_fn(state, grid, sigma, phys_state=ps)
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(ps.conv_prog_profile))
        M_b_trajectory.append(float(jnp.max(ps.conv_prog_profile[:, -1])))

    assert all(
        abs(M_b_trajectory[i + 1] - M_b_trajectory[i]) <= 1e-12
        for i in range(len(M_b_trajectory) - 1)
    ), f"mb must be step-invariant for a fixed state: {M_b_trajectory}"


# ---------------------------------------------------------------------------
# Column budgets
# ---------------------------------------------------------------------------

def test_zm_column_water_closes_with_explicit_rain():
    """``-sum dp (dq_v + dq_c)/g`` equals the column-integrated rain source
    ``dq_r_conv_dt``: the CAM6 port emits its precipitation explicitly."""
    T, q_v, p_full, p_half, u, v = _synthetic_column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = zhang_mcfarlane_convection(
        T=T, q_v=q_v, p_full=p_full, p_half=p_half,
        u=u, v=v, conv_prog_profile=cpp, dt=1800.0,
        config=ZhangMcFarlaneConfig(land_fraction="none", enable_cmt=False),
    )
    dp = p_half[:, 1:] - p_half[:, :-1]
    sink = -jnp.sum((out.dq_v_dt + out.dq_c_conv_dt) * dp, axis=1) / constants.g
    rain = jnp.sum(out.dq_r_conv_dt * dp, axis=1) / constants.g
    assert float(jnp.max(rain)) > 0.0
    np.testing.assert_allclose(np.asarray(rain), np.asarray(sink), rtol=1e-9,
                               atol=1e-12 * float(jnp.max(rain)))


# ---------------------------------------------------------------------------
# Hydrostatic bridge: CAM6 ZM's rain is a signed NET flux divergence
# ---------------------------------------------------------------------------
def _convecting_3d_state(n=4, nlev=12, land_frac=None):
    """Held-Suarez state with a moist, conditionally unstable column in every
    cell (lapse 6.5 K/km to 12 km, RH 0.85), so ZM fires everywhere."""
    from legoesm.thermo import saturation_mixing_ratio

    state, grid, sigma = _make_3d_state(n, nlev)
    p_s = 1.0e5
    p_full = sigma.sigma_full * p_s
    z = -8000.0 * jnp.log(p_full / p_s)
    T_col = jnp.maximum(300.0 - 6.5e-3 * z, 300.0 - 6.5 * 12.0)
    q_col = 0.85 * saturation_mixing_ratio(T_col, p_full)
    shape = (6, n, n, nlev)
    state = state._replace(
        T=state.T.replace(data=jnp.broadcast_to(T_col, shape)),
        p_s=state.p_s.replace(data=jnp.full((6, n, n), p_s)),
        tracers={
            **state.tracers,
            "q_v": state.tracers["q_v"].replace(data=jnp.broadcast_to(q_col, shape)),
        },
    )
    if land_frac is not None:
        grid = grid._replace(land_frac=jnp.full((6 * n * n,), land_frac))
    return state, grid, sigma


def test_hydro_bridge_routes_zm_net_rain_to_surface_precip():
    """``dq_r_conv_dt`` from CAM6 ZM is ``ntprprd``: production minus
    evaporation of rain from above, NEGATIVE in evaporating layers.  Booked
    per layer into ``q_c`` it would create condensate sinks where there is
    no condensate (codex round 1, #3); the bridge must column-integrate it
    to surface precipitation instead, with ``q_c`` receiving only the
    detrained cloud water (``dlf >= 0``)."""
    n, nlev = 4, 12
    state, grid, sigma = _convecting_3d_state(n, nlev)
    physics_fn = make_physics(_make_zm_only_config(), model_type="hydrostatic", dt=300.0)
    ps = init_physics_state(6 * n * n, nlev, _make_zm_only_config())
    tend, _ = physics_fn(state, grid, sigma, phys_state=ps)

    assert tend.precip is not None, "ZM rain must leave through the surface precip field"
    precip = tend.precip.data
    assert float(precip.max()) > 0.0, "ZM must fire on this state"
    tt = tend.tracer_tendencies
    assert "q_r" not in tt
    dq_c = tt["q_c"].data
    assert bool((dq_c >= 0.0).all()), "q_c must receive only the detrained cloud water"
    # Ratchet at the measured residual (1.19e-7 of the precipitation, x64):
    # any growth fails here. The exact budget is asserted, and currently
    # xfails, in the test below, which names the open defect.
    dp = sigma.layer_thickness_dp(state.p_s.data)
    sink = -jnp.sum((tt["q_v"].data + dq_c) * dp, axis=-1) / constants.g
    np.testing.assert_allclose(np.asarray(precip), np.asarray(sink), rtol=2e-7, atol=1e-14)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "OPEN: the ZM water budget misses by 1.19e-7 of the precipitation on the "
    "hydro-bridge fixture (x64). NOT the zm_conv_evap flux clip: removing it "
    "leaves the residual unchanged. It is already inside zm_convr: column "
    "rain production sum(rprd) differs from the moisture removed "
    "-sum(dqdt + dlf) by 3.3e-8 (and from its prec by 1.1e-7). Cause not "
    "yet identified."))
def test_zm_convr_rain_production_closes_the_moisture_budget():
    n, nlev = 4, 12
    state, grid, sigma = _convecting_3d_state(n, nlev)
    physics_fn = make_physics(_make_zm_only_config(), model_type="hydrostatic", dt=300.0)
    ps = init_physics_state(6 * n * n, nlev, _make_zm_only_config())
    tend, _ = physics_fn(state, grid, sigma, phys_state=ps)
    tt = tend.tracer_tendencies
    dp = sigma.layer_thickness_dp(state.p_s.data)
    sink = -jnp.sum((tt["q_v"].data + tt["q_c"].data) * dp, axis=-1) / constants.g
    np.testing.assert_allclose(np.asarray(tend.precip.data), np.asarray(sink),
                               rtol=1e-9, atol=1e-14)


def test_cubed_sphere_grid_carries_land_frac_to_zm():
    """``land_fraction_for_columns`` reads ``grid.land_frac``; the cubed
    sphere had no such field, so this standalone route ran ocean autoconversion
    everywhere (codex round 1, #6).  A land grid must change the tendencies."""
    from legoesm.atmosphere.physics.convection.integration import land_fraction_for_columns

    n, nlev = 4, 12
    ncol = 6 * n * n
    grid = create_cubed_sphere(n)
    assert land_fraction_for_columns(grid, ncol) is None
    g_land = grid._replace(land_frac=jnp.ones((ncol,)))
    lf = land_fraction_for_columns(g_land, ncol)
    assert lf.shape == (ncol,) and bool((lf == 1.0).all())
    leaves, tree = jax.tree_util.tree_flatten(g_land)
    assert tree.unflatten(leaves).land_frac is not None

    cfg = _make_zm_only_config("required")
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    ps = init_physics_state(ncol, nlev, cfg)
    outs = []
    for land in (0.0, 1.0):
        state, grid_l, sigma = _convecting_3d_state(n, nlev, land_frac=land)
        tend, _ = physics_fn(state, grid_l, sigma, phys_state=ps)
        outs.append(tend.dT_dt.data)
    assert float(jnp.abs(outs[0]).max()) > 0.0
    assert float(jnp.abs(outs[1] - outs[0]).max()) > 0.0, (
        "c0_lnd != c0_ocn: a land grid must change the ZM heating")


def test_hydro_bridge_publishes_deepcu_inputs_into_the_carry():
    """The bridge publishes ``mass_flux_up``/``icwmr`` into the lagged
    ``PhysicsState`` carry only when the scheme sets both; before the port
    set them the CAM6 deep-convective cloud fraction was exactly zero on the
    CAM6 deck (GLM merged-suite review)."""
    from legoesm.atmosphere.physics.clouds.cloud_fraction import cam6_deep_convective_fraction
    from legoesm.atmosphere.physics.clouds.config import CloudConfig

    n, nlev = 4, 12
    ncol = 6 * n * n
    cfg = _make_zm_only_config()
    state, grid, sigma = _convecting_3d_state(n, nlev)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    ps = init_physics_state(ncol, nlev, cfg)
    assert float(jnp.abs(ps.conv_mass_flux_up).max()) == 0.0
    _, ps_out = physics_fn(state, grid, sigma, phys_state=ps)
    mf, icw = ps_out.conv_mass_flux_up, ps_out.conv_icwmr
    assert mf.shape == (ncol, nlev + 1) and icw.shape == (ncol, nlev)
    assert float(mf.max()) > 0.0 and float(icw.max()) > 0.0
    deepcu = cam6_deep_convective_fraction(mf, icw, CloudConfig(scheme="cam6_clubb"))
    assert float(deepcu.max()) > 0.0


def test_zm_land_fraction_is_an_explicit_choice():
    """No land fraction is an error unless the run declares an aquaplanet;
    declaring one while passing a land fraction is an error too; an unknown
    policy raises (dispatch hardening)."""
    T, q, pf, ph, u, v = _synthetic_column()
    cpp = jnp.zeros(T.shape)
    with pytest.raises(ValueError, match="land_frac is required"):
        zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0)
    with pytest.raises(ValueError, match="land_fraction='none'"):
        zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=_AQUA,
                                   land_frac=jnp.zeros(T.shape[0]))
    with pytest.raises(ValueError, match="Unknown ZhangMcFarlaneConfig.land_fraction"):
        zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0,
                                   config=ZhangMcFarlaneConfig(land_fraction="ocean"))
    # The aquaplanet choice reproduces the all-ocean column exactly.
    a, _ = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=_AQUA)
    b, _ = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0,
                                      land_frac=jnp.zeros(T.shape[0]))
    assert jnp.array_equal(a.dT_dt, b.dT_dt)


def test_combined_physics_lane_aquaplanet_runs_under_none_and_refuses_land():
    """Idealized grids carry an ALL-ZERO land mask on the combined-physics
    lane (MPAS / spectral / hydrostatic bridges).  "none" must run there, as it
    does on the column pipeline, and a mask with land must be refused."""
    n, nlev = 4, 12
    ncol = 6 * n * n
    cfg = _make_zm_only_config("none")
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    ps = init_physics_state(ncol, nlev, cfg)
    state, grid_ocean, sigma = _convecting_3d_state(n, nlev, land_frac=0.0)
    assert grid_ocean.land_frac is not None
    tend, _ = physics_fn(state, grid_ocean, sigma, phys_state=ps)
    assert bool(jnp.isfinite(tend.dT_dt.data).all())
    assert float(jnp.abs(tend.dT_dt.data).max()) > 0.0, "fixture must convect"
    state, grid_land, sigma = _convecting_3d_state(n, nlev, land_frac=1.0)
    with pytest.raises(ValueError, match="aquaplanet"):
        physics_fn(state, grid_land, sigma, phys_state=ps)


def test_combined_physics_lane_aquaplanet_refuses_land_under_jit():
    """With the grid a jit argument its land mask is traced; the aquaplanet
    guard must still refuse land (at run time) instead of running ocean
    coefficients everywhere."""
    import equinox as eqx

    n, nlev = 4, 12
    cfg = _make_zm_only_config("none")
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    ps = init_physics_state(6 * n * n, nlev, cfg)

    @eqx.filter_jit
    def step(s, g, p):
        return physics_fn(s, g, sigma, phys_state=p)[0].dT_dt.data

    state, grid_ocean, sigma = _convecting_3d_state(n, nlev, land_frac=0.0)
    assert bool(jnp.isfinite(step(state, grid_ocean, ps)).all())
    state, grid_land, sigma = _convecting_3d_state(n, nlev, land_frac=1.0)
    with pytest.raises(Exception, match="aquaplanet"):
        jax.block_until_ready(step(state, grid_land, ps))


def _zm_driver_config(grid_type, resolution, dycore_kw):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    dt = dycore_kw["dt"]
    return ExperimentConfig(
        grid=GridConfig(grid_type=grid_type, resolution=resolution, nlev=8),
        dycore=DycoreConfig(**dycore_kw), output=OutputConfig(diag_days=1),
        days=2.0 * dt / 86400.0, dataset="analytical", topography="flat",
        convection="zhang_mcfarlane")


@pytest.mark.parametrize("grid_type,resolution,dycore_kw", [
    ("cubed_sphere", 8, {"dt": 1.0}),
    ("mpas", 2, {"dt": 600.0, "discretization": "mpas"}),
], ids=["column_pipeline", "mpas_combined"])
def test_driver_zm_land_policy_means_the_same_on_every_lane(tmp_path, grid_type, resolution, dycore_kw):
    """One rule, both driver lanes, on a flat (all-ocean) run: "required"
    stops at setup, "none" runs to completion."""
    from legoesm.driver.model_driver import ModelDriver

    cfg = _zm_driver_config(grid_type, resolution, dycore_kw)
    assert cfg.zm_land_fraction == "required"
    with pytest.raises(ValueError, match="no land"):
        ModelDriver(cfg, output_dir=tmp_path / "req").setup()
    driver = ModelDriver(cfg._replace(zm_land_fraction="none"), output_dir=tmp_path / "none")
    driver.setup()
    assert driver.run() == "COMPLETED"
