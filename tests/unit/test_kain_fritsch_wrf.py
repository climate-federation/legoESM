"""Unit tests for ``kain_fritsch_wrf`` (WRF v4.8.0 KF-Eta port, OPT-IN).

The port is selected by ``KainFritschConfig.variant = "wrf_kfeta"`` and is NOT
RULE 5 compliant (#2280); the default ``"smooth"`` stays the surrogate.  The
oracle match lives in tests/atmosphere/test_kain_fritsch_oracle.py; these pin
the variant dispatch, the wrapper's interface (shapes, no momentum transport,
carry layout, quiescence without a trigger, the hydrostatic bridge end to
end) and the refusal of inputs the port cannot run without.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.convection.config import ConvectionConfig, KainFritschConfig
from legoesm.atmosphere.physics.convection.kain_fritsch import kain_fritsch_convection
from legoesm.atmosphere.physics.convection.kain_fritsch_wrf import (
    check_wrf_kfeta_config,
    kain_fritsch_wrf_convection,
)
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_WRF = KainFritschConfig(variant="wrf_kfeta")


def _column(ncol=2, nlev=16, T_sfc=305.0, q_sfc=18.0e-3, lapse_rate=8.0, w=0.0):
    p_s, p_top = 1.0e5, 5.0e3
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half = jnp.concatenate([jnp.full((ncol, 1), p_top * 0.5),
                              0.5 * (p_full[:, :-1] + p_full[:, 1:]),
                              jnp.full((ncol, 1), p_s)], axis=1)
    z = -8500.0 * jnp.log(p_full / p_s)
    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z
    q = q_sfc * jnp.exp(-z / 3000.0)
    u = jnp.full_like(T, 10.0)
    return dict(T=T, q_v=q, p_full=p_full, p_half=p_half, w_grid=jnp.full_like(T, w),
                conv_prog_profile=jnp.zeros_like(T), dt=300.0, u=u, v=jnp.zeros_like(T))


def test_kf_shapes_finite_no_cmt_and_carry():
    x = _column(ncol=3, nlev=16, w=0.1)
    out, carry = kain_fritsch_convection(**x, config=_WRF)
    ncol, nlev = x["T"].shape
    for a in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.dq_r_conv_dt):
        assert a.shape == (ncol, nlev) and bool(jnp.all(jnp.isfinite(a)))
    assert out.cape.shape == (ncol,) and out.convective_mask.shape == (ncol,)
    assert out.du_dt_conv is None and out.dv_dt_conv is None
    assert carry.shape == (ncol, nlev)
    assert float(jnp.max(jnp.abs(carry[:, :-1]))) == 0.0


def test_kf_fires_on_unstable_column_with_ascent():
    out, carry = kain_fritsch_convection(**_column(w=0.1), config=_WRF)
    assert bool(jnp.all(out.convective_mask == 1.0))
    assert float(jnp.max(out.dT_dt)) > 0.0          # heating aloft
    assert float(jnp.min(out.dq_v_dt)) < 0.0        # drying
    assert bool(jnp.all(carry[:, -1] > 0.0))        # cloud-base mass flux
    dp = _column()["p_half"][:, 1:] - _column()["p_half"][:, :-1]
    precip = jnp.sum(out.dq_r_conv_dt * dp, axis=1) / constants.g
    assert bool(jnp.all(precip > 0.0))


def test_kf_quiescent_on_stable_column():
    out, carry = kain_fritsch_convection(**_column(lapse_rate=3.0, q_sfc=2.0e-3, w=-0.5), config=_WRF)
    assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.dq_r_conv_dt))) == 0.0
    assert float(jnp.max(out.convective_mask)) == 0.0
    assert float(jnp.max(jnp.abs(carry))) == 0.0


def test_kf_requires_winds_and_rejects_surrogate_fields():
    x = _column()
    with pytest.raises(ValueError, match="winds"):
        kain_fritsch_convection(**{**x, "v": None}, config=_WRF)
    for field, value in (("trigger_sharpness", 1.0e6), ("M_b_max", 0.1),
                         ("parcel_perturb_T", 0.0), ("precip_efficiency", 0.5),
                         ("subsidence_solve", "advective")):
        with pytest.raises(ValueError, match=field):
            kain_fritsch_convection(**x, config=_WRF._replace(**{field: value}))


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
    return state._replace(tracers=tracers), grid, sigma


def test_kf_orchestrator_one_step_finite():
    """Hydrostatic bridge with KF selected: winds reach the scheme, no NaN,
    no convective momentum tendency."""
    state, grid, sigma = _make_3d_state()
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="kain_fritsch", kain_fritsch=_WRF),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    n, nlev = 4, 12
    ncol = 6 * n * n
    ps = init_physics_state(ncol, nlev, cfg)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)
    assert ps_out.conv_prog_profile.shape == (ncol, nlev)
    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
        assert jnp.all(jnp.isfinite(f.data))
    assert float(jnp.max(jnp.abs(tend.du_dt.data))) == 0.0
    assert float(jnp.max(jnp.abs(tend.dv_dt.data))) == 0.0


def test_variant_dispatch_default_smooth_and_unknown_raises():
    """Default "smooth" runs the surrogate (winds ignored), "wrf_kfeta" runs the
    port (same answer as calling it directly), anything else raises."""
    x = _column(w=0.1)
    no_wind = {k: v for k, v in x.items() if k not in ("u", "v")}
    smooth, _ = kain_fritsch_convection(**no_wind)
    smooth_w, _ = kain_fritsch_convection(**x)
    wrf, _ = kain_fritsch_convection(**x, config=_WRF)
    direct, _ = kain_fritsch_wrf_convection(**x, config=_WRF)
    assert bool(jnp.all(smooth.dT_dt == smooth_w.dT_dt))
    assert bool(jnp.all(wrf.dT_dt == direct.dT_dt))
    # the two implementations differ (the default is NOT the port)
    assert float(jnp.max(jnp.abs(smooth.dT_dt - wrf.dT_dt))) > 0.0
    with pytest.raises(ValueError, match="variant"):
        kain_fritsch_convection(**x, config=KainFritschConfig(variant="wrf"))


def test_check_wrf_kfeta_config_accepts_default_rejects_surrogate_field():
    check_wrf_kfeta_config(_WRF)
    with pytest.raises(ValueError, match="cape_consumption_time"):
        check_wrf_kfeta_config(_WRF._replace(cape_consumption_time=2400.0))


def test_wrf_entrain_const_gradient_finite_nonzero():
    """The port's REI entrainment coefficient (WRF 0.03) carries a gradient."""
    x = _column(w=0.1)

    def loss(c):
        out = kain_fritsch_convection(**x, config=_WRF._replace(entrain_const=c))[0]
        return jnp.sum(out.dT_dt ** 2)

    g = jax.grad(loss)(0.03)
    assert bool(jnp.isfinite(g)) and float(jnp.abs(g)) > 0.0
