"""CATEGORY 4 (Land) differentiability tests.

Verifies that ``jax.grad`` produces finite, non-zero, spatially-structured
gradients through the land surface components:

  4a) Slab land model        (``step_land``)
  4b) Multi-layer land model (``step_multilayer_land``, Richards solver)
  4c) Snow budget            (``update_snow`` energy-limited melt feedback)
  4d) Carbon cycle           (``compute_gpp`` w.r.t. T and CO2)
  4e) Stomatal conductance   (Jarvis / coupled Farquhar w.r.t. VPD/PAR)

All loss functions are scalar reductions of the model output. We
differentiate w.r.t. the ``.data`` array of Field-wrapped state members
(slab land) and w.r.t. raw arrays (multi-layer land), per the agent's
"Key pattern" guidance.

Run:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/unit/test_diff_land.py -v --tb=short

(The Metal JAX backend is broken in this env; JAX_PLATFORMS=cpu is required.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.state import LandState
from legoesm.land.slab_land import step_land
from legoesm.land.multilayer_land import (
    step_multilayer_land,
    init_multilayer_land_state,
)
from legoesm.land.snow_budget import update_snow
from legoesm.land.carbon.carbon_cycle import compute_gpp
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.stomata import (
    StomataConfig,
    coupled_farquhar_stomata,
    jarvis_gs,
)


# ----------------------------------------------------------------------
# Shared helper
# ----------------------------------------------------------------------

def assert_gradient_ok(
    loss_fn,
    x0,
    name: str = "",
    *,
    require_structure: bool = True,
    min_nonzero_frac: float = 0.1,
):
    """Check that grad(loss_fn)(x0) is finite, non-zero, and structured.

    Parameters
    ----------
    loss_fn : callable
        Scalar-valued function of x0.
    x0 : array
        Point at which to evaluate the gradient.
    name : str
        Label used in assertion messages.
    require_structure : bool
        If True (and x0 has >1 element) assert the gradient is not
        spatially uniform (catches dead/broadcast-only paths).
    min_nonzero_frac : float
        Minimum fraction of gradient entries that must be non-zero.
    """
    grad = jax.grad(loss_fn)(x0)
    grad = jnp.asarray(grad)

    assert jnp.all(jnp.isfinite(grad)), (
        f"[{name}] gradient has non-finite entries: "
        f"{grad[~jnp.isfinite(grad)][:5]}"
    )

    nonzero_frac = float(jnp.mean(jnp.abs(grad) > 0.0))
    assert nonzero_frac >= min_nonzero_frac, (
        f"[{name}] only {nonzero_frac:.1%} of gradient entries non-zero "
        f"(need >= {min_nonzero_frac:.0%}); grad sample={grad.ravel()[:5]}"
    )

    if require_structure and grad.size > 1:
        spread = float(jnp.max(grad) - jnp.min(grad))
        scale = float(jnp.max(jnp.abs(grad))) + 1e-30
        assert spread / scale > 1e-10, (
            f"[{name}] gradient is spatially uniform (no structure): "
            f"min={float(jnp.min(grad)):.3e} max={float(jnp.max(grad)):.3e}"
        )
    return grad


# ----------------------------------------------------------------------
# Builders (small grids — run in seconds)
# ----------------------------------------------------------------------

NCOL = 8  # tiny column count


def _make_forcing(ncol: int, **overrides) -> AtmToSurface:
    """AtmToSurface with mild spatial variation so gradients are structured."""
    base = jnp.linspace(0.9, 1.1, ncol)
    defaults = dict(
        sw_down=200.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
        T_lowest=280.0, q_lowest=5e-3, u_lowest=5.0, v_lowest=2.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.7,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    defaults.update(overrides)
    return AtmToSurface(**{k: jnp.asarray(v, dtype=jnp.float64) * base
                           for k, v in defaults.items()})


def _make_slab_state(ncol: int, T_init=280.0, W_init=50.0,
                     snow_init=0.0, snow_age_init=0.0) -> LandState:
    return LandState(
        T_soil=Field(jnp.full(ncol, float(T_init)), name="T_soil"),
        W_bucket=Field(jnp.full(ncol, float(W_init)), name="W_bucket"),
        snow_depth=Field(jnp.full(ncol, float(snow_init)), name="snow_depth"),
        snow_age=Field(jnp.full(ncol, float(snow_age_init)), name="snow_age"),
    )


# ======================================================================
# 4a) Slab land model
# ======================================================================

def test_slab_land_grad_wrt_T_soil():
    """grad of next-step T_soil energy w.r.t. initial T_soil."""
    config = LandConfig()
    forcing = _make_forcing(NCOL)
    state = _make_slab_state(NCOL)
    T0 = state.T_soil.data

    def loss(T_data):
        st = state._replace(T_soil=state.T_soil.replace(data=T_data))
        out, _, _ = step_land(st, forcing, config, U_min=1.0, dt=600.0)
        return jnp.sum(out.T_soil.data ** 2)

    assert_gradient_ok(loss, T0, name="slab T_soil")


def test_slab_land_grad_wrt_sw_down_cross_component():
    """Cross-component: grad of T_soil energy w.r.t. forcing sw_down.

    Tests that a forcing field (not a state field) reaches the soil
    temperature through the radiation + energy-balance chain.
    """
    config = LandConfig()
    forcing = _make_forcing(NCOL)
    state = _make_slab_state(NCOL)
    sw0 = forcing.sw_down

    def loss(sw):
        f = forcing._replace(sw_down=sw)
        out, _, _ = step_land(state, f, config, U_min=1.0, dt=600.0)
        return jnp.sum(out.T_soil.data ** 2)

    grad = assert_gradient_ok(loss, sw0, name="slab sw_down->T_soil")
    # More incoming SW must raise the soil temperature (and hence T^2),
    # so the gradient must be strictly positive everywhere.
    assert jnp.all(grad > 0.0), (
        f"d(T_soil^2)/d(sw_down) must be positive (warming), got {grad}"
    )


def test_slab_land_grad_through_5_chained_steps():
    """Gradient accumulation through 5 chained slab-land steps."""
    config = LandConfig()
    forcing = _make_forcing(NCOL)
    state = _make_slab_state(NCOL)
    T0 = state.T_soil.data

    def loss(T_data):
        st = state._replace(T_soil=state.T_soil.replace(data=T_data))
        for _ in range(5):
            st, _, _ = step_land(st, forcing, config, U_min=1.0, dt=600.0)
        return jnp.sum(st.T_soil.data ** 2)

    assert_gradient_ok(loss, T0, name="slab T_soil x5")


def test_slab_land_grad_wrt_W_bucket():
    """Soil-moisture sensitivity: grad of latent flux energy w.r.t. W_bucket.

    W controls beta (moisture availability), which scales evaporation and
    therefore the surface latent-heat flux.
    """
    config = LandConfig()
    # Hot dry forcing so the bucket actually limits evaporation (non-zero grad).
    forcing = _make_forcing(NCOL, T_lowest=270.0, q_lowest=1e-3,
                            sw_down=500.0, precip_total=0.0)
    state = _make_slab_state(NCOL, T_init=305.0, W_init=20.0)
    W0 = state.W_bucket.data

    def loss(W_data):
        st = state._replace(W_bucket=state.W_bucket.replace(data=W_data))
        out, resp, _ = step_land(st, forcing, config, U_min=1.0, dt=600.0)
        return jnp.sum(resp.lhflx ** 2)

    assert_gradient_ok(loss, W0, name="slab W_bucket->lhflx")


# ======================================================================
# 4b) Multi-layer land model
# ======================================================================

def _ml_config():
    # 5 soil layers per the agent's "small grids" guidance.
    base = MultiLayerLandConfig()
    sg = base.soil_grid._replace(n_layers=5)
    return base._replace(soil_grid=sg)


def test_multilayer_land_grad_wrt_T_soil():
    """grad of next-step soil-T energy w.r.t. initial multi-layer T_soil."""
    config = _ml_config()
    forcing = _make_forcing(NCOL, T_lowest=290.0, q_lowest=8e-3, sw_down=300.0,
                            lw_down=350.0, precip_total=1e-4)
    state = init_multilayer_land_state(NCOL, config, T_init=280.0)
    T0 = state.T_soil

    def loss(T_data):
        st = state._replace(T_soil=T_data)
        out, _, _ = step_multilayer_land(st, forcing, config, U_min=1.0, dt=600.0)
        return jnp.sum(out.T_soil ** 2)

    assert_gradient_ok(loss, T0, name="multilayer T_soil")


def test_multilayer_land_grad_wrt_psi_soil():
    """grad through the Richards equation solver w.r.t. matric potential psi.

    psi (matric potential) drives unsaturated flow in the Richards solver;
    a finite, non-zero gradient confirms the Picard-iteration solver is
    differentiable end-to-end.
    """
    config = _ml_config()
    forcing = _make_forcing(NCOL, T_lowest=290.0, q_lowest=8e-3, sw_down=300.0,
                            lw_down=350.0, precip_total=1e-4)
    state = init_multilayer_land_state(NCOL, config, T_init=280.0)
    psi0 = state.psi_soil

    def loss(psi_data):
        st = state._replace(psi_soil=psi_data)
        out, _, _ = step_multilayer_land(st, forcing, config, U_min=1.0, dt=600.0)
        return jnp.sum(out.theta_soil ** 2) + jnp.sum(out.psi_soil ** 2)

    assert_gradient_ok(loss, psi0, name="multilayer psi_soil (Richards)")


def test_multilayer_land_grad_through_3_chained_steps():
    """Gradient accumulation through 3 chained multi-layer steps."""
    config = _ml_config()
    forcing = _make_forcing(NCOL, T_lowest=290.0, q_lowest=8e-3, sw_down=300.0,
                            lw_down=350.0, precip_total=1e-4)
    state = init_multilayer_land_state(NCOL, config, T_init=280.0)
    T0 = state.T_soil

    def loss(T_data):
        st = state._replace(T_soil=T_data)
        for _ in range(3):
            st, _, _ = step_multilayer_land(st, forcing, config, U_min=1.0, dt=600.0)
        return jnp.sum(st.T_soil ** 2)

    assert_gradient_ok(loss, T0, name="multilayer T_soil x3")


# ======================================================================
# 4c) Snow budget differentiability
# ======================================================================

def test_snow_melt_feedback_grad_wrt_Q_net():
    """Energy-limited melt: grad of remaining snow w.r.t. net energy Q_net.

    With snow present and T_sfc >= freezing, more net energy melts more
    snow, so d(snow_new^2)/d(Q_net) must be negative and non-zero.
    """
    ncol = NCOL
    snow = jnp.full(ncol, 5.0)               # kg/m2 (snow present)
    snow_age = jnp.zeros(ncol)
    T_sfc = jnp.full(ncol, constants.T_freeze + 2.0)  # above freezing
    precip_snow = jnp.zeros(ncol)
    Q0 = jnp.linspace(50.0, 150.0, ncol)     # positive net energy [W/m2]

    def loss(Q_net):
        snow_new, _, _ = update_snow(
            snow, snow_age, T_sfc, precip_snow, dt=600.0, Q_net=Q_net, snow_age_activation_K=0.0
        )
        return jnp.sum(snow_new ** 2)

    grad = assert_gradient_ok(loss, Q0, name="snow melt Q_net")
    # snow_new = snow - melt, melt grows with Q_net, so d(snow^2)/dQ < 0.
    assert jnp.all(grad < 0.0), (
        f"more melting energy must shrink the snowpack, got grad={grad}"
    )


def test_snow_grad_wrt_T_lowest_through_slab_step():
    """4c via step_land: grad of next-step snow w.r.t. T_lowest with snow present.

    Warmer near-surface air raises the surface energy budget, drives the
    energy-limited melt feedback, and changes the remaining snowpack.
    """
    config = LandConfig()
    # Snow present, surface near freezing so melt can switch on.
    forcing = _make_forcing(NCOL, T_lowest=constants.T_freeze + 3.0,
                            sw_down=300.0, lw_down=320.0,
                            precip_total=0.0, precip_snow=0.0)
    state = _make_slab_state(NCOL, T_init=constants.T_freeze + 0.5,
                             W_init=50.0, snow_init=10.0)
    T_low0 = forcing.T_lowest

    def loss(T_low):
        f = forcing._replace(T_lowest=T_low)
        out, _, _ = step_land(state, f, config, U_min=1.0, dt=1800.0)
        return jnp.sum(out.snow_depth.data ** 2)

    assert_gradient_ok(loss, T_low0, name="slab snow vs T_lowest")


# ======================================================================
# 4d) Carbon cycle differentiability
# ======================================================================

def test_carbon_gpp_grad_wrt_temperature():
    """grad of GPP w.r.t. surface temperature (smooth Gaussian T response)."""
    config = CarbonConfig(scheme="differland")
    ncol = NCOL
    sw_down = jnp.full(ncol, 300.0)
    LAI = jnp.full(ncol, 3.0)
    co2 = jnp.full(ncol, 400.0)
    beta = jnp.full(ncol, 0.8)
    # Span both sides of the optimum so the Gaussian response slope is
    # non-zero (at exactly T_opt the derivative would vanish).
    T0 = jnp.linspace(285.0, 305.0, ncol)

    def loss(T):
        return jnp.sum(compute_gpp(sw_down, T, LAI, co2, beta, config) ** 2)

    assert_gradient_ok(loss, T0, name="GPP vs T")


def test_carbon_gpp_grad_wrt_co2():
    """grad of GPP w.r.t. CO2 (Michaelis-Menten fertilization). Must be > 0."""
    config = CarbonConfig(scheme="differland")
    ncol = NCOL
    sw_down = jnp.full(ncol, 300.0)
    T = jnp.full(ncol, constants.T_freeze + 25.0)  # at T_opt -> max f_T
    LAI = jnp.full(ncol, 3.0)
    beta = jnp.full(ncol, 0.8)
    co2_0 = jnp.linspace(350.0, 450.0, ncol)

    def loss(co2):
        return jnp.sum(compute_gpp(sw_down, T, LAI, co2, beta, config) ** 2)

    grad = assert_gradient_ok(loss, co2_0, name="GPP vs CO2")
    # Michaelis-Menten is monotone increasing in CO2 -> GPP^2 increases.
    assert jnp.all(grad > 0.0), (
        f"d(GPP^2)/d(CO2) must be positive (fertilization), got {grad}"
    )


# ======================================================================
# 4e) Stomatal conductance differentiability
# ======================================================================

def test_jarvis_gs_grad_wrt_par():
    """Jarvis gs gradient w.r.t. sw_down (PAR forcing). Must be > 0."""
    config = StomataConfig(enabled=True)
    ncol = NCOL
    T = jnp.full(ncol, constants.T_freeze + 25.0)
    q_air = jnp.full(ncol, 5e-3)
    p_surface = jnp.full(ncol, 1.013e5)
    beta_soil = jnp.full(ncol, 0.8)
    sw0 = jnp.linspace(100.0, 300.0, ncol)

    def loss(sw_down):
        gs = jarvis_gs(T, sw_down, q_air, p_surface, beta_soil, config)
        return jnp.sum(gs ** 2)

    grad = assert_gradient_ok(loss, sw0, name="jarvis gs vs PAR")
    # f_PAR is hyperbolic-increasing in PAR -> gs (and gs^2) increase.
    assert jnp.all(grad > 0.0), (
        f"d(gs^2)/d(PAR) must be positive (light limitation eases), got {grad}"
    )


def test_jarvis_gs_grad_wrt_vpd_via_humidity():
    """Jarvis gs gradient w.r.t. near-surface humidity (drives VPD).

    Drier air -> larger VPD -> stomatal closure (lower gs). So
    d(gs^2)/d(q_air) must be >= 0 (more humidity -> less closure -> higher gs)
    and non-zero where the VPD response is unsaturated.
    """
    config = StomataConfig(enabled=True)
    ncol = NCOL
    T = jnp.full(ncol, constants.T_freeze + 25.0)
    p_surface = jnp.full(ncol, 1.013e5)
    beta_soil = jnp.full(ncol, 0.8)
    sw_down = jnp.full(ncol, 250.0)
    # Humidity range that keeps f_VPD in its linear (unsaturated) regime.
    q0 = jnp.linspace(2e-3, 8e-3, ncol)

    def loss(q_air):
        gs = jarvis_gs(T, sw_down, q_air, p_surface, beta_soil, config)
        return jnp.sum(gs ** 2)

    grad = assert_gradient_ok(loss, q0, name="jarvis gs vs humidity(VPD)")
    assert jnp.all(grad >= 0.0), (
        f"d(gs^2)/d(q_air) must be >= 0 (less VPD stress), got {grad}"
    )


@pytest.mark.parametrize("stomata_model", ["ball_berry", "medlyn"])
def test_coupled_farquhar_grad_wrt_par(stomata_model):
    """Coupled Farquhar-stomata: grad of GPP w.r.t. PAR through the A-gs solver.

    This exercises the unrolled fixed-point Ci iteration. Both Ball-Berry
    and Medlyn closures should pass gradients.
    """
    config = StomataConfig(enabled=True, stomata_model=stomata_model)
    ncol = NCOL
    T_leaf = jnp.full(ncol, constants.T_freeze + 25.0)
    co2 = 400.0
    q_air = jnp.full(ncol, 6e-3)
    p_surface = jnp.full(ncol, 1.013e5)
    LAI = jnp.full(ncol, 3.0)
    beta_soil = jnp.full(ncol, 0.8)
    sw0 = jnp.linspace(150.0, 400.0, ncol)

    def loss(sw_down):
        _, gpp = coupled_farquhar_stomata(
            T_leaf, sw_down, co2, q_air, p_surface, LAI, beta_soil, config,
        )
        return jnp.sum(gpp ** 2)

    assert_gradient_ok(loss, sw0, name=f"coupled Farquhar GPP vs PAR ({stomata_model})")


def test_coupled_farquhar_grad_wrt_vpd():
    """Coupled Farquhar-stomata: grad of conductance w.r.t. humidity (VPD)."""
    config = StomataConfig(enabled=True, stomata_model="medlyn")
    ncol = NCOL
    T_leaf = jnp.full(ncol, constants.T_freeze + 25.0)
    co2 = 400.0
    p_surface = jnp.full(ncol, 1.013e5)
    LAI = jnp.full(ncol, 3.0)
    beta_soil = jnp.full(ncol, 0.8)
    sw_down = jnp.full(ncol, 300.0)
    q0 = jnp.linspace(3e-3, 9e-3, ncol)

    def loss(q_air):
        gs, _ = coupled_farquhar_stomata(
            T_leaf, sw_down, co2, q_air, p_surface, LAI, beta_soil, config,
        )
        return jnp.sum(gs ** 2)

    assert_gradient_ok(loss, q0, name="coupled Farquhar gs vs VPD")
