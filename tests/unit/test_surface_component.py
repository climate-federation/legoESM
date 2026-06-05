"""SurfaceComponentAdapter — live land/ice/ocean surface steps as StepComponent bricks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.components import AbstractComponent, StepComponent
from legoesm.components.surface_component import SurfaceComponentAdapter
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.field import Field

_needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="surface energy-balance / soil solve needs JAX_ENABLE_X64=1",
)

_SHAPE = (6, 4, 4)
_DIMS = ("face", "x", "y")
_DT = 3600.0


def _forcing() -> AtmToSurface:
    f = jnp.ones(_SHAPE)
    return AtmToSurface(
        sw_down=200.0 * f, lw_down=300.0 * f, precip_total=1e-5 * f,
        precip_snow=0.0 * f, T_lowest=285.0 * f, q_lowest=6e-3 * f,
        u_lowest=7.0 * f, v_lowest=3.0 * f, p_lowest=95000.0 * f,
        p_surface=101325.0 * f, rho_lowest=1.2 * f, cos_zenith=0.5 * f,
        co2_ppmv=jnp.array(400.0), has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )


def _land_brick():
    from legoesm.land.config import LandConfig
    from legoesm.land.slab_land import step_land
    from legoesm.land.state import LandState

    config = LandConfig()
    state = LandState(
        T_soil=Field(jnp.full(_SHAPE, 270.0), name="T_soil", dims=_DIMS, units="K"),
        W_bucket=Field(jnp.full(_SHAPE, 50.0), name="W_bucket", dims=_DIMS,
                       units="kg/m^2"),
        snow_depth=Field(jnp.zeros(_SHAPE), name="snow_depth", dims=_DIMS, units="m"),
        snow_age=Field(jnp.zeros(_SHAPE), name="snow_age", dims=_DIMS, units="s"),
    )

    def step_fn(grid, st, frc, dt):
        new_state, resp, _carbon = step_land(st, frc, config, U_min=1.0, dt=dt)
        return new_state, resp

    brick = SurfaceComponentAdapter(
        step_fn,
        prognostic_variables=("T_soil", "W_bucket", "snow_depth", "snow_age"),
        required_forcing=("atm_forcing",),
        provided_fluxes=("shflx", "lhflx", "lw_up", "T_sfc"),
    )
    return brick, state, "T_soil", lambda r: r.shflx


def _ice_brick():
    from legoesm import constants
    from legoesm.ice.config import SeaIceConfig
    from legoesm.ice.sea_ice import step_sea_ice
    from legoesm.ice.state import SeaIceState

    config = SeaIceConfig(bulk_scheme="most")
    state = SeaIceState(
        h_ice=Field(jnp.full(_SHAPE, 1.0), name="h_ice", dims=_DIMS, units="m"),
        T_ice=Field(jnp.full(_SHAPE, 260.0), name="T_ice", dims=_DIMS, units="K"),
        concentration=Field(jnp.full(_SHAPE, 0.8), name="concentration", dims=_DIMS,
                            units="1"),
    )
    sst = jnp.full(_SHAPE, constants.T_freeze_ocean)
    zero = jnp.zeros(_SHAPE)

    def step_fn(grid, st, frc, dt):
        return step_sea_ice(st, frc, sst, zero, zero, config, 1.0, dt)

    brick = SurfaceComponentAdapter(
        step_fn,
        prognostic_variables=("h_ice", "T_ice", "concentration"),
        required_forcing=("atm_forcing", "ocean_state"),
        provided_fluxes=("shflx", "lhflx", "tau_x", "tau_y", "lw_up", "T_sfc"),
    )
    return brick, state, "T_ice", lambda r: r.shflx


def _ocean_brick():
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig,
        init_slab_state,
        make_ocean,
    )

    ocean_step = make_ocean(SimpleOceanConfig(mode="slab"))
    state = init_slab_state(_SHAPE, T_sfc_init=290.0)

    def step_fn(grid, st, frc, dt):
        # make_ocean returns (state, sst, u_sfc, v_sfc); repackage the surface
        # fields the coupler turns into atmospheric fluxes as the named handoff.
        new_state, sst, u_sfc, v_sfc = ocean_step(st, frc, dt)
        return new_state, {"sst": sst, "u_ocean_sfc": u_sfc, "v_ocean_sfc": v_sfc}

    brick = SurfaceComponentAdapter(
        step_fn,
        prognostic_variables=("T_sfc",),
        required_forcing=("atm_forcing",),
        provided_fluxes=("sst", "u_ocean_sfc", "v_ocean_sfc"),
    )
    return brick, state, "T_sfc", lambda r: r["sst"]


_BUILDERS = [_land_brick, _ice_brick, _ocean_brick]


@pytest.mark.parametrize("builder", _BUILDERS)
def test_surface_brick_is_stepcomponent_with_metadata(builder) -> None:
    brick, _state, _gf, _flux = builder()
    assert isinstance(brick, StepComponent)
    # implicit-step brick, NOT the explicit-RHS kind (no faked tendency)
    assert not isinstance(brick, AbstractComponent)
    assert brick.prognostic_variables  # evolves a surface state
    assert brick.required_forcing       # needs atmospheric forcing
    assert brick.provided_fluxes        # provides surface fluxes/fields


@_needs_x64
@pytest.mark.parametrize("builder", _BUILDERS)
def test_surface_brick_flux_is_differentiable(builder) -> None:
    """The wrapped live step runs as the brick's uniform step(.., dt) seam, and the
    PROVIDED FLUX is jax.grad differentiable wrt the surface temperature through the
    implicit energy balance (D1/D3) — proving the flux path, not just the state."""
    brick, state, grad_field, flux = builder()
    forcing = _forcing()

    new_state, resp = brick.step(None, state, forcing, _DT)
    leaves = jax.tree.leaves((new_state, resp))
    assert leaves and all(jnp.all(jnp.isfinite(x)) for x in leaves)

    fld = getattr(state, grad_field)

    def loss(arr):
        st = state._replace(**{grad_field: fld.replace(data=arr)})
        _ns, r = brick.step(None, st, forcing, _DT)
        return jnp.sum(flux(r) ** 2)

    g = jax.grad(loss)(fld.data)
    assert jnp.all(jnp.isfinite(g)) and jnp.max(jnp.abs(g)) > 0.0


def test_response_validation_catches_a_missing_declared_flux() -> None:
    """A step omitting a declared flux fails loudly (no silent metadata drift)."""
    bad = SurfaceComponentAdapter(
        lambda grid, state, forcing, dt: (state, {"sst": jnp.zeros(2)}),
        prognostic_variables=("x",), required_forcing=(),
        provided_fluxes=("sst", "missing"))
    with pytest.raises(ValueError, match="missing"):
        bad.step(None, None, None, 1.0)


def test_non_callable_step_fn_rejected() -> None:
    with pytest.raises(TypeError, match="callable"):
        SurfaceComponentAdapter(
            object(), prognostic_variables=("h",), required_forcing=(),
            provided_fluxes=())
