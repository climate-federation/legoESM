"""End-to-end: the ocean and land complexity ladders actually RUN.

Every complexity rung is built through the component factory and stepped once on
real state — the user's "ocean must run at increasing complexity (fixed SST ->
slab -> multilayer -> 3D); land can be slab or multilayer".  The dispatch itself
is unit-tested in ``test_component_complexity``; this is the integration check
that each rung produces a *finite* step (simpler -> more complex).  ``full_3d``
ocean is exercised by the ocean dynamics suite, not here.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.components import LandComplexity, OceanComplexity
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.driver.component_factory import (
    create_land_component,
    create_ocean_component,
)

_DT = 600.0
_SHAPE_2D = (6, 4, 4)  # cubed-sphere-shaped slab fields
_DIMS_2D = ("face", "x", "y")
# Minimum |Δ| that counts as the model genuinely advancing a prognostic field
# (well above float round-off, well below the real one-step tendencies here —
# slab SST moves ~1e-3 K/step, which is *within* jnp.allclose's default rtol,
# so an explicit threshold is needed rather than `not allclose`).
_EVOLVE_EPS = 1e-6


def _changed(new, old) -> bool:
    return bool(jnp.max(jnp.abs(new - old)) > _EVOLVE_EPS)


def _forcing(shape: tuple[int, ...]) -> AtmToSurface:
    """A physically-sane AtmToSurface at the given (gridded or column) shape."""

    def full(v: float) -> jnp.ndarray:
        return jnp.full(shape, v)

    return AtmToSurface(
        sw_down=full(200.0),
        lw_down=full(300.0),
        precip_total=full(1e-5),
        precip_snow=full(0.0),
        T_lowest=full(280.0),
        q_lowest=full(5e-3),
        u_lowest=full(5.0),
        v_lowest=full(-3.0),
        p_lowest=full(95_000.0),
        p_surface=full(1.0e5),
        rho_lowest=full(1.15),
        cos_zenith=full(0.6),
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )


def _assert_response_ok(resp, shape: tuple[int, ...]) -> None:
    """Every coupler-facing TileResponse channel is finite and full-field.

    TileResponse is a per-tile *spatial* response: ``blend_tiles`` and the
    accumulator consume every channel as a per-cell array, so each must carry
    the full tile shape.  A regression that NaNs OR collapses a spatially
    varying channel (freshwater_flux, co2_flux, ocean_stress, salt_flux, ...)
    to a broadcast scalar would silently erase spatial budgets in the coupler
    — so scalars are rejected here, not only the two heat fluxes checked.
    """
    for name, val in resp._asdict().items():
        arr = jnp.asarray(val)
        assert jnp.all(jnp.isfinite(arr)), f"TileResponse.{name} not finite"
        assert arr.shape == shape, (
            f"TileResponse.{name} shape {arr.shape} != full-field {shape}"
        )


# --------------------------------------------------------------------------
# Ocean ladder: fixed_sst -> slab -> slab_multilayer (full_3d covered elsewhere)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rung",
    [
        OceanComplexity.FIXED_SST,
        OceanComplexity.SLAB,
        OceanComplexity.SLAB_MULTILAYER,
    ],
)
def test_ocean_rung_runs_one_step(rung) -> None:
    from legoesm.ocean.simple_ocean import init_slab_state

    # The factory resolves the rung to the simple-ocean mode and returns a
    # closure bound to that config: step(state, forcing, dt).
    step = create_ocean_component(config=None, grid=None, ocean_config=rung)
    state = init_slab_state(_SHAPE_2D)

    # JAX is functional: `state` still holds the pre-step values after the call.
    new_state, sst, u_sfc, v_sfc = step(state, _forcing(_SHAPE_2D), _DT)

    assert sst.shape == _SHAPE_2D
    assert jnp.all(jnp.isfinite(sst))
    assert jnp.all(jnp.isfinite(new_state.T_sfc.data))
    assert jnp.all(jnp.isfinite(new_state.T_deep.data))
    assert jnp.all(jnp.isfinite(u_sfc)) and jnp.all(jnp.isfinite(v_sfc))

    # Behavioural: a higher rung must actually ADVANCE its prognostic state,
    # not just return a finite container.  fixed_sst is prescribed (inert);
    # slab evolves the mixed layer; two-layer also evolves the deep layer.
    if rung is OceanComplexity.FIXED_SST:
        assert not _changed(new_state.T_sfc.data, state.T_sfc.data), (
            "fixed SST must stay prescribed"
        )
    else:
        assert _changed(new_state.T_sfc.data, state.T_sfc.data), "slab T_sfc must evolve"
    if rung is OceanComplexity.SLAB_MULTILAYER:
        assert _changed(new_state.T_deep.data, state.T_deep.data), (
            "two-layer T_deep must evolve"
        )


# --------------------------------------------------------------------------
# Land ladder: slab -> multilayer (column)
# --------------------------------------------------------------------------


def test_land_slab_rung_runs_one_step() -> None:
    from legoesm.core.field import Field
    from legoesm.land import LandConfig, LandState

    # The factory selects step_land for the slab rung; the slab config it
    # resolves the rung to is LandConfig().
    step = create_land_component(config=None, grid=None, land_config=LandComplexity.SLAB)
    config = LandConfig()

    def field(name: str, v: float, units: str) -> Field:
        return Field(jnp.full(_SHAPE_2D, v), name=name, dims=_DIMS_2D, units=units)

    state = LandState(
        T_soil=field("T_soil", 280.0, "K"),
        W_bucket=field("W_bucket", 75.0, "kg/m2"),
        snow_depth=field("snow_depth", 0.0, "kg/m2"),
        snow_age=field("snow_age", 0.0, "s"),
    )

    new_state, resp, _carbon = step(
        state, _forcing(_SHAPE_2D), config, 1.0, _DT
    )

    assert jnp.all(jnp.isfinite(new_state.T_soil.data))
    # The slab land model must actually integrate: soil temperature and/or
    # bucket moisture advance under the supplied radiative/turbulent/precip
    # forcing (not a no-op return of the initial state).
    advanced = _changed(new_state.T_soil.data, state.T_soil.data) or _changed(
        new_state.W_bucket.data, state.W_bucket.data
    )
    assert advanced, "slab land left both T_soil and W_bucket unchanged"
    _assert_response_ok(resp, _SHAPE_2D)


def test_land_multilayer_rung_runs_one_step() -> None:
    from legoesm.land import MultiLayerLandConfig
    from legoesm.land.multilayer_land import init_multilayer_land_state

    ncol = 16
    step = create_land_component(
        config=None, grid=None, land_config=LandComplexity.MULTILAYER
    )
    config = MultiLayerLandConfig()
    state = init_multilayer_land_state(ncol, config)

    new_state, resp, _carbon = step(
        state, _forcing((ncol,)), config, 1.0, _DT
    )

    assert new_state.T_soil.shape[0] == ncol
    assert jnp.all(jnp.isfinite(new_state.T_soil))
    assert jnp.all(jnp.isfinite(new_state.theta_soil))
    # The multi-layer column must integrate: soil temperature and/or moisture
    # advance under the forcing rather than returning the initial column.
    advanced = _changed(new_state.T_soil, state.T_soil) or _changed(
        new_state.theta_soil, state.theta_soil
    )
    assert advanced, "multilayer land left both T_soil and theta_soil unchanged"
    _assert_response_ok(resp, (ncol,))
