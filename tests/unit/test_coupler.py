"""Tests for the surface coupler and all tile models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.coupler.config import CouplerConfig, TileConfig
from legoesm.coupler.coupling_fields import AtmToSurface, SurfaceToAtm, TileResponse
from legoesm.coupler.tile_fractions import (
    TileFractions,
    blend_tiles,
    compute_tile_fractions,
)
from legoesm.coupler.accumulator import (
    FluxAccumulator,
    accumulate,
    mean_accumulator,
    reset_accumulator,
)
from legoesm.coupler.coupler import (
    SurfaceState,
    init_surface_state,
    make_coupler,
    ocean_tile_response,
)
from legoesm.land.config import LandConfig
from legoesm.land.state import LandState
from legoesm.land.slab_land import step_land
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.coupler.lake.two_layer_lake import step_lake


# ---- Helpers ----

SHAPE = (6, 4, 4)
DT = 600.0
DIMS_2D = ("face", "x", "y")


def _make_forcing(shape=SHAPE, T_lowest=280.0, sw=200.0, lw=300.0,
                  precip=1e-5):
    """Create a realistic AtmToSurface forcing."""
    z = jnp.zeros(shape)
    return AtmToSurface(
        sw_down=jnp.full(shape, sw),
        lw_down=jnp.full(shape, lw),
        precip_total=jnp.full(shape, precip),
        precip_snow=z,
        T_lowest=jnp.full(shape, T_lowest),
        q_lowest=jnp.full(shape, 5e-3),
        u_lowest=jnp.full(shape, 5.0),
        v_lowest=jnp.full(shape, -3.0),
        p_lowest=jnp.full(shape, 95000.0),
        p_surface=jnp.full(shape, 1e5),
        rho_lowest=jnp.full(shape, 1.15),
        cos_zenith=jnp.full(shape, 0.6),
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )


def _make_land_state(shape=SHAPE, T=280.0, W=75.0):
    return LandState(
        T_soil=Field(jnp.full(shape, T), name="T_soil", dims=DIMS_2D, units="K"),
        W_bucket=Field(jnp.full(shape, W), name="W_bucket", dims=DIMS_2D, units="kg/m2"),
        snow_depth=Field(jnp.zeros(shape), name="snow_depth", dims=DIMS_2D, units="kg/m2"),
        snow_age=Field(jnp.zeros(shape), name="snow_age", dims=DIMS_2D, units="s"),
    )


def _make_ice_state(shape=SHAPE, h=0.5, T=260.0, conc=0.5):
    return SeaIceState(
        h_ice=Field(jnp.full(shape, h), name="h_ice", dims=DIMS_2D, units="m"),
        T_ice=Field(jnp.full(shape, T), name="T_ice", dims=DIMS_2D, units="K"),
        concentration=Field(jnp.full(shape, conc), name="ice_conc", dims=DIMS_2D, units="1"),
    )


def _make_lake_state(shape=SHAPE, T_epi=285.0, T_hypo=278.0):
    return LakeState(
        T_epi=Field(jnp.full(shape, T_epi), name="T_epi", dims=DIMS_2D, units="K"),
        T_hypo=Field(jnp.full(shape, T_hypo), name="T_hypo", dims=DIMS_2D, units="K"),
    )


# ==============================================================================
# Test coupling fields
# ==============================================================================

def test_coupling_fields_shapes():
    """AtmToSurface and SurfaceToAtm have correct number of fields."""
    forcing = _make_forcing()
    assert len(forcing) == 15
    assert forcing.sw_down.shape == SHAPE

    z = jnp.zeros(SHAPE)
    sfc = SurfaceToAtm(z, z, z, z, z, z, z, z, z, z, z, z, z)
    assert len(sfc) == 13

    tile = TileResponse(z, z, z, z, z, z, z, z, z, z, z, z, z)
    assert len(tile) == 13


# ==============================================================================
# Test land
# ==============================================================================

def test_land_step_finite():
    """Land model produces finite outputs."""
    state = _make_land_state()
    forcing = _make_forcing()
    config = LandConfig()

    new_state, resp = step_land(state, forcing, config, U_min=1.0, dt=DT)

    assert jnp.all(jnp.isfinite(new_state.T_soil.data))
    assert jnp.all(jnp.isfinite(new_state.W_bucket.data))
    assert jnp.all(jnp.isfinite(resp.shflx))
    assert jnp.all(jnp.isfinite(resp.lhflx))
    assert jnp.all(jnp.isfinite(resp.lw_up))


def test_land_energy_balance_sign():
    """When surface is warmer than air, SH is positive (upward)."""
    state = _make_land_state(T=300.0)  # Warm soil
    forcing = _make_forcing(T_lowest=270.0)  # Cold air
    config = LandConfig()

    _, resp = step_land(state, forcing, config, U_min=1.0, dt=DT)
    assert jnp.all(resp.shflx > 0)


def test_land_bucket_limits():
    """Bucket moisture stays in [0, W_max]."""
    config = LandConfig()

    # Saturated bucket with zero precip
    state = _make_land_state(W=config.W_max)
    forcing = _make_forcing(precip=0.0)
    new_state, _ = step_land(state, forcing, config, U_min=1.0, dt=DT)
    assert jnp.all(new_state.W_bucket.data <= config.W_max + 1e-6)

    # Empty bucket
    state = _make_land_state(W=0.0)
    new_state, _ = step_land(state, forcing, config, U_min=1.0, dt=DT)
    assert jnp.all(new_state.W_bucket.data >= -1e-6)


# ==============================================================================
# Test sea ice
# ==============================================================================

def test_sea_ice_step_finite():
    """Sea ice model produces finite outputs."""
    state = _make_ice_state()
    forcing = _make_forcing()
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, 275.0)
    ocean_u = jnp.zeros(SHAPE)
    ocean_v = jnp.zeros(SHAPE)

    new_state, resp = step_sea_ice(
        state, forcing, ocean_sst, ocean_u, ocean_v, config, U_min=1.0, dt=DT)

    assert jnp.all(jnp.isfinite(new_state.h_ice.data))
    assert jnp.all(jnp.isfinite(new_state.T_ice.data))
    assert jnp.all(jnp.isfinite(new_state.concentration.data))
    assert jnp.all(jnp.isfinite(resp.shflx))


def test_sea_ice_thickness_nonneg():
    """Ice thickness stays non-negative."""
    state = _make_ice_state(h=0.01)  # Thin ice
    forcing = _make_forcing(T_lowest=280.0, sw=400.0)  # Warm forcing
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, 280.0)  # Warm ocean melts from below

    new_state, _ = step_sea_ice(
        state, forcing, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT)

    assert jnp.all(new_state.h_ice.data >= 0.0)


def test_sea_ice_concentration_bounds():
    """Ice concentration stays in [0, 1]."""
    state = _make_ice_state(conc=0.99, h=2.0)
    forcing = _make_forcing()
    config = SeaIceConfig()

    new_state, _ = step_sea_ice(
        state, forcing, jnp.full(SHAPE, 270.0),
        jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT)

    assert jnp.all(new_state.concentration.data >= -1e-6)
    assert jnp.all(new_state.concentration.data <= 1.0 + 1e-6)


def test_sea_ice_no_spurious_growth_from_open_water():
    """Cells without ice should not grow ice under net warming conditions."""
    state = _make_ice_state(h=0.0, conc=0.0, T=271.35)
    forcing = _make_forcing(T_lowest=280.0, sw=300.0, lw=350.0)
    config = SeaIceConfig()
    ocean_sst = jnp.full(SHAPE, 275.0)

    new_state, _ = step_sea_ice(
        state, forcing, ocean_sst, jnp.zeros(SHAPE), jnp.zeros(SHAPE),
        config, U_min=1.0, dt=DT)

    assert jnp.allclose(new_state.h_ice.data, 0.0, atol=1e-8)
    assert jnp.allclose(new_state.concentration.data, 0.0, atol=1e-8)


# ==============================================================================
# Test lake
# ==============================================================================

def test_lake_step_finite():
    """Lake model produces finite outputs."""
    state = _make_lake_state()
    forcing = _make_forcing()
    config = LakeConfig()

    new_state, resp = step_lake(state, forcing, config, U_min=1.0, dt=DT)

    assert jnp.all(jnp.isfinite(new_state.T_epi.data))
    assert jnp.all(jnp.isfinite(new_state.T_hypo.data))
    assert jnp.all(jnp.isfinite(resp.shflx))


def test_lake_mixing_direction():
    """Heat flows from epilimnion to hypolimnion when epi is warmer."""
    state = _make_lake_state(T_epi=290.0, T_hypo=275.0)
    forcing = _make_forcing(sw=0.0, lw=0.0, T_lowest=290.0)
    config = LakeConfig()

    new_state, _ = step_lake(state, forcing, config, U_min=1.0, dt=DT)

    # Hypolimnion should warm, epilimnion should cool (from mixing alone)
    assert jnp.all(new_state.T_hypo.data > 275.0)


# ==============================================================================
# Test tile blending
# ==============================================================================

def test_tile_fractions_sum_to_one():
    """Tile fractions always sum to 1."""
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.1),
    )
    fracs = compute_tile_fractions(tile_cfg, ice_concentration=jnp.full(SHAPE, 0.2))

    total = fracs.f_ocean + fracs.f_ice + fracs.f_land + fracs.f_lake
    assert jnp.allclose(total, 1.0, atol=1e-10)


def test_tile_fractions_sanitize_invalid_static_masks():
    """Invalid static masks are clipped/rescaled to a conservative partition."""
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 1.2),
        f_lake=jnp.full(SHAPE, 0.7),
    )
    fracs = compute_tile_fractions(tile_cfg, ice_concentration=jnp.full(SHAPE, 1.5))

    for frac in (fracs.f_ocean, fracs.f_ice, fracs.f_land, fracs.f_lake):
        assert jnp.all(frac >= -1e-12)
        assert jnp.all(frac <= 1.0 + 1e-12)

    total = fracs.f_ocean + fracs.f_ice + fracs.f_land + fracs.f_lake
    assert jnp.allclose(total, 1.0, atol=1e-10)


def test_blending_is_area_weighted():
    """Blended T_surface is correct area-weighted average."""
    z = jnp.zeros(SHAPE)
    one = jnp.ones(SHAPE)

    def _make_tile_resp(T):
        return TileResponse(
            T_surface=jnp.full(SHAPE, T),
            albedo=z, emissivity=z, z0=z, q_surface=z,
            shflx=z, lhflx=z, tau_x=z, tau_y=z, lw_up=z,
            u_ocean_sfc=z, v_ocean_sfc=z, co2_flux=z,
        )

    fracs = TileFractions(
        f_ocean=jnp.full(SHAPE, 0.5),
        f_ice=jnp.full(SHAPE, 0.1),
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.1),
    )

    blended = blend_tiles(
        _make_tile_resp(300.0),
        _make_tile_resp(260.0),
        _make_tile_resp(280.0),
        _make_tile_resp(285.0),
        fracs,
    )

    expected = 0.5 * 300 + 0.1 * 260 + 0.3 * 280 + 0.1 * 285
    assert jnp.allclose(blended.T_surface, expected, atol=1e-6)


# ==============================================================================
# Test accumulator
# ==============================================================================

def test_accumulator_reset():
    """Reset accumulator has zero total_dt."""
    acc = reset_accumulator(SHAPE)
    assert acc.total_dt == 0.0
    assert acc.sum_shflx.shape == SHAPE


def test_accumulator_mean():
    """Accumulating two steps gives correct dt-weighted mean."""
    z = jnp.zeros(SHAPE)
    acc = reset_accumulator(SHAPE)

    # Step 1: shflx=10, dt=100
    sfc1 = SurfaceToAtm(
        T_surface=jnp.full(SHAPE, 280.0),
        albedo=z, emissivity=z, z0=z, q_surface=z,
        shflx=jnp.full(SHAPE, 10.0), lhflx=z,
        tau_x=z, tau_y=z, lw_up=z,
        u_ocean_sfc=z, v_ocean_sfc=z, co2_flux=z,
    )
    acc = accumulate(acc, sfc1, 100.0)

    # Step 2: shflx=30, dt=300
    sfc2 = sfc1._replace(shflx=jnp.full(SHAPE, 30.0))
    acc = accumulate(acc, sfc2, 300.0)

    result = mean_accumulator(acc)

    # Expected: (10*100 + 30*300) / 400 = 10000/400 = 25
    assert jnp.allclose(result.shflx, 25.0, atol=1e-6)
    assert jnp.allclose(result.T_surface, 280.0, atol=1e-6)


# ==============================================================================
# Test full coupler
# ==============================================================================

def test_full_coupler_step():
    """End-to-end: step all tiles and get blended output."""
    coupler_cfg = CouplerConfig()
    land_cfg = LandConfig()
    ice_cfg = SeaIceConfig()
    lake_cfg = LakeConfig()

    step_fn = make_coupler(coupler_cfg, land_cfg, ice_cfg, lake_cfg)

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()

    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.05),
    )

    ocean_sst = jnp.full(SHAPE, 300.0)
    ocean_u = jnp.zeros(SHAPE)
    ocean_v = jnp.zeros(SHAPE)

    new_state, blended = step_fn(
        sfc_state, forcing, tile_cfg, ocean_sst, ocean_u, ocean_v, DT)

    # All outputs finite
    assert jnp.all(jnp.isfinite(blended.T_surface))
    assert jnp.all(jnp.isfinite(blended.shflx))
    assert jnp.all(jnp.isfinite(blended.lhflx))
    assert jnp.all(jnp.isfinite(blended.tau_x))
    assert jnp.all(jnp.isfinite(blended.lw_up))

    # Accumulator advanced
    assert new_state.accumulator.total_dt > 0.0


def test_coupler_multiple_steps():
    """Multiple coupler steps stay finite and accumulator grows."""
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.4),
        f_lake=jnp.full(SHAPE, 0.0),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    for _ in range(5):
        sfc_state, blended = step_fn(
            sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)

    assert jnp.all(jnp.isfinite(blended.T_surface))
    assert jnp.allclose(sfc_state.accumulator.total_dt, 5 * DT, atol=1e-6)


def test_coupler_flushes_at_coupling_interval():
    """Accumulator should flush when coupling_dt is reached."""
    coupler_cfg = CouplerConfig(coupling_dt=2 * DT)
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.4),
        f_lake=jnp.full(SHAPE, 0.0),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    sfc_state, _ = step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)
    assert jnp.allclose(sfc_state.accumulator.total_dt, DT, atol=1e-6)

    sfc_state, _ = step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)
    assert jnp.allclose(sfc_state.accumulator.total_dt, 0.0, atol=1e-6)


def test_coupler_carries_excess_dt_after_flush():
    """If dt overshoots coupling_dt, residual dt should carry to next window."""
    coupler_cfg = CouplerConfig(coupling_dt=900.0)
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.4),
        f_lake=jnp.full(SHAPE, 0.0),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    sfc_state, _ = step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, 600.0)
    assert jnp.allclose(sfc_state.accumulator.total_dt, 600.0, atol=1e-6)

    sfc_state, _ = step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, 600.0)
    assert jnp.allclose(sfc_state.accumulator.total_dt, 300.0, atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coupling_dt": 0.0}, "coupling_dt"),
        ({"U_min": -1.0}, "U_min"),
        ({"ocean_albedo": -0.1}, "ocean_albedo"),
        ({"ocean_albedo": 1.1}, "ocean_albedo"),
        ({"ocean_emissivity": -0.1}, "ocean_emissivity"),
        ({"ocean_emissivity": 1.1}, "ocean_emissivity"),
        ({"ocean_z0": 0.0}, "ocean_z0"),
        ({"Cd_ocean": -1.0e-3}, "Cd_ocean"),
        ({"Ch_ocean": -1.0e-3}, "Ch_ocean"),
    ],
)
def test_make_coupler_rejects_invalid_config(kwargs, match):
    """Coupler factory should fail fast on invalid configuration."""
    with pytest.raises(ValueError, match=match):
        make_coupler(CouplerConfig(**kwargs), LandConfig(), SeaIceConfig(), LakeConfig())


def test_make_coupler_warns_for_unused_blend_sharpness():
    """Non-default blend_sharpness should surface an explicit warning."""
    with pytest.warns(RuntimeWarning, match="blend_sharpness"):
        make_coupler(
            CouplerConfig(blend_sharpness=10.0),
            LandConfig(),
            SeaIceConfig(),
            LakeConfig(),
        )


def test_coupler_step_validates_dt():
    """Coupler step should reject non-positive dt."""
    step_fn = make_coupler(CouplerConfig(), LandConfig(), SeaIceConfig(), LakeConfig())
    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.05),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    with pytest.raises(ValueError, match="dt"):
        step_fn(sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, 0.0)


# ==============================================================================
# Test smooth operations
# ==============================================================================

def test_smooth_wind_floor():
    """Wind floor sqrt(u^2 + v^2 + U_min^2) is smooth (differentiable at zero)."""
    U_min = 1.0

    def wind_speed(u, v):
        return jnp.sqrt(u ** 2 + v ** 2 + U_min ** 2)

    # At u=v=0 the gradient should be finite (not NaN)
    grad_u = jax.grad(lambda u: wind_speed(u, 0.0).sum())(jnp.array(0.0))
    grad_v = jax.grad(lambda v: wind_speed(0.0, v).sum())(jnp.array(0.0))

    assert jnp.isfinite(grad_u)
    assert jnp.isfinite(grad_v)
    # Gradient should be zero at u=v=0 (symmetric minimum)
    assert jnp.abs(grad_u) < 1e-6
    assert jnp.abs(grad_v) < 1e-6


def test_init_surface_state():
    """init_surface_state creates valid state with correct shapes."""
    state = init_surface_state(SHAPE)
    assert state.land.T_soil.shape == SHAPE
    assert state.land.W_bucket.shape == SHAPE
    assert state.ice.h_ice.shape == SHAPE
    assert state.ice.T_ice.shape == SHAPE
    assert state.lake.T_epi.shape == SHAPE
    assert state.lake.T_hypo.shape == SHAPE
    assert state.accumulator.total_dt == 0.0


# ==============================================================================
# Test simple ocean modes
# ==============================================================================

from legoesm.ocean.simple_ocean import (
    SimpleOceanConfig,
    SlabOceanState,
    make_ocean,
    init_slab_state,
)


def test_fixed_sst_constant():
    """Fixed mode with constant SST returns that constant, state unchanged."""
    cfg = SimpleOceanConfig(mode="fixed", sst_constant=305.0)
    step_fn = make_ocean(cfg)

    state = init_slab_state(SHAPE, T_sfc_init=300.0)
    forcing = _make_forcing()

    new_state, sst, u_sfc, v_sfc = step_fn(state, forcing, DT)

    assert jnp.allclose(sst, 305.0)
    # State unchanged
    assert jnp.allclose(new_state.T_sfc.data, state.T_sfc.data)
    # Surface currents are zero
    assert jnp.allclose(u_sfc, 0.0)
    assert jnp.allclose(v_sfc, 0.0)


def test_fixed_sst_map():
    """Fixed mode with spatial SST map returns that map."""
    sst_map = jnp.linspace(280.0, 310.0, SHAPE[0] * SHAPE[1] * SHAPE[2]).reshape(SHAPE)
    cfg = SimpleOceanConfig(mode="fixed")
    step_fn = make_ocean(cfg, sst_map=sst_map)

    state = init_slab_state(SHAPE)
    forcing = _make_forcing()

    new_state, sst, _, _ = step_fn(state, forcing, DT)

    assert jnp.allclose(sst, sst_map)
    assert jnp.allclose(new_state.T_sfc.data, state.T_sfc.data)


def test_slab_ocean_finite():
    """Slab ocean step produces all-finite outputs."""
    cfg = SimpleOceanConfig(mode="slab")
    step_fn = make_ocean(cfg)

    state = init_slab_state(SHAPE, T_sfc_init=300.0)
    forcing = _make_forcing()

    new_state, sst, u_sfc, v_sfc = step_fn(state, forcing, DT)

    assert jnp.all(jnp.isfinite(new_state.T_sfc.data))
    assert jnp.all(jnp.isfinite(sst))
    assert jnp.all(jnp.isfinite(u_sfc))
    assert jnp.all(jnp.isfinite(v_sfc))


def test_slab_ocean_warming():
    """Warm atmosphere heats a cold slab ocean."""
    cfg = SimpleOceanConfig(mode="slab")
    step_fn = make_ocean(cfg)

    T_init = 275.0
    state = init_slab_state(SHAPE, T_sfc_init=T_init)
    # Warm air + strong SW drives ocean warming
    forcing = _make_forcing(T_lowest=300.0, sw=400.0, lw=350.0)

    new_state, sst, _, _ = step_fn(state, forcing, DT)

    assert jnp.all(new_state.T_sfc.data > T_init)


def test_slab_ocean_freezing_clamp():
    """SST stays >= T_freeze even with extreme cooling."""
    cfg = SimpleOceanConfig(mode="slab", T_freeze=271.35)
    step_fn = make_ocean(cfg)

    state = init_slab_state(SHAPE, T_sfc_init=272.0)
    # Strong cooling: no SW, cold air
    forcing = _make_forcing(T_lowest=200.0, sw=0.0, lw=50.0)

    # Multiple steps to drive temperature down
    for _ in range(100):
        state, sst, _, _ = step_fn(state, forcing, DT)

    assert jnp.all(state.T_sfc.data >= cfg.T_freeze - 1e-6)


def test_two_layer_ocean_finite():
    """Two-layer ocean step produces all-finite outputs."""
    cfg = SimpleOceanConfig(mode="two_layer")
    step_fn = make_ocean(cfg)

    state = init_slab_state(SHAPE, T_sfc_init=300.0, T_deep_init=278.0)
    forcing = _make_forcing()

    new_state, sst, u_sfc, v_sfc = step_fn(state, forcing, DT)

    assert jnp.all(jnp.isfinite(new_state.T_sfc.data))
    assert jnp.all(jnp.isfinite(new_state.T_deep.data))
    assert jnp.all(jnp.isfinite(sst))


def test_two_layer_mixing_direction():
    """Heat flows from warm surface to cold deep layer via mixing."""
    cfg = SimpleOceanConfig(mode="two_layer", k_mix=1.0e-2)  # Strong mixing
    step_fn = make_ocean(cfg)

    T_sfc_init = 300.0
    T_deep_init = 275.0
    state = init_slab_state(SHAPE, T_sfc_init=T_sfc_init, T_deep_init=T_deep_init)
    forcing = _make_forcing(sw=0.0, lw=0.0, T_lowest=300.0)

    new_state, _, _, _ = step_fn(state, forcing, DT)

    # Deep layer should warm
    assert jnp.all(new_state.T_deep.data > T_deep_init)


def test_two_layer_deep_restoring():
    """Deep layer relaxes toward T_deep_ref when restoring is on."""
    T_deep_ref = 280.0
    cfg = SimpleOceanConfig(
        mode="two_layer",
        restore_deep=True,
        T_deep_ref=T_deep_ref,
        tau_deep=3600.0,  # Fast restoring for test
        k_mix=0.0,        # No mixing, isolate restoring
    )
    step_fn = make_ocean(cfg)

    # Start deep layer away from reference
    T_deep_init = 290.0
    state = init_slab_state(SHAPE, T_sfc_init=300.0, T_deep_init=T_deep_init)
    forcing = _make_forcing()

    new_state, _, _, _ = step_fn(state, forcing, DT)

    # Deep layer should move toward T_deep_ref (cool down)
    assert jnp.all(new_state.T_deep.data < T_deep_init)
    assert jnp.all(new_state.T_deep.data > T_deep_ref)  # Not overshoot


def test_make_ocean_factory():
    """Factory returns a callable for each valid mode."""
    for mode in ("fixed", "slab", "two_layer"):
        cfg = SimpleOceanConfig(mode=mode)
        step_fn = make_ocean(cfg)
        assert callable(step_fn)

    # Invalid mode raises
    with pytest.raises(ValueError, match="Unknown ocean mode"):
        make_ocean(SimpleOceanConfig(mode="invalid"))


# ==============================================================================
# Test differentiability
# ==============================================================================

def test_coupler_differentiable():
    """Full coupler step is differentiable w.r.t. ocean SST."""
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(SHAPE)
    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.3),
        f_lake=jnp.full(SHAPE, 0.05),
    )
    zu = jnp.zeros(SHAPE)

    def loss(ocean_sst):
        _, blended = step_fn(
            sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)
        return jnp.sum(blended.shflx)

    ocean_sst = jnp.full(SHAPE, 300.0)
    grad_sst = jax.grad(loss)(ocean_sst)
    assert grad_sst.shape == SHAPE
    assert jnp.all(jnp.isfinite(grad_sst))
    # Non-zero gradient (shflx depends on SST via bulk formula)
    assert float(jnp.max(jnp.abs(grad_sst))) > 0.0


def test_coupler_differentiable_through_surface_state():
    """Coupler step is differentiable w.r.t. land soil temperature."""
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    forcing = _make_forcing()
    tile_cfg = TileConfig(
        f_land=jnp.full(SHAPE, 0.5),
        f_lake=jnp.full(SHAPE, 0.0),
    )
    ocean_sst = jnp.full(SHAPE, 300.0)
    zu = jnp.zeros(SHAPE)

    def loss(T_soil):
        land = LandState(
            T_soil=Field(T_soil, name="T_soil", dims=DIMS_2D, units="K"),
            W_bucket=Field(jnp.full(SHAPE, 75.0), name="W_bucket",
                           dims=DIMS_2D, units="kg/m2"),
            snow_depth=Field(jnp.zeros(SHAPE), name="snow_depth",
                             dims=DIMS_2D, units="kg/m2"),
            snow_age=Field(jnp.zeros(SHAPE), name="snow_age",
                           dims=DIMS_2D, units="s"),
        )
        sfc_state = SurfaceState(
            land=land,
            ice=_make_ice_state(),
            lake=_make_lake_state(),
            accumulator=reset_accumulator(SHAPE),
        )
        _, blended = step_fn(
            sfc_state, forcing, tile_cfg, ocean_sst, zu, zu, DT)
        return jnp.sum(blended.T_surface)

    T_soil = jnp.full(SHAPE, 280.0)
    grad_T = jax.grad(loss)(T_soil)
    assert grad_T.shape == SHAPE
    assert jnp.all(jnp.isfinite(grad_T))


def test_coupler_with_3d_ocean_fc_gram():
    """Coupler receives SST from FC-Gram ocean model (integration test)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init import rest_state_ocean
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.state import OceanConfig

    grid = create_cubed_sphere(8)
    z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
    ocean_state = rest_state_ocean(
        grid, z_coord, T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )

    # Step ocean with FC-Gram
    ocean_config = OceanConfig(
        use_conservation_fixer=False,
        A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0,
    )
    ocean_model = OceanModel(
        grid, z_coord, config=ocean_config, discretization="fc_gram",
    )
    ocean_state_new = ocean_model.step(ocean_state, 60.0)

    # Extract SST (top-level T) for coupler
    sst = ocean_state_new.T.data[:, :, :, 0]  # (6, n, n)
    assert jnp.all(jnp.isfinite(sst))

    # Feed to coupler
    shape = (6, 8, 8)
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(shape)
    forcing = _make_forcing(shape=shape)
    tile_cfg = TileConfig(
        f_land=jnp.full(shape, 0.3),
        f_lake=jnp.full(shape, 0.0),
    )
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    new_sfc, blended = step_fn(
        sfc_state, forcing, tile_cfg,
        sst + 273.15,  # degC → K
        ocean_u, ocean_v, DT,
    )
    assert jnp.all(jnp.isfinite(blended.T_surface))
    assert jnp.all(jnp.isfinite(blended.shflx))


def test_coupler_with_fv_ocean_tracer_transport():
    """Coupler receives SST from FV-tracer ocean model (integration test)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init import rest_state_ocean
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.state import OceanConfig

    grid = create_cubed_sphere(8)
    z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
    ocean_state = rest_state_ocean(
        grid, z_coord, T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )

    # Step ocean with FV tracer transport
    ocean_config = OceanConfig(
        use_fv_tracer_transport=True,
        use_conservation_fixer=False,
    )
    ocean_model = OceanModel(grid, z_coord, config=ocean_config)
    ocean_state_new = ocean_model.step(ocean_state, 60.0)

    # Extract SST for coupler
    sst = ocean_state_new.T.data[:, :, :, 0]
    assert jnp.all(jnp.isfinite(sst))

    # Feed to coupler
    shape = (6, 8, 8)
    coupler_cfg = CouplerConfig()
    step_fn = make_coupler(coupler_cfg, LandConfig(), SeaIceConfig(), LakeConfig())

    sfc_state = init_surface_state(shape)
    forcing = _make_forcing(shape=shape)
    tile_cfg = TileConfig(
        f_land=jnp.full(shape, 0.3),
        f_lake=jnp.full(shape, 0.0),
    )

    new_sfc, blended = step_fn(
        sfc_state, forcing, tile_cfg,
        sst + 273.15,
        jnp.zeros(shape), jnp.zeros(shape), DT,
    )
    assert jnp.all(jnp.isfinite(blended.T_surface))
    assert jnp.all(jnp.isfinite(blended.shflx))
