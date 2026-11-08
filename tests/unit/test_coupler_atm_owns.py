"""Coupler mode where the atmosphere owns land + open-water fluxes (P5c K1)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.field import Field
from legoesm.coupler.config import CouplerConfig, TileConfig
from legoesm.coupler.coupler import (
    OwnedTileForcing, SurfaceHandback, init_surface_state, make_coupler,
)
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.coupler.lake.two_layer_lake import lake_surface_albedo, step_lake
from legoesm.coupler.tile_fractions import compute_tile_fractions
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.sea_ice import slab_surface_albedo, step_sea_ice
from legoesm.land.config import LandConfig

SHAPE = (6, 2, 2)
DT = 1800.0
DIMS = ("face", "x", "y")
ICE_CFG = SeaIceConfig(temp_dependent_albedo=True)
LAKE_CFG = LakeConfig()


def _f(vals):
    return jnp.asarray(vals, dtype=jnp.float64).reshape(SHAPE)


def _forcing(sw=250.0, lw=290.0):
    z = jnp.zeros(SHAPE)
    one = jnp.ones(SHAPE)
    return AtmToSurface(
        sw_down=sw * one, lw_down=lw * one, precip_total=1e-5 * one,
        precip_snow=z, T_lowest=268.0 * one, q_lowest=2e-3 * one,
        u_lowest=6.0 * one, v_lowest=-2.0 * one, p_lowest=95000.0 * one,
        p_surface=1e5 * one, rho_lowest=1.25 * one, cos_zenith=0.5 * one,
        co2_ppmv=jnp.array(400.0), has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0))


def _state():
    s = init_surface_state(SHAPE)
    n = SHAPE[0] * SHAPE[1] * SHAPE[2]
    # Cells span: no ice (h=0), thin cold ice, ice near melt (albedo ramp);
    # frozen and liquid lakes.
    h = _f([0.0, 0.3, 1.2, 2.0] * (n // 4))
    T_ice = _f([271.0, 255.0, 272.5, 265.0] * (n // 4))
    conc = _f([0.0, 0.4, 0.9, 0.7] * (n // 4))
    T_epi = _f([280.0, 272.0, 274.0, 270.0] * (n // 4))
    ice = s.ice._replace(
        h_ice=Field(h, name="h_ice", dims=DIMS, units="m"),
        T_ice=Field(T_ice, name="T_ice", dims=DIMS, units="K"),
        concentration=Field(conc, name="ice_conc", dims=DIMS, units="1"))
    lake = LakeState(
        T_epi=Field(T_epi, name="T_epi", dims=DIMS, units="K"),
        T_hypo=Field(jnp.full(SHAPE, 277.0), name="T_hypo", dims=DIMS,
                     units="K"))
    return s._replace(ice=ice, lake=lake)


TILES = TileConfig(f_land=jnp.full(SHAPE, 0.2), f_lake=jnp.full(SHAPE, 0.1))
SST = jnp.full(SHAPE, 271.5)
ZU = jnp.zeros(SHAPE)


def _owned_coupler():
    return make_coupler(CouplerConfig(), LandConfig(), ICE_CFG, LAKE_CFG,
                        atm_owns=("land", "ocean"))


def _booked_from(state, sw, lw):
    """Absorption the tiles would book from gross downward fluxes sw, lw."""
    a_i = slab_surface_albedo(state.ice.h_ice.data, state.ice.T_ice.data, ICE_CFG)
    a_k = lake_surface_albedo(state.lake.T_epi.data, LAKE_CFG)
    return OwnedTileForcing(
        sw_net_ice=(1.0 - a_i) * sw, lw_abs_ice=ICE_CFG.emissivity_ice * lw,
        sw_net_lake=(1.0 - a_k) * sw, lw_abs_lake=LAKE_CFG.emissivity_lake * lw)


def test_tiles_absorb_exactly_the_booked_radiation():
    """Booked absorption inverted with the tiles' own start albedo/eps must
    step ice and lake exactly as the gross fluxes that produced it, and the
    gross fluxes in the atmosphere forcing must not reach the tiles."""
    s0 = _state()
    sw, lw = 250.0, 290.0
    forcing = _forcing(sw, lw)
    s1, _ = _owned_coupler()(s0, _forcing(17.0, 3.0), TILES, SST, ZU, ZU, DT,
                             owned_forcing=_booked_from(s0, sw, lw))
    ice_ref, _ = step_sea_ice(s0.ice, forcing, SST, ZU, ZU, ICE_CFG,
                              CouplerConfig().U_min, DT)
    lake_ref, _ = step_lake(s0.lake, forcing, LAKE_CFG, CouplerConfig().U_min, DT)
    for got, ref in ((s1.ice.T_ice.data, ice_ref.T_ice.data),
                     (s1.ice.h_ice.data, ice_ref.h_ice.data),
                     (s1.ice.concentration.data, ice_ref.concentration.data),
                     (s1.lake.T_epi.data, lake_ref.T_epi.data),
                     (s1.lake.T_hypo.data, lake_ref.T_hypo.data)):
        npt.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    # The test only discriminates if the albedo actually varies across cells.
    a = slab_surface_albedo(s0.ice.h_ice.data, s0.ice.T_ice.data, ICE_CFG)
    assert float(jnp.max(a) - jnp.min(a)) > 0.1
    k = lake_surface_albedo(s0.lake.T_epi.data, LAKE_CFG)
    assert float(jnp.max(k) - jnp.min(k)) > 0.1


def test_handback_fields_are_start_fraction_weighted():
    s0 = _state()
    sw, lw = 250.0, 290.0
    forcing = _forcing(sw, lw)
    s1, hb = _owned_coupler()(s0, _forcing(17.0, 3.0), TILES, SST, ZU, ZU, DT,
                              owned_forcing=_booked_from(s0, sw, lw))
    assert isinstance(hb, SurfaceHandback)
    U = CouplerConfig().U_min
    _, ir = step_sea_ice(s0.ice, forcing, SST, ZU, ZU, ICE_CFG, U, DT)
    _, kr = step_lake(s0.lake, forcing, LAKE_CFG, U, DT)
    fr = compute_tile_fractions(TILES, s0.ice.concentration.data)
    fw = fr.f_ocean + fr.f_ice
    sb = constants.sigma_sb
    T_i, T_k = s0.ice.T_ice.data, s0.lake.T_epi.data
    exp = {
        "shflx": fr.f_ice * ir.shflx + fr.f_lake * kr.shflx,
        "lhflx": fw * ir.lhflx_exchange + fr.f_lake * kr.lhflx,
        "evap": fw * ir.surface_mass_flux + fr.f_lake * kr.surface_mass_flux,
        "tau_x": fr.f_ice * ir.tau_x + fr.f_lake * kr.tau_x,
        "tau_y": fr.f_ice * ir.tau_y + fr.f_lake * kr.tau_y,
        "lw_emit": (fr.f_ice * ICE_CFG.emissivity_ice * sb * T_i ** 4
                    + fr.f_lake * LAKE_CFG.emissivity_lake * sb * T_k ** 4),
    }
    for name, val in exp.items():
        npt.assert_allclose(getattr(hb, name), val, rtol=1e-12, atol=1e-15,
                            err_msg=name)
    # Ice evaporation must not be scaled by the ice fraction a second time.
    assert float(jnp.max(jnp.abs(ir.surface_mass_flux))) > 0.0
    fr1 = compute_tile_fractions(TILES, s1.ice.concentration.data)
    npt.assert_allclose(hb.f_ice, fr1.f_ice, rtol=0, atol=0)
    npt.assert_allclose(hb.f_lake, fr1.f_lake, rtol=0, atol=0)
    npt.assert_allclose(
        hb.albedo_ice,
        slab_surface_albedo(s1.ice.h_ice.data, s1.ice.T_ice.data, ICE_CFG),
        rtol=0, atol=0)
    npt.assert_allclose(hb.albedo_lake,
                        lake_surface_albedo(s1.lake.T_epi.data, LAKE_CFG),
                        rtol=0, atol=0)


def test_owned_mode_leaves_land_and_accumulator_untouched():
    s0 = _state()
    s1, _ = _owned_coupler()(s0, _forcing(), TILES, SST, ZU, ZU, DT,
                             owned_forcing=_booked_from(s0, 250.0, 290.0))
    for a, b in zip(jax.tree.leaves((s0.land, s0.accumulator, s0.carbon)),
                    jax.tree.leaves((s1.land, s1.accumulator, s1.carbon))):
        assert a is b


def test_double_count_and_mode_guards_raise():
    s0 = _state()
    owned = _owned_coupler()
    with pytest.raises(ValueError, match="count them twice"):
        owned(s0, _forcing(), TILES, SST, ZU, ZU, DT)
    full = make_coupler(CouplerConfig(), LandConfig(), ICE_CFG, LAKE_CFG)
    with pytest.raises(ValueError, match="owns every tile"):
        full(s0, _forcing(), TILES, SST, ZU, ZU, DT,
             owned_forcing=_booked_from(s0, 250.0, 290.0))
    with pytest.raises(ValueError, match="atm_owns must be"):
        make_coupler(CouplerConfig(), LandConfig(), ICE_CFG, LAKE_CFG,
                     atm_owns=("land",))
    with pytest.raises(ValueError, match="single-category slab"):
        make_coupler(CouplerConfig(), LandConfig(),
                     SeaIceConfig(n_categories=5), LAKE_CFG,
                     atm_owns=("land", "ocean"))


def test_handback_is_differentiable_in_booked_radiation():
    s0 = _state()
    step = _owned_coupler()

    def loss(sw):
        s1, hb = step(s0, _forcing(), TILES, SST, ZU, ZU, DT,
                      owned_forcing=_booked_from(s0, sw, 290.0))
        return (jnp.sum(s1.lake.T_epi.data) + jnp.sum(s1.ice.T_ice.data)
                + jnp.sum(hb.albedo_lake))

    g = jax.grad(loss)(250.0)
    eps = 1e-3
    fd = (loss(250.0 + eps) - loss(250.0 - eps)) / (2 * eps)
    assert jnp.isfinite(g) and abs(float(g)) > 1e-6
    npt.assert_allclose(g, fd, rtol=1e-5, atol=1e-10)


def test_owned_coupler_builds_with_traced_parameters():
    """A trained (traced) emissivity must flow through the owned step."""
    s0 = _state()

    def loss(e_ice):
        cfg = ICE_CFG._replace(emissivity_ice=e_ice)
        step = make_coupler(CouplerConfig(), LandConfig(), cfg, LAKE_CFG,
                            atm_owns=("land", "ocean"))
        _, hb = step(s0, _forcing(), TILES, SST, ZU, ZU, DT,
                     owned_forcing=_booked_from(s0, 250.0, 290.0))
        return jnp.sum(hb.lw_emit)

    g = jax.grad(loss)(ICE_CFG.emissivity_ice)
    assert jnp.isfinite(g) and abs(float(g)) > 0.0
