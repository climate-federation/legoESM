"""Every surface tile uses one air-temperature reference for its MOST fluxes (#1818).

Model-level forcing (``AtmToSurface.z_lowest`` set) is brought dry-adiabatically
to the surface, ``T + g z/c_pd``, and paired with its own height -- the
convention of the land tile and the atmosphere surface layer
(``core.bulk_flux.surface_reference_state``).  Observed forcing
(``z_lowest=None``) is used as given.

Neutral check: when the surface temperature equals the surface-referenced air
temperature, the potential-temperature difference is zero and so is the
sensible heat flux.  A tile that still passes the absolute lowest-level
temperature sees a spurious ~0.5 K contrast at 50 m and a flux of several
W/m2, so each case fails if its caller is reverted.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.field import Field


_N = 3
_T_LOW = 285.0
_Z_LOW = 50.0
_T_NEUTRAL = _T_LOW + constants.g / constants.c_pd * _Z_LOW


def _forcing(z_lowest, T_lowest=_T_LOW):
    f = lambda v: jnp.full((_N,), v, jnp.float64)  # noqa: E731
    return AtmToSurface(
        sw_down=f(200.0), lw_down=f(300.0), precip_total=f(0.0),
        precip_snow=f(0.0), T_lowest=f(T_lowest), q_lowest=f(0.006),
        u_lowest=f(6.0), v_lowest=f(2.0), p_lowest=f(99400.0),
        p_surface=f(1.0e5), rho_lowest=f(1.2), cos_zenith=f(0.5),
        co2_ppmv=f(415.0), has_radiation=f(1.0), has_precipitation=f(1.0),
        z_lowest=None if z_lowest is None else f(z_lowest),
    )


def _ocean(scheme, forcing, T_sfc, z=None):
    from legoesm.coupler.config import CouplerConfig
    from legoesm.coupler.coupler import ocean_tile_response
    zero = jnp.zeros((_N,))
    cfg = CouplerConfig(bulk_scheme=scheme)
    if z is not None:
        cfg = cfg._replace(z_ref=z, z_t_atm=z, z_q_atm=z)
    return ocean_tile_response(forcing, jnp.full((_N,), T_sfc), zero, zero,
                               cfg).shflx


def _sea_ice(forcing, T_sfc, z=None):
    from legoesm.ice.config import SeaIceConfig
    from legoesm.ice.sea_ice import _bulk_flux_dispatch
    cfg = SeaIceConfig(bulk_scheme="most")
    if z is not None:
        cfg = cfg._replace(z_ref=z)
    return _bulk_flux_dispatch(jnp.full((_N,), T_sfc), forcing, cfg, 0.5)[2]


def _lake(forcing, T_sfc, z=None):
    from legoesm.coupler.lake.config import LakeConfig
    from legoesm.coupler.lake.state import LakeState
    from legoesm.coupler.lake.two_layer_lake import step_lake
    fld = lambda v: Field(data=jnp.full((_N,), v), name="", dims=("ncol",),  # noqa: E731
                          units="K")
    cfg = LakeConfig(bulk_scheme="most")
    if z is not None:
        cfg = cfg._replace(z_ref=z)
    state = LakeState(T_epi=fld(T_sfc), T_hypo=fld(280.0))
    _, resp = step_lake(state, forcing, cfg, 0.5, 600.0)
    return resp.shflx


def _slab(forcing, T_sfc, z=None):
    from legoesm.core.bulk_flux import compute_most_fluxes
    from legoesm.ocean.simple_ocean import SimpleOceanConfig, _ocean_turbulent_fluxes
    T = jnp.full((_N,), T_sfc)
    q = jnp.full((_N,), 0.01)
    cfg = SimpleOceanConfig(bulk_scheme="coare3")
    if z is None:
        return _ocean_turbulent_fluxes(T, q, forcing, cfg)[0]
    # The slab has no height setting: the observed-forcing reference is the
    # solver itself, called with the same arguments the slab passes.
    return compute_most_fluxes(
        forcing.u_lowest, forcing.v_lowest, forcing.T_lowest, forcing.q_lowest,
        T, q, forcing.rho_lowest, z_ref=z, scheme=cfg.bulk_scheme,
        gustiness_w_zi=cfg.gustiness_w_zi,
        thermo_convention=cfg.thermo_convention)[2]


_TILES = {
    "ocean_most": lambda f, T, z=None: _ocean("most", f, T, z),
    "ocean_coare3": lambda f, T, z=None: _ocean("coare3", f, T, z),
    "ocean_large_yeager": lambda f, T, z=None: _ocean("large_yeager", f, T, z),
    "ocean_large_yeager_cesm":
        lambda f, T, z=None: _ocean("large_yeager_cesm", f, T, z),
    "sea_ice_most": _sea_ice,
    "lake_most": _lake,
    "slab_coare3": _slab,
}


@pytest.mark.parametrize("tile", sorted(_TILES))
def test_model_level_forcing_is_surface_referenced(tile):
    shflx = _TILES[tile](_forcing(_Z_LOW), _T_NEUTRAL)
    assert jnp.all(jnp.abs(shflx) < 1e-6), (
        f"{tile}: sensible heat {shflx} W/m2 at zero potential-temperature "
        "contrast -> the tile used the absolute lowest-level temperature")


@pytest.mark.parametrize("tile", sorted(_TILES))
def test_observed_forcing_is_used_as_given(tile):
    shflx = _TILES[tile](_forcing(None), _T_LOW)
    assert jnp.all(jnp.abs(shflx) < 1e-6), f"{tile}: {shflx}"


@pytest.mark.parametrize("tile", sorted(_TILES))
def test_warmer_surface_gives_upward_heat(tile):
    shflx = _TILES[tile](_forcing(_Z_LOW), _T_NEUTRAL + 2.0)
    assert jnp.all(shflx > 0.0), f"{tile}: {shflx}"


@pytest.mark.parametrize("tile", sorted(_TILES))
def test_model_level_equals_observed_forcing_at_that_height(tile):
    """Non-neutral: model-level forcing at 50 m must give exactly the flux of
    observed forcing of the surface-referenced temperature AT 50 m.  Catches a
    wrong height (z_ref / z_t / z_q / z_bot) as well as a wrong temperature."""
    T_sfc = _T_NEUTRAL + 3.0
    model = _TILES[tile](_forcing(_Z_LOW), T_sfc)
    observed = _TILES[tile](_forcing(None, T_lowest=_T_NEUTRAL), T_sfc, _Z_LOW)
    assert jnp.all(jnp.abs(model) > 1.0)
    assert jnp.allclose(model, observed, rtol=1e-10, atol=1e-10), (
        f"{tile}: model-level {model} vs observed-at-{_Z_LOW} m {observed}")
