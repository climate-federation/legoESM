"""Integration test for the CATKE vertical-mixing scheme (Wagner et al. 2025).

Verifies that ``vertical_mixing scheme="catke"`` flows through:

1. ``make_vertical_mixing_physics`` accepts the literal (implicit-only).
2. ``compute_vertical_K_profiles`` with scheme="catke" returns finite,
   non-negative K_v / A_v + an advanced prognostic TKE on a non-trivial state
   (catke is ALWAYS prognostic -> needs dt_tke).
3. A ``LatLonCGridOceanModel`` builds with CATKE + implicit vertical mixing and
   ``model.step`` runs to completion without NaN, advancing ``state.tke``.

Mirrors ``test_tke_integration.py``; end-to-end realism is the recipe-acceptance
harness's job — this verifies the dispatch wiring.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    CATKEConfig, VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.integration import (
    make_vertical_mixing_physics,
)
from legoesm.ocean.physics.vertical_mixing.k_profiles import (
    compute_vertical_K_profiles,
)
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star

jax.config.update("jax_enable_x64", True)


def _perturbed_cc_state(grid, z_coord):
    s = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    rng = np.random.default_rng(0)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    # Cell-centre velocities (compute_vertical_K_profiles expects centres).
    u = 0.1 * rng.standard_normal((n_lat, n_lon, nlev))
    v = 0.1 * rng.standard_normal((n_lat, n_lon, nlev))
    T_prof = 5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))
    T = T_prof[None, None, :] + 0.1 * rng.standard_normal((n_lat, n_lon, nlev))
    return s._replace(
        u=s.u.replace(data=jnp.asarray(u)),
        v=s.v.replace(data=jnp.asarray(v)),
        T=s.T.replace(data=jnp.asarray(T)),
    )


def _catke_physics():
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="catke", catke=CATKEConfig()),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def test_catke_dispatch_registered():
    cfg = VerticalMixingConfig(scheme="catke", catke=CATKEConfig())
    fn = make_vertical_mixing_physics(cfg, apply_diffusion=False)
    assert callable(fn)


def test_catke_explicit_mode_raises():
    cfg = VerticalMixingConfig(scheme="catke", catke=CATKEConfig())
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        make_vertical_mixing_physics(cfg, apply_diffusion=True)


def test_compute_K_profiles_catke_branch():
    """compute_vertical_K_profiles scheme='catke' returns finite, non-negative
    K_v / A_v + an advanced prognostic TKE (catke needs dt_tke)."""
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
    cc_state = _perturbed_cc_state(grid, z_coord)
    K_v, A_v, tke_new = compute_vertical_K_profiles(
        cc_state, z_coord, surface_forcing=None, physics_config=_catke_physics(),
        A_v_background=1.0e-6, K_v_background=1.0e-7,
        tke_old=None, dt_tke=3600.0, return_tke=True,
    )
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    assert K_v.shape == (n_lat, n_lon, nlev - 1)
    assert A_v.shape == (n_lat, n_lon, nlev - 1)
    assert tke_new is not None and tke_new.shape == (n_lat, n_lon, nlev - 1)
    for arr in (K_v, A_v, tke_new):
        assert jnp.all(jnp.isfinite(arr))
        assert jnp.all(arr >= 0.0)


def test_catke_branch_requires_dt_tke():
    """catke is prognostic — compute_vertical_K_profiles without dt_tke raises."""
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
    cc_state = _perturbed_cc_state(grid, z_coord)
    with pytest.raises(ValueError, match="dt_tke"):
        compute_vertical_K_profiles(
            cc_state, z_coord, surface_forcing=None,
            physics_config=_catke_physics(),
            A_v_background=1.0e-6, K_v_background=1.0e-7,
            return_tke=True,
        )


def test_model_construction_and_step_with_catke():
    """Build a LatLonCGridOceanModel with CATKE + implicit vertical mixing and
    run one step without NaN, advancing state.tke."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
    cfg = LatLonCGridOceanConfig(
        A_h=1.0e4, implicit_vertical_mixing=True, physics=_catke_physics(),
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )
    # LatLonCGridOceanModel.step returns the new state (a NamedTuple) directly;
    # do NOT index it (a NamedTuple is a tuple -> [0] would grab its first Field).
    state_new = model.step(state, 3600.0)
    assert jnp.all(jnp.isfinite(state_new.T.data))
    assert jnp.all(jnp.isfinite(state_new.u.data))
    # CATKE carries a prognostic TKE field that advanced off the cold start.
    assert state_new.tke is not None
    assert jnp.all(jnp.isfinite(state_new.tke.data))
    assert jnp.all(state_new.tke.data >= 0.0)
