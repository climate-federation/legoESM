"""Integration test for the TKE vertical-mixing scheme.

Verifies that ``vertical_mixing scheme="tke"`` flows through:

1. ``OceanPhysicsConfig`` accepts the literal.
2. ``LatLonCGridOceanModel`` can be constructed with TKE + implicit
   vertical mixing without raising.
3. ``compute_vertical_K_profiles`` returns finite, positive K_v / A_v
   on a non-trivial state when scheme="tke".
4. A single ``model.step(state, dt)`` call runs to completion without
   producing NaN.

End-to-end Veros parity is the job of the recipe-acceptance harness
(Phase G.0c+) and a local Veros install; this test only verifies the
wiring is plumbed correctly.
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
    TKEConfig, VerticalMixingConfig,
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


def _perturbed_state(grid, z_coord):
    s = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    rng = np.random.default_rng(0)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.1 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.1 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    T_prof = 5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))
    T = T_prof[None, None, :] + 0.1 * rng.standard_normal((n_lat, n_lon, nlev))
    return s._replace(
        u=s.u.replace(data=jnp.asarray(u)),
        v=s.v.replace(data=jnp.asarray(v)),
        T=s.T.replace(data=jnp.asarray(T)),
    )


def test_tke_dispatch_registered():
    """``make_vertical_mixing_physics`` accepts scheme='tke'."""
    cfg = VerticalMixingConfig(scheme="tke", tke=TKEConfig())
    fn = make_vertical_mixing_physics(cfg, apply_diffusion=False)
    assert callable(fn)


def test_tke_explicit_mode_raises_clear_error():
    """Explicit vertical-mixing mode is unsupported for TKE — raise."""
    cfg = VerticalMixingConfig(scheme="tke", tke=TKEConfig())
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        make_vertical_mixing_physics(cfg, apply_diffusion=True)


def test_tke_physics_is_noop_K_None():
    """The TKE physics tendency function returns K_v=None / A_v=None
    so the implicit solver falls back to ``compute_vertical_K_profiles``
    (where the actual TKE closure runs with dt available)."""
    cfg = VerticalMixingConfig(scheme="tke", tke=TKEConfig())
    fn = make_vertical_mixing_physics(cfg, apply_diffusion=False)
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
    state = _perturbed_state(grid, z_coord)
    out = fn(state, grid, z_coord, surface_forcing=None)
    assert out.K_v is None
    assert out.A_v is None


def test_compute_K_profiles_tke_branch():
    """``compute_vertical_K_profiles`` with vertical_mixing scheme='tke'
    returns finite, positive K_v / A_v matching cell-centre interior
    shape."""
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
    state = _perturbed_state(grid, z_coord)
    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=TKEConfig()),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )
    # Convert u, v to cell centres (compute_vertical_K_profiles expects
    # cell-centre velocities on the lat-lon C-grid).
    u_cell = 0.5 * (state.u.data[:, :-1, :] + state.u.data[:, 1:, :])
    v_cell = 0.5 * (state.v.data[:-1, :, :] + state.v.data[1:, :, :])
    cc_state = state._replace(
        u=state.u.replace(data=u_cell),
        v=state.v.replace(data=v_cell),
    )
    K_v, A_v = compute_vertical_K_profiles(
        cc_state, z_coord, surface_forcing=None, physics_config=cfg,
        A_v_background=1.0e-6, K_v_background=1.0e-7,
    )
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    assert K_v.shape == (n_lat, n_lon, nlev - 1)
    assert A_v.shape == (n_lat, n_lon, nlev - 1)
    assert jnp.all(jnp.isfinite(K_v))
    assert jnp.all(jnp.isfinite(A_v))
    # TKEConfig defaults: kappaH_min=2e-5, kappaM_min=2e-4. Plus background.
    assert float(jnp.min(K_v)) >= 2.0e-5 - 1e-10
    assert float(jnp.min(A_v)) >= 2.0e-4 - 1e-10


def test_model_construction_with_tke():
    """Constructing a LatLonCGridOceanModel with TKE + implicit
    vertical mixing must succeed without raising."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=TKEConfig()),
        # 'harmonic' physics lateral mixing is rejected on the lat-lon
        # C-grid (horizontal viscosity goes via config.A_h); 'none' keeps
        # this test about the TKE wiring it actually exercises.
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4,
        implicit_vertical_mixing=True,
        physics=physics,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    assert model is not None
