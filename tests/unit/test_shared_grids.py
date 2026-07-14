"""The global grids are GENUINELY shared: one grid object, built ONCE via the
uniform :func:`create_grid` factory, constructs BOTH an atmosphere dycore AND an
ocean dycore.

Confirms the design claim that lat-lon FV, cubed-sphere C-D, MPAS/Voronoi
(TRiSK), spectral Gaussian and SFNO grids are shared between the atmosphere and
the ocean — not re-implemented per component.  What each case asserts:

* cubed-sphere / MPAS — the atmosphere and ocean dycores hold the *same* grid
  object (``is`` identity); each derives its staggered C-D / edge geometry from
  that one base internally.
* lat-lon — the atmosphere holds the base grid directly; the ocean derives a
  ``LatLonCGridGeometry`` from it, so we assert the derived T-point latitudes
  reproduce the base grid's (coordinate fidelity, not just matching dimensions).
* spectral / SFNO — compared by coordinate arrays rather than ``is`` identity,
  because a non-CPU backend may device-copy the Gaussian grid into the model;
  array equality proves logical sharing regardless of that copy.

These are *construction* checks (the grid object flows into both constructors),
not end-to-end stepping; that is the load-bearing claim for "shared grid".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.factory import create_grid

_needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="Gaussian/spectral grids require JAX_ENABLE_X64=1",
)


def _assert_same_gaussian_grid(model_grid, base) -> None:
    """Logical-sharing check robust to a backend device-copy of the grid."""
    assert (model_grid.n_lat, model_grid.n_lon) == (base.n_lat, base.n_lon)
    assert jnp.allclose(jnp.asarray(model_grid.lat), jnp.asarray(base.lat))
    assert jnp.allclose(jnp.asarray(model_grid.lon), jnp.asarray(base.lon))


def test_cubed_sphere_grid_shared_by_atmosphere_and_ocean() -> None:
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        CDGridShallowWaterModel,
    )
    from legoesm.ocean import OceanModel
    from legoesm.ocean.state import OceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_grid("cubed_sphere", 8)  # built ONCE

    atm = CDGridShallowWaterModel(grid, CDGridShallowWaterConfig())
    ocn = OceanModel(grid=grid, z_coord=create_ocean_z_star(n_levels=4),
                     config=OceanConfig())

    assert hasattr(atm, "step") and hasattr(ocn, "step")
    assert atm.grid is grid          # the SAME grid object feeds the atmosphere ...
    assert ocn.grid is grid          # ... and the ocean (each derives its own C-D)


def test_latlon_grid_shared_by_atmosphere_and_ocean() -> None:
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterModel,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanConfig,
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_grid("latlon", 16)  # built ONCE, fed to BOTH constructors

    atm = CGridLatLonShallowWaterModel(grid)
    ocn = LatLonCGridOceanModel(grid, create_ocean_z_star(n_levels=4),
                                config=LatLonCGridOceanConfig.from_flat())

    assert hasattr(atm, "step") and hasattr(ocn, "step")
    # Atmosphere holds the base grid directly; the ocean derives its staggered
    # C-grid geometry from the SAME base grid.  Assert the derived T-point
    # latitudes reproduce the base grid coordinates (fidelity, not just dims).
    assert atm.grid is grid
    assert (ocn.grid.n_lat, ocn.grid.n_lon) == (grid.n_lat, grid.n_lon)
    assert jnp.allclose(jnp.asarray(ocn.grid.lat_T)[:, 0], jnp.asarray(grid.lat))


def test_mpas_mesh_shared_by_atmosphere_and_ocean() -> None:
    from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import MPASShallowWaterModel
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh = create_grid("mpas", 2)  # one Voronoi/TRiSK mesh

    atm = MPASShallowWaterModel(mesh)
    ocn = MPASOceanModel(mesh, create_ocean_z_star(n_levels=4))

    assert hasattr(atm, "step") and hasattr(ocn, "step")
    assert atm.mesh is mesh and ocn.mesh is mesh  # the SAME edge-normal mesh


@_needs_x64
def test_gaussian_grid_shared_by_spectral_atmosphere_and_ocean() -> None:
    from legoesm.atmosphere.dynamics.gcm.spectral_sw import SpectralShallowWaterModel
    from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_grid("gaussian", 32)  # one spectral/Gaussian transform grid

    atm = SpectralShallowWaterModel(grid)
    # SpectralOceanModel is reference-only (issue #99: Gibbs ringing at coasts) —
    # it constructs on the shared grid but warns it is unsupported.  Assert the
    # warning so the "shared" claim does not silently overstate ocean support.
    with pytest.warns(FutureWarning, match="unsupported"):
        ocn = SpectralOceanModel(grid, create_ocean_z_star(n_levels=4))

    assert hasattr(atm, "step") and hasattr(ocn, "step")
    _assert_same_gaussian_grid(atm.grid, grid)
    _assert_same_gaussian_grid(ocn.grid, grid)


@_needs_x64
def test_gaussian_grid_shared_by_sfno_atmosphere_and_ocean() -> None:
    from legoesm.atmosphere.dynamics.neural.sfno_pe import SFNOPrimitiveEquationModel
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.ocean.dynamics.sfno_ocean import SFNOOceanModel
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_grid("gaussian", 32)  # one Gaussian grid for both neural cores
    k_atm, k_ocn = jax.random.split(jax.random.PRNGKey(0))

    atm = SFNOPrimitiveEquationModel(grid, create_sigma_coordinate(4), key=k_atm)
    ocn = SFNOOceanModel(grid, create_ocean_z_star(n_levels=4), key=k_ocn)

    assert hasattr(atm, "step") and hasattr(ocn, "step")
    _assert_same_gaussian_grid(atm.grid, grid)
    _assert_same_gaussian_grid(ocn.grid, grid)
