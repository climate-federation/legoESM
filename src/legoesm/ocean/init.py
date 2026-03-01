"""Ocean initialization: bathymetry, initial conditions, test cases."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import OceanState


def idealized_bathymetry(
    grid: CubedSphereGrid,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate idealized bathymetry and land mask.

    Land is placed at high latitudes (|lat| > threshold).
    Ocean has uniform depth H_max elsewhere.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude [degrees] above which cells are land.

    Returns
    -------
    H_bathy : array
        Bathymetry depth [m], shape (6, n, n). Positive for ocean, 1.0 for land.
    land_mask : array
        Ocean mask [0/1], shape (6, n, n). 1=ocean, 0=land.
    """
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if land_lat_threshold < 0.0 or land_lat_threshold > 90.0:
        raise ValueError(
            "land_lat_threshold must be in [0, 90] degrees, "
            f"got {land_lat_threshold!r}",
        )

    lat_deg = jnp.abs(grid.lat) * (180.0 / jnp.pi)
    land_mask = jnp.where(lat_deg < land_lat_threshold, 1.0, 0.0).astype(jnp.float32)

    # H_bathy = H_max everywhere (including land) so that the Jacobian
    # (eta + H_bathy) / H_max is smooth across coastlines.  The land_mask
    # prevents any actual flow on land cells.
    H_bathy = jnp.full_like(land_mask, H_max, dtype=jnp.float32)

    return H_bathy, land_mask


def rest_state_ocean(
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    T_surface: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
) -> OceanState:
    """Create a rest-state initial condition with stratification.

    Temperature: exponential profile T(z) = T_deep + (T_surface - T_deep) * exp(z/scale)
    Salinity: uniform.
    Velocity: zero.
    Eta: zero.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    T_surface : float
        Surface temperature [degC].
    T_deep : float
        Deep ocean temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude threshold for land mask [degrees].

    Returns
    -------
    OceanState : Rest state with stratification.
    """
    n = grid.n
    nlev = z_coord.n_levels

    # Bathymetry and land mask
    H_bathy, land_mask = idealized_bathymetry(grid, H_max, land_lat_threshold)

    # Temperature: exponential stratification
    # z_full_ref is negative, scale depth = 1000m
    scale_depth = 1000.0
    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(z_coord.z_full_ref / scale_depth)
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
        (6, n, n, nlev),
    ).astype(jnp.float32)
    # Note: T is NOT zeroed on land. Keeping the same profile on land
    # ensures smooth gradients at coastlines, preventing pressure gradient
    # errors. Land tendencies are zeroed by the mask in the dynamics.

    # Salinity: uniform (same on land and ocean for smooth gradients)
    S_3d = jnp.full((6, n, n, nlev), S_uniform, dtype=jnp.float32)

    # Zero velocity
    zeros_3d = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    zeros_2d = jnp.zeros((6, n, n), dtype=jnp.float32)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return OceanState(
        u=Field(data=zeros_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=zeros_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
    )


def wind_driven_gyre_init(
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    tau_max: float = 0.1,
    H_max: float = 5500.0,
) -> OceanState:
    """Create initial condition for a wind-driven double-gyre test.

    Starts from rest with stratification. Wind forcing is applied
    externally as a tendency (not part of the initial condition).

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    tau_max : float
        Maximum wind stress [N/m^2] (not stored in state, for reference).
    H_max : float
        Maximum ocean depth [m].

    Returns
    -------
    OceanState : Initial condition for double-gyre experiment.
    """
    return rest_state_ocean(grid, z_coord, H_max=H_max)
