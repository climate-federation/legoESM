"""Conservation budget diagnostic for the ocean model.

Computes the actual change in volume, heat, and salt between two states,
and optionally compares against the expected change from external forcing.
Reports the residual (actual - expected), which should be zero for a
perfectly conservative scheme.

This is a **diagnostic** — it does not modify the state.  Use it to
validate conservation in forced/coupled runs without active intervention.

Works with any ocean state that has eta, T, S, H_bathy, land_mask fields,
plus an area array from the grid.  Grid-agnostic: supports cubed-sphere,
latlon, and MPAS.

Example
-------
>>> budget = conservation_budget(state_new, state_old, area, z_coord)
>>> print(f"Volume residual: {budget.volume_residual:.3e} m^3")
>>> print(f"Heat residual:   {budget.heat_residual:.3e} degC*m^3")
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.core.precision import cast
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness

_M = "ocean_diagnostics"


class ConservationBudget(NamedTuple):
    """Conservation budget for a single timestep.

    All quantities are global integrals (area- or volume-weighted sums).
    Units assume eta in [m], T in [degC], S in [PSU], area in [m^2].

    Fields
    ------
    volume_old : float
        sum(eta_old * area) [m^3]
    volume_new : float
        sum(eta_new * area) [m^3]
    volume_change : float
        volume_new - volume_old [m^3]
    volume_forcing : float
        Expected volume change from external forcing [m^3].
        Zero if no forcing provided.
    volume_residual : float
        volume_change - volume_forcing [m^3].
        Should be ~0 for a conservative scheme.
    heat_old : float
        sum(T_old * h_k_old * area) [degC * m^3]
    heat_new : float
        sum(T_new * h_k_new * area) [degC * m^3]
    heat_change : float
        heat_new - heat_old [degC * m^3]
    heat_forcing : float
        Expected heat change from external forcing [degC * m^3].
    heat_residual : float
        heat_change - heat_forcing [degC * m^3].
    salt_old : float
        sum(S_old * h_k_old * area) [PSU * m^3]
    salt_new : float
        sum(S_new * h_k_new * area) [PSU * m^3]
    salt_change : float
    salt_forcing : float
    salt_residual : float
    """
    volume_old: float = 0.0
    volume_new: float = 0.0
    volume_change: float = 0.0
    volume_forcing: float = 0.0
    volume_residual: float = 0.0
    heat_old: float = 0.0
    heat_new: float = 0.0
    heat_change: float = 0.0
    heat_forcing: float = 0.0
    heat_residual: float = 0.0
    salt_old: float = 0.0
    salt_new: float = 0.0
    salt_change: float = 0.0
    salt_forcing: float = 0.0
    salt_residual: float = 0.0


def conservation_budget(
    state_new,
    state_old,
    area: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    min_water_column_m: float = 0.5,
    volume_forcing: float = 0.0,
    heat_forcing: float = 0.0,
    salt_forcing: float = 0.0,
) -> ConservationBudget:
    """Compute the conservation budget between two ocean states.

    Parameters
    ----------
    state_new : OceanState or similar
        State after the timestep.  Must have .eta, .T, .S, .H_bathy,
        .land_mask fields with .data attributes.
    state_old : OceanState or similar
        State before the timestep.
    area : jnp.ndarray
        Cell area array.  Shape must broadcast with eta.
        For cubed-sphere: grid.area (6, n, n).
        For latlon: grid.area (n_lat, n_lon).
        For MPAS: mesh.areaCell (nCells,).
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    min_water_column_m : float
        Minimum water column for layer thickness computation.
    volume_forcing : float
        Expected total volume change from external forcing [m^3].
        Pass sum(freshwater_flux * dt * area) for forced runs.
    heat_forcing : float
        Expected total heat change [degC * m^3].
        Pass sum(Q_net / (rho * cp) * dt * area) for forced runs.
    salt_forcing : float
        Expected total salt change [PSU * m^3].

    Returns
    -------
    ConservationBudget
    """
    mask_old = state_old.land_mask.data
    mask_new = state_new.land_mask.data

    # Upcast for precise global sums
    area_acc = cast(area, _M, "accumulate")

    # --- Volume ---
    eta_old_acc = cast(state_old.eta.data, _M, "accumulate")
    eta_new_acc = cast(state_new.eta.data, _M, "accumulate")
    mask_old_acc = cast(mask_old, _M, "accumulate")
    mask_new_acc = cast(mask_new, _M, "accumulate")

    vol_old = float(jnp.sum(eta_old_acc * mask_old_acc * area_acc))
    vol_new = float(jnp.sum(eta_new_acc * mask_new_acc * area_acc))
    vol_change = vol_new - vol_old

    # --- Layer thicknesses ---
    h_k_old = compute_layer_thickness(
        state_old.eta.data, state_old.H_bathy.data, z_coord,
        min_water_column_m=min_water_column_m,
    )
    h_k_new = compute_layer_thickness(
        state_new.eta.data, state_new.H_bathy.data, z_coord,
        min_water_column_m=min_water_column_m,
    )

    # Broadcast area and mask to 3D
    # Handle different grid shapes: (6,n,n), (nlat,nlon), (nCells,)
    if area.ndim == 1:
        # MPAS: (nCells,) -> (nCells, 1)
        area_3d = area_acc[:, jnp.newaxis]
        mask_old_3d = mask_old_acc[:, jnp.newaxis]
        mask_new_3d = mask_new_acc[:, jnp.newaxis]
    else:
        # Cubed-sphere or latlon: (..., ) -> (..., 1)
        area_3d = area_acc[..., jnp.newaxis]
        mask_old_3d = mask_old_acc[..., jnp.newaxis]
        mask_new_3d = mask_new_acc[..., jnp.newaxis]

    h_k_old_acc = cast(h_k_old, _M, "accumulate")
    h_k_new_acc = cast(h_k_new, _M, "accumulate")

    # --- Heat ---
    T_old_acc = cast(state_old.T.data, _M, "accumulate")
    T_new_acc = cast(state_new.T.data, _M, "accumulate")
    ht_old = float(jnp.sum(T_old_acc * h_k_old_acc * mask_old_3d * area_3d))
    ht_new = float(jnp.sum(T_new_acc * h_k_new_acc * mask_new_3d * area_3d))
    ht_change = ht_new - ht_old

    # --- Salt ---
    S_old_acc = cast(state_old.S.data, _M, "accumulate")
    S_new_acc = cast(state_new.S.data, _M, "accumulate")
    st_old = float(jnp.sum(S_old_acc * h_k_old_acc * mask_old_3d * area_3d))
    st_new = float(jnp.sum(S_new_acc * h_k_new_acc * mask_new_3d * area_3d))
    st_change = st_new - st_old

    return ConservationBudget(
        volume_old=vol_old,
        volume_new=vol_new,
        volume_change=vol_change,
        volume_forcing=volume_forcing,
        volume_residual=vol_change - volume_forcing,
        heat_old=ht_old,
        heat_new=ht_new,
        heat_change=ht_change,
        heat_forcing=heat_forcing,
        heat_residual=ht_change - heat_forcing,
        salt_old=st_old,
        salt_new=st_new,
        salt_change=st_change,
        salt_forcing=salt_forcing,
        salt_residual=st_change - salt_forcing,
    )


def print_budget(budget: ConservationBudget, label: str = "") -> None:
    """Print a human-readable conservation budget summary."""
    prefix = f"[{label}] " if label else ""
    vol_rel = abs(budget.volume_residual) / max(abs(budget.volume_old), 1e-30)
    ht_rel = abs(budget.heat_residual) / max(abs(budget.heat_old), 1e-30)
    st_rel = abs(budget.salt_residual) / max(abs(budget.salt_old), 1e-30)
    print(f"{prefix}Conservation budget:")
    print(f"  Volume: change={budget.volume_change:.3e}, "
          f"forcing={budget.volume_forcing:.3e}, "
          f"residual={budget.volume_residual:.3e} "
          f"(rel={vol_rel:.3e})")
    print(f"  Heat:   change={budget.heat_change:.3e}, "
          f"forcing={budget.heat_forcing:.3e}, "
          f"residual={budget.heat_residual:.3e} "
          f"(rel={ht_rel:.3e})")
    print(f"  Salt:   change={budget.salt_change:.3e}, "
          f"forcing={budget.salt_forcing:.3e}, "
          f"residual={budget.salt_residual:.3e} "
          f"(rel={st_rel:.3e})")
