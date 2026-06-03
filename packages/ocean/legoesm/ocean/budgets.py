"""Global energy + tracer budget diagnostics for the ocean dycores.

Two callables exposed:

* :func:`compute_energy_budget` -- domain-integrated kinetic energy +
  available potential energy on every supported grid (cube C-D, lat-lon
  C-grid, MPAS Voronoi). Used as the input to a closure check of the
  form ``|d(KE+APE)/dt - sources| < 1 %`` per day on steady state.

* :func:`compute_tracer_budget` -- total volume / heat content / salt
  content integrals on the same grids. The drift in these should be at
  floating-point precision for closed basins and balance applied
  surface / sponge fluxes when the model is forced.

Both helpers reuse the same area + layer-thickness extraction logic as
``rpe.compute_rpe`` so the three diagnostics report on identical cell
volumes. Returns plain Python floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np

from legoesm import constants


def _grid_area(grid_type: str, grid) -> np.ndarray:
    if grid_type in ("mpas", "mpas_regional", "mpas_channel"):
        return np.asarray(grid.areaCell, dtype=np.float64)
    if grid_type == "cubed_sphere":
        return np.asarray(
            getattr(grid, "grid_area", getattr(grid, "area", None)),
            dtype=np.float64,
        )
    return np.asarray(grid.area, dtype=np.float64)


def _layer_thickness(state, z_coord) -> np.ndarray:
    from legoesm.ocean.vertical import compute_layer_thickness
    eta = jnp.asarray(state.eta.data)
    H_bathy = jnp.asarray(state.H_bathy.data)
    return np.asarray(
        compute_layer_thickness(eta, H_bathy, z_coord),
        dtype=np.float64,
    )


def _cell_velocity(state, grid_type: str, grid) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-centred (u, v) on each supported grid."""
    if grid_type in ("latlon", "latlon_regional", "latlon_channel"):
        u_east = np.asarray(state.u.data, dtype=np.float64)
        v_north = np.asarray(state.v.data, dtype=np.float64)
        u_mask = np.asarray(state.u_mask.data, dtype=np.float64)
        v_mask = np.asarray(state.v_mask.data, dtype=np.float64)
        u_safe = u_east * u_mask[..., None]
        v_safe = v_north * v_mask[..., None]
        u_cell = 0.5 * (u_safe[:, :-1, :] + u_safe[:, 1:, :])
        v_cell = 0.5 * (v_safe[:-1, :, :] + v_safe[1:, :, :])
        return u_cell, v_cell
    if grid_type in ("mpas", "mpas_regional", "mpas_channel"):
        # Edge-normal -> per-cell weighted-least-squares fit using the
        # same algorithm the cross-model comparison harness uses.
        u_edge = np.asarray(state.u.data, dtype=np.float64)
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
        cellsOnEdge = np.asarray(grid.cellsOnEdge, dtype=np.int64)
        edgesOnCell = np.asarray(grid.edgesOnCell, dtype=np.int64)
        angleEdge = np.asarray(grid.angleEdge, dtype=np.float64)
        c1 = cellsOnEdge[0, :]
        c2 = cellsOnEdge[1, :]
        edge_wet = ((mask[c1] > 0.5) & (mask[c2] > 0.5)).astype(np.float64)
        u_edge_masked = u_edge * edge_wet[:, None]
        cos_e = np.cos(angleEdge)
        sin_e = np.sin(angleEdge)
        edge_idx = np.where(edgesOnCell >= 0, edgesOnCell, 0)
        valid = (edgesOnCell >= 0).astype(np.float64)
        contrib = valid * edge_wet[edge_idx]
        u_gather = u_edge_masked[edge_idx]
        cos_gather = cos_e[edge_idx]
        sin_gather = sin_e[edge_idx]
        A = (cos_gather ** 2 * contrib).sum(axis=0)
        B = (cos_gather * sin_gather * contrib).sum(axis=0)
        C = (sin_gather ** 2 * contrib).sum(axis=0)
        rhs_x = (u_gather * (cos_gather * contrib)[..., None]).sum(axis=0)
        rhs_y = (u_gather * (sin_gather * contrib)[..., None]).sum(axis=0)
        det = A * C - B ** 2
        safe = det > 1e-12
        inv_det = np.where(safe, 1.0 / np.where(safe, det, 1.0), 0.0)[..., None]
        u_x = (C[..., None] * rhs_x - B[..., None] * rhs_y) * inv_det
        u_y = (A[..., None] * rhs_y - B[..., None] * rhs_x) * inv_det
        return u_x, u_y
    # cubed_sphere / spectral default: state stores collocated u, v.
    u = np.asarray(state.u.data, dtype=np.float64)
    v = np.asarray(state.v.data, dtype=np.float64)
    return u, v


@dataclass(frozen=True)
class EnergyBudget:
    KE: float          # Volume-integrated kinetic energy [J]
    APE: float         # Volume-integrated available potential energy [J]
    total: float       # KE + APE [J]
    volume: float      # Total wet volume [m^3]
    area_total: float  # Wet surface area [m^2]


def compute_energy_budget(state, z_coord, *, grid_type: str, grid,
                          rho_0: Optional[float] = None,
                          g_val: Optional[float] = None) -> EnergyBudget:
    """Domain-integrated KE + APE.

    APE is computed *relative to the rest state* as
    ``APE = 0.5 * rho_0 * g * sum(eta^2 * area)`` (linear free-surface
    approximation, consistent with the legoESM rigid-lid / barotropic
    treatment); the full nonlinear APE would need RPE.
    """
    if rho_0 is None:
        rho_0 = float(constants.rho_ocean)
    if g_val is None:
        g_val = float(constants.g)

    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    area = _grid_area(grid_type, grid)
    h_k = _layer_thickness(state, z_coord)
    u_cell, v_cell = _cell_velocity(state, grid_type, grid)
    speed_sq = u_cell ** 2 + v_cell ** 2
    vol = (area * mask)[..., None] * h_k
    KE = float(0.5 * rho_0 * (speed_sq * vol).sum())

    eta = np.asarray(state.eta.data, dtype=np.float64)
    APE = float(0.5 * rho_0 * g_val * ((eta ** 2) * area * mask).sum())

    total_vol = float(vol.sum())
    A_total = float((area * mask).sum())
    return EnergyBudget(
        KE=KE, APE=APE, total=KE + APE,
        volume=total_vol, area_total=A_total,
    )


@dataclass(frozen=True)
class TracerBudget:
    volume: float           # Total wet ocean volume [m^3]
    heat_content: float     # rho_0 * c_p * sum(T * vol) [J]
    salt_mass: float        # rho_0 * sum(S * vol) [g]  (PSU = g/kg → with rho_0 yields g)
    eta_integral: float     # sum(eta * area) [m^3]


def compute_tracer_budget(state, z_coord, *, grid_type: str, grid,
                          rho_0: Optional[float] = None,
                          c_p: Optional[float] = None) -> TracerBudget:
    """Total volume / heat content / salt content / SSH integral.

    Drift in these between two states quantifies tracer-conservation
    error. For closed basins (no surface forcing) drift should be at
    floating-point precision; for forced runs the drift must match the
    applied surface fluxes.
    """
    if rho_0 is None:
        rho_0 = float(constants.rho_ocean)
    if c_p is None:
        c_p = float(constants.c_sw)

    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    area = _grid_area(grid_type, grid)
    h_k = _layer_thickness(state, z_coord)
    T = np.asarray(state.T.data, dtype=np.float64)
    S = np.asarray(state.S.data, dtype=np.float64)
    eta = np.asarray(state.eta.data, dtype=np.float64)
    vol = (area * mask)[..., None] * h_k
    return TracerBudget(
        volume=float(vol.sum()),
        heat_content=float(rho_0 * c_p * (T * vol).sum()),
        salt_mass=float(rho_0 * (S * vol).sum()),
        eta_integral=float((eta * area * mask).sum()),
    )


__all__ = [
    "EnergyBudget", "TracerBudget",
    "compute_energy_budget", "compute_tracer_budget",
]
