"""Factory for ocean surface forcing physics."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.surface_forcing.prescribed import prescribed_surface_forcing
from legoesm.ocean.physics.surface_forcing.external import external_surface_forcing
from legoesm.ocean.physics.surface_forcing.flux_feedback import (
    flux_feedback_surface_forcing,
)
from legoesm.ocean.physics.surface_forcing.restoring import restoring_surface_forcing
from legoesm.ocean.physics.surface_forcing.bulk_formulas import bulk_formula_surface_forcing
from legoesm.ocean.physics.tendencies import (
    make_none_physics_fn,
    wrap_ocean_tendencies,
    zero_ocean_tendencies,
)


def make_surface_forcing_physics(
    config: SurfaceForcingConfig,
) -> Callable:
    """Create a surface forcing physics function.

    Parameters
    ----------
    config : SurfaceForcingConfig

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    """
    scheme = config.scheme

    if scheme == "none":
        return make_none_physics_fn()
    elif scheme == "prescribed":
        return _make_prescribed(config)
    elif scheme == "restoring":
        return _make_restoring(config)
    elif scheme == "combined":
        return _make_combined(config)
    elif scheme == "bulk_formulas":
        return _make_bulk_formulas(config)
    elif scheme == "external":
        return _make_external(config)
    elif scheme == "flux_feedback":
        return _make_flux_feedback(config)
    else:
        raise ValueError(f"Unknown surface forcing scheme: {scheme!r}")


def _make_prescribed(config: SurfaceForcingConfig) -> Callable:
    cfg = config.prescribed

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = prescribed_surface_forcing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            z_coord, J, grid, cfg,
        )
        return wrap_ocean_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_external(config: SurfaceForcingConfig) -> Callable:
    """Coupler-provided surface forcing: apply the passed OceanSurfaceForcing
    (tau / q_net / freshwater / salt) through the physics path (cubed-sphere /
    MPAS two-way coupling)."""
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        if surface_forcing is None:
            return zero_ocean_tendencies(state)
        # Partial-cell-aware ACTUAL top-layer thickness (not dz_ref[0]*J) so the
        # flux-to-tendency conversion is conservative on shallow top cells.
        h = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)
        out = external_surface_forcing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            h[..., 0], surface_forcing,
        )
        return wrap_ocean_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_flux_feedback(config: SurfaceForcingConfig) -> Callable:
    """Veros-style flux + SST-feedback heat forcing + SSS restoring + ice mask
    (global_4deg), driven by the traced ``OceanSurfaceForcing`` channels
    ``q_prescribed`` / ``q_feedback`` / ``T_feedback_target`` /
    ``S_restore_target``.  Returns zero tendencies when no forcing is passed.
    A ``make_surface_forcing_physics`` citizen, so
    ``surface_forcing_implicit=True`` routes its dT/dS rates into the
    backward-Euler solve via the existing ``surface_tracer_forcing_fn`` seam."""
    cfg = config.flux_feedback

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        if surface_forcing is None:
            # Tracer-shaped zeros (NOT zero_ocean_tendencies, whose du/dT
            # share state.u's shape): the implicit ``surface_tracer_forcing_fn``
            # seam calls this with the FULL C-grid state (staggered u), and
            # only dT_dt/dS_dt are consumed there.
            return wrap_ocean_tendencies(
                None, None, jnp.zeros_like(state.T.data),
                jnp.zeros_like(state.S.data), state,
            )
        # Partial-cell-aware ACTUAL top-layer thickness (mirrors "external"):
        # the W/m² → K/s conversion is conservative on shallow top cells and
        # equals Veros's fixed dzt[-1] under a rigid lid with full top cells.
        h = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)
        # Column geometry for the penetrative-shortwave (q_solar) channel:
        # built only when the option is on (static config bool) so the
        # default path is structurally untouched.  ``h > 0`` is the per-cell
        # wet mask (Veros maskT; zero below kbot under partial cells).
        solar_kwargs = {}
        if cfg.penetrative_shortwave:
            solar_kwargs = dict(
                dz_ref=z_coord.dz_ref,
                z_half_ref=z_coord.z_half_ref,
                jacobian=compute_ocean_jacobian(
                    state.eta.data, state.H_bathy.data, z_coord
                ),
                wet_3d=(h > 0.0).astype(state.T.data.dtype),
            )
        out = flux_feedback_surface_forcing(
            state.T.data, state.S.data, h[..., 0], surface_forcing, cfg,
            **solar_kwargs,
        )
        return wrap_ocean_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_restoring(config: SurfaceForcingConfig) -> Callable:
    cfg = config.restoring

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        out = restoring_surface_forcing(state.T.data, state.S.data, grid, cfg)
        return wrap_ocean_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_combined(config: SurfaceForcingConfig) -> Callable:
    """Prescribed wind stress + temperature/salinity restoring."""
    cfg_p = config.prescribed
    cfg_r = config.restoring

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        p = prescribed_surface_forcing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            z_coord, J, grid, cfg_p,
        )
        r = restoring_surface_forcing(state.T.data, state.S.data, grid, cfg_r)
        return wrap_ocean_tendencies(
            p.du_dt + r.du_dt,
            p.dv_dt + r.dv_dt,
            p.dT_dt + r.dT_dt,
            p.dS_dt + r.dS_dt,
            state,
        )
    return physics_fn


def _make_bulk_formulas(config: SurfaceForcingConfig) -> Callable:
    cfg = config.bulk_formulas

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = bulk_formula_surface_forcing(
            state.T.data, state.S.data, z_coord, J, cfg,
        )
        return wrap_ocean_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


