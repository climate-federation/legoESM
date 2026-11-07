"""Shared potential-energy extraction for ocean matrix runners.

The monolithic and modular runners must not carry independent versions of
this diagnostic: PE is part of their emitted acceptance record, and a stale
copy can make identical trajectories report different verdict inputs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants

_MPAS_GRID_TYPES = ("mpas", "mpas_regional", "mpas_channel")
_CELL_AREA_GRID_TYPES = (
    "latlon", "latlon_regional", "latlon_channel", "cubed_sphere",
    "cs_regional", "tripole", "fesom",
)


def extract_energy_fields(state, grid_type, grid, z_coord):
    """Return tracer, horizontal, and moving-volume inputs for PE/RPE.

    The tuple is ``(T, S, area, wet_mask, z_centre, thickness)`` with the
    horizontal arrays reshaped to the tracer's spatial shape.
    """
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d
        temperature = np.asarray(
            sh_synthesis_3d(grid, state.T_hat.data), dtype=np.float64)
        salinity = np.asarray(
            sh_synthesis_3d(grid, state.S_hat.data), dtype=np.float64)
        area = np.asarray(getattr(grid, "grid_area", None), dtype=np.float64)
        mask_attr = "land_mask_grid"
    elif grid_type in _MPAS_GRID_TYPES:
        temperature = np.asarray(state.T.data, dtype=np.float64)
        salinity = np.asarray(state.S.data, dtype=np.float64)
        area = np.asarray(grid.areaCell, dtype=np.float64)
        mask_attr = "land_mask"
    elif grid_type in _CELL_AREA_GRID_TYPES:
        temperature = np.asarray(state.T.data, dtype=np.float64)
        salinity = np.asarray(state.S.data, dtype=np.float64)
        area = np.asarray(grid.area, dtype=np.float64)
        mask_attr = "land_mask"
    else:
        raise ValueError(
            f"extract_energy_fields: unknown grid_type {grid_type!r}. Add it "
            "to the area-attribute registry instead of relying on a "
            "fall-through.")

    missing = object()
    mask_obj = getattr(state, mask_attr, missing)
    if mask_obj is missing:
        raise ValueError(
            f"state for grid_type={grid_type!r} has no attribute "
            f"state.{mask_attr}; cannot compute a land-masked energy integral. "
            "Refusing to fall back to an unmasked sum.")
    mask_raw = np.asarray(mask_obj.data, dtype=np.float64)

    spatial_shape = temperature.shape[:-1]
    expected_size = int(np.prod(spatial_shape))
    if area.size != expected_size:
        raise ValueError(
            f"area has {area.size} entries but the tracer spatial shape is "
            f"{spatial_shape} for grid_type={grid_type!r}.")
    if mask_raw.size != expected_size:
        raise ValueError(
            f"land mask has {mask_raw.size} entries but the tracer spatial "
            f"shape is {spatial_shape} for grid_type={grid_type!r}.")

    from legoesm.ocean.vertical import compute_layer_thickness
    if grid_type == "spectral":
        # Spectral state carries eta/H_bathy spectrally; this diagnostic has no
        # grid-point moving-volume representation for that arm.
        h = np.broadcast_to(
            np.asarray(z_coord.dz_ref, dtype=np.float64),
            temperature.shape,
        ).copy()
    else:
        h = np.asarray(
            compute_layer_thickness(
                jnp.asarray(state.eta.data), jnp.asarray(state.H_bathy.data),
                z_coord),
            dtype=np.float64,
        )
        if h.shape != temperature.shape:
            raise ValueError(
                f"layer thickness shape {h.shape} != tracer shape "
                f"{temperature.shape} "
                f"for grid_type={grid_type!r}.")

    h_cum = np.cumsum(h, axis=-1)
    if grid_type == "spectral":
        z_centre = np.broadcast_to(
            np.asarray(z_coord.z_full_ref, dtype=np.float64),
            temperature.shape,
        ).copy()
    else:
        eta = np.asarray(state.eta.data, dtype=np.float64)
        z_centre = eta[..., np.newaxis] - (h_cum - 0.5 * h)

    return (
        temperature, salinity, area.reshape(spatial_shape),
        mask_raw.reshape(spatial_shape),
        z_centre, h,
    )


def compute_potential_energy(state, grid_type, grid, z_coord) -> float:
    """Return wet-cell, moving-control-volume potential energy [J]."""
    temperature, salinity, area, wet_mask, z_centre, h = extract_energy_fields(
        state, grid_type, grid, z_coord)

    from legoesm.ocean.eos import linear_eos
    rho = np.asarray(
        linear_eos(
            jnp.asarray(temperature), jnp.asarray(salinity),
            jnp.zeros_like(jnp.asarray(temperature)),
            rho_ref=constants.rho_ocean, alpha_T=2.0e-4, beta_S=0.0,
            T_ref=15.0),
        dtype=np.float64,
    )

    pe = 0.0
    for k in range(z_centre.shape[-1]):
        cell = rho[..., k] * z_centre[..., k] * h[..., k] * area
        # ``where``, not multiplication: 0 * NaN must not contaminate land.
        weighted = np.where(wet_mask > 0.5, cell, 0.0)
        if not np.all(np.isfinite(weighted)):
            bad = np.argwhere(~np.isfinite(weighted))[:5].tolist()
            raise ValueError(
                f"Non-finite PE summand at grid_type={grid_type!r}, level "
                f"k={k}; first bad index(es): {bad}.")
        pe += float(np.sum(weighted))
    return float(constants.g) * pe


__all__ = ["compute_potential_energy", "extract_energy_fields"]
