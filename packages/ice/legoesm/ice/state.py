"""Sea ice model state containers.

Two state types:
- ``SeaIceState``: Slab thermodynamics only (3 fields, backward compatible).
- ``DynamicSeaIceState``: Full dynamic ice with velocity, stress tensor,
  optional multi-category ice, snow-on-ice layer, bulk ice salinity, and
  melt-pond tracking.  The optional fields are always allocated (zero-
  initialised when their config gate is off) so the pytree structure is
  static across all configurations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.ice.itd import aggregate_state, distribute_to_categories


class SeaIceState(NamedTuple):
    """Thermodynamic slab sea ice state.

    All fields have shape (6, n, n).
    """
    h_ice: Field               # Ice thickness [m]
    T_ice: Field               # Ice surface temperature [K]
    concentration: Field       # Ice areal fraction [0-1]


class DynamicSeaIceState(NamedTuple):
    """Full dynamic sea ice state.

    Per-category thickness/temperature/concentration may have shape
    ``(6, n, n)`` (single-category) or ``(6, n, n, n_cat)`` (multi-
    category).  Velocity and stress are always ``(6, n, n)``.

    Optional physics fields (``h_snow``, ``S_ice``, ``pond_area``,
    ``pond_depth``) follow the same per-category shape as ``h_ice`` and
    are zero-initialised when the matching ``SeaIceConfig`` module gate
    is off — this preserves pytree-uniformity across all configs.
    """
    h_ice: Field               # Ice thickness [m]
    T_ice: Field               # Ice surface temperature [K]
    concentration: Field       # Ice areal fraction [0-1]
    u_ice: Field               # Ice velocity x-component [m/s]
    v_ice: Field               # Ice velocity y-component [m/s]
    sigma_11: Field            # Stress tensor component [N/m]
    sigma_22: Field            # Stress tensor component [N/m]
    sigma_12: Field            # Stress tensor component [N/m]
    h_snow: Field              # Snow depth on ice [m] (zero when snow disabled)
    S_ice: Field               # Bulk ice salinity [g/kg] (zero when brine disabled)
    pond_area: Field           # Melt pond area fraction of ice area [0-1]
    pond_depth: Field          # Mean melt pond depth [m]


# Per-grid spatial Field dim names, keyed by spatial rank (no category axis) —
# matches sea_ice._base_spatial_ndim and the codebase Field convention: MPAS
# Voronoi (nCells,), lat-lon / tripole (lat, lon), cubed-sphere (face, x, y).
_SPATIAL_DIMS_BY_RANK = {
    1: ("nCells",),
    2: ("lat", "lon"),
    3: ("face", "x", "y"),
}


def init_dynamic_ice_state(
    shape: tuple[int, ...],
    *,
    n_categories: int = 1,
    S_ice_init: float = constants.S_ice_bulk_default,
    T_ice_init: float = 260.0,  # coeff-ok: initial ice temperature [K]
) -> DynamicSeaIceState:
    """Create a zero-initialized DynamicSeaIceState.

    Parameters
    ----------
    shape : tuple
        Per-cell field shape INCLUDING the trailing category axis when
        ``n_categories > 1``: cubed-sphere ``(6, n, n[, n_cat])``, lat-lon /
        tripole ``(n_lat, n_lon[, n_cat])``, MPAS ``(nCells[, n_cat])`` —
        the same per-grid spatial base ranks ``step_sea_ice`` validates
        (``sea_ice._base_spatial_ndim``).
    n_categories : int
        Number of ITD categories.  Used only for default shape
        validation; ``shape`` should already include the trailing
        category axis when ``n_categories > 1``.
    S_ice_init : float
        Initial bulk ice salinity [g/kg].  Default
        ``constants.S_ice_bulk_default`` ≈ 4 PSU (first-year-ice
        salinity).  Pass 0.0 to seed with fresh ice — but note that
        enabling brine on fresh seed ice releases no stored salt
        until new ice forms.
    """
    if n_categories < 1:
        raise ValueError(
            f"init_dynamic_ice_state: n_categories={n_categories}; must be a "
            "positive integer (>= 1). Single-category ice is n_categories=1."
        )
    if n_categories > 1 and (len(shape) < 2 or shape[-1] != n_categories):
        raise ValueError(
            f"init_dynamic_ice_state: shape {shape} inconsistent with "
            f"n_categories={n_categories}.  Pass shape that ends with "
            f"the category axis (e.g. (6, n, n, {n_categories}) on the "
            f"cubed sphere, (n_lat, n_lon, {n_categories}) on lat-lon, "
            f"(nCells, {n_categories}) on MPAS)."
        )

    cat_dim = ("category",) if n_categories > 1 else ()
    # Velocity / stress are always spatial-only — strip the trailing category
    # axis (which is LAST by contract, whatever the grid's spatial rank; the
    # old ``shape[:3]`` slice was cubed-sphere-only and leaked the category
    # axis into u_ice/v_ice on lat-lon / MPAS shapes).
    spatial_shape = shape[:-1] if n_categories > 1 else shape
    # Dim NAMES follow the spatial rank so a lat-lon (n_lat, n_lon[, n_cat]) or
    # MPAS (nCells[, n_cat]) state is not mislabelled with the cubed-sphere
    # ("face", "x", "y") — the earlier hard-coding gave a (nCells, n_cat) field
    # four dims and corrupted coordinate-aware consumers.
    spatial_dims = _SPATIAL_DIMS_BY_RANK.get(len(spatial_shape))
    if spatial_dims is None:
        raise ValueError(
            f"init_dynamic_ice_state: unsupported spatial rank "
            f"{len(spatial_shape)} for shape {shape} (expected MPAS 1-D, "
            "lat-lon 2-D, or cubed-sphere 3-D)."
        )
    dims = spatial_dims + cat_dim
    return DynamicSeaIceState(
        h_ice=Field(data=jnp.zeros(shape), name="h_ice", dims=dims, units="m"),
        T_ice=Field(data=jnp.full(shape, T_ice_init), name="T_ice", dims=dims, units="K"),
        concentration=Field(data=jnp.zeros(shape), name="ice_concentration", dims=dims, units="1"),
        u_ice=Field(data=jnp.zeros(spatial_shape), name="u_ice", dims=spatial_dims, units="m/s"),
        v_ice=Field(data=jnp.zeros(spatial_shape), name="v_ice", dims=spatial_dims, units="m/s"),
        sigma_11=Field(data=jnp.zeros(spatial_shape), name="sigma_11", dims=spatial_dims, units="N/m"),
        sigma_22=Field(data=jnp.zeros(spatial_shape), name="sigma_22", dims=spatial_dims, units="N/m"),
        sigma_12=Field(data=jnp.zeros(spatial_shape), name="sigma_12", dims=spatial_dims, units="N/m"),
        h_snow=Field(data=jnp.zeros(shape), name="h_snow", dims=dims, units="m"),
        S_ice=Field(data=jnp.full(shape, S_ice_init), name="S_ice", dims=dims, units="g/kg"),
        pond_area=Field(data=jnp.zeros(shape), name="pond_area", dims=dims, units="1"),
        pond_depth=Field(data=jnp.zeros(shape), name="pond_depth", dims=dims, units="m"),
    )


def dynamic_to_slab(state: DynamicSeaIceState) -> SeaIceState:
    """Convert DynamicSeaIceState to SeaIceState (drops dynamics fields).

    If multi-category, aggregates to single category first.
    """
    h = state.h_ice.data
    T = state.T_ice.data
    a = state.concentration.data

    # Multi-category: aggregate.  Detected by RANK against the spatial-only
    # u_ice, not the old cubed-sphere-only ``h.ndim > 3`` heuristic — lat-lon
    # multi-category is rank 3 and MPAS rank 2, and both must aggregate here
    # rather than smuggle a category axis into a scalar SeaIceState (codex).
    if h.ndim == state.u_ice.data.ndim + 1:
        h_agg, T_agg, a_agg = aggregate_state(h, T, a)
        return SeaIceState(
            h_ice=state.h_ice.replace(data=h_agg),
            T_ice=state.T_ice.replace(data=T_agg),
            concentration=state.concentration.replace(data=a_agg),
        )

    return SeaIceState(
        h_ice=state.h_ice,
        T_ice=state.T_ice,
        concentration=state.concentration,
    )


def slab_to_dynamic(
    state: SeaIceState,
    *,
    S_ice_init: float = constants.S_ice_bulk_default,
) -> DynamicSeaIceState:
    """Convert SeaIceState to DynamicSeaIceState (adds zero dynamics fields).

    All new fields adopt the slab state's spatial shape and dims —
    callers that need multi-category ice should build the state via
    :func:`init_dynamic_ice_state` + :func:`distribute_to_categories`,
    not through this lift.

    ``S_ice`` is seeded with ``constants.S_ice_bulk_default`` so the
    brine-aware path sees realistic first-year-ice salinity when a
    slab state is lifted into ``_step_dynamic_v2``.  Pass
    ``S_ice_init=0.0`` for an explicitly fresh seed.
    """
    s = state.h_ice.data.shape
    dims = state.h_ice.dims
    return DynamicSeaIceState(
        h_ice=state.h_ice,
        T_ice=state.T_ice,
        concentration=state.concentration,
        u_ice=Field(data=jnp.zeros(s), name="u_ice", dims=dims, units="m/s"),
        v_ice=Field(data=jnp.zeros(s), name="v_ice", dims=dims, units="m/s"),
        sigma_11=Field(data=jnp.zeros(s), name="sigma_11", dims=dims, units="N/m"),
        sigma_22=Field(data=jnp.zeros(s), name="sigma_22", dims=dims, units="N/m"),
        sigma_12=Field(data=jnp.zeros(s), name="sigma_12", dims=dims, units="N/m"),
        h_snow=Field(data=jnp.zeros(s), name="h_snow", dims=dims, units="m"),
        S_ice=Field(data=jnp.full(s, S_ice_init), name="S_ice", dims=dims, units="g/kg"),
        pond_area=Field(data=jnp.zeros(s), name="pond_area", dims=dims, units="1"),
        pond_depth=Field(data=jnp.zeros(s), name="pond_depth", dims=dims, units="m"),
    )


def distribute_dynamic_state_to_categories(
    state: DynamicSeaIceState,
    n_categories: int,
) -> DynamicSeaIceState:
    """Lift a SINGLE-category :class:`DynamicSeaIceState` into an
    ``n_categories``-bin ITD state (trailing category axis, any grid rank).

    The delta-function seeding of :func:`legoesm.ice.itd.
    distribute_to_categories`: each cell's ice lands entirely in the one
    thickness bin containing its mean thickness (thickness above the last
    upper bound goes to the last bin).  The per-cell tracers that ride on the
    ice — snow depth and bulk salinity — are placed in that SAME occupied bin
    (zero elsewhere), so the aggregate snow volume ``sum_k a_k h_snow_k`` and
    salt content are conserved exactly by construction.  Melt-pond fields
    are distributed into the SAME occupied bin (pond water is prognostic
    liquid — volume = concentration*area*depth carries mass and enthalpy;
    zeroing it silently deleted pond water on any ponded lift).  Velocity and internal-stress fields are spatial-
    only and pass through unchanged.

    This is the missing driver-side step the ``step_sea_ice`` multi-category
    guard points at ("build one via init_dynamic_ice_state(...) and
    distribute_to_categories"): apply any single-category IC first, then lift.
    """
    if n_categories < 2:
        raise ValueError(
            f"distribute_dynamic_state_to_categories: n_categories="
            f"{n_categories}; a multi-category lift needs >= 2 bins "
            "(single-category states need no distribution)."
        )
    h = state.h_ice.data
    # u_ice is spatial-only by contract, so a category-carrying state has
    # h_ice rank == u_ice rank + 1.  Exact (a size test against n_categories
    # could false-match a spatial axis, e.g. n_lon == n_categories).
    if h.ndim == state.u_ice.data.ndim + 1:
        raise ValueError(
            "distribute_dynamic_state_to_categories: state already carries "
            f"a trailing category axis (h_ice shape {tuple(h.shape)} vs "
            f"spatial u_ice shape {tuple(state.u_ice.data.shape)}); "
            "refusing to double-lift."
        )
    t_srf = state.T_ice.data
    conc = state.concentration.data
    h_mc, t_mc, conc_mc = distribute_to_categories(h, t_srf, conc,
                                                   n_categories)
    # Occupied-bin mask: the delta ITD puts each cell in exactly one bin, so
    # (h_mc > 0) | (conc_mc > 0) marks it (h-only ice and conc-only ice both
    # count; ice-free cells have no occupied bin and keep zero tracers).
    occ = ((h_mc > 0.0) | (conc_mc > 0.0)).astype(h.dtype)
    cat_dims = state.h_ice.dims + ("category",)

    def _lift(field, data_mc):
        return Field(data=data_mc, name=field.name, dims=cat_dims,
                     units=field.units)

    return state._replace(
        h_ice=_lift(state.h_ice, h_mc),
        T_ice=_lift(state.T_ice, t_mc),
        concentration=_lift(state.concentration, conc_mc),
        h_snow=_lift(state.h_snow, state.h_snow.data[..., jnp.newaxis] * occ),
        S_ice=_lift(state.S_ice, state.S_ice.data[..., jnp.newaxis] * occ),
        # Melt-pond water is PROGNOSTIC liquid (volume = area*depth carrying
        # mass + enthalpy): distribute into the occupied bin exactly like
        # snow/salt above — the earlier zeroing silently DELETED pond water
        # on any lift of a ponded IC / restart / mid-run state (unaccounted
        # mass+energy sink; conservation is a hard invariant).
        pond_area=_lift(
            state.pond_area, state.pond_area.data[..., jnp.newaxis] * occ),
        pond_depth=_lift(
            state.pond_depth, state.pond_depth.data[..., jnp.newaxis] * occ),
    )
