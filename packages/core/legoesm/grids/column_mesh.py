"""Column mesh + column-model contract of the shared column loop.

A column lane presents its dycore to the driver's column loop
(``ModelDriver._run_column``) as a :class:`ColumnModel` whose ``mesh`` is a
:class:`ColumnMesh`: per-column lat/lon/area, no edge topology.  The FV3 duo
(``FV3DuoColumnModel``) is the first adapter; MPAS runs the loop natively.

Exchange convention (the MPAS lane's): ``T`` [K]; cell geographic ``u, v``
[m/s]; ``p_s`` TOTAL (moist) surface pressure [Pa]; tracers mass fractions
of total air [kg/kg]; physics tendencies are rates [unit/s], applied inside
``step`` to the native prognostics (``dp_s_dt = 0``: p_s and tracer
re-weighting follow the water tendencies, so column dry mass is exact).

Rules: ``driver.grid is model.mesh``; only ``step`` changes u/v/p_s/phis
(the driver edits T and tracers only); full winds never round-trip through
the native stagger; polar filters / safety rails are the native dycore's.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp


class ColumnMesh(NamedTuple):
    """What the column loop and its physics read off ``model.mesh``: cell
    lat/lon [rad] and area [m^2] per column, no edge topology (a frontal
    GWD source, which needs gradients, is refused on it).  Per-cell 1-D
    like the Voronoi mesh (``grid_shape_2d == (nCells,)``), so every
    setup-time regrid (topography, SST/SIC, land, ozone) lands on the
    model's own columns."""
    latCell: jax.Array
    lonCell: jax.Array
    areaCell: jax.Array
    nCells: int
    grid_lat: jax.Array
    grid_lon: jax.Array
    grid_shape_2d: tuple
    lat: jax.Array
    lon: jax.Array
    #: cell corners ``(nCells, 4)`` [rad] in ring order: the terrain
    #: product's exact quad ownership
    cornerLat: jax.Array
    cornerLon: jax.Array
    #: the NATIVE grid's planet radius [m] and rotation rate [1/s] (its
    #: areas and Coriolis use these; the duo passes FV3's)
    radius: float
    omega: float
    #: per-column subgrid orographic stddev [m] for the orographic GWD launch,
    #: attached by the driver (``grid._replace``) like the Voronoi mesh's;
    #: None = the scheme's scalar fallback
    subgrid_topo_stddev: Any = None
    #: per-column land fraction for convection (ZM autoconversion split)
    #: and orographic GWD, attached by the driver like the Voronoi mesh's;
    #: None = no land field (ZM land_fraction="required" then refuses)
    land_frac: Any = None

    # the rest of GridProtocol, as the Voronoi mesh defines them
    @property
    def grid_area(self) -> jax.Array:
        return self.areaCell

    @property
    def grid_total_area(self) -> jax.Array:
        return jnp.sum(self.areaCell)

    @property
    def grid_coriolis(self) -> jax.Array:
        return 2.0 * self.omega * jnp.sin(self.latCell)

    @property
    def grid_radius(self) -> float:
        return self.radius

    @property
    def grid_n_columns(self) -> int:
        return self.nCells

    def to_columns(self, field):
        return field

    def from_columns(self, cols):
        return cols


@runtime_checkable
class ColumnModel(Protocol):
    """A dycore behind the column loop's step contract."""
    mesh: ColumnMesh
    sigma_coord: Any
    config: Any
    _state_type: type

    def step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        """Dynamics + physics increments; sets ``self._phys_state`` and
        ``self._sfc_diag`` (slot-merged)."""

    def column_view(self, native):
        """The column state of the native prognostics."""

    def to_bundle(self, state):
        """Native arrays *state* stands for (checkpoint writer's entry)."""

    def from_bundle(self, bundle):
        """Column state of a native bundle (IC / restart)."""

    def init_state(self, driver) -> None:
        """Fresh IC on the native grid, set on *driver* as its column view."""
