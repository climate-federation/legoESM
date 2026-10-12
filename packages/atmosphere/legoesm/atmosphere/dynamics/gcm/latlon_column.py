"""The lat-lon C-grid primitive-equation model as a COLUMN model (composable
physics P3): the dycore behind the shared column loop's step contract
(:mod:`legoesm.grids.column_mesh`), the FV3 duo's pattern on a structured
grid.

The native lat-lon step is already operator-split (dynamics-only RK, ONE
physics evaluation on the cell-centred view, increments through
``apply_physics_water_mass`` + the polar filter + the dry-mass fixer), so this
adapter is a column-reshaping physics wrapper plus a forcing pass-through:

* :class:`LatLonColumnState` -- ``u``/``v`` cell-centred geographic winds
  (the dycore's own face->cell average), ``T``, ``p_s``, ``phis``, tracers,
  all on ``(n_lat*n_lon[, nlev])`` row-major columns; the native C-grid state
  (face winds) rides along as ``native``.  The columns are a VIEW of it.
* :meth:`LatLonColumnModel.step` -- the native step with
  ``physics_fn = wrap(fn)``: the wrapper reshapes the grid-shaped
  ``HydrostaticState`` to columns, calls the column physics with the mesh,
  the coordinate, the carry and the forcing, and reshapes the tendencies
  back; the dycore applies them.  Surface diagnostics come out of the jit
  explicitly (``return_phys_tend``) and are slot-merged into ``_sfc_diag``
  exactly as the duo / MPAS models do, so the land gets its downwelling
  radiation.

Exchange convention: the MPAS lane's (``column_mesh`` docstring).  Rules: only
``step`` moves ``u``/``v``/``p_s``; the driver's post-step T / tracer edits
are written back into the native state by :meth:`to_bundle`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.column_mesh import ColumnMesh
from legoesm.timestepping.integration import (
    refuse_unthreaded_stateful_physics)

from .primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState, CGridLatLonPrimitiveEquationModel,
    cgrid_to_hydrostatic, hydrostatic_to_cgrid,
)

#: checkpoint keys of the native face winds the column view cannot rebuild
LATLON_CKPT_KEYS = ("llc_u_face", "llc_v_face")


def build_latlon_column_mesh(grid) -> ColumnMesh:
    """The column mesh of a :class:`~legoesm.grids.latlon.LatLonGrid`: the
    A-grid cell centres and areas flattened ``(n_lat, n_lon) -> (ncol,)``
    row-major (``reshape(-1)``, the one order every field in this module
    uses), corners from the v-face latitudes and the lon interfaces in ring
    order (sw, se, ne, nw), the grid's own radius and rotation rate."""
    lat2 = np.asarray(grid.lat2d, dtype=np.float64)
    lon2 = np.asarray(grid.lon2d, dtype=np.float64)
    n_lat, n_lon = lat2.shape
    lat_v = np.asarray(grid.lat_v, dtype=np.float64)
    if lat_v.shape != (n_lat + 1,):
        raise ValueError(f"build_latlon_column_mesh: lat_v has shape "
                         f"{lat_v.shape}, expected {(n_lat + 1,)}")
    half = 0.5 * float(grid.dlon)
    lon_w = (lon2 - half) % (2.0 * np.pi)
    lon_e = (lon2 + half) % (2.0 * np.pi)
    lat_s = np.broadcast_to(lat_v[:-1, None], (n_lat, n_lon))
    lat_n = np.broadcast_to(lat_v[1:, None], (n_lat, n_lon))
    c_lat = np.stack([lat_s, lat_s, lat_n, lat_n], axis=-1).reshape(-1, 4)
    c_lon = np.stack([lon_w, lon_e, lon_e, lon_w], axis=-1).reshape(-1, 4)
    jlat, jlon = jnp.asarray(lat2.reshape(-1)), jnp.asarray(lon2.reshape(-1))
    ncol = n_lat * n_lon
    return ColumnMesh(
        latCell=jlat, lonCell=jlon,
        areaCell=jnp.asarray(grid.area).reshape(-1),
        nCells=ncol, grid_lat=jlat, grid_lon=jlon, grid_shape_2d=(ncol,),
        lat=jlat, lon=jlon,
        cornerLat=jnp.asarray(c_lat), cornerLon=jnp.asarray(c_lon),
        radius=float(grid.radius), omega=float(grid.omega))


class LatLonColumnState(NamedTuple):
    """``HydrostaticState`` field order plus the native C-grid state."""
    u: Field
    T: Field
    p_s: Field
    phis: Field
    v: Field | None = None
    tracers: dict | None = None
    native: CGridLatLonHydrostaticState | None = None


class LatLonColumnModel:
    """:class:`CGridLatLonPrimitiveEquationModel` behind the column contract."""

    def __init__(self, dyn: CGridLatLonPrimitiveEquationModel, *,
                 tracer_names):
        self.dyn = dyn
        self.config = dyn.config
        self.native_grid = dyn.grid
        self.sigma_coord = dyn.sigma_coord
        self.mesh = build_latlon_column_mesh(dyn.grid)
        self.n_lat, self.n_lon = int(dyn.grid.n_lat), int(dyn.grid.n_lon)
        names = tuple(tracer_names)
        if len(set(names)) != len(names):
            raise ValueError(f"LatLonColumnModel: duplicate tracer names {names}")
        self.tracer_names = names
        self._state_type = LatLonColumnState
        self._phys_state = None
        self._sfc_diag = None
        self._wrapped = {}      # physics_fn -> its grid<->column wrapper (static jit arg)
        self._last = None

    @property
    def effective_dt(self):
        return getattr(self.dyn, "effective_dt", self.dyn.dt)

    @property
    def grid(self):
        return self.native_grid

    # ------------------------------------------------------------------
    # layouts
    # ------------------------------------------------------------------

    def _cols(self, a):
        """``(n_lat, n_lon, ...) -> (ncol, ...)``."""
        return jnp.reshape(a, (self.mesh.nCells,) + tuple(a.shape[2:]))

    def _grid2(self, a):
        """``(ncol, ...) -> (n_lat, n_lon, ...)``."""
        return jnp.reshape(a, (self.n_lat, self.n_lon) + tuple(a.shape[1:]))

    def _column_state(self, hs: HydrostaticState, native) -> LatLonColumnState:
        def fld(f, nm, un):
            return Field(data=self._cols(f.data), name=nm, units=un)
        tracers = None
        if hs.tracers:
            missing = [nm for nm in self.tracer_names if nm not in hs.tracers]
            if missing:
                raise ValueError(
                    f"LatLonColumnModel: the native state carries no "
                    f"{missing}; registered tracers {self.tracer_names}")
            tracers = {nm: fld(hs.tracers[nm], nm,
                               "1/kg" if nm.startswith("N_") else "kg/kg")
                       for nm in self.tracer_names}
        return LatLonColumnState(
            u=fld(hs.u, "u", "m/s"), v=fld(hs.v, "v", "m/s"),
            T=fld(hs.T, "T", "K"), p_s=fld(hs.p_s, "p_s", "Pa"),
            phis=fld(hs.phis, "phis", "m^2/s^2"), tracers=tracers,
            native=native)

    def column_view(self, native: CGridLatLonHydrostaticState):
        """The physics-facing columns of a native C-grid state (the dycore's
        own face->cell wind average, as its physics call sees them)."""
        return self._column_state(
            cgrid_to_hydrostatic(native, self.native_grid), native)

    def _native_of(self, state: LatLonColumnState) -> CGridLatLonHydrostaticState:
        """The native state *state* stands for, with the driver's post-step
        column edits (T, tracers; the only fields it may touch) written back.
        A reshape is a view: the write-back is exact."""
        nat = state.native
        if nat is None:
            raise ValueError("LatLonColumnModel: a column state without its "
                             "native C-grid state (built outside from_bundle)")
        T = self._grid2(state.T.data)
        tr = ({nm: self._grid2(f.data) for nm, f in state.tracers.items()}
              if state.tracers is not None else nat.tracers)
        return nat._replace(T=T, tracers=tr)

    def to_bundle(self, state):
        return self._native_of(state)

    def from_bundle(self, bundle):
        """Column state of a native C-grid state (IC / restart); a dict of
        the state's arrays is accepted too."""
        if isinstance(bundle, dict):
            bundle = CGridLatLonHydrostaticState(**bundle)
        self._last = self.column_view(bundle)
        return self._last

    def from_native(self, hs: HydrostaticState, tracers: dict | None):
        """The column state of a cell-centred native IC (the driver's
        lat-lon builders): winds re-staggered ONCE onto the faces by the
        dycore's own interpolator, the registered tracers attached."""
        if tracers:
            hs = hs._replace(tracers={
                nm: Field(data=jnp.asarray(tracers[nm]), name=nm,
                          dims=("lat", "lon", "level"),
                          units="1/kg" if nm.startswith("N_") else "kg/kg")
                for nm in self.tracer_names})
        return self.from_bundle(hydrostatic_to_cgrid(hs, self.native_grid))

    def init_state(self, driver) -> None:
        raise RuntimeError(
            "LatLonColumnModel.init_state: the driver builds the IC on the "
            "native lat-lon grid (ModelDriver._init_state, native_grid "
            "branch) and wraps it through from_native")

    # ------------------------------------------------------------------
    # the step
    # ------------------------------------------------------------------

    def _wrap(self, physics_fn):
        """The grid<->column wrapper of *physics_fn*, ONE object per
        physics_fn (it is the jit's static argument: a fresh closure per
        step would retrace every step)."""
        w = self._wrapped.get(physics_fn)
        if w is not None:
            return w
        mesh, coord = self.mesh, self.sigma_coord
        cols, grid2 = self._cols, self._grid2

        def wrapped(hs, grid, sigma, phys_state=None, forcing=None):
            def c(f, nm, un):
                return Field(data=cols(f.data), name=nm, units=un)
            # registered order first (the column view's), then any extra
            # native tracer: the physics reads by name, the dycore applies
            # tendencies by name, so the order is cosmetic but STABLE
            names = [nm for nm in self.tracer_names if nm in (hs.tracers or {})]
            names += sorted(nm for nm in (hs.tracers or {}) if nm not in names)
            tracers = ({nm: c(hs.tracers[nm], nm,
                              "1/kg" if nm.startswith("N_") else "kg/kg")
                        for nm in names}
                       if hs.tracers else None)
            col_state = HydrostaticState(
                u=c(hs.u, "u", "m/s"), v=c(hs.v, "v", "m/s"),
                T=c(hs.T, "T", "K"), p_s=c(hs.p_s, "p_s", "Pa"),
                phis=c(hs.phis, "phis", "m^2/s^2"), tracers=tracers)
            pr = physics_fn(col_state, mesh, coord, phys_state=phys_state,
                            forcing=forcing)
            tend, pout = (pr[0], pr[1]) if type(pr) is tuple else (pr, phys_state)
            if tend.dv_dt is None:
                raise ValueError(
                    "LatLonColumnModel: physics returned dv_dt=None; the "
                    "column model carries both cell wind components "
                    "(make_physics with model_type='mpas')")

            def g(f):
                return None if f is None else f.replace(data=grid2(f.data))
            tt = tend.tracer_tendencies
            tend = tend._replace(
                du_dt=g(tend.du_dt), dv_dt=g(tend.dv_dt), dT_dt=g(tend.dT_dt),
                dp_s_dt=g(tend.dp_s_dt), dphis_dt=g(tend.dphis_dt),
                tracer_tendencies=(None if tt is None
                                   else {k: g(v) for k, v in tt.items()}))
            # the surface / TOA diagnostics stay (ncol,): the driver reads
            # them as columns off _sfc_diag
            return (tend, pout) if type(pr) is tuple else tend
        self._wrapped[physics_fn] = wrapped
        return wrapped

    def step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        """The native lat-lon step on the state *state* views, physics once
        on the post-dynamics columns; stashes ``_phys_state`` and the
        slot-merged ``_sfc_diag`` eagerly like the MPAS model."""
        from legoesm.core.state import MPAS_SFC_DIAG_EXTRA_KEYS
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="LatLonColumnModel.step()")
        native = self._native_of(state)
        if physics_fn is None:
            out = self.dyn.step(native, dt)
            self._last = self.column_view(out)
            return self._last
        out, tend = self.dyn.step(
            native, dt, physics_fn=self._wrap(physics_fn),
            phys_state=phys_state, forcing=forcing, return_phys_tend=True)
        phys_out = self.dyn._phys_state
        if phys_out is not None and not any(
                isinstance(leaf, jax.core.Tracer)
                for leaf in jax.tree_util.tree_leaves(phys_out)):
            self._phys_state = phys_out
        sfc = (getattr(tend, "sw_net_sfc", None),
               getattr(tend, "lw_net_sfc", None),
               getattr(tend, "precip", None)) + tuple(
            getattr(tend, k, None) for k in MPAS_SFC_DIAG_EXTRA_KEYS)
        if any(s is not None for s in sfc) and not any(
                isinstance(leaf, jax.core.Tracer)
                for leaf in jax.tree_util.tree_leaves(sfc)):
            # slot-wise merge as the MPAS / duo models do: sw/lw refresh
            # only on a radiation step, precip every step; pad so a
            # longer tuple never truncates held slots
            prev = self._sfc_diag or ()
            n = max(len(sfc), len(prev))
            prev = prev + (None,) * (n - len(prev))
            sfc = sfc + (None,) * (n - len(sfc))
            self._sfc_diag = tuple(new if new is not None else old
                                   for new, old in zip(sfc, prev))
        self._last = self.column_view(out)
        return self._last
