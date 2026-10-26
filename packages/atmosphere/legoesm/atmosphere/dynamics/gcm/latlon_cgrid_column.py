"""Lat-lon C-grid hydrostatic dycore as a COLUMN model (route C).

Presents :class:`CGridLatLonPrimitiveEquationModel` to the shared column
loop (``ModelDriver._run_column``) through the
:class:`~legoesm.grids.column_mesh.ColumnModel` contract, the way
``FV3DuoColumnModel`` presents the FV3 duo:

* the native prognostic state is the C-grid ``CGridLatLonHydrostaticState``
  (face winds); the physics-facing columns are a VIEW of it
  (``cgrid_to_hydrostatic`` cell winds, every field flattened row-major
  ``(n_lat, n_lon) -> (n_lat * n_lon,)``, the ``ColumnMesh`` order);
* ``step`` runs the dycore's own operator-split step (dynamics RK, energy
  fixer, ONE physics evaluation applied by ``_apply_physics_increment``:
  wind INCREMENTS lifted to the faces and polar-filtered, water mass through
  ``apply_physics_water_mass``, then the safety rails with the dry-mass
  fixer and water positivity) with ``physics_fn`` bridged from grid fields
  to columns.  Full winds never round-trip cell -> face.

Single device only (no MPI band, ensembles or tiling).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.grids.column_mesh import ColumnMesh
from legoesm.timestepping.integration import refuse_unthreaded_stateful_physics

from .primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState, CGridLatLonPrimitiveEquationModel,
    cgrid_to_hydrostatic,
)


def build_latlon_column_mesh(grid) -> ColumnMesh:
    """The column mesh of a global lat-lon grid: cell centres, areas and
    quad corners ``(sw, se, ne, nw)`` flattened row-major (latitude rows
    S->N, longitude fastest)."""
    lat2d = np.asarray(grid.lat2d, dtype=np.float64)
    lon2d = np.asarray(grid.lon2d, dtype=np.float64)
    lat_v = np.asarray(grid.lat_v, dtype=np.float64)
    if lat_v.shape != (grid.n_lat + 1,):
        raise ValueError(
            f"build_latlon_column_mesh: lat_v has shape {lat_v.shape}, "
            f"expected ({grid.n_lat + 1},) (a global grid, not an MPI band)")
    lon = np.asarray(grid.lon, dtype=np.float64)
    half = 0.5 * float(grid.dlon)
    s, n = lat_v[:-1, None], lat_v[1:, None]
    w, e = (lon - half)[None, :], (lon + half)[None, :]
    shp = (grid.n_lat, grid.n_lon)
    corner_lat = np.stack([np.broadcast_to(s, shp), np.broadcast_to(s, shp),
                           np.broadcast_to(n, shp), np.broadcast_to(n, shp)],
                          axis=-1).reshape(-1, 4)
    corner_lon = np.stack([np.broadcast_to(w, shp), np.broadcast_to(e, shp),
                           np.broadcast_to(e, shp), np.broadcast_to(w, shp)],
                          axis=-1).reshape(-1, 4)
    jlat = jnp.asarray(lat2d.reshape(-1))
    jlon = jnp.asarray(lon2d.reshape(-1))
    return ColumnMesh(
        latCell=jlat, lonCell=jlon,
        areaCell=jnp.asarray(np.asarray(grid.area).reshape(-1)),
        nCells=int(jlat.shape[0]), grid_lat=jlat, grid_lon=jlon,
        grid_shape_2d=(int(jlat.shape[0]),), lat=jlat, lon=jlon,
        cornerLat=jnp.asarray(corner_lat), cornerLon=jnp.asarray(corner_lon),
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


class LatLonCGridColumnModel:
    """A :class:`CGridLatLonPrimitiveEquationModel` behind the column loop's
    step contract (``legoesm.grids.column_mesh.ColumnModel``)."""

    def __init__(self, dyn: CGridLatLonPrimitiveEquationModel):
        self.dyn = dyn
        self.config = dyn.config
        self.grid = dyn.grid
        self.sigma_coord = dyn.sigma_coord
        self.n_lat, self.n_lon = int(dyn.grid.n_lat), int(dyn.grid.n_lon)
        self.mesh = build_latlon_column_mesh(dyn.grid)
        self._state_type = LatLonColumnState
        self._phys_state = None
        self._sfc_diag = None
        self._last = None
        self._fns = {}
        # the dry-mass anchor, taken once from the first state stepped (the
        # dycore's own ``step`` contract: anchor_mass_to_initial)
        self._target_mass = None

    # ------------------------------------------------------------------
    # layouts
    # ------------------------------------------------------------------

    def columns_of(self, a):
        """``(n_lat, n_lon[, k])`` -> ``(n_lat * n_lon[, k])``."""
        return jnp.reshape(a, (self.n_lat * self.n_lon,) + tuple(a.shape[2:]))

    def grid_of(self, a):
        """``(n_lat * n_lon[, k])`` -> ``(n_lat, n_lon[, k])``."""
        return jnp.reshape(a, (self.n_lat, self.n_lon) + tuple(a.shape[1:]))

    def _hs_to_columns(self, hs, native=None):
        """A gridded ``HydrostaticState`` as columns."""
        def f(x):
            return (None if x is None
                    else x.replace(data=self.columns_of(x.data), dims=()))
        tr = (None if hs.tracers is None
              else {k: f(v) for k, v in hs.tracers.items()})
        return LatLonColumnState(u=f(hs.u), T=f(hs.T), p_s=f(hs.p_s),
                                 phis=f(hs.phis), v=f(hs.v), tracers=tr,
                                 native=native)

    def _tend_to_grid(self, tend):
        """Column physics tendencies back on the grid (the fields the
        dycore's increment applier reads)."""
        def g(x):
            return (None if x is None
                    else x.replace(data=self.grid_of(x.data), dims=()))
        if tend.dv_dt is None:
            raise ValueError(
                "LatLonCGridColumnModel: physics returned dv_dt=None; the "
                "column model carries both cell wind components")
        dps = (tend.dp_s_dt if tend.dp_s_dt is not None else
               Field(data=jnp.zeros_like(tend.dT_dt.data[:, 0]),
                     name="dp_s_dt", units="Pa/s"))
        tt = (None if tend.tracer_tendencies is None
              else {k: g(v) for k, v in tend.tracer_tendencies.items()})
        # the surface diagnostics (sw/lw/precip/...) stay column-shaped:
        # the column loop reads them off ``_sfc_diag``, not the dycore
        return tend._replace(du_dt=g(tend.du_dt), dv_dt=g(tend.dv_dt),
                             dT_dt=g(tend.dT_dt), dp_s_dt=g(dps),
                             dphis_dt=g(tend.dphis_dt),
                             tracer_tendencies=tt)

    def _view(self, native):
        return self._hs_to_columns(cgrid_to_hydrostatic(native, self.grid),
                                   native=native)

    # ------------------------------------------------------------------
    # ColumnModel contract
    # ------------------------------------------------------------------

    def column_view(self, native):
        return self._view(native)

    def from_bundle(self, native):
        """The column state of a native C-grid state (IC, restart)."""
        fn = self._fns.get("view")
        if fn is None:
            fn = jax.jit(self._view)
            self._fns["view"] = fn
        self._last = fn(native)
        return self._last

    def to_bundle(self, state):
        """The native state *state* stands for, with the driver's post-step
        T / tracer edits written back (the checkpoint writer's entry)."""
        return self._native_of(state)

    def init_state(self, driver) -> None:
        """Fresh IC: the driver's own lat-lon IC (rest / Held-Suarez / ERA5)
        built on the NATIVE grid, lifted to the C-grid once
        (``hydrostatic_to_cgrid``, as the lat-lon lane's first step does),
        then seen through the column view."""
        driver._latlon_column_init_state(self)

    def _native_of(self, state):
        """The native state a column state stands for.  The columns are a
        VIEW: only T and the tracers may be edited by the driver (written
        back here); a wind / p_s / phis edit is refused (a wind edit would
        need the face lift, p_s follows the layer mass)."""
        last = self._last
        if last is None or state.native is not last.native:
            raise ValueError(
                "LatLonCGridColumnModel.step: the column state is not the one "
                "this model returned (build states with from_bundle and thread "
                "step's output)")
        for nm in ("u", "v", "p_s", "phis"):
            if getattr(state, nm) is not getattr(last, nm):
                raise ValueError(
                    f"LatLonCGridColumnModel.step: state.{nm} was edited "
                    "outside the model; the columns are a view of the C-grid "
                    "state and only T and the tracers can be written back")
        native = state.native
        if state.T is last.T and state.tracers is last.tracers:
            return native
        T = native.T if state.T is last.T else self.grid_of(state.T.data)
        tracers = dict(native.tracers)
        if state.tracers is not last.tracers:
            if state.tracers is None or set(state.tracers) != set(native.tracers):
                raise ValueError(
                    f"LatLonCGridColumnModel.step: tracers "
                    f"{sorted(state.tracers or ())} != the model's "
                    f"{sorted(native.tracers)}")
            for nm in native.tracers:
                if state.tracers[nm] is not last.tracers[nm]:
                    tracers[nm] = self.grid_of(state.tracers[nm].data)
        return native._replace(T=T, tracers=tracers)

    def _anchor(self, native):
        c = self.config
        if not (c.fix_mass and c.anchor_mass_to_initial):
            return None
        if self._target_mass is None:
            self._target_mass = self.dyn.compute_mass(native)
        return self._target_mass

    def step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        """The dycore's operator-split step with ``physics_fn(columns, mesh,
        sigma_coord, phys_state=, forcing=)`` bridged onto its grid fields.
        Returns the new column state; stashes ``_phys_state`` /
        ``_sfc_diag`` (slot-merged) like the MPAS and duo models."""
        from legoesm.core.state import MPAS_SFC_DIAG_EXTRA_KEYS
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="LatLonCGridColumnModel.step()")
        native = self._native_of(state)
        target = self._anchor(native)
        fn = self._fns.get(physics_fn)
        if fn is None:
            def _step(nat, dt_, tm, fo, ps):
                box = {}
                bridge = None
                if physics_fn is not None:
                    def bridge(hs, grid, sigma, ps_in=None):
                        pr = physics_fn(self._hs_to_columns(hs), self.mesh,
                                        self.sigma_coord, phys_state=ps_in,
                                        forcing=fo)
                        tend, out = ((pr[0], pr[1]) if type(pr) is tuple
                                     else (pr, ps_in))
                        box["tend"], box["out"] = tend, out
                        return self._tend_to_grid(tend), out
                new, _ = self.dyn._step_cgrid_impl(nat, dt_, tm, bridge, ps)
                tend, ps_out = box.get("tend"), box.get("out", ps)
                sfc = None
                if tend is not None:
                    sfc = (getattr(tend, "sw_net_sfc", None),
                           getattr(tend, "lw_net_sfc", None),
                           getattr(tend, "precip", None)) + tuple(
                        getattr(tend, k, None) for k in MPAS_SFC_DIAG_EXTRA_KEYS)
                    if all(s is None for s in sfc):
                        sfc = None
                return self._view(new), ps_out, sfc
            fn = jax.jit(_step)
            self._fns[physics_fn] = fn
        state_new, phys_out, sfc_diag = fn(native, float(dt), target, forcing,
                                           phys_state)
        self._last = state_new
        if physics_fn is not None:
            self._phys_state = phys_out
        if sfc_diag is not None:
            prev = self._sfc_diag or (None,) * len(sfc_diag)
            self._sfc_diag = tuple(new if new is not None else old
                                   for new, old in zip(sfc_diag, prev))
        return state_new
