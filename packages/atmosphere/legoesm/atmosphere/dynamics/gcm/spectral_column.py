"""Spectral hydrostatic dycore as a COLUMN model (composable physics P4).

Presents :class:`SpectralPrimitiveEquationModel` to the shared column loop
(``ModelDriver._run_column``) through the
:class:`~legoesm.grids.column_mesh.ColumnModel` contract, the way
:class:`~legoesm.atmosphere.dynamics.gcm.latlon_cgrid_column.LatLonCGridColumnModel`
presents the lat-lon C-grid:

* the native prognostic state is the ``SpectralHydrostaticState`` (SH
  coefficients + grid-space tracers); the physics-facing columns are its
  SYNTHESIS (``grid_view_state``) flattened row-major ``(n_lat, n_lon) ->
  (n_lat * n_lon,)``, the mesh of :func:`build_latlon_column_mesh` on the
  Gaussian grid (areas = the quadrature areas);
* the driver's post-step T edits are written back as the spectral INCREMENT
  ``T_hat += SH(T - T_view)`` (exactly zero for no edit, so bitwise; a
  re-analysis of the view would not be), the tracers replaced (a reshape);
* ``step`` runs the dycore's operator split (``column_split_impl``: dynamics,
  energy fixer, ONE physics evaluation applied as grid increments, the
  tracer rails and the anchored dry-mass fixer) with ``physics_fn`` bridged
  from grid fields to columns, jitted here per ``(physics_fn, dt)`` because
  the dycore's filters are dt-dependent constants.

Single device only (no MPI, ensembles or tiling); ssp_rk54 only (D5).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.gaussian import sh_analysis_3d
from legoesm.timestepping.integration import refuse_unthreaded_stateful_physics

from .latlon_cgrid_column import build_latlon_column_mesh
from .spectral_pe import (
    SpectralHydrostaticState, SpectralPrimitiveEquationModel, grid_view_state,
)


class SpectralColumnState(NamedTuple):
    """``HydrostaticState`` field order plus the native spectral state and
    the synthesised ``T`` columns the view was built with (the write-back
    base of the driver's T edits)."""
    u: Field
    T: Field
    p_s: Field
    phis: Field
    v: Field | None = None
    tracers: dict | None = None
    native: SpectralHydrostaticState | None = None
    T_view: Field | None = None


class SpectralColumnModel:
    """A :class:`SpectralPrimitiveEquationModel` behind the column loop's
    step contract (``legoesm.grids.column_mesh.ColumnModel``)."""

    def __init__(self, dyn: SpectralPrimitiveEquationModel):
        self.dyn = dyn
        self.config = dyn.config
        self.grid = dyn.grid
        self.sigma_coord = dyn.sigma_coord
        self.n_lat, self.n_lon = int(dyn.grid.n_lat), int(dyn.grid.n_lon)
        self.mesh = build_latlon_column_mesh(dyn.grid)
        self._state_type = SpectralColumnState
        self._phys_state = None
        self._sfc_diag = None
        self._step_diag = None
        self._fns = {}

    # ------------------------------------------------------------------
    # layouts
    # ------------------------------------------------------------------

    def columns_of(self, a):
        """``(n_lat, n_lon[, k])`` -> ``(n_lat * n_lon[, k])``."""
        return jnp.reshape(a, (self.n_lat * self.n_lon,) + tuple(a.shape[2:]))

    def grid_of(self, a):
        """``(n_lat * n_lon[, k])`` -> ``(n_lat, n_lon[, k])``."""
        return jnp.reshape(a, (self.n_lat, self.n_lon) + tuple(a.shape[1:]))

    def _hs_to_columns(self, hs):
        """A gridded ``HydrostaticState`` as columns."""
        def f(x):
            return (None if x is None
                    else x.replace(data=self.columns_of(x.data), dims=()))
        tr = (None if hs.tracers is None
              else {k: f(v) for k, v in hs.tracers.items()})
        return SpectralColumnState(u=f(hs.u), T=f(hs.T), p_s=f(hs.p_s),
                                   phis=f(hs.phis), v=f(hs.v), tracers=tr)

    def _tend_to_grid(self, tend):
        """Column physics tendencies back on the grid (the fields
        ``apply_grid_physics_increments`` reads: winds, T, tracers)."""
        def g(x):
            return (None if x is None
                    else x.replace(data=self.grid_of(x.data), dims=()))
        if tend.dv_dt is None:
            raise ValueError(
                "SpectralColumnModel: physics returned dv_dt=None; the "
                "column model carries both wind components")
        tt = (None if tend.tracer_tendencies is None
              else {k: g(v) for k, v in tend.tracer_tendencies.items()})
        return tend._replace(du_dt=g(tend.du_dt), dv_dt=g(tend.dv_dt),
                             dT_dt=g(tend.dT_dt), tracer_tendencies=tt)

    def _view(self, native):
        cs = self._hs_to_columns(
            grid_view_state(native, self.grid, self.sigma_coord))
        return cs._replace(native=native, T_view=cs.T)

    # ------------------------------------------------------------------
    # ColumnModel contract
    # ------------------------------------------------------------------

    def column_view(self, native):
        return self._view(native)

    def from_bundle(self, native):
        """The column state of a native spectral state (IC, restart)."""
        fn = self._fns.get("view")
        if fn is None:
            fn = jax.jit(self._view)
            self._fns["view"] = fn
        return fn(native)

    def to_bundle(self, state):
        """The native state *state* stands for, with the driver's post-step
        T / tracer edits written back (the checkpoint writer's entry)."""
        return self._native_of(state)

    def init_state(self, driver) -> None:
        """Fresh IC: the driver's own spectral IC (rest / ERA5) built on the
        NATIVE Gaussian grid, seen through the column view."""
        driver._latlon_column_init_state(self)

    def _native_of(self, state):
        """The native state *state* stands for: T as the spectral INCREMENT
        of the column edit over the view it was made on (``SH(0) == 0``: no
        edit leaves ``T_hat`` bitwise), the tracers replaced (grid space, a
        reshape is exact).  Always computed, so a state rebuilt by a JAX
        transform (no object identity) is as good as the one returned."""
        nat = state.native
        if nat is None or state.T_view is None:
            raise ValueError(
                "SpectralColumnModel: a column state without its native "
                "spectral state / view (build states with from_bundle and "
                "thread step's output)")
        dT = self.grid_of(state.T.data) - self.grid_of(state.T_view.data)
        nat = nat._replace(T_hat=nat.T_hat.replace(
            data=nat.T_hat.data + sh_analysis_3d(self.grid, dT)))
        if state.tracers is not None:
            if set(state.tracers) != set(nat.tracers or ()):
                raise ValueError(
                    f"SpectralColumnModel.step: tracers "
                    f"{sorted(state.tracers)} != the model's "
                    f"{sorted(nat.tracers or ())}")
            nat = nat._replace(tracers={
                nm: nat.tracers[nm].replace(
                    data=self.grid_of(state.tracers[nm].data))
                for nm in nat.tracers})
        return nat

    def step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        """The dycore's operator-split step with ``physics_fn(columns, mesh,
        sigma_coord, phys_state=, forcing=)`` bridged onto its grid fields.
        Returns the new column state; stashes ``_phys_state`` /
        ``_sfc_diag`` (slot-merged) like the MPAS and lat-lon models, and
        the lane diagnostics (dry-mass truncation residual, min T) on
        ``_step_diag``."""
        from legoesm.core.state import MPAS_SFC_DIAG_EXTRA_KEYS
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="SpectralColumnModel.step()")
        native = self._native_of(state)
        if physics_fn is None:
            return self.from_bundle(self.dyn.step(native, dt))
        self.dyn.prepare_column_split(native, dt)
        key = (physics_fn, float(dt))
        fn = self._fns.get(key)
        if fn is None:
            dt_ = float(dt)

            def _step(nat, tm, fo, ps):
                box = {}

                def bridge(hs, grid, sigma, ps_in, fo_):
                    pr = physics_fn(self._hs_to_columns(hs), self.mesh,
                                    self.sigma_coord, phys_state=ps_in,
                                    forcing=fo_)
                    tend, out = ((pr[0], pr[1]) if type(pr) is tuple
                                 else (pr, ps_in))
                    box["tend"] = tend
                    return self._tend_to_grid(tend), out
                new, ps_out, _, diag = self.dyn.column_split_impl(
                    nat, dt_, bridge, fo, ps, tm)
                tend = box["tend"]
                sfc = (getattr(tend, "sw_net_sfc", None),
                       getattr(tend, "lw_net_sfc", None),
                       getattr(tend, "precip", None)) + tuple(
                    getattr(tend, k, None) for k in MPAS_SFC_DIAG_EXTRA_KEYS)
                if all(s is None for s in sfc):
                    sfc = None
                return self._view(new), ps_out, sfc, diag
            fn = jax.jit(_step)
            self._fns[key] = fn
        state_new, phys_out, sfc_diag, diag = fn(
            native, self.dyn._target_mass, forcing, phys_state)
        self._phys_state = phys_out
        self._step_diag = diag
        if sfc_diag is not None:
            prev = self._sfc_diag or (None,) * len(sfc_diag)
            self._sfc_diag = tuple(new if new is not None else old
                                   for new, old in zip(sfc_diag, prev))
        return state_new
