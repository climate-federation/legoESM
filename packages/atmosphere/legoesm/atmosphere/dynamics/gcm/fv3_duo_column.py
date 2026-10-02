"""FV3 duo as a COLUMN model: the dycore behind the MPAS lane's step contract.

Route A (docs/architecture/fv3_duo_amip_adapter_plan.md): the CAM6 AMIP
suite is orchestrated by the MPAS lane, whose physics runs on cell columns
``(nCells, nlev)`` and only ever touches the dycore through
``model.step(state, dt, physics_fn=, forcing=, phys_state=)``,
``model.mesh`` (lat/lon/area per column) and ``model.sigma_coord``.  This
wrapper owns an :class:`FV3DuoDynamicsModel` and presents its six-face
bundle as such a column model:

* :class:`FV3DuoColumnState` -- ``u``/``v`` geographic A-grid winds by the
  closed lane's ORDER-4 c2l, ``T``, ``p_s``, ``phis``, tracers, all on
  ``(6*n*n[, km])`` face-major columns; the native bundle rides along as
  ``native`` (the dynamics carry; the columns are a VIEW of it).
* :meth:`FV3DuoColumnModel.step` -- dynamics on the bundle, the column
  view, ``physics_fn`` once on the post-dynamics columns, then the
  increments applied FV3-style (``fv_update_phys``: winds through the
  D-grid lift, water tracers + layer mass + T through the moist block,
  pressures rebuilt).  NO additive surface-pressure fixer: FV3 conserves
  dry mass by construction and the layer mass follows the water.

Six faces only; the window (SPMD) layout is certification rung 7.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.fv3_native_physics_coupling import (
    FV3_WATER_SPECIES, apply_column_increments_sixface_jax,
    column_view_sixface_jax, stack_held_suarez_metrics)
from legoesm.grids.fv3_native_gridstruct import (
    FV3_KAPPA, FV3_OMEGA, FV3_RADIUS_M,
)
from legoesm.grids.vertical import create_hybrid_coordinate
from legoesm.timestepping.integration import (
    refuse_unthreaded_stateful_physics)

from .fv3_duo_dynamics import FV3DuoDynamicsModel

# the MPAS lane's tracer names for the duo's list slots: the warm-rain
# trio (nwat = 3); a deck carrying ice/snow/graupel names the six FV3
# water species in order (nwat = 6), passengers (N_*) after them
DUO_COLUMN_TRACER_NAMES = FV3_WATER_SPECIES[:3]


class DuoColumnMesh(NamedTuple):
    """What the MPAS lane and its physics read off ``model.mesh``: cell
    lat/lon [rad] and area [m^2] per column, no edge topology (a frontal
    GWD source, which needs gradients, is refused on it).  Per-cell 1-D
    like the Voronoi mesh (``grid_shape_2d == (nCells,)``), so every
    setup-time regrid (topography, SST/SIC, land, ozone) lands on the
    duo's own A-grid centres -- the driver's standard cubed-sphere
    centres are NOT these (MEASURED 2026-09-26: 1.6 deg offsets)."""
    latCell: jax.Array
    lonCell: jax.Array
    areaCell: jax.Array
    nCells: int
    grid_lat: jax.Array
    grid_lon: jax.Array
    grid_shape_2d: tuple
    lat: jax.Array
    lon: jax.Array
    #: cell corners ``(nCells, 4)`` [rad], the B-grid nodes of each compute
    #: cell in ring order: the terrain product's exact quad ownership
    cornerLat: jax.Array
    cornerLon: jax.Array

    # the rest of GridProtocol, as the Voronoi mesh defines them.  Radius
    # and rotation rate are the DUO GRID'S (FV3's gfs_constants, which the
    # six-face context is built with: areas, metrics, Coriolis), not
    # legoESM's -- 3.1e-5 relative apart; a mesh reporting one radius
    # while its areas use another would be a hidden choice.
    @property
    def grid_area(self) -> jax.Array:
        return self.areaCell

    @property
    def grid_total_area(self) -> jax.Array:
        return jnp.sum(self.areaCell)

    @property
    def grid_coriolis(self) -> jax.Array:
        return 2.0 * FV3_OMEGA * jnp.sin(self.latCell)

    @property
    def grid_radius(self) -> float:
        return FV3_RADIUS_M

    @property
    def radius(self) -> float:
        return FV3_RADIUS_M

    @property
    def grid_n_columns(self) -> int:
        return self.nCells

    def to_columns(self, field):
        return field

    def from_columns(self, cols):
        return cols


def build_duo_column_mesh(ctx_np, n: int, ng: int) -> DuoColumnMesh:
    """The column mesh of a duo grid context: the six faces' compute-
    window A-grid centres and areas, flattened ``(6, n, n) -> (6*n*n,)``
    row-major (the model's ``_columns`` order).  Built by the driver at
    grid-creation time so the forcings are regridded onto it, and by the
    model from the same context -- identical by construction."""
    ci = slice(ng, ng + n)
    def cols(key):
        return np.stack([ctx_np["gs6"][t][key][ci, ci]
                         for t in range(6)]).reshape(6 * n * n)
    lat, lon, area = cols("agrid_lat"), cols("agrid_lon"), cols("area")
    jlat, jlon = jnp.asarray(lat), jnp.asarray(lon)

    def corners(key):
        out = []
        for t in range(6):
            g = np.asarray(ctx_np["gs6"][t][key])
            if g.shape != (n + 2 * ng + 1, n + 2 * ng + 1):
                raise ValueError(f"build_duo_column_mesh: {key} has shape {g.shape}, "
                                 f"expected the padded B lattice {(n + 2 * ng + 1,) * 2}")
            sw, se = g[ng:ng + n, ng:ng + n], g[ng + 1:ng + n + 1, ng:ng + n]
            ne, nw = g[ng + 1:ng + n + 1, ng + 1:ng + n + 1], g[ng:ng + n, ng + 1:ng + n + 1]
            out.append(np.stack([sw, se, ne, nw], axis=-1))
        return jnp.asarray(np.stack(out).reshape(6 * n * n, 4))
    return DuoColumnMesh(
        latCell=jlat, lonCell=jlon, areaCell=jnp.asarray(area),
        nCells=int(lat.shape[0]), grid_lat=jlat, grid_lon=jlon,
        grid_shape_2d=(int(lat.shape[0]),), lat=jlat, lon=jlon,
        cornerLat=corners("grid_lat"), cornerLon=corners("grid_lon"))


class FV3DuoColumnState(NamedTuple):
    """``HydrostaticState`` field order plus the native duo bundle."""
    u: Field
    T: Field
    p_s: Field
    phis: Field
    v: Field | None = None
    tracers: dict | None = None
    native: dict | None = None


class FV3DuoColumnModel:
    def __init__(self, dyn: FV3DuoDynamicsModel, *, tracer_names=None,
                 conservative_tracer_clamp: bool = True,
                 energy_consistent_moisture_clip: bool = False):
        if dyn.window_layout is not None:
            raise NotImplementedError(
                "FV3DuoColumnModel runs on six faces; the window layout is "
                "certification rung 7")
        if not dyn.config.hydrostatic:
            raise NotImplementedError(
                "FV3DuoColumnModel: the column increments rebuild the "
                "hydrostatic pressures (p_var_hydrostatic); NH is not wired")
        self.dyn = dyn
        self.config = dyn.config
        self.grid = dyn.grid
        self.n, self.ng, self.km = dyn.grid.n, dyn.grid.ng, dyn.config.km
        names = tuple(DUO_COLUMN_TRACER_NAMES if tracer_names is None
                      else tracer_names)
        # nwat = how many leading names are FV3 water species, in FV3's
        # slot order: exactly 3 (warm rain) or 6 (with ice/snow/graupel);
        # a water species out of order or after a passenger would enter
        # the mass block as a passenger and its mass would be dropped
        nwat = 0
        for nm, want in zip(names, FV3_WATER_SPECIES):
            if nm != want:
                break
            nwat += 1
        stray = [nm for nm in names[nwat:] if nm in FV3_WATER_SPECIES]
        if nwat not in (3, 6) or stray or len(set(names)) != len(names):
            raise ValueError(
                f"FV3DuoColumnModel: tracer_names must start with the FV3 "
                f"water species {FV3_WATER_SPECIES[:3]} or "
                f"{FV3_WATER_SPECIES} in that order (the slots the nwat "
                f"mass block reads), then passengers, all unique; got "
                f"{names} (leading water species {nwat}, out-of-place "
                f"water species {stray})")
        self.tracer_names = names
        self.nwat = nwat
        # the MPAS lane's end-of-step positivity stage (#1354/#1515): the
        # SAME shared routine with the SAME two deck knobs -- borrow
        # (conservative_tracer_clamp=True, T untouched) or the hard floor
        # (False; with energy_consistent_moisture_clip the floor carries a
        # latent-heat T correction, which this lane does not plumb into pt:
        # refused rather than silently dropped)
        self.conservative_tracer_clamp = bool(conservative_tracer_clamp)
        self.energy_consistent_moisture_clip = bool(energy_consistent_moisture_clip)
        if self.energy_consistent_moisture_clip and not self.conservative_tracer_clamp:
            raise NotImplementedError(
                "FV3DuoColumnModel: conservative_tracer_clamp=False with "
                "energy_consistent_moisture_clip=True needs the floor's "
                "latent-heat T correction written back into pt, which this "
                "lane does not do; use the borrow (True) or drop the flag")
        self._state_type = FV3DuoColumnState
        self._phys_state = None
        self._sfc_diag = None
        ctx = dyn.grid.ctx_np
        self._tab = dyn.sixface_halo_tables
        self._amat6, _, self._wv6 = stack_held_suarez_metrics(ctx)
        ci = slice(self.ng, self.ng + self.n)
        self.mesh = build_duo_column_mesh(ctx, self.n, self.ng)
        hs6 = ctx.get("hs6")
        self._phis = (np.zeros(self.mesh.nCells) if hs6 is None else
                      self._columns(np.stack([np.asarray(hs6[t])[ci, ci]
                                              for t in range(6)])))
        # FV3 p = ak + bk*ps [Pa]; the coordinate's A is ak/p_ref
        self.sigma_coord = create_hybrid_coordinate(
            self.km, np.asarray(dyn.ak) / constants.p_ref, np.asarray(dyn.bk),
            p_ref=constants.p_ref, dtype=jnp.float64)
        self._post_fns = {}
        self._last = None      # the column state this model last returned

    # ------------------------------------------------------------------
    # layouts
    # ------------------------------------------------------------------

    def _columns(self, a6):
        """``(6, n, n[, km])`` compute window -> ``(6*n*n[, km])``."""
        return np.asarray(a6).reshape((6 * self.n * self.n,) + a6.shape[3:])

    def _faces(self, cols):
        """``(6*n*n[, km])`` -> ``(6, n, n[, km])``."""
        return jnp.reshape(cols, (6, self.n, self.n) + cols.shape[1:])

    def _tracer_index(self, name):
        try:
            return self.tracer_names.index(name)
        except ValueError:
            raise KeyError(
                f"FV3DuoColumnModel: physics returned a tendency for tracer "
                f"{name!r}; the bundle carries {self.tracer_names}") from None

    def column_view(self, bundle):
        """The physics-facing columns of a native bundle (and the D-wind
        view the increments need): ``(state, view)``."""
        view = column_view_sixface_jax(bundle["state"], self._tab, self._amat6,
                                       n=self.n, ng=self.ng, km=self.km)
        ci = slice(self.ng, self.ng + self.n)
        _, _, ua6, va6 = view
        cols = lambda a: jnp.reshape(a[:, ci, ci],  # noqa: E731
                                     (self.mesh.nCells,) + a.shape[3:])
        fld = lambda a, nm, un: Field(data=a, name=nm, units=un)  # noqa: E731
        q = bundle["q"]
        if len(q) < len(self.tracer_names):
            raise ValueError(
                f"FV3DuoColumnModel: {len(self.tracer_names)} tracer names "
                f"{self.tracer_names} for a bundle carrying {len(q)}")
        tracers = {nm: fld(cols(q[i]), nm,
                           "1/kg" if nm.startswith("N_") else "kg/kg")
                   for i, nm in enumerate(self.tracer_names)}
        state = FV3DuoColumnState(
            u=fld(cols(ua6), "u", "m/s"), v=fld(cols(va6), "v", "m/s"),
            T=fld(cols(bundle["state"]["pt"]), "T", "K"),
            p_s=fld(cols(bundle["press"]["ps"]), "p_s", "Pa"),
            phis=fld(jnp.asarray(self._phis), "phis", "m^2/s^2"),
            tracers=tracers, native=bundle)
        return state, view

    @property
    def zvir(self) -> float:
        """The dycore's thermodynamic mode (the checkpoint stamps it)."""
        return self.dyn.zvir

    def to_bundle(self, state):
        """The native bundle *state* stands for, with the driver's
        post-step column edits written back (the checkpoint writer's
        entry: what the next step would consume)."""
        return self._native_of(state)

    def from_bundle(self, bundle):
        fn = self._post_fns.get("view")
        if fn is None:
            fn = jax.jit(lambda b: self.column_view(b)[0])
            self._post_fns["view"] = fn
        self._last = fn(bundle)
        return self._last

    def _native_of(self, state):
        """The bundle a column state stands for, with the driver's
        post-step column edits (T, tracers -- e.g. a hard-saturation
        drain) written back into it.  The columns are a VIEW: an edit
        the bundle does not receive would be a silent no-op, so every
        leaf is checked by object identity against the state this model
        last returned; wind / p_s / phis edits are REFUSED (a wind edit
        needs the D-grid lift, p_s follows the layer mass)."""
        last = self._last
        if last is None or state.native is not last.native:
            raise ValueError(
                "FV3DuoColumnModel.step: the column state is not the one "
                "this model returned (build states with from_bundle and "
                "thread step's output)")
        for nm in ("u", "v", "p_s", "phis"):
            if getattr(state, nm) is not getattr(last, nm):
                raise ValueError(
                    f"FV3DuoColumnModel.step: state.{nm} was edited outside "
                    f"the model; the columns are a view of the duo bundle "
                    f"and only T and the tracers can be written back "
                    f"(a wind edit needs the D-grid lift)")
        bundle = state.native
        if state.T is last.T and state.tracers is last.tracers:
            return bundle
        ci = slice(self.ng, self.ng + self.n)
        st = dict(bundle["state"])
        if state.T is not last.T:
            st["pt"] = jnp.asarray(bundle["state"]["pt"]).at[:, ci, ci].set(
                self._faces(state.T.data))
        q = list(bundle["q"])
        if state.tracers is not last.tracers:
            # key SET, not order: a jitted output dict comes back with
            # its keys sorted (pytree flattening), the driver copies it
            if set(state.tracers) != set(self.tracer_names):
                raise ValueError(
                    f"FV3DuoColumnModel.step: tracers {tuple(state.tracers)} "
                    f"!= the model's {self.tracer_names}")
            for i, nm in enumerate(self.tracer_names):
                if state.tracers[nm] is not last.tracers[nm]:
                    q[i] = jnp.asarray(q[i]).at[:, ci, ci].set(
                        self._faces(state.tracers[nm].data))
        return {**bundle, "state": st, "q": q}

    # ------------------------------------------------------------------
    # the MPAS lane's step contract
    # ------------------------------------------------------------------

    def step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        """Dynamics, then ``physics_fn(columns, mesh, sigma_coord,
        phys_state=, forcing=)`` once on the post-dynamics columns, its
        tendencies applied over ``dt`` FV3-style.  Returns the new column
        state (native bundle inside); stashes ``_phys_state`` /
        ``_sfc_diag`` eagerly like the MPAS model.  No dry-mass fixer:
        FV3 conserves dry mass by construction."""
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="FV3DuoColumnModel.step()")
        bundle = self.dyn.step(self._native_of(state), dt)
        if physics_fn is None:
            fn = self._post_fns.get(None)
            if fn is None:
                fn = jax.jit(lambda b: {**b, "q": self._positivity(
                    b["state"], list(b["q"]))})
                self._post_fns[None] = fn
            return self.from_bundle(fn(bundle))
        fn = self._post_fns.get(physics_fn)
        if fn is None:
            fn = jax.jit(lambda b, dt_, fo, ps: self._physics_and_apply(
                b, dt_, physics_fn, fo, ps))
            self._post_fns[physics_fn] = fn
        state_new, phys_out, sfc_diag = fn(bundle, float(dt), forcing,
                                           phys_state)
        self._last = state_new
        if not any(isinstance(leaf, jax.core.Tracer)
                   for leaf in jax.tree_util.tree_leaves(phys_out)):
            self._phys_state = phys_out
        if sfc_diag is not None and not any(
                isinstance(leaf, jax.core.Tracer)
                for leaf in jax.tree_util.tree_leaves(sfc_diag)):
            # slot-wise merge as the MPAS model does: sw/lw refresh only
            # on a radiation step, precip every step -- keep the last
            # non-None value per slot
            prev = self._sfc_diag or (None,) * len(sfc_diag)
            self._sfc_diag = tuple(
                new if new is not None else old
                for new, old in zip(sfc_diag, prev))
        return state_new

    def _positivity(self, st, q):
        """The MPAS lane's end-of-step positivity stage, the shared routine
        verbatim (``apply_water_positivity``: column-conserving BORROW
        weighted by the layer mass for every borrow-eligible species and a
        plain floor for the rest, or the hard floor when the borrow is
        off; T untouched on both arms this lane admits), on the compute
        window, EVERY step as on MPAS.  Weight = delp * area: the column
        borrow is per column (a per-column factor cancels) but the global
        residual redistribution sums over cells, and duo cells differ 1.4x
        corner to centre -- delp alone would mis-conserve mass.  Applied
        per tracer only where a negative exists: identity in exact
        arithmetic otherwise, but the global rescale is 1 ulp off under
        jit, and the certified bitwise identities with the closed lane
        (rung 1) must hold on non-negative fields."""
        from legoesm.core.conservation import (
            apply_water_positivity, cell_mass_weight)
        ci = slice(self.ng, self.ng + self.n)
        dp_w = cell_mass_weight(st["delp"][:, ci, ci, :],
                                self._faces(self.mesh.areaCell))
        tr_w = {nm: q[i][:, ci, ci, :]
                for i, nm in enumerate(self.tracer_names)}
        fixed, _ = apply_water_positivity(
            tr_w, None, dp_w, conservative=self.conservative_tracer_clamp,
            energy_consistent=False)
        return [q[i].at[:, ci, ci, :].set(jnp.where(
                    jnp.any(tr_w[nm] < 0.0), fixed[nm], tr_w[nm]))
                for i, nm in enumerate(self.tracer_names)]

    def _physics_and_apply(self, bundle, dt, physics_fn, forcing, phys_state):
        from legoesm.core.state import MPAS_SFC_DIAG_EXTRA_KEYS
        cols, view = self.column_view(bundle)
        pr = physics_fn(cols, self.mesh, self.sigma_coord,
                        phys_state=phys_state, forcing=forcing)
        tend, phys_out = (pr[0], pr[1]) if type(pr) is tuple else (pr, phys_state)
        d = lambda x: getattr(x, "data", x)  # noqa: E731
        if tend.dv_dt is None:
            raise ValueError(
                "FV3DuoColumnModel: physics returned dv_dt=None; the column "
                "model carries both cell wind components (make_physics with "
                "model_type='mpas' on a state whose v is present)")
        u_dt_c = self._faces(d(tend.du_dt))
        v_dt_c = self._faces(d(tend.dv_dt))
        # moist deck: legoESM c_pd heating rescaled to cp_air and applied
        # on cvm inside the applier; dry deck (idealized): applied as given
        t_dt_c = self._faces(d(tend.dT_dt))
        # dp_s_dt is NOT consumed: p_s follows the layer mass the water
        # tendencies move (MPAS physics returns zeros there, survey
        # 2026-09-26); there is no other pressure source on this lane
        q_dt_c = {}
        if tend.tracer_tendencies is not None:
            for nm, tq in tend.tracer_tendencies.items():
                if nm not in self.tracer_names:
                    # the MPAS model's contract (primitive_eq_mpas.py:
                    # tracer loop over state.tracers): a tendency for a
                    # tracer the state does not carry is dropped (the
                    # integrations emit the full warm/ice set)
                    continue
                q_dt_c[self._tracer_index(nm)] = self._faces(d(tq))
        if q_dt_c and not self.config.moist:
            raise ValueError(
                "FV3DuoColumnModel: physics returned tracer tendencies on "
                "the DRY deck (FV3DuoConfig.moist=False ignores humidity in "
                "the dycore); run the moist deck")
        st, press, q = apply_column_increments_sixface_jax(
            bundle["state"], bundle["press"], list(bundle["q"]), view,
            self._tab, self._wv6, u_dt_c, v_dt_c, t_dt_c, q_dt_c, dt=dt,
            n=self.n, ng=self.ng, km=self.km, ptop=self.dyn.ptop,
            akap=FV3_KAPPA, moist_cp=self.config.moist, nwat=self.nwat)
        q = self._positivity(st, q)
        new_bundle = {**bundle, "state": st, "press": press, "q": q}
        sfc = (getattr(tend, "sw_net_sfc", None),
               getattr(tend, "lw_net_sfc", None),
               getattr(tend, "precip", None)) + tuple(
            getattr(tend, k, None) for k in MPAS_SFC_DIAG_EXTRA_KEYS)
        sfc_diag = sfc if any(s is not None for s in sfc) else None
        return self.column_view(new_bundle)[0], phys_out, sfc_diag
