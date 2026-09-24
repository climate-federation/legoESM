"""FV3 six-face duo-cube dycore — ModelDriver wrapper over the certified lane.

Wires ``legoesm.core.fv3_dynamics`` (the JAX ``fv_dynamics`` twin,
module 6 of the duo port) into the model: DRY dynamics, fp64, DCMIP16
baroclinic wave IC.  The only physics is the certified Held-Suarez step,
applied by the driver lane (``_run_fv3_duo``) when
``held_suarez_forcing`` is set — this model class itself stays
dynamics-only.  Every restriction is the certified lane's own contract,
enforced loudly here and at the component factory rather than assumed:

* moist coupling is routed ONLY by ``FV3DuoConfig.moist`` (2026-09-24):
  it passes the oracle's ``zvir`` with tracer 0 as specific humidity and
  builds the moist IC; the default deck stays adiabatic (``zvir = 0``,
  ``dp1`` never formed).  ``consv_te != 0`` is still refused by the core;
* f64 is the DEFAULT and certified storage dtype (``storage_dtype``);
  the phase gates enforce dtype UNIFORMITY and ``step`` enforces the
  configured storage dtype at the boundary. fp32/mixed storage is
  accepted by the config but the RUNTIME is not yet wired (refused at
  construction);
* the vertical coordinate is ``set_eta_analytic``'s ``km in {5, 10}``
  branch (fv_eta.F90:334-344) — any other km raises there;
* ``kord_tm`` must be NEGATIVE: a positive value selects a different
  remap operator (fv_mapz.F90:495-501) that is not ported.

State contract: ``step(state, dt)`` takes and returns ONE bundled pytree
``{"state", "press", "q", "omga", "nh"}`` — the exact dicts
``fv_dynamics_step`` threads (face-stacked prognostics; the pressure
bundle ``p_var`` builds; ``q`` a list of face-stacked passenger tracers;
``omga`` an output-only passenger whose values are meaningless on this
lane; ``nh`` the NH carry, ``None`` on the hydrostatic arm).  The caller
threads the WHOLE bundle into the next call (C4); the wrapper never
mutates an argument.

Constants: ``FV3_KAPPA`` / ``FV3_CP_AIR`` are the FMS **GFS** set the
pinned oracle binary links (asserted from the oracle run's own log by the
parity runner) — deliberately NOT ``legoesm.constants``; the two differ
at ~1e-4 relative and mixing them was the 2026-08-07 IC-parity confound.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_acoustic_3d import build_nh_carry
from legoesm.core.fv3_cgrid_phase_3d import state_3d_to_jax
from legoesm.core.fv3_dynamics import (
    make_fv_dynamics_step_jit,
    p_var_hydrostatic,
    p_var_nonhydrostatic,
)
from legoesm.core.fv3_native_dcmip16_ic import (
    dcmip16_bc_six_face_state,
    dcmip16_terminator_six_face,
)
from legoesm.core.fv3_native_eta import set_eta_analytic
from legoesm.core.fv3_native_state_3d import field_shape
from legoesm.core.fv3_tracer2d import check_nsplt_schedule
from legoesm.grids.fv3_native_gridstruct import (FV3_CP_AIR, FV3_KAPPA,
                                                 FV3_RDGAS, FV3_RVGAS)

#: The bundled-pytree keys ``step`` threads — the ``fv_dynamics_step``
#: return keys minus its non-array metadata.
FV3_DUO_STATE_KEYS = ("state", "press", "q", "omga", "nh")


class FV3DuoConfig(NamedTuple):
    """Deck constants of the certified duo lane (all STATIC — baked into
    the compiled step; a new value is a new compile).

    Defaults are the pinned oracle deck's resolved namelist
    (``full_step_oracle_parity.py``: k_split=1, n_split=8,
    kord_mt=9, kord_tm=-9, kord_tr=9).  ``km`` must be in {5, 10}
    (fv_eta.F90:334-344 — the only analytic ``set_eta`` branch).
    """
    km: int = 5
    k_split: int = 1
    n_split: int = 8
    hydrostatic: bool = True
    kord_mt: int = 9
    kord_tm: int = -9
    kord_tr: int = 9
    # Storage/compute dtype of the prognostic carry. DEFAULT "float64" is
    # the certified oracle path (byte-identical to every prior run). A
    # COARSE precision policy: "float32" runs the whole step in fp32 (the
    # gates enforce dtype UNIFORMITY, not strict fp64, so an f32 IC flows
    # through). Per-op mixed (fp64 pressure column / energy fixer, fp32
    # elsewhere) is a SEPARATE later increment; this field only selects a
    # single uniform storage dtype. NOTE (2026-08-28): fp32 is NOT yet
    # runnable end-to-end -- the in-phase workspace allocations and grid
    # metrics are still fp64-pinned; those must be threaded to this dtype
    # before an fp32 step passes the uniformity gates. Setting "float32"
    # today fails LOUDLY at the first fp64 workspace, by design.
    storage_dtype: str = "float64"
    #: MOIST coupling (2026-09-24): ``True`` routes the oracle's own
    #: ``zvir = rvgas/rdgas - 1`` with tracer 0 as specific humidity into
    #: the step (virtual temperature in pt_to_theta_v and the remap; the
    #: core arm certified at 1.19e-09 vs the Fortran moist deck) and
    #: builds the MOIST DCMIP16 IC (pt divided by 1 + zvir*q).  ``False``
    #: is the certified adiabatic deck: humidity a passenger.  Selected
    #: automatically by Kessler in the driver (user 2026-09-24), never a
    #: knob there.
    moist: bool = False


def lon_modulated_tracer(sphum, agrid_lon, n: int, ng: int, iq: int):
    """``sphum * (1 + 0.5 sin(iq * lon))`` on the compute window of one
    padded face array; halos untouched (zero, as sphum's are)."""
    cs = slice(ng, ng + n)
    q = np.array(sphum, dtype=np.float64)
    lon = np.asarray(agrid_lon, dtype=np.float64)[cs, cs]
    q[cs, cs, :] *= (1.0 + 0.5 * np.sin(iq * lon))[:, :, None]
    return q


class FV3DuoDynamicsModel:
    """One ``fv_dynamics`` outer step per ``step(state, dt)`` call.

    Constructed on a :class:`legoesm.grids.factory.FV3DuoGridBundle`
    (never a plain cubed-sphere grid — the duo halo tables and bounded
    gridstructs live only in the bundle).  The jitted step is built ONCE
    here; ``dt`` is the only dynamic deck quantity (``bdt``), so a new
    ``dt`` does not recompile.
    """

    def __init__(self, grid, config: FV3DuoConfig | None = None, *,
                 step_out_shardings=None, step_spmd_mesh=None,
                 step_face_batched: bool = False,
                 step_windows=None):
        # step_windows: ENGINEERING knob (M6, the tiled port) -- (kt, pad)
        # runs the step on 6*kt*kt sub-face WINDOWS (fv3_duo_windows,
        # padded partition, kt | m_a) instead of six faces: with
        # step_spmd_mesh a (6, kt, kt) tile mesh, one window per device
        # (fv3_duo_window_spmd); without a mesh, the single-device window
        # arm (every exchange through the flat impl).  The kernels are
        # the certified ones vmapped over the windows (batched arm
        # forced); the state bundle is WINDOW-stacked (to_windows /
        # to_flat convert).  Selects no scientific configuration: bitwise
        # against the face-sharded flat step (jobs 9632470-3).  None keeps
        # the certified six-face path byte-identical.
        # step_face_batched: ENGINEERING knob (face-batching ladder) --
        # routes the 3-D phases' per-face loops through their vmapped
        # arms (batched==loop gated at rtol 1e-13 per phase). Selects no
        # scientific configuration; default False = the certified loop
        # trace, byte-identical.
        # step_out_shardings: ENGINEERING knob -- ONE jax.sharding.Sharding
        # applied to each face-stacked output leaf (state/press/q/omga/nh;
        # NOT a jit out_shardings pytree prefix).  It selects no scientific
        # configuration; sharding can move last-bit float results via
        # reduction/fusion order (measured 4e-15 rel, inside the parity
        # gate).  Default None = the unconstrained jit the certified lane
        # always used. SPMD callers pin the face sharding here because an
        # unconstrained jit resolves sharded-input outputs REPLICATED
        # (measured, spmd_face_shard_parity 2026-08-24) and a stepping
        # loop then decays after one step.
        # step_spmd_mesh: ENGINEERING knob (M3) -- a jax.sharding.Mesh
        # with one 'face' axis.  When set, the jitted step's halo
        # exchanges route through the O(halo) shard_map ring built on
        # that mesh (bitwise-equal to the certified exchanges,
        # test_fv3_duo_spmd), instead of leaving the face-stacked
        # gathers to GSPMD.  Selects no scientific configuration.
        # Default None = the certified single-device trace, byte-
        # identical.  SPMD callers (spmd_face_shard_parity --ring) pass
        # BOTH knobs -- the mesh for the interior exchanges and
        # step_out_shardings for the output boundary; they are not
        # coupled here.
        if config is None:
            config = FV3DuoConfig()
        if not isinstance(config, FV3DuoConfig):
            raise TypeError(
                f"FV3DuoDynamicsModel: config must be an FV3DuoConfig, got "
                f"{type(config).__name__}")
        # COARSE precision policy (2026-08-28). The dtype gates are now
        # uniformity gates (require_uniform_float_jax) and this config
        # carries a storage_dtype, but the fp32/mixed RUNTIME is not yet
        # wired: the in-phase workspace allocations (~57 jnp.float64 zeros
        # across the phase modules) and the grid metrics / halo tables are
        # still fp64-pinned, so an fp32 step would trip a uniformity gate
        # the moment an fp64 workspace meets the f32 carry. Refuse it HERE,
        # loudly, with the remainder named -- rather than deep in a phase
        # -- until that surgery lands. fp64 is the certified default and is
        # byte-identical (uniform-f64 passes every gate exactly as strict-
        # f64 did).
        # Normalise once (np.dtype gives a loud ValueError on a bad
        # string, not a silent mismatch later). self._storage_dtype is the
        # single source of truth the per-step boundary guard checks the
        # incoming carry against.
        self._storage_dtype = np.dtype(config.storage_dtype)
        if self._storage_dtype not in (np.float32, np.float64):
            raise NotImplementedError(
                f"fv3_duo storage_dtype={config.storage_dtype!r}: only "
                f"'float32' (coarse fp32) and 'float64' (certified default) "
                f"are wired. A true per-op MIXED mode (fp64 pressure column "
                f"/ energy fixer, fp32 elsewhere) is a later increment.")
        for attr in ("ctx_np", "ctx_jax", "n", "ng"):
            if not hasattr(grid, attr):
                raise TypeError(
                    f"FV3DuoDynamicsModel: grid must be an FV3DuoGridBundle "
                    f"(legoesm.grids.factory.create_fv3_duo_grid), got "
                    f"{type(grid).__name__} without {attr!r}")
        if config.kord_tm >= 0:
            raise ValueError(
                f"kord_tm={config.kord_tm} must be NEGATIVE: a positive "
                f"kord_tm selects the remap-theta_v-in-linear-p operator "
                f"(fv_mapz.F90:495-501), which the certified lane does not "
                f"port. The pinned deck runs kord_tm=-9.")
        _kords = (config.kord_mt, config.kord_tm, config.kord_tr)
        if _kords != (9, -9, 9):
            raise ValueError(
                f"(kord_mt, kord_tm, kord_tr)={_kords} is outside the "
                f"certified deck: the duo-lane certification (full-step "
                f"oracle parity) is DECK-PINNED to (9, -9, 9). Other remap "
                f"orders would run OUTSIDE the certified configuration "
                f"(or fail only at the first remap, step 1), so they are "
                f"refused at construction.")
        if config.k_split < 1 or config.n_split < 1:
            raise ValueError(
                f"k_split={config.k_split} / n_split={config.n_split} must "
                f"both be >= 1 (fv_dynamics.F90:451 / dyn_core.F90:337).")
        # km in {5, 10} — set_eta_analytic raises with the fv_eta citation.
        ak, bk, ptop, _ks = set_eta_analytic(config.km)

        self.grid = grid
        self.config = config
        # Retained (not just consumed) so a multi-process restart loader
        # can reconstruct GSPMD-sharded arrays against the EXACT sharding
        # object the compiled step uses, instead of an independently
        # rebuilt "similar" one that could drift in mesh/device order
        # (codex MAJOR, mp-driver-io design review 2026-08-27).
        self.step_spmd_mesh = step_spmd_mesh
        # The grid bundle's ctx_jax was built WITHOUT a mesh; a ring-
        # enabled context is rebuilt here from ctx_np rather than
        # mutating the shared bundle's tables in place (the tables hash
        # by identity as a STATIC jit arg, so mutating them under an
        # already-traced certified step would leave a stale cache).
        # Context dtype (fp32/mixed increment 2): the grid metrics are fp64
        # host constants; for a coarse fp32 run they must be cast to the
        # storage dtype or they promote the f32 state back to f64 (and trip
        # the uniformity gates). Rebuild the context at storage_dtype when
        # it is not fp64. For the certified fp64 default we keep the shared
        # grid.ctx_jax verbatim (no rebuild -> byte-identical). A ring
        # (spmd_mesh) run already rebuilds; it now also carries the dtype.
        _fp32 = self._storage_dtype == np.float32
        _ctx_dtype = jnp.float32 if _fp32 else None   # None = uncast f64
        self.window_layout = None
        self._window_comm = None
        self._window_sharding = None
        if step_windows is not None:
            from legoesm.core.fv3_duo_stepper import (
                build_jax_duo_stepper_context,
            )
            if _fp32:
                raise NotImplementedError(
                    "step_windows with storage_dtype float32: the tiled "
                    "lane is fp64-only (scope: fp32 on the tiled lane is "
                    "out of scope)")
            kt, pad = (int(v) for v in step_windows)
            # a FRESH context: the window comm rides on its tables, and
            # the shared grid bundle's tables must not be mutated
            flat_ctx = build_jax_duo_stepper_context(grid.ctx_np)
            if step_spmd_mesh is not None:
                from legoesm.grids.fv3_duo_window_spmd import (
                    attach_window_spmd_comm)
                if tuple(step_spmd_mesh.devices.shape) != (6, kt, kt):
                    raise ValueError(
                        f"step_windows kt={kt} needs a (6, {kt}, {kt}) "
                        f"mesh, got {tuple(step_spmd_mesh.devices.shape)}")
                self._ctx_jax, self._window_comm = attach_window_spmd_comm(
                    flat_ctx, step_spmd_mesh, pad)
                self._window_sharding = self._window_comm.sharding
                if step_out_shardings is None:
                    step_out_shardings = self._window_sharding
            else:
                from legoesm.grids.fv3_duo_windows import attach_window_comm
                self._ctx_jax, self._window_comm = attach_window_comm(
                    flat_ctx, kt, pad, "padded")
            self._flat_ctx_jax = flat_ctx
            self.window_layout = self._window_comm.lay
            step_face_batched = True
            # the window ctx's n_w and the vertical extent must not
            # collide: to_windows/to_flat classify horizontal axes by
            # extent (fv3_duo_windows.horizontal_axes) and a km+1 inside
            # [n, m_a+1] would be ambiguous -- refuse HERE, not deep in
            # the IC (codex 2026-09-05; only toy grids can trigger it)
            if grid.n <= config.km + 1 <= grid.n + 2 * grid.ng + 1:
                raise NotImplementedError(
                    f"step_windows: km+1 = {config.km + 1} lies inside the "
                    f"horizontal extent range [{grid.n}, "
                    f"{grid.n + 2 * grid.ng + 1}] of C{grid.n}; the window "
                    f"layout cannot tell levels from a horizontal axis")
        elif step_spmd_mesh is None and not _fp32:
            self._ctx_jax = grid.ctx_jax
        else:
            from legoesm.core.fv3_duo_stepper import (
                build_jax_duo_stepper_context,
            )
            self._ctx_jax = build_jax_duo_stepper_context(
                grid.ctx_np, spmd_mesh=step_spmd_mesh, dtype=_ctx_dtype)
        # recorded AFTER the window default so a caller (the multiprocess
        # driver) reads the sharding the step was actually built with
        self.step_out_shardings = step_out_shardings
        # ak/bk (eta coefficients) enter p_var alongside delp -> they must
        # be the storage dtype too. fp64 default is byte-identical.
        self._ak = np.asarray(ak, dtype=self._storage_dtype)
        self._bk = np.asarray(bk, dtype=self._storage_dtype)
        self._ptop = float(ptop)
        # The resolved NH deck runs W_LIMITER=T (fv_mapz.F90:368); the core
        # refuses hydrostatic=False without an explicit choice.
        self._step_fn = make_fv_dynamics_step_jit(
            self._ctx_jax, config.km,
            k_split=config.k_split, n_split=config.n_split,
            ptop=self._ptop, ak=self._ak, bk=self._bk,
            akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
            kord_mt=config.kord_mt, kord_tm=config.kord_tm,
            kord_tr=config.kord_tr,
            hydrostatic=config.hydrostatic,
            w_limiter=(None if config.hydrostatic else True),
            out_shardings=step_out_shardings,
            batched=step_face_batched,
            zvir=self.zvir, sphum_index=(0 if config.moist else None),
        )

    @property
    def zvir(self) -> float:
        """``rvgas/rdgas - 1`` with the oracle's gas constants when the
        deck is moist (atmosphere.F90:156-161), else exactly 0.0."""
        return (FV3_RVGAS / FV3_RDGAS - 1.0) if self.config.moist else 0.0

    # ------------------------------------------------------------------
    # DycoreProtocol
    # ------------------------------------------------------------------

    def step(self, state, dt):
        """Advance the bundled pytree by ``dt`` seconds (one fv_dynamics
        call: ``k_split`` remaps over ``k_split * n_split`` acoustic
        sub-steps).  Pure ``state -> state``; the input bundle is not
        mutated.

        Per-step HOST SYNC: ``check_nsplt_schedule`` inspects a concrete
        output scalar (``np.asarray`` + ``bool``), so every ``step`` call
        carries one device-to-host transfer that acts as a completion
        barrier -- the step is jitted but NOT async-dispatch-pipelined.
        This is the accepted cost of the C5 fail-closed guard (an nsplt
        above NSPLT_MAX would otherwise under-advect tracers silently); a
        future alternative is a fused in-graph check with a sticky
        failure flag read at a coarser (segment) cadence."""
        if not isinstance(state, dict) or set(state) != set(
                FV3_DUO_STATE_KEYS):
            got = sorted(state) if isinstance(state, dict) else type(
                state).__name__
            raise ValueError(
                f"FV3DuoDynamicsModel.step: state must be a dict with keys "
                f"{sorted(FV3_DUO_STATE_KEYS)}, got {got}. Build it with "
                f"dcmip16_initial_state().")
        # BOUNDARY dtype guard (2026-08-28, triple-review fix): the phase
        # gates now check dtype UNIFORMITY, not strict fp64, so a
        # uniformly-f32 carry fed to an fp64-configured run would sail
        # through them and run SILENTLY in f32 -- the exact
        # "lost bits before step 1" bug the old strict gate caught
        # incidentally. The relocated protection lives HERE: every inexact
        # leaf of the incoming bundle MUST equal the configured storage
        # dtype (symmetric -- also blocks a silent f64->f32 upcast for a
        # future fp32 run). Host-side aval read, no device work, so the
        # certified fp64 step stays byte-identical. `raise`, never
        # `assert` (elidable under -O). Non-inexact leaves (int indices,
        # bool masks) are skipped.
        for leaf in jax.tree_util.tree_leaves(state):
            _ldt = getattr(leaf, "dtype", None)   # NOT `dt` -- that is the
            if _ldt is not None and jnp.issubdtype(_ldt, jnp.inexact) \
                    and np.dtype(_ldt) != self._storage_dtype:  # timestep arg
                raise TypeError(
                    f"FV3DuoDynamicsModel.step: a state leaf is {np.dtype(_ldt)} "
                    f"but the configured storage_dtype is "
                    f"{self._storage_dtype} -- refusing a silent precision "
                    f"change. Cast the bundle to storage_dtype (a stray "
                    f"float32 IC, e.g. a bare jnp.zeros or a restart read, "
                    f"is the usual cause).")
        out = self._step_fn(state["state"], state["press"], state["q"],
                            dt, state["omga"], state["nh"])
        # C5, on concrete outputs OUTSIDE jit: an nsplt above NSPLT_MAX
        # would otherwise under-advect the tracers silently.
        check_nsplt_schedule(out)
        # the resolved per-level sub-cycle schedule of this step, kept for
        # instruments (the rank ladder prints it per rank: under
        # jax.distributed it must be identical on every process)
        self.last_nsplt = np.asarray(out["nsplt"])
        if out["pt_units"] != "K":
            # unreachable while km in {5,10} (> REMAP_MIN_NPZ = 4), but a
            # future km table below 5 would hand back theta_v — fail loudly.
            raise RuntimeError(
                f"fv_dynamics left pt in {out['pt_units']!r}, not K — the "
                f"remap did not run (km <= 4?); this wrapper's state "
                f"contract is temperature.")
        return {k: out[k] for k in FV3_DUO_STATE_KEYS}

    # ------------------------------------------------------------------
    # Initial condition (slice 1: DCMIP16 baroclinic wave only)
    # ------------------------------------------------------------------

    def dcmip16_initial_state(self, *, do_pert: bool = True,
                              n_tracers: int = 1,
                              terminator: bool = False) -> dict:
        """The DCMIP16_BC (test_case = -13 / -12) bundle on this grid.

        ``terminator=True`` appends the oracle's own DCMIP16 terminator
        pair ``cl, cl2`` (``dcmip16_terminator_six_face``) AFTER the
        ``n_tracers`` list -- the same nonzero, longitude-dependent
        passengers the Fortran cold start fills when its field_table lists
        them, so a run can be scored tracer-for-tracer against that deck.

        ``n_tracers > 1`` appends passenger tracers ``iq = 1..n-1`` built
        as ``sphum * (1 + 0.5 sin(iq * lon))`` on the compute window (halos
        stay zero, like sphum's).  sphum itself is zonally symmetric, so a
        longitude-shifted copy would be the SAME field; the modulation is
        what makes tracer ``iq`` distinguishable from tracer 0 and from
        every other ``iq`` -- an index swap or a dropped tracer in the
        sharded transport shows up, a copy would not.  Every extra tracer
        is strictly positive where sphum is.

        ``sphum`` rides as a PASSENGER tracer (the lane advects it;
        ``zvir`` stays 0 so it never feeds back).  Pressures come from
        the lane's own ``p_var`` (hydro or NH per the config), which is
        the ONLY producer of the ``pkz`` the first ``pt -> theta_v``
        conversion divides by.  The NH carry is prebuilt here so every
        subsequent ``step`` call is ONE compiled program (an ``nh=None``
        first call would compile a second, carry-building variant).
        """
        cfg = self.config
        st6, sphum6 = dcmip16_bc_six_face_state(
            self.grid.ctx_np, self._ak, self._bk, cfg.km,
            hydrostatic=cfg.hydrostatic, do_pert=do_pert, zvir=self.zvir)
        jstate = state_3d_to_jax(st6)
        n, ng = self.grid.n, self.grid.ng
        flat_ctx = (self._flat_ctx_jax if self.window_layout is not None
                    else self._ctx_jax)
        if cfg.hydrostatic:
            press = p_var_hydrostatic(
                jstate["delp"], ptop=self._ptop, akap=FV3_KAPPA,
                n=n, ng=ng, km=cfg.km)
            nh = None
        else:
            press = p_var_nonhydrostatic(
                jstate["delp"], jstate["delz"], jstate["pt"],
                ptop=self._ptop, akap=FV3_KAPPA, n=n, ng=ng, km=cfg.km)
            nh = build_nh_carry(flat_ctx, cfg.km, flat_ctx.hs6)
        q = [jnp.asarray(np.stack(sphum6))]
        if n_tracers < 1:
            raise ValueError(f"n_tracers must be >= 1, got {n_tracers}")
        for iq in range(1, n_tracers):
            q.append(jnp.asarray(np.stack([
                lon_modulated_tracer(
                    sphum6[t], self.grid.ctx_np["gs6"][t]["agrid_lon"],
                    n, ng, iq)
                for t in range(6)])))
        if terminator:
            pairs = dcmip16_terminator_six_face(self.grid.ctx_np, cfg.km)
            for iq in range(2):
                q.append(jnp.asarray(np.stack([pairs[t][iq]
                                               for t in range(6)])))
        omga = jnp.zeros(
            (6,) + tuple(field_shape("delp", n, ng, cfg.km)),
            dtype=jnp.float64)
        bundle = {"state": jstate, "press": press, "q": q, "omga": omga,
                  "nh": nh}
        # Correct-by-construction (GLM review): the primary IC provenance
        # ends AT the configured storage dtype, so the boundary guard in
        # step() never has to reject the model's own IC. For the certified
        # fp64 default this is an identity cast (every leaf is already
        # float64 -> astype is a no-op, byte-identical). For a future fp32
        # run it is the one place the IC is downcast; every OTHER
        # provenance (restart reads, external ICs) is caught by step()'s
        # boundary guard instead.
        bundle = jax.tree_util.tree_map(
            lambda a: (a.astype(self._storage_dtype)
                       if jnp.issubdtype(getattr(a, "dtype", np.int64),
                                         jnp.inexact) else a),
            bundle)
        return self.to_windows(bundle) if self.window_layout else bundle

    @property
    def sixface_halo_tables(self):
        """The six-face ``DuoHaloTables`` this model's exchanges are built
        on (the flat context's tables under a window layout).  Public so
        the driver's face-stacked physics step can reuse the certified
        exchange tables instead of reaching into private context."""
        ctx = (self._flat_ctx_jax if self.window_layout is not None
               else self._ctx_jax)
        return ctx.tab

    # ------------------------------------------------------------------
    # window layout conversions (M6)
    # ------------------------------------------------------------------

    def to_windows(self, bundle):
        """Six-face bundle -> window-stacked bundle (every horizontal array
        gathered into its 6*kt*kt windows, placed on the window sharding
        when the step is SPMD).  Identity when the model runs on faces."""
        if self.window_layout is None:
            return bundle
        if self.window_layout.kt == 1 and self._window_sharding is None:
            return bundle                      # W == m_a: the flat layout
        from legoesm.grids.fv3_duo_windows import (gather_windows,
                                                   horizontal_axes)
        lay, sh = self.window_layout, self._window_sharding

        def conv(a):
            if (hasattr(a, "ndim")
                    and horizontal_axes(lay, a.shape, 6) is not None):
                # host-side gather: on device every one of the nb window
                # slices is its own eagerly compiled XLA program per leaf
                # (nb*leaves executables cached per process -- ~1.9 GB at
                # C96 kt=3, growing with kt^2; 216-rank C192 OOM 2026-09-05)
                w = gather_windows(lay, np.asarray(a), np)
                if sh is None:
                    return jnp.asarray(w)
                # NOT device_put: for a multi-process sharding device_put
                # first asserts the host array is identical on every process
                # by gathering the WHOLE array across all ranks, per leaf
                # (jax dispatch._device_put_sharding_impl) -- 27 GB/rank and
                # 5 h for the 216-rank C192 IC (job 9654155).  The callback
                # form transfers only this process's shards.
                return jax.make_array_from_callback(
                    w.shape, sh, lambda idx, w=w: w[idx])
            return a
        return jax.tree_util.tree_map(conv, bundle)

    def to_flat(self, bundle):
        """Window-stacked bundle -> six-face bundle: each window's OWNED
        cells (host-side scatter, no device collective).  Identity when
        the model runs on faces."""
        if self.window_layout is None:
            return bundle
        if self.window_layout.kt == 1:
            return jax.tree_util.tree_map(
                lambda a: np.asarray(a) if hasattr(a, "ndim") else a, bundle)
        from legoesm.grids.fv3_duo_windows import (scatter_owned,
                                                   horizontal_axes)
        lay = self.window_layout

        def conv(a):
            if (hasattr(a, "ndim")
                    and horizontal_axes(lay, a.shape, lay.nb) is not None):
                return scatter_owned(lay, np.asarray(a), np)
            return a
        return jax.tree_util.tree_map(conv, bundle)
