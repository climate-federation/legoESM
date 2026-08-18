"""FV3 six-face duo-cube dycore — ModelDriver wrapper over the certified lane.

Slice 1 of wiring ``legoesm.core.fv3_dynamics`` (the JAX ``fv_dynamics``
twin, module 6 of the duo port) into the model: DRY, physics-off, fp64,
DCMIP16 baroclinic wave only.  Every restriction is the certified lane's
own contract, enforced loudly here and at the component factory rather
than assumed:

* moist coupling is REFUSED by the core (``zvir != 0`` / ``consv_te != 0``
  raise, fv3_dynamics.py:301-311) — this wrapper never passes either;
* f64 is REQUIRED (``require_f64_jax`` gates every leaf);
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

import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_acoustic_3d import build_nh_carry
from legoesm.core.fv3_cgrid_phase_3d import state_3d_to_jax
from legoesm.core.fv3_dynamics import (
    make_fv_dynamics_step_jit,
    p_var_hydrostatic,
    p_var_nonhydrostatic,
)
from legoesm.core.fv3_native_dcmip16_ic import dcmip16_bc_six_face_state
from legoesm.core.fv3_native_eta import set_eta_analytic
from legoesm.core.fv3_native_state_3d import field_shape
from legoesm.core.fv3_tracer2d import check_nsplt_schedule
from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA

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


class FV3DuoDynamicsModel:
    """One ``fv_dynamics`` outer step per ``step(state, dt)`` call.

    Constructed on a :class:`legoesm.grids.factory.FV3DuoGridBundle`
    (never a plain cubed-sphere grid — the duo halo tables and bounded
    gridstructs live only in the bundle).  The jitted step is built ONCE
    here; ``dt`` is the only dynamic deck quantity (``bdt``), so a new
    ``dt`` does not recompile.
    """

    def __init__(self, grid, config: FV3DuoConfig | None = None):
        if config is None:
            config = FV3DuoConfig()
        if not isinstance(config, FV3DuoConfig):
            raise TypeError(
                f"FV3DuoDynamicsModel: config must be an FV3DuoConfig, got "
                f"{type(config).__name__}")
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
        if config.k_split < 1 or config.n_split < 1:
            raise ValueError(
                f"k_split={config.k_split} / n_split={config.n_split} must "
                f"both be >= 1 (fv_dynamics.F90:451 / dyn_core.F90:337).")
        # km in {5, 10} — set_eta_analytic raises with the fv_eta citation.
        ak, bk, ptop, _ks = set_eta_analytic(config.km)

        self.grid = grid
        self.config = config
        self._ak = np.asarray(ak, dtype=np.float64)
        self._bk = np.asarray(bk, dtype=np.float64)
        self._ptop = float(ptop)
        # The resolved NH deck runs W_LIMITER=T (fv_mapz.F90:368); the core
        # refuses hydrostatic=False without an explicit choice.
        self._step_fn = make_fv_dynamics_step_jit(
            grid.ctx_jax, config.km,
            k_split=config.k_split, n_split=config.n_split,
            ptop=self._ptop, ak=self._ak, bk=self._bk,
            akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
            kord_mt=config.kord_mt, kord_tm=config.kord_tm,
            kord_tr=config.kord_tr,
            hydrostatic=config.hydrostatic,
            w_limiter=(None if config.hydrostatic else True),
        )

    # ------------------------------------------------------------------
    # DycoreProtocol
    # ------------------------------------------------------------------

    def step(self, state, dt):
        """Advance the bundled pytree by ``dt`` seconds (one fv_dynamics
        call: ``k_split`` remaps over ``k_split * n_split`` acoustic
        sub-steps).  Pure ``state -> state``; the input bundle is not
        mutated."""
        if not isinstance(state, dict) or set(state) != set(
                FV3_DUO_STATE_KEYS):
            got = sorted(state) if isinstance(state, dict) else type(
                state).__name__
            raise ValueError(
                f"FV3DuoDynamicsModel.step: state must be a dict with keys "
                f"{sorted(FV3_DUO_STATE_KEYS)}, got {got}. Build it with "
                f"dcmip16_initial_state().")
        out = self._step_fn(state["state"], state["press"], state["q"],
                            dt, state["omga"], state["nh"])
        # C5, on concrete outputs OUTSIDE jit: an nsplt above NSPLT_MAX
        # would otherwise under-advect the tracers silently.
        check_nsplt_schedule(out)
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

    def dcmip16_initial_state(self, *, do_pert: bool = True) -> dict:
        """The DCMIP16_BC (test_case = -13 / -12) bundle on this grid.

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
            hydrostatic=cfg.hydrostatic, do_pert=do_pert)
        jstate = state_3d_to_jax(st6)
        n, ng = self.grid.n, self.grid.ng
        if cfg.hydrostatic:
            press = p_var_hydrostatic(
                jstate["delp"], ptop=self._ptop, akap=FV3_KAPPA,
                n=n, ng=ng, km=cfg.km)
            nh = None
        else:
            press = p_var_nonhydrostatic(
                jstate["delp"], jstate["delz"], jstate["pt"],
                ptop=self._ptop, akap=FV3_KAPPA, n=n, ng=ng, km=cfg.km)
            nh = build_nh_carry(self.grid.ctx_jax, cfg.km,
                                self.grid.ctx_jax.hs6)
        q = [jnp.asarray(np.stack(sphum6))]
        omga = jnp.zeros(
            (6,) + tuple(field_shape("delp", n, ng, cfg.km)),
            dtype=jnp.float64)
        return {"state": jstate, "press": press, "q": q, "omga": omga,
                "nh": nh}
