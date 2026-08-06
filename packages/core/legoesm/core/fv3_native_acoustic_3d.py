"""One acoustic sub-step of the duo cadence, km-general, six faces.

STAGE 1 unit 6: the driver that assembles units 3-5 in the oracle's order.

ORACLE ORDER (published, checksum-pinned tree; `dyn_core.F90`):

    :339  do it=1,n_split
    :470  ext_scalar(delp), ext_scalar(pt)     [it == 1 only]
    :504  ext_vector(u, v)                     [every it]
    :489  c_sw                                 [do k=1,npz]
    :533  geopk   (C-grid)
    :629  p_grad_c
    :831  d_sw1                                [do k=1,npz]
    :872  BARRIER 1  (CGRID_NE)
    :950  d_sw2      :961  d_sw3
    :984  BARRIER 2  (BGRID_NE)
    :1102 d_sw4      :1107 d_sw5      :1256 d_sw6
    :1401 geopk   (D-grid)
    :1531 one_grad_p

The `it == 1` gate on the scalar exchange is real and load-bearing
(`dyn_core.F90:465`): delp/pt are exchanged once per acoustic loop, the
winds every sub-step. Exchanging scalars every sub-step would be a
different model, so `first_substep` is an explicit argument rather than
something inferred.

WHAT IS AND IS NOT PERSISTED. `fv_arrays.F90` declares u, v and delp
PROGNOSTIC on the D grid and says the C grid "is diagnostic in that it is
predicted every time step from the D grid variables". So uc/vc are
rebuilt from scratch each sub-step and never carried; only delp, pt, u, v
survive the call.
"""
from __future__ import annotations

import numpy as np

from legoesm.core.fv3_native_cgrid_phase_3d import (
    cgrid_pressure_phase_3d,
    csw_phase_3d,
)
from legoesm.core.fv3_native_dsw_phase_3d import dsw_transport_phase_3d
from legoesm.core.fv3_native_dsw_tail_3d import (
    dgrid_pressure_phase_3d,
    dsw_tail_phase_3d,
)
from legoesm.core.fv3_native_state_3d import (
    require_no_remap_needed,
    validate_state_3d,
)


# Loose physical sanity bounds for a dry hydrostatic column. Not tuning --
# these are orders of magnitude above anything valid, so tripping one means
# the integration has diverged, not that a coefficient needs adjusting.
_SANE_MAX = {"delp": 1.0e6, "pt": 1.0e4, "u": 1.0e4, "v": 1.0e4}


def exchange_state_halos_3d(ctx: dict, state: list, km: int, *,
                            scalars: bool, winds: bool) -> None:
    """Duo halo exchanges, applied per level.

    `dyn_core.F90:470/471` exchanges delp and pt (it == 1 only);
    `:504` exchanges the D winds every sub-step.
    """
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_agrid_scalar_halos,
        exchange_dgrid_vector_halos,
    )
    n, ng = ctx["n"], ctx["ng"]
    for k in range(km):
        if scalars:
            for name in ("delp", "pt"):
                f6 = [state[t][name][:, :, k] for t in range(6)]
                _pad_scalars_6(f6, ctx, n, ng, exchange_agrid_scalar_halos)
        if winds:
            u6 = [state[t]["u"][:, :, k] for t in range(6)]
            v6 = [state[t]["v"][:, :, k] for t in range(6)]
            for t in range(1, 7):
                exchange_dgrid_vector_halos(u6, v6, t, n, ng)


def _pad_scalars_6(f6, ctx, n, ng, fallback) -> None:
    """A-grid scalar halo fill for one level, all six faces.

    Prefers ``duo_pad_scalars`` -- the certified k2e Lagrange halo remap
    (``pad_halo`` with the real ``DuoGridData``), i.e. the faithful
    ``ext_scalar`` analog. It replaces the WHOLE padded array, so the
    corner-diagonal regions get filled.

    ``exchange_agrid_scalar_halos`` is the interim index-copy exchange: it
    fills the four edge halos but leaves the corner diagonals at their
    sentinel, and PPM stencils reach into those corners. That is the
    suspected source of the sub-step-2 blow-up, so the interim path is only
    a fallback for a context built without the ext bundle, and it says so.
    """
    from legoesm.core.fv3_native_duo_stepper import duo_pad_scalars
    if ctx.get("dg") is not None:
        duo_pad_scalars(f6, ctx)
        return
    for t in range(1, 7):
        fallback(f6, t, n, ng)


def acoustic_substep_3d(ctx: dict, state: list, dt: float, km: int, *,
                        first_substep: bool,
                        ptop: float, akap: float, cp_air: float,
                        cfg: dict | None = None,
                        a2b_ord: int = 4,
                        exchange: bool = True) -> list:
    """One `it` of `do it=1,n_split`. Returns the updated six-face state.

    `state` is mutated in place for delp/pt/u/v (matching the Fortran's
    intent(inout) dummies) and also returned for convenience.
    """
    require_no_remap_needed(km)
    dt2 = 0.5 * dt

    if exchange:
        exchange_state_halos_3d(ctx, state, km,
                                scalars=first_substep, winds=True)

    csw = csw_phase_3d(ctx, state, dt2=dt2, km=km, nord=2)
    cgrid_pressure_phase_3d(ctx, csw, km, dt2=dt2, ptop=ptop, akap=akap,
                            cp_air=cp_air, a2b_ord=a2b_ord)
    dsw = dsw_transport_phase_3d(ctx, state, csw, dt=dt, km=km, cfg=cfg)
    tail = dsw_tail_phase_3d(ctx, state, csw, dsw, dt=dt, km=km, cfg=cfg)

    # POST-d_sw scalar exchange, BEFORE the D-grid geopk.
    # dyn_core.F90:1336-1337 -- ext_scalar(delp,...,0,0) and
    # ext_scalar(pt,...,0,0), inside the duogrid branch, every sub-step,
    # ahead of geopk at :1401. d_sw2 has just rewritten delp/pt, so without
    # this their halos are stale going into the pressure chain and the NEXT
    # sub-step: sub-step 1 stayed clean while sub-step 2 produced 278
    # non-finite delp values before this was added.
    if exchange:
        from legoesm.grids.fv3_native_gridstruct import (
            exchange_agrid_scalar_halos,
        )
        n_, ng_ = ctx["n"], ctx["ng"]
        for k in range(km):
            for _nm in ("delp", "pt"):
                f6 = [dsw[t][_nm][:, :, k] for t in range(6)]
                _pad_scalars_6(f6, ctx, n_, ng_,
                               exchange_agrid_scalar_halos)

    dgrid_pressure_phase_3d(ctx, dsw, tail, km, dt=dt, ptop=ptop,
                            akap=akap, cp_air=cp_air, a2b_ord=a2b_ord)

    # Write the prognostic fields back. delp/pt come from d_sw2 (unit 4);
    # u/v from d_sw6 as updated in place by one_grad_p (unit 5).
    for t in range(6):
        state[t]["delp"][:] = dsw[t]["delp"]
        state[t]["pt"][:] = dsw[t]["pt"]
        state[t]["u"][:] = tail[t]["u"]
        state[t]["v"][:] = tail[t]["v"]
    return state


def acoustic_loop_3d(ctx: dict, state: list, dt_atmos: float, km: int, *,
                     n_split: int, ptop: float, akap: float, cp_air: float,
                     cfg: dict | None = None, validate: bool = True) -> list:
    """`do it=1,n_split` -- one outer dynamics step.

    `dt = bdt/n_split` (`dyn_core.F90:249`). The shipped duo decks run
    `k_split = 1`, so one call of this is one `dt_atmos`.
    """
    require_no_remap_needed(km)
    if n_split < 1:
        raise ValueError(f"n_split must be >= 1, got {n_split}")
    dt = dt_atmos / float(n_split)
    n, ng = ctx["n"], ctx["ng"]
    for it in range(1, n_split + 1):
        acoustic_substep_3d(ctx, state, dt, km, first_substep=(it == 1),
                            ptop=ptop, akap=akap, cp_air=cp_air, cfg=cfg)
        if validate:
            # Fail at the sub-step that broke, not many steps later with a
            # field of NaN and no idea which stage produced it.
            try:
                validate_state_3d(state, n, ng, km, require_finite=False)
            except Exception as exc:
                raise RuntimeError(
                    f"state invalid after acoustic sub-step {it}/{n_split}: "
                    f"{exc}") from None
            bd = ctx["bd"]
            i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
            ni, nj = bd.ie - bd.is_ + 1, bd.je - bd.js + 1
            for t in range(6):
                for name in ("delp", "pt", "u", "v"):
                    # Compute window only: the corner-diagonal halo regions
                    # carry `sentinel` by construction, so scoring the full
                    # padded array would flag every healthy step. Scoring
                    # `a[isfinite(a)]` instead -- as an earlier version did --
                    # is a tautology that can never fail.
                    win = state[t][name][i0:i0 + ni, j0:j0 + nj, :]
                    bad = int(np.count_nonzero(~np.isfinite(win)))
                    if bad:
                        raise RuntimeError(
                            f"sub-step {it}/{n_split}: face {t + 1} {name} "
                            f"has {bad} non-finite value(s) inside the "
                            f"compute window")
                    # FINITENESS IS NOT ENOUGH. A delp of 1e9 is finite and
                    # passes every isfinite() check, then drives delpc
                    # negative one sub-step later and only becomes NaN
                    # inside geopk's log. Catching the magnitude here names
                    # the sub-step that actually diverged.
                    if name == "delp" and win.min() <= 0.0:
                        raise RuntimeError(
                            f"sub-step {it}/{n_split}: face {t + 1} delp min "
                            f"{win.min():.6g} <= 0 -- non-positive layer mass")
                    lim = _SANE_MAX.get(name)
                    if lim is not None:
                        peak = float(np.abs(win).max())
                        if peak > lim:
                            raise RuntimeError(
                                f"sub-step {it}/{n_split}: face {t + 1} "
                                f"{name} peaked at {peak:.6g}, above the "
                                f"sanity bound {lim:g}. Finite but "
                                f"non-physical -- the divergence starts "
                                f"HERE, not where the NaN appears.")
    return state
