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
_SANE_MAX = {"delp": 1.0e6, "pt": 1.0e4, "u": 1.0e4, "v": 1.0e4,
             "w": 1.0e3}


def exchange_state_halos_3d(ctx: dict, state: list, km: int, *,
                            scalars: bool, winds: bool,
                            w_field: bool = False) -> None:
    """Duo halo exchanges, applied per level.

    `dyn_core.F90:470/471` exchanges delp and pt (it == 1 only);
    `:504` exchanges the D winds every sub-step.

    THE D-WIND EXCHANGE IS ``ext_vector``, NOT THE INDEX-COPY HELPER.
    ``dyn_core.F90:504`` reads

        if (duogrid) call ext_vector(u, v, gridstruct%dg, bd, domain,
                                     gridstruct, flagstruct, 0,1,1,0)

    i.e. the k2e Lagrange duo exchange, which replaces the WHOLE padded
    array including the corner-diagonal regions.  This function used to
    call ``exchange_dgrid_vector_halos``, the interim mpp-analog whose
    own docstring says the corner diagonals are left untouched, and the
    km=1 lane has used the authoritative one since the ext bundle
    existed (``fv3_native_duo_stepper`` at the same oracle site).

    That gap was NOT visible in the km=1 certified corpus because its
    fixtures come from ``analytic_swcore_state``, which evaluates the
    halos analytically at the kinked lattice, so the interim exchange had
    almost nothing to correct.  With a real IC whose halos start at zero
    it is the dominant error: MEASURED at C48/npz=5 on the J&W IC, one
    240 s sub-step produced a spurious increment of 3.34 m/s in u and v
    whose every cell above 10% of peak sat within 3 cells of a panel
    boundary (90 of 90, 298 of 298, ... on the six faces), growing
    linearly to 26 m/s over the 8 sub-steps of one dt_atmos, against an
    oracle tendency of 0.03-0.2 m/s.

    It FAILS CLOSED: a context without the ext bundle must say so with
    ``ext_exclude=('dvec',)`` rather than get the interim exchange
    silently, for the same reason ``exchange_post_pgrad_sixface`` does.
    """
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_agrid_scalar_halos,
        exchange_dgrid_vector_halos,
    )
    n, ng = ctx["n"], ctx["ng"]
    use_ext = bool(ctx.get("use_ext_bundle")) and ctx.get("ectx") is not None
    dvec_excluded = "dvec" in tuple(ctx.get("ext_exclude", ()))
    if winds and not use_ext and not dvec_excluded:
        raise ValueError(
            "exchange_state_halos_3d: dyn_core.F90:504 exchanges the D "
            "winds with ext_vector on the duo lane, and this context has "
            "no ext bundle. Substituting exchange_dgrid_vector_halos "
            "leaves the corner diagonals stale and puts a spurious "
            "~3 m/s per sub-step forcing on every panel boundary. Build "
            "the context with use_ext_bundle=True, or declare the "
            "substitution with ext_exclude=('dvec',).")
    for k in range(km):
        if scalars:
            for name in ("delp", "pt"):
                f6 = [state[t][name][:, :, k] for t in range(6)]
                _pad_scalars_6(f6, ctx, n, ng, exchange_agrid_scalar_halos)
        if winds:
            u6 = [state[t]["u"][:, :, k] for t in range(6)]
            v6 = [state[t]["v"][:, :, k] for t in range(6)]
            if use_ext and not dvec_excluded:
                from legoesm.grids.fv3_native_ext_vector import (
                    ext_vector_dgrid_sixface,
                )
                ext_vector_dgrid_sixface(u6, v6, ctx["ectx"])
            else:
                for t in range(1, 7):
                    exchange_dgrid_vector_halos(u6, v6, t, n, ng)
            # ext_vector_dgrid_sixface may replace the level arrays
            # rather than writing through the views, so copy back.
            for t in range(6):
                state[t]["u"][:, :, k] = u6[t]
                state[t]["v"][:, :, k] = v6[t]
        if w_field:
            # NH only: dyn_core.F90:479-480 -- ext_scalar(w, ..., 0,0)
            # every substep, AFTER the D winds (:466-482 order).
            f6 = [state[t]["w"][:, :, k] for t in range(6)]
            _pad_scalars_6(f6, ctx, n, ng, exchange_agrid_scalar_halos)
            for t in range(6):
                state[t]["w"][:, :, k] = f6[t]


def _pad_scalars_6(f6, ctx, n, ng, fallback) -> None:
    """A-grid scalar halo fill for one level, all six faces.

    ``dyn_core.F90:470-471`` is ``ext_scalar(delp, dg, bd, domain, 0,0)``
    and the same for ``pt``, so the authoritative analog is
    ``ext_scalar_sixface(f6, "A", ectx)``: mpp exchange, then
    ``k2e_remap_halo_rings``, then the Lagrange corner-region fill.  That
    is what the km=1 lane calls at this site.

    IT IS NOT ``duo_pad_scalars``.  That is the LEGACY jax duo pad, and
    ``fv3_native_duo_stepper._check_exchange_flags`` refuses to let it
    coexist with the ext bundle precisely because "the legacy pad would
    shadow the required ext_scalar refreshes" (codex ext r1 P1-2).  This
    function called it unconditionally whenever ``ctx["dg"]`` existed --
    which is every duo context -- so the 3-D lane has been shadowing its
    own ext_scalar since it was written.

    MEASURED, C48/npz=5, the J&W IC, entry exchanges only
    (``scripts/validate/fv3_native/halo_vs_analytic_probe.py``): with the
    legacy pad, ``pt`` in the four EDGE halo bands (576 cells per face,
    where the analytic reference is well defined) missed the analytic
    value by **36.7 K, 12% relative, on all six faces**, while the D
    winds -- which go through ``ext_vector`` -- were within 0.2% and
    ``delp`` within 1.8e-16.  ``delp`` agreeing proves nothing here: it
    is exactly 10000 Pa everywhere on this IC, so any exchange
    reproduces it.  The 37 K halo is what drives ``geopk``'s ``gz`` at
    ``ifirst = is-2 .. ie+2``, and hence the panel-boundary pressure
    gradient that put 3.34 m/s per sub-step on every seam.

    Fails closed for the same reason the wind exchange does: a context
    without the bundle must declare the substitution.
    """
    use_ext = bool(ctx.get("use_ext_bundle")) and ctx.get("ectx") is not None
    excluded = "ascalar" in tuple(ctx.get("ext_exclude", ()))
    if use_ext and not excluded:
        from legoesm.grids.fv3_native_ext_vector import ext_scalar_sixface
        ext_scalar_sixface(f6, "A", ctx["ectx"])
        return
    if not use_ext and not excluded:
        raise ValueError(
            "_pad_scalars_6: dyn_core.F90:470-471 fills the delp/pt halos "
            "with ext_scalar on the duo lane, and this context has no ext "
            "bundle. Build it with use_ext_bundle=True, or declare the "
            "substitution with ext_exclude=('ascalar',).")
    for t in range(1, 7):
        fallback(f6, t, n, ng)


def acoustic_substep_3d(ctx: dict, state: list, dt: float, km: int, *,
                        first_substep: bool,
                        ptop: float, akap: float, cp_air: float,
                        cfg: dict | None = None,
                        a2b_ord: int = 4,
                        exchange: bool = True,
                        remap_step: bool = False,
                        remap_follows: bool = False,
                        hydrostatic: bool = True,
                        nh: dict | None = None,
                        p_fac: float = 0.05, a_imp: float = 1.0,
                        dp0: np.ndarray | None = None,
                        use_logp: bool = False,
                        press_out: list | None = None) -> list:
    """One `it` of `do it=1,n_split`. Returns the updated six-face state.

    `state` is mutated in place for delp/pt/u/v (matching the Fortran's
    intent(inout) dummies) and also returned for convenience.

    `remap_step` is `dyn_core.F90:344-348` (true on `it == n_split`); it
    makes the D-grid geopk snapshot `pk` before `one_grad_p` destroys it.
    `press_out`, when given, is filled in place with the six per-face
    geopk bundles (`pe`, `peln`, `pk`, `pkz`, `gz`, plus `pk_remap` on a
    remap step) that `Lagrangian_to_Eulerian` consumes.

    ``hydrostatic=False`` runs the NH cadence: the first-substep gz
    seed/rebuild from ``delz`` (:384-416), the per-substep ``w``
    exchange (:479-480), the gz<->zh cadence around the C stage
    (:535-581), ``update_dz_c``/``Riem_Solver_c``/NH ``p_grad_c``, the
    d_sw w arms, and the NH D tail through ``nh_p_grad``.  ``nh`` is
    the persistent carry built by :func:`build_nh_carry`; ``dp0`` is
    ``dp_ref`` (both REQUIRED then).
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    dt2 = 0.5 * dt
    if not hydrostatic and (nh is None or dp0 is None):
        raise ValueError(
            "acoustic_substep_3d: hydrostatic=False needs the persistent "
            "nh carry (build_nh_carry) and dp0 (dp_ref)")

    bd = ctx["bd"]
    i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
    n = ctx["n"]

    if not hydrostatic and first_substep:
        # :384-416 -- gz(:,:,km+1) = zs over the PADDED box (duogrid
        # forces bounded_domain), then gz(k) = gz(k+1) - delz over the
        # COMPUTE window only.
        for t in range(6):
            gz = nh["gz6"][t]
            gz[:, :, km] = nh["zs6"][t]
            for k in range(km - 1, -1, -1):
                gz[i0:i0 + n, j0:j0 + n, k] = (
                    gz[i0:i0 + n, j0:j0 + n, k + 1]
                    - state[t]["delz"][:, :, k])

    if exchange:
        exchange_state_halos_3d(ctx, state, km,
                                scalars=first_substep, winds=True,
                                w_field=not hydrostatic)

    csw = csw_phase_3d(ctx, state, dt2=dt2, km=km, nord=2,
                       hydrostatic=hydrostatic,
                       remap_follows=remap_follows)

    if hydrostatic:
        cgrid_pressure_phase_3d(ctx, csw, km, dt2=dt2, ptop=ptop,
                                akap=akap, cp_air=cp_air, a2b_ord=a2b_ord,
                                remap_follows=remap_follows)
        csw_press = None
    else:
        from legoesm.core.fv3_native_cgrid_phase_3d import (
            cgrid_nh_pressure_phase_3d,
        )
        from legoesm.core.fv3_native_dsw_tail_3d import _ext_scalar_planes_6
        if first_substep:
            # :535-557 -- duo-exchange gz, then save zh = gz (padded).
            if exchange:
                for k in range(km + 1):
                    gz_k = [nh["gz6"][t][:, :, k] for t in range(6)]
                    _ext_scalar_planes_6(ctx, gz_k)
                    for t in range(6):
                        nh["gz6"][t][:, :, k] = gz_k[t]
            for t in range(6):
                nh["zh6"][t][:] = nh["gz6"][t]
        else:
            # :559-581 -- restore gz = zh (padded).
            for t in range(6):
                nh["gz6"][t][:] = nh["zh6"][t]
        csw_press = cgrid_nh_pressure_phase_3d(
            ctx, csw, nh["gz6"], nh["ws3_6"], km, dt2=dt2, ptop=ptop,
            akap=akap, cp_air=cp_air, p_fac=p_fac, a_imp=a_imp, dp0=dp0,
            hs6=nh["hs6"], zs6=nh["zs6"], remap_follows=remap_follows)

    dsw = dsw_transport_phase_3d(ctx, state, csw, dt=dt, km=km, cfg=cfg,
                                 hydrostatic=hydrostatic,
                                 remap_follows=remap_follows)
    tail = dsw_tail_phase_3d(ctx, state, csw, dsw, dt=dt, km=km, cfg=cfg,
                             hydrostatic=hydrostatic,
                             remap_follows=remap_follows)

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

    if hydrostatic:
        press = dgrid_pressure_phase_3d(ctx, dsw, tail, km, dt=dt,
                                        ptop=ptop, akap=akap,
                                        cp_air=cp_air, a2b_ord=a2b_ord,
                                        remap_step=remap_step,
                                        remap_follows=remap_follows)
    else:
        from legoesm.core.fv3_native_dsw_tail_3d import (
            dgrid_nh_pressure_phase_3d,
        )
        press = dgrid_nh_pressure_phase_3d(
            ctx, csw_press, dsw, tail,
            nh, km, dt=dt, ptop=ptop, akap=akap, cp_air=cp_air,
            p_fac=p_fac, a_imp=a_imp, dp0=dp0,
            delz6=[state[t]["delz"] for t in range(6)],
            remap_step=remap_step, use_logp=use_logp, cfg=cfg,
            remap_follows=remap_follows)
    if press_out is not None:
        press_out[:] = press

    # Write the prognostic fields back. delp/pt come from d_sw2 (unit 4);
    # u/v from d_sw6 as updated in place by nh_p_grad/one_grad_p; on the
    # NH lane w carries Riem_Solver3's update (delz was mutated in place).
    for t in range(6):
        state[t]["delp"][:] = dsw[t]["delp"]
        state[t]["pt"][:] = dsw[t]["pt"]
        state[t]["u"][:] = tail[t]["u"]
        state[t]["v"][:] = tail[t]["v"]
        if not hydrostatic:
            state[t]["w"][:] = tail[t]["w"]
    return state


def build_nh_carry(ctx: dict, km: int, hs6: list) -> dict:
    """The persistent NH arrays one acoustic loop carries across substeps.

    ``hs6`` is surface GEOPOTENTIAL (phis) per face; ``zs = phis/grav``
    (``dyn_core.F90:262-278``).  ``gz6``/``zh6`` are padded (m_a, m_a,
    km+1); ``ws3_6`` padded 2-D (the C-stage surface velocity);
    ``ws6`` compute-window 2-D (the D-stage one); ``pk3_6`` padded
    interfaces; ``pe6``/``pk6``/``peln6`` in their oracle layouts
    (``field_shape``).
    """
    from legoesm.core.fv3_native_state_3d import field_shape
    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV

    n, ng = ctx["n"], ctx["ng"]
    m_a = n + 2 * ng
    z = np.zeros
    hs6 = [np.asarray(h, dtype=np.float64) for h in hs6]
    return {
        "hs6": hs6,
        "zs6": [h / FV3_GRAV for h in hs6],
        "gz6": [z((m_a, m_a, km + 1)) for _ in range(6)],
        "zh6": [z((m_a, m_a, km + 1)) for _ in range(6)],
        "ws3_6": [z((m_a, m_a)) for _ in range(6)],
        "ws6": [z((n, n)) for _ in range(6)],
        "pk3_6": [z((m_a, m_a, km + 1)) for _ in range(6)],
        "pe6": [z(field_shape("pe", n, ng, km)) for _ in range(6)],
        # dyn_core's pk is COMPUTE-window (is:ie, js:je, npz+1) -- the
        # padded field_shape("pk") is the hydro geopk's own layout, a
        # different array.  Riem_Solver3 writes pk[:, jc, k] with ni rows.
        "pk6": [z((n, n, km + 1)) for _ in range(6)],
        "peln6": [z(field_shape("peln", n, ng, km)) for _ in range(6)],
    }


def acoustic_loop_3d(ctx: dict, state: list, dt_atmos: float, km: int, *,
                     n_split: int, ptop: float, akap: float, cp_air: float,
                     cfg: dict | None = None, validate: bool = True,
                     remap_follows: bool = False,
                     hydrostatic: bool = True,
                     nh: dict | None = None,
                     p_fac: float = 0.05, a_imp: float = 1.0,
                     dp0: np.ndarray | None = None,
                     use_logp: bool = False,
                     press_out: list | None = None) -> list:
    """`do it=1,n_split` -- one outer dynamics step.

    `dt = bdt/n_split` (`dyn_core.F90:249`). The shipped duo decks run
    `k_split = 1`, so one call of this is one `dt_atmos`.

    `press_out`, when given, receives the pressure bundle of the FINAL
    sub-step -- the one `dyn_core.F90:344-348` marks `remap_step` and the
    one `Lagrangian_to_Eulerian` reads.

    ``hydrostatic=False``: pass ``dp0`` (dp_ref) and either a carry from
    :func:`build_nh_carry` or None to have one built from ``ctx['hs6']``.
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    if n_split < 1:
        raise ValueError(f"n_split must be >= 1, got {n_split}")
    if not hydrostatic and nh is None:
        hs6 = ctx.get("hs6")
        if hs6 is None:
            raise ValueError(
                "acoustic_loop_3d: hydrostatic=False needs ctx['hs6'] "
                "(phis) to build the NH carry")
        nh = build_nh_carry(ctx, km, hs6)
    dt = dt_atmos / float(n_split)
    n, ng = ctx["n"], ctx["ng"]
    for it in range(1, n_split + 1):
        remap_step = (it == n_split)
        acoustic_substep_3d(ctx, state, dt, km, first_substep=(it == 1),
                            ptop=ptop, akap=akap, cp_air=cp_air, cfg=cfg,
                            remap_step=remap_step,
                            remap_follows=remap_follows,
                            hydrostatic=hydrostatic, nh=nh,
                            p_fac=p_fac, a_imp=a_imp, dp0=dp0,
                            use_logp=use_logp,
                            press_out=(press_out if remap_step else None))
        if validate:
            # Fail at the sub-step that broke, not many steps later with a
            # field of NaN and no idea which stage produced it.
            try:
                validate_state_3d(state, n, ng, km, require_finite=False,
                                  remap_follows=remap_follows)
            except Exception as exc:
                raise RuntimeError(
                    f"state invalid after acoustic sub-step {it}/{n_split}: "
                    f"{exc}") from None
            bd = ctx["bd"]
            i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
            ni, nj = bd.ie - bd.is_ + 1, bd.je - bd.js + 1
            fields = (("delp", "pt", "u", "v") if hydrostatic
                      else ("delp", "pt", "u", "v", "w"))
            for t in range(6):
                for name in fields:
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
