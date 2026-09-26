"""FV3 ``tracer_2d_1L`` on the six-face duo lane -- JAX twin of
``legoesm.core.fv3_native_tracer2d`` (module 5 of 6), functional and
jit-compilable.

ORACLE CADENCE (the spec's block, checksum-pinned tree)::

    fv_dynamics.F90:472-478   dp1 = delp            (BEFORE dyn_core)
    dyn_core.F90:313-316      mfx/mfy/cx/cy capacitors zeroed
    sw_core.F90:903-920       d_sw1 accumulates crx_adv/fx into them,
                              EVERY acoustic sub-step, BEFORE the
                              inter-panel allflux averaging
    fv_dynamics.F90:534       tracer_2d_1L(q, dp1, mfx, mfy, cx, cy, ...)

SEAMS ARE THE HALO MODULE'S: the q halos come from
``fv3_duo_halos.ext_scalar_sixface`` (fv_tracer2d.F90:276-281 for q,
:371-377 for qn2 between subcycles) and the per-tracer flux_adj blend
is ``fv3_duo_halos.average_shared_edge_cgrid`` -- the CGRID_NE blend
barrier 1 uses -- applied per level with all six faces of that level
present.  This module never re-derives an extent.

SUBCYCLING (the node that does not port mechanically): nsplt is
data-derived, so the level loop runs a fixed ``NSPLT_MAX``-long
``lax.scan`` whose inactive iterations are a PROVEN no-op (jnp.where
returns the old state bit-exactly); the schedule carries
stop_gradient, so the derivative is piecewise with no term through
the trip-count decision -- the true derivative almost everywhere.  A
resolved nsplt above NSPLT_MAX is never truncated: the schedule and a
flag come back and ``check_nsplt_schedule`` raises; the default
caller produced by ``make_tracer_2d_1l_sixface_jit`` performs that
check.

CONVENTIONS inherited from the sibling phases: C1 one dict of
face-stacked arrays (face t = Fortran tile t+1, Fortran index 1-ng at
numpy 0; the d_sw1-lane capacitors/fluxes carry a TRAILING km axis);
C2 face and level loops are Python loops, not vmap; C3 ctx/km/deck
constants are static, dt is dynamic; C4 every array the spec mutates
is returned; C5 config guards are static and refuse rather than skip;
C6 stage observation is by return -- the spec's tracer_2d_1L has no
stage_hook, so nothing beyond the C4 returns and the D3 schedule is
returned; C7 deck constants come from the caller's config, never
restated here.

WHAT THIS MODULE DOES NOT DO: it introduces no numerics of its own --
every arithmetic operation lives in fv_tp_2d or the halo module; the
barrier/exchange extents are the halo module's; ``d_con = 0.0``
throughout, so it is not a parameter here and the KE-to-heat pathway
(dyn_core.F90:322-325, gated on d_con > 1.0E-5) is inactive; and the
moist path is dead -- the tracers are passive (rainwat inert, DNATS=1)
and nothing here reads a moisture field.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.fv3_phase3d_common import (
    build_batched_gs,
    require_bool,
    require_f64_jax,
)
from legoesm.core.fv3_tp_core import fv_tp_2d
from legoesm.grids.fv3_duo_halos import (
    average_shared_edge_cgrid,
    ext_scalar_sixface_allk,
)

# fv_tracer2d.F90:243 -- `if (trdm > 1.e-4)` gates the dp1 halo
# pre-exchange; this port refuses that arm (D6).
TRDM_ACTIVE_MIN = 1.0e-4

# D1-D3: STATIC cap on the per-level subcycle count, from the admitted
# CFL envelope (resolved nsplt = int(1+cmax) <= NSPLT_MAX).  Exceeding
# it is REPORTED through the returned schedule/flag, never truncated.
NSPLT_MAX = 8


def require_tracer_2d_1l_lane(*, z_tracer: bool, q_split: int,
                              nord_tr: int, trdm: float,
                              inline_q: bool = False) -> None:
    """Refuse every fv_tracer2d arm this port does not carry (C5/D6: a
    guard silently dropped is how a different run wears the same name)."""
    if inline_q:
        raise NotImplementedError(
            "inline_q=True advects tracers INSIDE d_sw1 (sw_core.F90:"
            "976-996) and skips tracer_2d entirely (fv_dynamics.F90:517); "
            "d_sw1_duo refuses that lane and so does this one. The "
            "resolved deck echoes INLINE_Q=F.")
    if not z_tracer:
        raise NotImplementedError(
            "z_tracer=False selects tracer_2d (fv_tracer2d.F90:391), the "
            "GLOBAL-split arm with its own q_split/ksplt logic -- a "
            "different routine, not a parameter of this one. The resolved "
            "deck echoes Z_TRACER=T, which selects tracer_2d_1L "
            "(fv_dynamics.F90:533-536).")
    if q_split != 0:
        raise NotImplementedError(
            f"q_split={q_split}: this tree's tracer_2d_1L IGNORES q_split "
            f"(its nsplt is always int(1+cmax), fv_tracer2d.F90:247); a "
            f"deck that sets q_split expects the fixed-split behaviour of "
            f"tracer_2d and must not get the dynamic one silently. The "
            f"resolved deck echoes Q_SPLIT=0.")
    if nord_tr != 0:
        raise NotImplementedError(
            f"nord_tr={nord_tr}: tracer del-n damping. This tree's "
            f"tracer_2d_1L fv_tp_2d call (fv_tracer2d.F90:302-304) carries "
            f"no nord/damp arguments, so nord_tr is dead there; refusing a "
            f"non-zero value rather than silently dropping it. The "
            f"resolved deck echoes NORD_TR=0.")
    if trdm > TRDM_ACTIVE_MIN:
        raise NotImplementedError(
            f"trdm={trdm} > {TRDM_ACTIVE_MIN}: the dp1 halo pre-exchange "
            f"(fv_tracer2d.F90:243-256) is not ported (it feeds nothing in "
            f"this tree's tracer_2d_1L, but a deck that sets trdm2 expects "
            f"upstream's damped-flux variant). The resolved deck echoes "
            f"TRDM2=0.0.")


def alloc_flux_capacitors(n: int, ng: int, km: int, *,
                          dtype=jnp.float64, nb: int = 6) -> dict:
    """The zeroed mfx/mfy/cx/cy capacitors (dyn_core.F90:313-316),
    face-stacked per C1: face axis 0, the spec's per-level d_sw1 dummy
    shapes with a trailing km axis --

        mfx (6, npx, m_a, km)  used i=is..ie+1, j=js..je
        mfy (6, m_a, npx, km)  used i=is..ie,   j=js..je+1
        cx  (6, npx, m_a, km)  used i=is..ie+1, j=jsd..jed
        cy  (6, m_a, npx, km)  used i=isd..ied, j=js..je+1

    with npx = n+1, m_a = n+2*ng.  These are d_sw1-lane staggerings,
    NOT field_shape-table fields (rule 10): check face axis 6 and level
    axis km only.  The structurally-zero mfx columns past je / mfy rows
    past ie exist because the caller-side arrays are rectangular."""
    # dtype follows the run's storage dtype (fp32/mixed increment 2):
    # these capacitors accumulate f32/f64 mass fluxes from d_sw1; a
    # hardcoded float64 would mix with an f32 carry. fp64 default is
    # byte-identical.
    npx = n + 1
    m_a = n + 2 * ng
    return {"mfx": jnp.zeros((nb, npx, m_a, km), dtype=dtype),
            "mfy": jnp.zeros((nb, m_a, npx, km), dtype=dtype),
            "cx": jnp.zeros((nb, npx, m_a, km), dtype=dtype),
            "cy": jnp.zeros((nb, m_a, npx, km), dtype=dtype)}


def _window_seam_refresh(tab, exch, nq: int):
    """WINDOW lane only (no-op on six faces): full seam-pad refresh of
    the tracer stack after a sub-iteration's exchange.

    That exchange is a band-restricted firing (face-edge bands only), so
    each sub-iteration eats one stencil reach of intra-face seam pad and
    nothing rebuilds it mid-phase; the phase's ENTRY refresh cannot help.
    MEASURED (C24 kt=2 pad=5, gate jobs 9913482/9913484): nsplt=2
    bitwise, nsplt=3 corrupts q (1056 -> 25883 of 27000 owned cells
    over 3 steps; dynamics untouched).  A full refresh per sub-iteration
    is the acoustic-substep entry pattern and makes the phase pad-depth
    safe for any nsplt.  One tracer per key: ``exch[:, iq]`` is
    ``(nb, W, W)`` per level or ``(nb, W, W, km)`` batched, the layouts
    the entry refresh and pt already rely on.  ponytail: fires on every
    masked scan slot (NSPLT_MAX); gate on the active mask if the ladder
    shows it."""
    wc = getattr(tab, "window_comm", None)
    if wc is None:
        return exch
    ref = wc.refresh({f"q{iq}": exch[:, iq] for iq in range(nq)})
    return jnp.stack([ref[f"q{iq}"] for iq in range(nq)], axis=1)


def check_nsplt_schedule(out: dict) -> None:
    """D3's loud check.  Must run OUTSIDE jit on the concrete outputs
    (handing it tracers raises, which is also loud).  D5: fixtures sit
    on both sides of every integer schedule transition and exercise
    this nsplt > NSPLT_MAX refusal."""
    if bool(np.asarray(out["nsplt_exceeded"])):
        raise ValueError(
            f"tracer_2d_1l_sixface: resolved nsplt="
            f"{np.asarray(out['nsplt']).tolist()} exceeds NSPLT_MAX="
            f"{NSPLT_MAX}; the subcycling is WRONG and the answer would "
            f"be silently under-transported.  Raise NSPLT_MAX or shrink "
            f"the large time step.")


def tracer_2d_1l_sixface(ctx: dict, q6, dp1_6, flux_cap: dict, *, km: int,
                         nq: int, hord_tr: int, dt, q_split: int = 0,
                         nord_tr: int = 0, trdm: float = 0.0,
                         lim_fac: float = 1.0, z_tracer: bool = True,
                         inline_q: bool = False,
                         batched: bool = False) -> dict:
    """``tracer_2d_1L`` (fv_tracer2d.F90:114-389), all six faces,
    FUNCTIONAL.

    Inputs (C1 face-stacked; shapes verified against the spec's own
    dummy declarations and alloc_flux_capacitors, NOT the field_shape
    table -- these are d_sw1-lane staggerings): ``q6``
    (nb, nq, m_a, m_a, km) [spec: [face][iq] of (m_a, m_a, km)];
    ``dp1_6`` (nb, m_a, m_a, km) = delp BEFORE dyn_core
    (fv_dynamics.F90:472-478); ``flux_cap`` = alloc_flux_capacitors'
    dict, the UN-averaged d_sw1 fluxes -- dp2 uses them as-is and the
    tracer fluxes get their own seam blend; do not "fix" this.

    Returned (C4 -- everything the spec mutates): ``q`` (interiors from
    the update nests of :284-388; halos/corners exactly as the Fortran
    leaves them), ``dp1`` (interiors advanced by :361-365 only where
    nsplt > 1), the frac-rescaled capacitors ``mfx``/``mfy``/``cx``/
    ``cy`` (:262-291), and D3's loud-failure pair ``nsplt``/
    ``nsplt_exceeded`` -- checked OUTSIDE jit by check_nsplt_schedule;
    the factory's default caller does it.  The spec has no stage_hook,
    so per C6 there are no further stage returns.

    Subcycling: lax.scan over the STATIC NSPLT_MAX, inactive iterations
    a proven no-op (D1/D2); the schedule is stop_gradient'ed, so the
    derivative is piecewise with no term through the trip count.

    ``dt`` is DYNAMIC (C3) and unused -- signature parity only
    (fv_dynamics.F90:413 passes mdt); the capacitors already carry the
    time integral.  ctx, km, nq, hord_tr, lim_fac and the lane switches
    are static.  No callee branches in Python on ptop/akap/cp_air/
    p_fac/a_imp, so none of those is a parameter of this routine.

    ``batched`` (STATIC, default False -- the certified loop path stays
    the default) selects the vmap-over-faces arm
    :func:`_tracer_2d_1l_sixface_batched` after the shared entry gates;
    it replaces convention C2's per-face Python loops with ``jax.vmap``
    over ``build_batched_gs``'s stacked view (face-batching ladder step
    6, pattern 4c7198bd1).  The halo exchanges, the flux_adj seam blend
    and the nsplt schedule machinery are cross-face and stay verbatim."""
    require_tracer_2d_1l_lane(z_tracer=z_tracer, q_split=q_split,
                              nord_tr=nord_tr, trdm=trdm,
                              inline_q=inline_q)
    require_bool("tracer_2d_1l_sixface", "batched", batched)
    del dt  # D7: parity only; see docstring -- it looks like a bug otherwise.
    # The JAX context is ATTRIBUTE-access; only the NumPy lane
    # uses a dict.  Same slip as module 4's first draft.
    n = int(ctx.n)
    ng = int(ctx.ng)
    if ng < 1:
        raise ValueError(f"ng={ng}: the tracer PPM stencils need ng >= 1")
    bd = ctx.bd       # fv_tp_2d's static bounds object, threaded from ctx
    # `ctx.tab` is what build_jax_duo_stepper_context names the
    # DuoHaloTables, and what every sibling phase passes.
    tab = getattr(ctx, "tab", None)
    if tab is None:
        raise ValueError(
            "ctx.tab: the q halo update is fv3_duo_halos."
            "ext_scalar_sixface's (fv_tracer2d.F90:276-281) and this "
            "context carries no duo halo tables.")
    m_a = n + 2 * ng
    npx = n + 1
    q6 = jnp.asarray(q6)
    dp1_6 = jnp.asarray(dp1_6)
    nb = len(ctx.gs6)             # 6 faces, or 6*kt*kt windows
    mfx6 = jnp.asarray(flux_cap["mfx"])
    mfy6 = jnp.asarray(flux_cap["mfy"])
    cx6 = jnp.asarray(flux_cap["cx"])
    cy6 = jnp.asarray(flux_cap["cy"])
    # The shared gate is (fname, arrays); the first draft passed the
    # arrays positionally, which is the same slip modules 3 and 4 made.
    require_f64_jax("tracer_2d_1l_sixface",
                    {"q6": q6, "dp1_6": dp1_6, "mfx": mfx6, "mfy": mfy6,
                     "cx": cx6, "cy": cy6})
    # Static shape gate on the d_sw1-lane staggerings (rule 10).
    if q6.shape != (nb, nq, m_a, m_a, km):
        raise ValueError(f"q6 shape {q6.shape}, expected "
                         f"{(nb, nq, m_a, m_a, km)}")
    if dp1_6.shape != (nb, m_a, m_a, km):
        raise ValueError(f"dp1_6 shape {dp1_6.shape}, expected "
                         f"{(nb, m_a, m_a, km)}")
    if cx6.shape != (nb, npx, m_a, km) or mfx6.shape != (nb, npx, m_a, km):
        raise ValueError("cx/mfx must be (nb, npx, m_a, km), got "
                         f"{cx6.shape}/{mfx6.shape}")
    if cy6.shape != (nb, m_a, npx, km) or mfy6.shape != (nb, m_a, npx, km):
        raise ValueError("cy/mfy must be (nb, m_a, npx, km), got "
                         f"{cy6.shape}/{mfy6.shape}")
    if batched:
        # Every entry gate above (lane guard, ctx.tab, f64, shapes) is
        # shared; the loop path below is byte-untouched and remains the
        # default.
        return _tracer_2d_1l_sixface_batched(
            ctx, q6, dp1_6, mfx6, mfy6, cx6, cy6, km=km, nq=nq,
            hord_tr=hord_tr, lim_fac=lim_fac)
    # The FLAGS live on ctx.flags6, never on ctx.gs6: the JAX context
    # builder keeps only the np.ndarray members of each gridstruct
    # (fv3_duo_stepper.py:484) and puts every scalar/bool into a
    # GridFlags NamedTuple alongside.  Reading them off gs6 with a
    # `.get(..., default)` therefore silently returns the DEFAULT on
    # every valid duo context -- which for bounded_domain is False, so
    # the guard below rejected everything, and for the corner flags
    # would have selected corner fills the oracle skips.  Module 3 shipped
    # the same mistake against update_dz_d; it is the "getattr fallback
    # counts as hardcoded" rule in CLAUDE.md, one level up.
    flg = []
    for t in range(nb):
        fl = ctx.flags6[t]
        if not bool(fl.bounded_domain):
            raise ValueError(
                f"face {t + 1}: gridstruct is not bounded_domain -- the "
                f"duo tracer lane (ext_scalar halos + flux_adj seam "
                f"averaging) is only certified on the bounded duo "
                f"gridstructs.")
        flg.append({"da_min": float(fl.da_min),
                    "bounded_domain": True,
                    "grid_type": int(fl.grid_type),
                    "sw_corner": bool(fl.sw_corner),
                    "se_corner": bool(fl.se_corner),
                    "nw_corner": bool(fl.nw_corner),
                    "ne_corner": bool(fl.ne_corner)})
    # C1: gridstruct metrics as ONE dict of face-stacked arrays (static
    # ctx data).  The spec's _tp_gsf fort-wrapper is replaced by direct
    # 0-based slicing: Fortran index f sits at numpy f + ng - 1.
    gmet = {key: jnp.stack([jnp.asarray(ctx.gs6[t][key]) for t in range(nb)])
         for key in ("dxa", "dya", "dx", "dy", "sin_sg", "area",
                     "rarea", "del6_v", "del6_u")}

    # ---- xfx/yfx from the Courant capacitors (:191-215) + cmax (:217-231)
    xfx6 = [[None] * km for _ in range(nb)]
    yfx6 = [[None] * km for _ in range(nb)]
    cmax_face = [[None] * km for _ in range(nb)]
    for t in range(nb):
        dxa_g, dya_g = gmet["dxa"][t], gmet["dya"][t]
        dx_g, dy_g = gmet["dx"][t], gmet["dy"][t]
        sin_g = gmet["sin_sg"][t]  # (m_a, m_a, 6)
        for k in range(km):
            cxk = cx6[t, :, :, k]  # origin (is, jsd): rows 0..n = is..ie+1
            cyk = cy6[t, :, :, k]  # origin (isd, js): cols 0..n = js..je+1
            # Vectorised (:191-215): every (i,j) writes xfx/yfx[i,j]
            # reading only cx/cy and static metrics; no iteration reads
            # another's write.  Both where arms total (slices in range
            # for ng >= 1) and finite -- rule 2.
            xfx = jnp.where(
                cxk > 0.0,
                cxk * dxa_g[ng - 1:ng + n, :] * dy_g[ng:ng + n + 1, :]
                * sin_g[ng - 1:ng + n, :, 2],
                cxk * dxa_g[ng:ng + n + 1, :] * dy_g[ng:ng + n + 1, :]
                * sin_g[ng:ng + n + 1, :, 0])
            yfx = jnp.where(
                cyk > 0.0,
                cyk * dya_g[:, ng - 1:ng + n] * dx_g[:, ng:ng + n + 1]
                * sin_g[:, ng - 1:ng + n, 3],
                cyk * dya_g[:, ng:ng + n + 1] * dx_g[:, ng:ng + n + 1]
                * sin_g[:, ng:ng + n + 1, 1])
            xfx6[t][k] = xfx
            yfx6[t][k] = yfx
            # :217-231; `k < npz/6` is INTEGER division on static km and
            # the Python level index -- a Python if (rule 1).
            cxi = jnp.abs(cxk[0:n, ng:ng + n])
            cyi = jnp.abs(cyk[ng:ng + n, 0:n])
            if (k + 1) < (km // 6):
                cm = jnp.maximum(cxi.max(), cyi.max())
            else:
                cm = (jnp.maximum(cxi, cyi)
                      + 1.0 - sin_g[ng:ng + n, ng:ng + n, 4]).max()
            cmax_face[t][k] = cm
    # :258 mp_reduce_max(cmax, npz) -- the six faces are this port's ranks.
    cmax = jnp.stack([jnp.stack(r) for r in cmax_face]).max(axis=0)
    # D1/D2: the trip count is data-derived; the schedule carries
    # stop_gradient so no derivative flows through the trip-count
    # decision (piecewise derivative, true almost everywhere).
    nsplt_f = jax.lax.stop_gradient(jnp.floor(1.0 + cmax))  # (km,) >= 1
    nsplt_i = nsplt_f.astype(jnp.int32)
    nsplt_exceeded = jnp.any(nsplt_i > NSPLT_MAX)
    frac = 1.0 / nsplt_f

    # ---- q halo update (:276-281): the halo module's ext_scalar over
    # every (level, tracer) slab -- a horizontal operator, so the
    # (iq, k) pair folds into ONE trailing batch axis and the whole
    # update is ONE batched exchange call (v2a; nq*km collectives -> 1
    # on the ring path).  moveaxis/reshape are bijective relabellings;
    # each slab still gets the certified per-level exchange, values
    # identical to the former per-(k, iq) loop (slabs independent).
    q = jnp.moveaxis(
        ext_scalar_sixface_allk(
            jnp.moveaxis(q6, 1, -1).reshape(nb, m_a, m_a, km * nq),
            tab, "A").reshape(nb, m_a, m_a, km, nq),
        -1, 1)

    # ---- the k loop (:284-388), one lax.scan of NSPLT_MAX per level ---
    dp1 = dp1_6
    for k in range(km):
        f_k = frac[k]
        # :262-291 frac rescale, returned (C4).  Unconditional *frac is
        # exact: nsplt >= 1 always, so frac == 1.0 exactly where
        # nsplt == 1 (D4); ranges mirror the Fortran exactly (cx/cy over
        # the whole slabs, mfx live columns and mfy live rows only).
        cx6 = cx6.at[:, :, :, k].multiply(f_k)
        cy6 = cy6.at[:, :, :, k].multiply(f_k)
        # FOUR selectors, not three: these are face-STACKED (C1), so a
        # three-index `.at[:, 0:n, k]` reads as `[:, 0:n, k, :]` -- it
        # scales one horizontal row at EVERY level instead of the live
        # columns at level k, and for mfy it also treats 0:n as the face
        # axis. Silent: the shapes broadcast, the run completes, and the
        # mass fluxes are wrong in a way that couples levels. The reader
        # two lines below (`mfx6[t, :, 0:n, k]`) already had the right
        # form, which is the tell.
        mfx6 = mfx6.at[:, :, 0:n, k].multiply(f_k)
        mfy6 = mfy6.at[:, 0:n, :, k].multiply(f_k)
        xf = [xfx6[t][k] * f_k for t in range(nb)]
        yf = [yfx6[t][k] * f_k for t in range(nb)]
        cxk = [cx6[t, :, :, k] for t in range(nb)]
        cyk = [cy6[t, :, :, k] for t in range(nb)]
        mxk = [mfx6[t, :, 0:n, k] for t in range(nb)]  # (n+1, n) dummy
        myk = [mfy6[t, 0:n, :, k] for t in range(nb)]  # (n, n+1) dummy
        ra_x, ra_y, div6 = [], [], []
        for t in range(nb):
            area_g = gmet["area"][t]
            # Vectorised (:286-296): each ra cell reads only xfx/yfx.
            ra_x.append(area_g[ng:ng + n, :]
                        + xf[t][0:n, :] - xf[t][1:n + 1, :])
            ra_y.append(area_g[:, ng:ng + n]
                        + yf[t][:, 0:n] - yf[t][:, 1:n + 1])
            # Vectorised (:302-306/:341-345): dp2 reads only mfx/mfy/dp1.
            div6.append((mxk[t][0:n, :] - mxk[t][1:n + 1, :]
                         + myk[t][:, 0:n] - myk[t][:, 1:n + 1])
                        * gmet["rarea"][t][ng:ng + n, ng:ng + n])
        div6 = jnp.stack(div6)
        nsp = nsplt_f[k]
        qn20 = q[:, :, :, :, k]          # halos already exchanged (:276-281)
        dp10 = dp1[:, ng:ng + n, ng:ng + n, k]

        def body(carry, it):
            qn2, dp1i, qfin = carry
            # D2 masks: scan index it is Fortran it+1.  `run` = active,
            # non-final iteration (writes qn2 and dp1); `last` writes q.
            # nsplt == 1 levels: only it == 0 fires, as `last` -- exactly
            # the spec's un-subcycled arm, so the two branches unify.
            run = (it + 1) < nsp
            last = (it + 1) == nsp
            dp2 = dp1i + div6
            base = jnp.zeros_like(qn2)
            for iq in range(nq):
                qm_l, fx_l, fy_l = [], [], []
                for t in range(nb):
                    # qm is fv_tp_2d's returned q (its copy_corners
                    # corner ghosts); its interior is untouched and every
                    # halo point is overwritten by the exchange below, so
                    # qm feeds only `base`, never the arithmetic.
                    qm, fxt, fyt = fv_tp_2d(
                        qn2[t, iq], cxk[t], cyk[t], npx, npx, hord_tr,
                        xf[t], yf[t], gmet["dxa"][t], gmet["dya"][t],
                        gmet["area"][t], gmet["del6_v"][t], gmet["del6_u"][t],
                        gmet["rarea"][t], flg[t]["da_min"], bd,
                        ra_x[t], ra_y[t], lim_fac,
                        flg[t]["bounded_domain"], flg[t]["grid_type"],
                        flg[t]["sw_corner"], flg[t]["se_corner"],
                        flg[t]["nw_corner"], flg[t]["ne_corner"],
                        mfx=mxk[t], mfy=myk[t], duogrid=True)
                    qm_l.append(qm)
                    fx_l.append(fxt)
                    fy_l.append(fyt)
                # flux_adj (:310-312, :46-108): the halo module's CGRID_NE
                # blend, all six faces of this level present (rule 6).
                # Signature READ, not recalled: the halo module's twin is
                # `(fx6, fy6, tab)` over STACKED (6, npx, n) / (6, n, npx)
                # arrays, not the NumPy lane's `(fx6, fy6, n, ng)` over
                # lists of per-face slabs. Same operator, different lane,
                # different call -- and the spec's home
                # (fv3_native_gridstruct) is a different module entirely.
                fx_l, fy_l = average_shared_edge_cgrid(
                    jnp.stack(fx_l), jnp.stack(fy_l), tab)
                for t in range(nb):
                    # Vectorised (update nests of :284-388): each q cell
                    # reads only its own flux pair; dp2 > 0 so the
                    # division is finite (rule 2).
                    d = (fx_l[t][0:n, :] - fx_l[t][1:n + 1, :]
                         + fy_l[t][:, 0:n] - fy_l[t][:, 1:n + 1]) \
                        * gmet["rarea"][t][ng:ng + n, ng:ng + n]
                    u = ((qn2[t, iq, ng:ng + n, ng:ng + n] * dp1i[t] + d)
                         / dp2[t])
                    base = base.at[t, iq].set(
                        qm_l[t].at[ng:ng + n, ng:ng + n].set(u))
            # :361-379: dp1 = dp2 (compute window), then the qn2 halo
            # refresh -- the halo module's ext_scalar (:371-377), all nq
            # tracers batched into the trailing axis: ONE exchange call
            # per subcycle iteration instead of nq (v2a).
            exch = jnp.moveaxis(
                ext_scalar_sixface_allk(
                    jnp.moveaxis(base, 1, -1), tab, "A"),
                -1, 1)
            exch = _window_seam_refresh(tab, exch, nq)
            # PROVEN no-op when inactive: where returns the old arrays
            # bit-exactly; both arms are total and finite (rule 2).
            return (jnp.where(run, exch, qn2),
                    jnp.where(run, dp2, dp1i),
                    jnp.where(last, base, qfin)), None

        (_, dp1k_f, qk_f), _ = jax.lax.scan(
            body, (qn20, dp10, qn20), jnp.arange(NSPLT_MAX))
        q = q.at[:, :, :, :, k].set(qk_f)
        dp1 = dp1.at[:, ng:ng + n, ng:ng + n, k].set(dp1k_f)

    return {"q": q, "dp1": dp1, "mfx": mfx6, "mfy": mfy6,
            "cx": cx6, "cy": cy6,
            "nsplt": nsplt_i, "nsplt_exceeded": nsplt_exceeded}



def _tracer_2d_1l_sixface_batched(ctx, q6, dp1_6, mfx6, mfy6, cx6, cy6, *,
                                  km: int, nq: int, hord_tr: int,
                                  lim_fac: float) -> dict:
    """The vmap-over-faces arm of :func:`tracer_2d_1l_sixface` (C2a --
    face-batching ladder step 6).

    Entry gates (lane guard, ``ctx.tab``, f64, static shapes) already
    ran in the caller.  The per-face traced reads become face-batched:
    the xfx/yfx/cmax build and the subcycle scan's ``fv_tp_2d``
    transport call are ``jax.vmap`` over the face axis of
    ``build_batched_gs``'s stacked view, and the ra/div/update
    arithmetic runs on face-stacked arrays (elementwise, so
    value-identical to the per-face expressions).  The CROSS-FACE
    collectives -- both ``ext_scalar_sixface_allk`` halo updates, the
    ``average_shared_edge_cgrid`` flux_adj blend -- and the nsplt
    schedule machinery (D1-D3: static NSPLT_MAX scan, stop_gradient'ed
    schedule, loud-failure pair) stay VERBATIM, exactly where the loop
    path puts them.

    in_axes: q/cx/cy/xfx/yfx planes, the gridstruct metrics
    (``view["gs"]``) and ``da_min6`` are 0 (per face); ``bd``/``npx``/
    ``hord_tr``/``lim_fac`` and the common-mode ``GridFlags`` fields
    (``grid_type``, corners, ``bounded_domain`` -- face-invariant by
    ``build_batched_gs``'s gate, which RAISES otherwise) close over as
    shared Python values.  ``da_min`` is UNREAD on this call path
    (``mass=None`` skips both deln_flux blocks, fv3_tp_core.py:
    1670-1687) but rides as a traced ``(6,)`` operand anyway, matching
    the DSW batched arm's threading -- arithmetic-only by the same
    audit, never compared.
    """
    fname = "tracer_2d_1l_sixface[batched]"
    n = int(ctx.n)
    ng = int(ctx.ng)
    bd = ctx.bd
    tab = ctx.tab
    m_a = n + 2 * ng
    npx = n + 1
    view = build_batched_gs(ctx)
    fl = view["flags"]
    if not bool(fl["bounded_domain"]):
        raise ValueError(
            f"{fname}: gridstructs are not bounded_domain -- the duo "
            f"tracer lane (ext_scalar halos + flux_adj seam averaging) "
            f"is only certified on the bounded duo gridstructs.")
    gs = view["gs"]
    da6 = view["da_min6"]
    grid_type = int(fl["grid_type"])
    corners = (bool(fl["sw_corner"]), bool(fl["se_corner"]),
               bool(fl["nw_corner"]), bool(fl["ne_corner"]))

    # ---- xfx/yfx from the Courant capacitors (:191-215) + cmax
    # (:217-231): the loop path's per-(t, k) expressions VERBATIM inside
    # a per-face function, vmapped over the face axis.  max/abs/where
    # are exact (no rounding), so neither the batching nor the reduction
    # regrouping in cmax can change a value.
    def _face_flux(cx, cy, dxa_g, dya_g, dx_g, dy_g, sin_g):
        xks, yks, cms = [], [], []
        for k in range(km):
            cxk = cx[:, :, k]
            cyk = cy[:, :, k]
            xfx = jnp.where(
                cxk > 0.0,
                cxk * dxa_g[ng - 1:ng + n, :] * dy_g[ng:ng + n + 1, :]
                * sin_g[ng - 1:ng + n, :, 2],
                cxk * dxa_g[ng:ng + n + 1, :] * dy_g[ng:ng + n + 1, :]
                * sin_g[ng:ng + n + 1, :, 0])
            yfx = jnp.where(
                cyk > 0.0,
                cyk * dya_g[:, ng - 1:ng + n] * dx_g[:, ng:ng + n + 1]
                * sin_g[:, ng - 1:ng + n, 3],
                cyk * dya_g[:, ng:ng + n + 1] * dx_g[:, ng:ng + n + 1]
                * sin_g[:, ng:ng + n + 1, 1])
            cxi = jnp.abs(cxk[0:n, ng:ng + n])
            cyi = jnp.abs(cyk[ng:ng + n, 0:n])
            if (k + 1) < (km // 6):
                cm = jnp.maximum(cxi.max(), cyi.max())
            else:
                cm = (jnp.maximum(cxi, cyi)
                      + 1.0 - sin_g[ng:ng + n, ng:ng + n, 4]).max()
            xks.append(xfx)
            yks.append(yfx)
            cms.append(cm)
        return (jnp.stack(xks, axis=-1), jnp.stack(yks, axis=-1),
                jnp.stack(cms))

    xfx6, yfx6, cm6 = jax.vmap(_face_flux, in_axes=(0,) * 7)(
        cx6, cy6, gs["dxa"], gs["dya"], gs["dx"], gs["dy"],
        gs["sin_sg"])
    # :258 mp_reduce_max + D1/D2, verbatim from the loop path.
    cmax = cm6.max(axis=0)
    nsplt_f = jax.lax.stop_gradient(jnp.floor(1.0 + cmax))  # (km,) >= 1
    nsplt_i = nsplt_f.astype(jnp.int32)
    nsplt_exceeded = jnp.any(nsplt_i > NSPLT_MAX)
    frac = 1.0 / nsplt_f

    # ---- q halo update (:276-281), verbatim: ONE batched cross-face
    # exchange call (v2a), not touched by face-batching.
    q = jnp.moveaxis(
        ext_scalar_sixface_allk(
            jnp.moveaxis(q6, 1, -1).reshape(q6.shape[0], m_a, m_a, km * nq),
            tab, "A").reshape(q6.shape[0], m_a, m_a, km, nq),
        -1, 1)

    # ---- the per-face transport kernel, vmapped over the face axis.
    # Positional order is fv_tp_2d's signature exactly (READ, not
    # recalled); bounded_domain is the gated True.
    def _face_tp(q2, cxk, cyk, xf, yf, dxa_g, dya_g, area_g, d6v, d6u,
                 rarea_g, da_t, rax, ray, mx, my):
        return fv_tp_2d(q2, cxk, cyk, npx, npx, hord_tr, xf, yf,
                        dxa_g, dya_g, area_g, d6v, d6u, rarea_g,
                        da_t, bd, rax, ray, lim_fac,
                        True, grid_type, *corners,
                        mfx=mx, mfy=my, duogrid=True)

    vtp = jax.vmap(_face_tp, in_axes=(0,) * 16)
    rarea_win = gs["rarea"][:, ng:ng + n, ng:ng + n]

    # ---- the sub-cycle loop (:284-388): ONE lax.scan of NSPLT_MAX for
    # ALL levels (M8-B, 2026-09-06).  The loop path scans per level; on
    # the window arm every scan iteration of every level fired its own
    # cross-face barrier + halo exchange (km * NSPLT_MAX * (nq + 1)
    # firings per step, most on masked-out iterations), and the exchange
    # ROUNDS were the measured overhead.  Every expression below is the
    # per-level one with k as a trailing axis: the frac rescale is the
    # same product per element, fv_tp_2d is vmapped over k on top of the
    # face vmap, the barrier/exchange take the trailing axis, and the
    # run/last masks are per level.
    dp1 = dp1_6
    fk = frac[None, None, None, :]                     # (1, 1, 1, km)
    cx6 = cx6 * fk
    cy6 = cy6 * fk
    mfx6 = mfx6.at[:, :, 0:n, :].multiply(fk)
    mfy6 = mfy6.at[:, 0:n, :, :].multiply(fk)
    xf6 = xfx6 * fk
    yf6 = yfx6 * fk
    mxk6 = mfx6[:, :, 0:n, :]   # (6, n+1, n, km) dummy
    myk6 = mfy6[:, 0:n, :, :]   # (6, n, n+1, km) dummy
    ra_x6 = (gs["area"][:, ng:ng + n, :, None]
             + xf6[:, 0:n, :, :] - xf6[:, 1:n + 1, :, :])
    ra_y6 = (gs["area"][:, :, ng:ng + n, None]
             + yf6[:, :, 0:n, :] - yf6[:, :, 1:n + 1, :])
    div6 = ((mxk6[:, 0:n, :, :] - mxk6[:, 1:n + 1, :, :]
             + myk6[:, :, 0:n, :] - myk6[:, :, 1:n + 1, :])
            * rarea_win[..., None])
    qn20 = q                                            # (6, nq, m, m, km)
    dp10 = dp1[:, ng:ng + n, ng:ng + n, :]              # (6, n, n, km)
    # level axis LAST on the level-varying operands, metrics shared
    vtpk = jax.vmap(vtp, in_axes=(-1, -1, -1, -1, -1, None, None, None,
                                  None, None, None, None, -1, -1, -1, -1),
                    out_axes=-1)
    run_q = lambda m: m[None, None, None, None, :]
    run_d = lambda m: m[None, None, None, :]

    def body(carry, it):
        qn2, dp1i, qfin = carry
        run = (it + 1) < nsplt_f                        # (km,)
        last = (it + 1) == nsplt_f
        dp2 = dp1i + div6
        base = jnp.zeros_like(qn2)
        for iq in range(nq):
            qm6, fx6, fy6 = vtpk(qn2[:, iq], cx6, cy6, xf6, yf6,
                                 gs["dxa"], gs["dya"], gs["area"],
                                 gs["del6_v"], gs["del6_u"],
                                 gs["rarea"], da6, ra_x6, ra_y6,
                                 mxk6, myk6)
            # flux_adj (:310-312): cross-face blend, verbatim, all
            # levels in one barrier (trailing axis).
            fx6, fy6 = average_shared_edge_cgrid(fx6, fy6, tab)
            d6 = ((fx6[:, 0:n, :, :] - fx6[:, 1:n + 1, :, :]
                   + fy6[:, :, 0:n, :] - fy6[:, :, 1:n + 1, :])
                  * rarea_win[..., None])
            u6 = ((qn2[:, iq, ng:ng + n, ng:ng + n, :] * dp1i + d6)
                  / dp2)
            base = base.at[:, iq].set(
                qm6.at[:, ng:ng + n, ng:ng + n, :].set(u6))
        # :361-379 qn2 halo refresh, verbatim cross-face exchange, all
        # levels and tracers in one call.
        nb_ = base.shape[0]
        exch = jnp.moveaxis(
            ext_scalar_sixface_allk(
                jnp.moveaxis(base, 1, -1).reshape(nb_, m_a, m_a, km * nq),
                tab, "A").reshape(nb_, m_a, m_a, km, nq),
            -1, 1)
        exch = _window_seam_refresh(tab, exch, nq)
        return (jnp.where(run_q(run), exch, qn2),
                jnp.where(run_d(run), dp2, dp1i),
                jnp.where(run_q(last), base, qfin)), None

    (_, dp1k_f, qk_f), _ = jax.lax.scan(
        body, (qn20, dp10, qn20), jnp.arange(NSPLT_MAX))
    q = qk_f
    dp1 = dp1.at[:, ng:ng + n, ng:ng + n, :].set(dp1k_f)

    return {"q": q, "dp1": dp1, "mfx": mfx6, "mfy": mfy6,
            "cx": cx6, "cy": cy6,
            "nsplt": nsplt_i, "nsplt_exceeded": nsplt_exceeded}


def make_alloc_flux_capacitors_jit(n: int, ng: int, km: int):
    """jit twin of alloc_flux_capacitors (every input is static)."""
    return jax.jit(functools.partial(alloc_flux_capacitors,
                                     n=n, ng=ng, km=km))


def make_require_tracer_2d_1l_lane_jit():
    """The guard reads only static config and never a tracer, so its
    jit twin is the guard itself; provided for the per-public-routine
    convention."""
    return require_tracer_2d_1l_lane


def make_tracer_2d_1l_sixface_jit(ctx: dict, *, km: int, nq: int,
                                  hord_tr: int, q_split: int = 0,
                                  nord_tr: int = 0, trdm: float = 0.0,
                                  lim_fac: float = 1.0,
                                  z_tracer: bool = True,
                                  inline_q: bool = False,
                                  batched: bool = False,
                                  check_nsplt: bool = True):
    """jit the phase with every static bound closed over (C3: dt, and
    only dt, stays dynamic -- a new time step never retraces).

    With check_nsplt=True (the default) the returned callable is D3's
    DEFAULT CALLER: it runs check_nsplt_schedule on the concrete
    outputs, so a resolved nsplt above NSPLT_MAX raises instead of
    silently under-transporting.  Pass check_nsplt=False to compose
    the raw jitted phase under further jit/grad/scan; then the caller
    owns the D3 check."""
    core = jax.jit(functools.partial(
        tracer_2d_1l_sixface, ctx, km=km, nq=nq, hord_tr=hord_tr,
        q_split=q_split, nord_tr=nord_tr, trdm=trdm, lim_fac=lim_fac,
        z_tracer=z_tracer, inline_q=inline_q, batched=batched))
    if not check_nsplt:
        return core

    def _checked(q6, dp1_6, flux_cap, dt):
        out = core(q6, dp1_6, flux_cap, dt=dt)
        check_nsplt_schedule(out)
        return out

    return _checked
