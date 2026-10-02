"""D-grid tail: d_sw3, BARRIER 2, d_sw4/5/6 -- km-general, JAX lane.

Functional, jit-compilable twin of ``legoesm.core.fv3_native_dsw_tail_3d``
(the ``dsw_tail_phase_3d`` half; this file continues in place with the
D-grid pressure phase and the NH D-grid pressure tail, authored as
further parts of this same module).

ORACLE CADENCE (the spec's block, checksum-pinned tree)::

    dyn_core.F90:961   call d_sw3
                :984   mpp_get_boundary BGRID_NE   <-- BARRIER 2
                :1102  call d_sw4
                :1107  call d_sw5
                :1256  call d_sw6
                :1401  call geopk        (D-grid)
                :1531  call one_grad_p   (hydrostatic, beta <= 0)

BARRIER 2'S EXTENT DIFFERS FROM BARRIER 1'S: it blends the B-grid corner
ingredients over ``i = is..ie+1``, ``j = js..je+1`` -- tile corners
included -- and it acts on ``ubb``/``vbbtemp``, NOT the final winds,
because the KE assembly that follows is
``ke = 0.5*(ubbtemp*vbbtemp + ubb*vbb)`` (``dyn_core.F90:1080-1085``).
It is applied ONE LEVEL AT A TIME with all six faces of that level
present -- a face blended against a stale neighbour is the silent
defect this unit exists to avoid.  The extent belongs to
``fv3_duo_halos.average_shared_edge_bgrid``; this module never
re-derives it.

CONVENTIONS inherited from the sibling 3-D phases: C1 one dict of
face-stacked arrays (face t = Fortran tile t+1, Fortran index 1-ng at
numpy 0, km at axis 2); C2 face and level loops are Python loops, not
vmap; C3 ctx/km/deck constants are static, dt is dynamic; C4 every
array the spec mutates comes back in the returned dict; C5 no
data-dependent guard (this phase has none, so it adds no check_*
keywords); C6 stage observation is by return, never a callback; C7 the
deck is built from the spec's own ``DUO_TAIL_CFG`` via
``SWConfig.from_mapping`` -- never restated here.

WHAT THIS MODULE DOES NOT DO: it introduces no numerics of its own --
every arithmetic operation lives in an already-ported kernel; the
barrier extents are the halo module's; ``d_con = 0.0`` throughout (the
shipped duo decks set it so), hence the KE-to-heat pathway and its
``heat_source`` allocation (``dyn_core.F90:322-325``, gated on
``d_con > 1.0E-5``) are inactive and d_con is not a parameter of this
module at all; and there is no moist or tracer path in this tail --
nothing between d_sw3 and d_sw6 reads a moisture field.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from legoesm.core.fv3_duo_stepper import SWConfig
from legoesm.core.fv3_duo_sw_core import (
    GridFlags,
    d_sw3_duo,
    d_sw4_duo,
    d_sw5_duo,
    d_sw6_duo,
)
from legoesm.core.fv3_native_dsw_tail_3d import (
    DUO_TAIL_CFG,
    require_real_area,
)
from legoesm.core.fv3_native_state_3d import (
    field_shape,
    require_no_remap_needed,
)
from legoesm.core.fv3_nh_core import riem_solver3, update_dz_d
from legoesm.core.fv3_pgrad import (
    geopk,
    nh_p_grad,
    one_grad_p,
    pe_halo,
    pk3_halo,
    pln_halo,
)
from legoesm.core.fv3_phase3d_common import (
    batch_size,
    build_batched_gs,
    require_bool,
    require_f64_jax,
    require_km,
    require_nord,
    stack_faces,
    stack_levels,
    validate_stacked,
)
from legoesm.grids.fv3_duo_halos import (
    average_shared_edge_bgrid,
    ext_scalar_sixface,
)

# The spec's own deck, imported not restated (C7).  "hord_tm" rides in it
# for update_dz_d's transport order (dyn_core.F90:1406), read by the NH
# tail further down this file.
_TAIL_DECK = SWConfig.from_mapping(DUO_TAIL_CFG)


def dsw_tail_phase_3d(ctx, state: dict, csw_outs: dict, dsw_outs: dict,
                      dt, km, *, cfg: SWConfig | None = None,
                      hydrostatic: bool = True,
                      remap_follows: bool = False,
                      damp_w: float | None = None,
                      batched: bool = False) -> dict:
    """``d_sw3`` -> BARRIER 2 -> ``d_sw4/5/6``, per level, all six faces.

    Stage observation is by RETURN (C6): the spec's ``stage_hook`` would
    be handed tracers under jit and is not ported.  Stage payloads, all
    face-stacked with km at axis 2:

    * S10 (``dyn_core.F90:961-966``) ``ubb_prebarrier`` /
      ``vbb_prebarrier`` / ``ubbtemp_prebarrier`` / ``vbbtemp_prebarrier``
      -- d_sw3's B-grid corner-velocity ingredients BEFORE the barrier,
      shape ``(6, npx, npx, km)`` from d_sw3's own contract;
    * S11 (``:969-1011``) ``ubb_postbarrier`` / ``vbbtemp_postbarrier``
      -- the blended pair, same shape (average_shared_edge_bgrid's
      ``(6, npx, npx)`` per level);
    * S12 (corner KE, ``:1015-1020``) ``ke_corner`` -- the assembly
      THIS phase performs, ``0.5*(ubbtemp_pre*vbbtemp_post +
      ubb_post*vbb_pre)`` on the B compute ring and exact zeros
      outside it.  Returned, and the reason is worth stating: a caller
      can recompute that formula from the barrier stacks, but
      recomputing a formula does not certify that this module used it,
      and the ``ke`` in the diagnostics below is d_sw5's OUTPUT ke --
      a different quantity, two stages later (measured 1.2e+02
      relative apart, job 9417474);
    * S13 (d_sw5 diagnostics, ``:1102-1123``) and S14 (d_sw6 winds,
      ``:1256-1288``) ARE recoverable and are not re-returned: S13 is
      the returned ``ke``/``wk``/``divg_d``/``delpc`` stacks as d_sw6
      consumed them, S14 is the returned ``u``/``v``.

    Field returns, each shape traced to its origin: ``u``
    ``(6, m_a, m_a+1, km)`` and ``v`` ``(6, m_a+1, m_a, km)`` (the
    spec's acc allocations for the D winds); ``ke``
    ``(6, m_a+1, m_a+1, km)`` (corner KE), and ``wk``/``divg_d``/
    ``delpc`` at whatever shapes d_sw5 itself returns -- stacked by
    ``stack_levels`` from level 0's actual shape, exactly why the spec
    refuses to guess (its (m_a+1, m_a+1) comment); ``w``
    ``(6, m_a, m_a, km)`` on the NH arm ONLY -- an absent key is an
    ABSENCE on the hydrostatic arm, mirroring the spec.

    ``dt`` is DYNAMIC (C3): it enters d_sw3/d_sw4/d_sw5 arithmetically
    only.  ``ctx``, ``km``, ``cfg`` and ``damp_w`` are STATIC.  None of
    ptop/akap/cp_air/p_fac/a_imp appears among this phase's callees, so
    none is declared here.  ``cfg`` is an :class:`SWConfig`; ``None``
    selects the deck built from the spec's ``DUO_TAIL_CFG``.  A dict
    raises.  ``damp_w=None`` reproduces the spec's fallback exactly
    (``damp_w -> cfg.damp_v`` on the NH arm, ``0.0`` on the
    hydrostatic arm).  ``d_con`` is not a parameter: it is 0.0
    throughout (the shipped duo decks set it so) and cannot be turned
    on from here.
    """
    fname = "dsw_tail_phase_3d"
    km = require_km(fname, km)
    require_bool(fname, "hydrostatic", hydrostatic)
    require_bool(fname, "remap_follows", remap_follows)
    require_bool(fname, "batched", batched)
    require_no_remap_needed(km, remap_follows=remap_follows)
    deck = _TAIL_DECK if cfg is None else cfg
    # decision B1: per-level additive del-2 sponge coefficient (static
    # floats; exactly 0.0 below the sponge; layers = 0 -> all zero, OFF)
    from legoesm.core.fv3_duo_stepper import sponge_d2_profile
    d2k = sponge_d2_profile(km, deck.sponge_del2_top_layers,
                            deck.sponge_del2_top_factor, deck.sponge_d2_top)
    sponge_on = bool(deck.sponge_del2_top_layers > 0 and deck.sponge_d2_top > 0.0)
    if not isinstance(deck, SWConfig):
        raise TypeError(
            f"{fname}: cfg must be an SWConfig or None, got {type(deck)!r}")
    validate_stacked(fname, state, ctx, km, ("u", "v"), what="D-grid state")
    validate_stacked(fname, csw_outs, ctx, km,
                     ("uc", "vc", "ua", "va", "divg_d"),
                     what="c_sw outputs")
    # `delp` (and `w` on the NH arm) are DECLARED fields, so they go
    # through the shape table.  The d_sw1 transport outputs do NOT: they
    # have their own staggerings that no table describes, and the sibling
    # transport phase says so where it stacks them ("d_sw1/d_sw2 publish
    # no such table -- re-deriving ten Fortran bound expressions here
    # would be exactly the kind of restatement that drifts").
    #
    # MEASURED, and it is why this split exists (job 9417450): routing
    # them through `validate_stacked` refused a CORRECT input --
    # `field_shape("ut")` is the C-grid `ut`, ``(m_a, m_a, km)``, while
    # d_sw1's `ut` is ``(m_a+1, m_a, km)``.  Two different quantities
    # under one name is exactly the hazard the table exists to catch, so
    # the table must not be applied across that boundary.
    dsw_declared = ("delp",) + (() if hydrostatic else ("w",))
    validate_stacked(fname, dsw_outs, ctx, km, dsw_declared,
                     what="d_sw outputs")
    dsw_stacks = ("ut", "vt", "crx_adv", "cry_adv", "xfx_adv", "yfx_adv",
                  "ra_x", "ra_y") + (() if hydrostatic else ("dw",))
    missing = [k for k in dsw_stacks if k not in dsw_outs]
    if missing:
        raise KeyError(
            f"{fname}: d_sw outputs is missing {missing}; keys are "
            f"{sorted(dsw_outs)}")
    for nm in dsw_stacks:
        arr = jnp.asarray(dsw_outs[nm])
        # What IS checkable without a table: the face axis and the level
        # axis.  A face or level slip is the defect that would otherwise
        # broadcast silently; the stagger axes are the kernels' own.
        if (arr.ndim != 4 or arr.shape[0] != batch_size(ctx)
                or arr.shape[3] != km):
            raise ValueError(
                f"{fname}: d_sw outputs[{nm!r}] has shape {arr.shape}; "
                f"expected (6, i, j, {km}) -- the face axis and the "
                f"level axis are fixed by convention C1 even where the "
                f"stagger is the kernel's own")
    # f64 entry gate -- reads static dtypes only, so it is jit-safe (C5).
    require_f64_jax(fname, {
        "state/u": state["u"], "state/v": state["v"],
        "csw/uc": csw_outs["uc"], "csw/vc": csw_outs["vc"],
        "csw/ua": csw_outs["ua"], "csw/va": csw_outs["va"],
        "csw/divg_d": csw_outs["divg_d"],
        "dsw/ut": dsw_outs["ut"], "dsw/vt": dsw_outs["vt"],
        "dsw/crx_adv": dsw_outs["crx_adv"],
        "dsw/cry_adv": dsw_outs["cry_adv"],
        "dsw/xfx_adv": dsw_outs["xfx_adv"],
        "dsw/yfx_adv": dsw_outs["yfx_adv"],
        "dsw/ra_x": dsw_outs["ra_x"], "dsw/ra_y": dsw_outs["ra_y"],
        "dsw/delp": dsw_outs["delp"],
        "dsw/w": dsw_outs.get("w"), "dsw/dw": dsw_outs.get("dw"),
    })

    n, ng, bd = ctx.n, ctx.ng, ctx.bd
    npx = ctx.npx
    m_a = n + 2 * ng
    ring = slice(ng, ng + npx)          # B compute ring in (m_a+1, m_a+1)
    fdt = state["u"].dtype              # f64 by the gate above
    nh_damp_w = 0.0 if hydrostatic else (
        deck.damp_v if damp_w is None else damp_w)

    if batched:
        return _dsw_tail_phase_3d_batched(
            ctx, state, csw_outs, dsw_outs, dt, km, deck=deck,
            d2k=d2k, sponge_on=sponge_on,
            hydrostatic=hydrostatic, nh_damp_w=nh_damp_w)

    # --- d_sw3 at every level, all faces (needed before barrier 2) -------
    # Python loops, not vmap (C2): each (t, k) body is independent, but
    # vectorising would be an optimisation, not a translation.
    nb = batch_size(ctx)
    s3 = [[None] * km for _ in range(nb)]   # per-face operands only (C1)
    for t in range(nb):
        # PER-FACE flags.  The first draft of this module took
        # `GridFlags.from_gridstruct(ctx.gs6[0])` -- a constructor that
        # does not exist (it is `from_gs`), and a per-RUN rather than
        # per-FACE value.  The context already carries `flags6`, built
        # face by face by `build_jax_duo_stepper_context`, and that is
        # what the km=1 stepper and the sibling 3-D phases pass.
        gs_t, fl_t = ctx.gs6[t], ctx.flags6[t]
        for k in range(km):
            # uc/vc are the c_sw outputs p_grad_c updated in place in the
            # C-grid phase; d_sw3 reads them and does not return them.
            s3[t][k] = d_sw3_duo(state["u"][t][..., k],
                                 state["v"][t][..., k],
                                 csw_outs["uc"][t][..., k],
                                 csw_outs["vc"][t][..., k],
                                 gs_t, fl_t, bd, npx, npx, dt=dt,
                                 hord_mt=deck.hord_mt)

    # --- BARRIER 2 (dyn_core.F90:984, BGRID_NE) ---------------------------
    # Blends ubb and vbbtemp over i = is..ie+1, j = js..je+1, corners
    # included -- an extent DIFFERENT from barrier 1's, owned by
    # average_shared_edge_bgrid and never re-derived here.  ONE LEVEL AT
    # A TIME so all six faces of that level are present; blending a face
    # against a stale neighbour is the silent defect this unit avoids.
    # The spec's np.array(copy=True) is dropped: a slice on this lane is
    # a fresh value, there is no buffer to alias.
    ubb_bld, vbbtemp_bld = [], []
    for k in range(km):
        xb6 = jnp.stack([s3[t][k]["ubb"] for t in range(nb)], axis=0)
        yb6 = jnp.stack([s3[t][k]["vbbtemp"] for t in range(nb)], axis=0)
        xb6, yb6 = average_shared_edge_bgrid(xb6, yb6, ctx.tab)
        ubb_bld.append(xb6)
        vbbtemp_bld.append(yb6)

    # --- d_sw4 / d_sw5 / d_sw6 on the blended ingredients ----------------
    per_face = []
    for t in range(batch_size(ctx)):
        gs_t, fl_t = ctx.gs6[t], ctx.flags6[t]
        u_lv, v_lv, w_lv = [], [], []
        ke_corner = []          # S12, the corner KE this phase assembles
        # The spec allocates each diagnostic from the shape the stage
        # actually returns; here the per-level lists carry that rôle and
        # stack_levels asserts no stage changes shape between levels.
        diag = {nm: [] for nm in ("ke", "wk", "divg_d", "delpc")}
        for k in range(km):
            u_k = state["u"][t][..., k]
            v_k = state["v"][t][..., k]
            ut_k = dsw_outs["ut"][t][..., k]
            vt_k = dsw_outs["vt"][t][..., k]
            # KE at the B corners (dyn_core.F90:1080-1085, Lin-Rood): the
            # product of PRE- and POST-barrier ingredients -- barrier 2
            # deliberately sat between d_sw3 and this assembly.  The
            # NumPy np.zeros + slice write becomes a returned index copy.
            kee = s3[t][k]["ubbtemp"] * vbbtemp_bld[k][t]
            ke = jnp.zeros((m_a + 1, m_a + 1), dtype=fdt).at[ring, ring].set(
                0.5 * (kee + ubb_bld[k][t] * s3[t][k]["vbb"]))
            # S12 (dyn_core.F90:1015-1020) IS RETURNED, and the earlier
            # claim that it is "recoverable" was wrong in the way that
            # matters: a caller can recompute the formula from the
            # returned barrier stacks, but recomputing a formula does
            # not certify that THIS module used it -- and `ke` in the
            # returned diagnostics is d_sw5's OUTPUT ke, a different
            # quantity that has been through two more stages.  Measured
            # (job 9417474): the reconstruction and the returned `ke`
            # differ by 1.2e+02 relative, exactly as they should.  So
            # the assembly is exposed at the boundary it happens on.
            ke_corner.append(ke)
            s4 = d_sw4_duo(u_k, v_k, ut_k, vt_k, ke, fl_t, bd, npx, npx,
                           dt=dt)
            s5 = d_sw5_duo(
                dsw_outs["delp"][t][..., k], u_k, v_k,
                csw_outs["uc"][t][..., k], csw_outs["vc"][t][..., k],
                csw_outs["ua"][t][..., k], csw_outs["va"][t][..., k],
                csw_outs["divg_d"][t][..., k],
                dsw_outs["crx_adv"][t][..., k],
                dsw_outs["cry_adv"][t][..., k],
                dsw_outs["xfx_adv"][t][..., k],
                dsw_outs["yfx_adv"][t][..., k],
                dsw_outs["ra_x"][t][..., k],
                dsw_outs["ra_y"][t][..., k],
                s4["ke"], gs_t, fl_t, bd, npx, npx, dt=dt,
                hord_vt=deck.hord_vt, nord=deck.nord,
                dddmp=deck.dddmp, d2_bg=deck.d2_bg + d2k[k],
                d4_bg=deck.d4_bg,
                d_con=0.0, hydrostatic=hydrostatic,
                w=None if hydrostatic else dsw_outs["w"][t][..., k],
                dw=None if hydrostatic else dsw_outs["dw"][t][..., k],
                damp_w=nh_damp_w)
            s6 = d_sw6_duo(u_k, v_k, s5["ut"], s5["vt"], s5["ke"],
                           s5["wk"], s5["vortfluxx"], s5["vortfluxy"],
                           gs_t, fl_t, bd, npx, npx,
                           nord_v=deck.nord_v, damp_v=deck.damp_v,
                           # per-level STATIC gate on the loop path: levels
                           # below the sponge trace no del-2 chain at all
                           # (bitwise the pre-B1 step there; the batched arm
                           # adds an exact 0.0 flux instead)
                           d_con=0.0, sponge_on=bool(sponge_on and d2k[k] > 0.0),
                           d2_sponge=d2k[k])
            u_lv.append(s6["u"])
            v_lv.append(s6["v"])
            if not hydrostatic:
                # w finalised against the UPDATED delp plus the d_sw2 dw
                # increment (d_sw5's NH block); stacked into the return.
                w_lv.append(s5["w"])
            for nm in diag:
                if nm in s5:       # dict membership is static under jit
                    diag[nm].append(s5[nm])
        face = {"u": stack_levels(fname, "u", u_lv),
                "v": stack_levels(fname, "v", v_lv),
                "ke_corner": stack_levels(fname, "ke_corner", ke_corner)}
        if not hydrostatic:
            face["w"] = stack_levels(fname, "w", w_lv)
        for nm, lvls in diag.items():
            if lvls:
                face[nm] = stack_levels(fname, nm, lvls)
        per_face.append(face)
    outs = stack_faces(fname, per_face)

    # --- stage payloads (C6: by return, not callback) --------------------
    # stack_levels yields the per-face (..., km) stack; jnp.stack over a
    # new leading axis gives the C1 layout (6, ..., km).
    for nm in ("ubb", "vbb", "ubbtemp", "vbbtemp"):
        outs[nm + "_prebarrier"] = jnp.stack(
            [stack_levels(fname, nm, [s3[t][k][nm] for k in range(km)])
             for t in range(batch_size(ctx))], axis=0)
    outs["ubb_postbarrier"] = jnp.stack(
        [stack_levels(fname, "ubb", [ubb_bld[k][t] for k in range(km)])
         for t in range(batch_size(ctx))], axis=0)
    outs["vbbtemp_postbarrier"] = jnp.stack(
        [stack_levels(fname, "vbbtemp",
                      [vbbtemp_bld[k][t] for k in range(km)])
         for t in range(batch_size(ctx))], axis=0)
    return outs


def _dsw_tail_phase_3d_batched(ctx, state, csw_outs, dsw_outs, dt, km,
                               *, deck, hydrostatic, nh_damp_w, d2k=None, sponge_on=False) -> dict:
    """The vmap-over-faces arm of :func:`dsw_tail_phase_3d` (C2a).

    Entry gates already ran in the caller.  The ``d_sw3`` and
    ``d_sw4/5/6`` FACE loops become one ``jax.vmap`` per level over
    ``build_batched_gs``'s stacked view, nested in an outer ``jax.vmap``
    over the LEVEL axis (independent, identical bodies: R1a);
    and BARRIER 2 sits exactly where the loop path puts it -- between
    d_sw3 and the KE assembly, one level at a time, on the full
    six-face stack (dyn_core.F90:984).  It is a cross-face collective
    and is not face-batched (its operands arrive already stacked from
    the vmapped d_sw3, so the loop path's per-face ``jnp.stack`` is
    simply absent -- a pure index copy either way).

    ``da_min`` / ``da_min_c`` ride the vmap as traced per-face scalars
    and are rebuilt into a ``GridFlags`` inside each batched body --
    their consumers are arithmetic-only (d_sw5's ``dd8 =
    (da_min_c*d4_bg)**(nord+1)`` / ``damp2 = da_min_c*max(...)`` and
    ``fv_tp_2d``'s ``damp``; d_sw6's ``(damp_v*da_min_c)**(nord_v+1)``)
    -- while every other flag field is face-invariant by the
    common-mode gate and closes over as the one shared Python value.
    ``d_sw4_duo`` reads no gridstruct array at all, only the flags.

    in_axes: every per-face field plane, the d_sw3/barrier stage
    planes, the gridstruct dict and the ``da_min6``/``da_min_c6``
    scalars are 0 (per face); ``bd``/``npx``/the deck knobs/
    ``nh_damp_w``/``hydrostatic`` are closed over (static Python
    values), plus ``dt`` (traced but shared).  On the hydrostatic arm
    the ``w``/``dw`` operands are ``None`` -- an empty pytree under
    vmap, exactly the loop path's ``w=None``.
    """
    fname = "dsw_tail_phase_3d[batched]"
    bview = build_batched_gs(ctx)
    shared = bview["flags"]
    da6, dac6 = bview["da_min6"], bview["da_min_c6"]
    bd = ctx.bd
    n, ng = ctx.n, ctx.ng
    npx = ctx.npx
    m_a = n + 2 * ng
    ring = slice(ng, ng + npx)
    fdt = state["u"].dtype

    # --- d_sw3 at every level, faces vmapped (:961) -------------------
    def one_face_sw3(u2, v2, uc2, vc2, gs_t, da_t, dac_t):
        fl = GridFlags(da_min=da_t, da_min_c=dac_t, **shared)
        return d_sw3_duo(u2, v2, uc2, vc2, gs_t, fl, bd, npx, npx,
                         dt=dt, hord_mt=deck.hord_mt)

    vf3 = jax.vmap(one_face_sw3, in_axes=(0,) * 7)
    # LEVELS (2026-09-29, the km=32 compile): the per-level bodies are
    # independent and identical (R1a: only data varies with k; deck
    # statics and dt are shared), so the former Python level loops --
    # which traced d_sw3 and the d_sw4/5/6 tail km times each -- are an
    # outer vmap over the trailing level axis, levels outside faces.
    # Level axis 3 in and out: the (6, i, j, km) layout the loop path
    # and the former stack_levels_batched produced.
    vfk3 = jax.vmap(vf3, in_axes=(3, 3, 3, 3, None, None, None),
                    out_axes=3)
    s3 = vfk3(state["u"], state["v"], csw_outs["uc"], csw_outs["vc"],
              bview["gs"], da6, dac6)

    # --- BARRIER 2, verbatim blend, ALL LEVELS IN ONE CALL (M8-B) -----
    # (dyn_core.F90:984, BGRID_NE; extent owned by
    # average_shared_edge_bgrid.)  The blend at a cell reads only that
    # level, so the levels ride as a trailing axis: the flat path vmaps
    # the certified 2-D blend over it and the window arm fires ONE
    # exchange instead of km (the exchange rounds were the measured
    # overhead, 2026-09-06).
    xb6, yb6 = average_shared_edge_bgrid(s3["ubb"], s3["vbbtemp"],
                                         ctx.tab)

    # --- KE assembly + d_sw4/5/6, faces vmapped -----------------------
    def one_face_tail(u_k, v_k, ut_k, vt_k, delp_k, uc_k, vc_k, ua_k,
                      va_k, dg_k, crx_k, cry_k, xfx_k, yfx_k, rax_k,
                      ray_k, ubbtemp_k, vbb_k, ubb_b, vbbtemp_b, w_k,
                      dw_k, d2_k, gs_t, da_t, dac_t):
        fl = GridFlags(da_min=da_t, da_min_c=dac_t, **shared)
        # S12 (dyn_core.F90:1015-1020): pre- x post-barrier product,
        # identical formula and window to the loop path.
        kee = ubbtemp_k * vbbtemp_b
        ke = jnp.zeros((m_a + 1, m_a + 1), dtype=fdt).at[ring, ring].set(
            0.5 * (kee + ubb_b * vbb_k))
        s4 = d_sw4_duo(u_k, v_k, ut_k, vt_k, ke, fl, bd, npx, npx,
                       dt=dt)
        s5 = d_sw5_duo(delp_k, u_k, v_k, uc_k, vc_k, ua_k, va_k, dg_k,
                       crx_k, cry_k, xfx_k, yfx_k, rax_k, ray_k,
                       s4["ke"], gs_t, fl, bd, npx, npx, dt=dt,
                       hord_vt=deck.hord_vt, nord=deck.nord,
                       dddmp=deck.dddmp, d2_bg=deck.d2_bg + d2_k,
                       d4_bg=deck.d4_bg, d_con=0.0,
                       hydrostatic=hydrostatic, w=w_k, dw=dw_k,
                       damp_w=nh_damp_w)
        s6 = d_sw6_duo(u_k, v_k, s5["ut"], s5["vt"], s5["ke"],
                       s5["wk"], s5["vortfluxx"], s5["vortfluxy"],
                       gs_t, fl, bd, npx, npx, nord_v=deck.nord_v,
                       damp_v=deck.damp_v, d_con=0.0,
                       sponge_on=sponge_on, d2_sponge=d2_k)
        face = {"u": s6["u"], "v": s6["v"], "ke_corner": ke,
                "ke": s5["ke"], "wk": s5["wk"],
                "divg_d": s5["divg_d"], "delpc": s5["delpc"]}
        if not hydrostatic:
            face["w"] = s5["w"]
        return face

    # d2_k (the per-level sponge coefficient) is one scalar per LEVEL:
    # shared across faces (None here), mapped over its own axis 0 in the
    # level vmap below.
    vft = jax.vmap(one_face_tail, in_axes=(0,) * 22 + (None, 0, 0, 0))
    # w/dw are None in the hydrostatic arm: a None operand has no
    # leaves, so its in_axes entry is irrelevant (same as on the face
    # vmap above); every array operand carries its level axis at 3.
    w_ax = None if hydrostatic else 3
    vfkt = jax.vmap(vft, in_axes=(3,) * 20 + (w_ax, w_ax, 0, None, None, None),
                    out_axes=3)
    d2k_arr = jnp.asarray(d2k if d2k is not None else (0.0,) * km, dtype=fdt)
    per = vfkt(
        state["u"], state["v"],
        dsw_outs["ut"], dsw_outs["vt"], dsw_outs["delp"],
        csw_outs["uc"], csw_outs["vc"], csw_outs["ua"], csw_outs["va"],
        csw_outs["divg_d"],
        dsw_outs["crx_adv"], dsw_outs["cry_adv"],
        dsw_outs["xfx_adv"], dsw_outs["yfx_adv"],
        dsw_outs["ra_x"], dsw_outs["ra_y"],
        s3["ubbtemp"], s3["vbb"], xb6, yb6,
        None if hydrostatic else dsw_outs["w"],
        None if hydrostatic else dsw_outs["dw"],
        d2k_arr,
        bview["gs"], da6, dac6)

    # --- assembly: identical keys and layouts to the loop path --------
    names = ("u", "v", "ke_corner", "ke", "wk", "divg_d",
             "delpc") + (() if hydrostatic else ("w",))
    outs = {}
    for nm in names:
        if nm not in per or per[nm].shape[3] != km:
            raise KeyError(
                f"{fname}: tail returned no level-batched {nm!r} (keys "
                f"{sorted(per)}); the 3-D assembler must not silently "
                f"drop a stage output")
        outs[nm] = per[nm]
    for nm in ("ubb", "vbb", "ubbtemp", "vbbtemp"):
        outs[nm + "_prebarrier"] = s3[nm]
    outs["ubb_postbarrier"] = xb6
    outs["vbbtemp_postbarrier"] = yb6
    return outs


def make_dsw_tail_phase_3d_jit(ctx, km, *, cfg: SWConfig | None = None,
                               hydrostatic: bool = True,
                               remap_follows: bool = False,
                               damp_w: float | None = None,
                               batched: bool = False):
    """Pre-bound jit closure ``f(state, csw_outs, dsw_outs, dt) -> dict``.

    ``ctx``, ``km``, ``cfg`` and ``damp_w`` are baked in as STATIC
    values (C3) -- the entry gates re-run at trace time on shapes,
    dtypes and static Python values only.  ``dt`` stays a traced
    argument, so a new acoustic time step does not retrace the phase.
    No ``donate_argnums`` (it conflicts with reverse-mode AD, which is
    the point of this lane).
    """
    return jax.jit(partial(
        dsw_tail_phase_3d, ctx, km=km, cfg=cfg, hydrostatic=hydrostatic,
        remap_follows=remap_follows, damp_w=damp_w, batched=batched))


# ============================================================================
# PART B -- dgrid_pressure_phase_3d: D-grid geopk then one_grad_p.
# Spec's own citations, none added: dyn_core.F90:1401 call geopk (D-grid);
# :1511-1519 the pk copy into the compute window (remap_step only), taken
# BEFORE :1531 call one_grad_p overwrites pk with B-grid corner values.
# ============================================================================


def dgrid_pressure_phase_3d(ctx, dsw_outs, tail_outs, km, *, dt, ptop, akap,
                            cp_air, a2b_ord=4, d_ext=0.0, remap_step=False,
                            remap_follows=False, batched=False):
    """D-grid ``geopk`` (dyn_core.F90:1401) then ``one_grad_p`` (:1531).

    JAX twin of the spec routine of the same name.  No numerics live
    here: every operation happens inside ``fv3_pgrad.geopk`` /
    ``fv3_pgrad.one_grad_p``; this routine owns only the CADENCE (per
    face: ``geopk``, then the :1511-1519 ``pk`` snapshot when
    ``remap_step``, then ``one_grad_p``) and the face stacking.

    Static / dynamic (C3): ``ctx``, ``km``, ``ptop``, ``akap``,
    ``cp_air``, ``a2b_ord``, ``d_ext``, ``remap_step``,
    ``remap_follows`` are STATIC; ``dt`` is DYNAMIC.  ``ptop``/
    ``akap``/``cp_air`` are static because geopk's contract declares
    every keyword static ("deck constant or #ifdef/branch selector")
    and one_grad_p takes the same ``ptop``/``akap``; they never vary
    inside a run.  ``d_ext`` is static because it selects one_grad_p's
    a2b_ord2/mass-weighted branch in Python (at ``d_ext = 0`` the
    oracle sets ``divg2 = 0`` and skips it).  ``dt`` enters one_grad_p
    arithmetically only, so a new time step must not recompile this
    phase.  The spec's ``stage_hook`` is NOT ported (C6); the S16
    payload is RETURNED instead.

    Returns ONE face-stacked dict (C1; the spec returns a list of six
    per-face dicts), level axis at position 2:

    ``pk``, ``gz``
        POST-``one_grad_p`` values -- exactly what the spec's
        ``press[t]["pk"]``/``["gz"]`` hold on return: B-grid corners
        on ``[is,ie+1] x [js,je+1]`` (a2b_ord4 ``replace=.true.``),
        A-grid elsewhere.  Per-face shape ``(m, m, km+1)`` from
        one_grad_p's stated storage convention; stacked to
        ``(6, m, m, km+1)``.
    ``pe``, ``peln``, ``pkz``
        geopk's own outputs with geopk's own shapes (its contract:
        same shapes as its NumPy twin); untouched by one_grad_p.
    ``u``, ``v``
        one_grad_p's updated winds -- the S17_onegradp payload.  The
        spec mutates ``tail_outs[t]["u"]``/``["v"]`` in place; C4
        returns them instead.  Stacked per ``field_shape("u"/"v",
        n, ng, km)``.
    ``pk_pre_onegradp``, ``gz_pre_onegradp``
        the S16_geopkD payload: geopk's ``pk``/``gz`` as they stood
        BEFORE one_grad_p scratched them; unrecoverable from the other
        returns, hence returned (C6).  The spec's ``_s16`` also
        carried ``pkz``, which one_grad_p does not touch and is
        therefore recoverable from ``pkz`` above -- not duplicated.
    ``pk_remap``
        present ONLY when ``remap_step`` is true (the spec's gating):
        the dyn_core.F90:1511-1519 ``pk`` snapshot taken BEFORE :1531
        overwrites ``pk`` with corner values.
    """
    fname = "dgrid_pressure_phase_3d"
    km = require_km(fname, km)
    require_bool(fname, "remap_step", remap_step)
    require_bool(fname, "remap_follows", remap_follows)
    require_bool(fname, "batched", batched)
    # Rule 5: a2b_ord is a static scheme selector; one_grad_p's arms
    # are a2b_ord2 (the d_ext branch) and _a2b_ord4_k -- any other
    # value selects no arm.
    a2b_ord = require_nord(fname, "a2b_ord", a2b_ord)
    if a2b_ord not in (2, 4):
        raise ValueError(f"{fname}: a2b_ord must be 2 or 4, got {a2b_ord!r}")
    # C5-style tracer screen on the STATIC scalars: the callees branch
    # on these in Python at trace time, so a tracer must fail loudly
    # here rather than deep inside geopk / one_grad_p.
    for _nm, _v in (("ptop", ptop), ("akap", akap),
                    ("cp_air", cp_air), ("d_ext", d_ext)):
        if isinstance(_v, jax.Array):
            raise TypeError(f"{fname}: {_nm} is STATIC; got a traced array")
    require_no_remap_needed(km, remap_follows=remap_follows)
    validate_stacked(fname, dsw_outs, ctx, km, ("delp", "pt"),
                     what="dsw_outs (D-grid transport output, DSW1_OUT_2D keys)")
    validate_stacked(fname, tail_outs, ctx, km, ("u", "v"),
                     what="tail_outs (d_sw6 winds)")
    require_f64_jax(fname, {"delp": dsw_outs["delp"],
                            "pt": dsw_outs["pt"],
                            "u": tail_outs["u"],
                            "v": tail_outs["v"]})

    bd = ctx.bd
    n, ng = ctx.n, ctx.ng
    m = n + 2 * ng
    # ctx is an object (attribute access); the spec's ctx.get("hs6")
    # becomes getattr with the same None default.  Static either way.
    hs6 = getattr(ctx, "hs6", None)

    if batched:
        return _dgrid_pressure_phase_3d_batched(
            ctx, dsw_outs, tail_outs, km, dt=dt, ptop=ptop, akap=akap,
            cp_air=cp_air, a2b_ord=a2b_ord, d_ext=d_ext,
            remap_step=remap_step)

    per_face = []
    for t in range(batch_size(ctx)):
        # C2: the face loop is a Python loop, not vmap.  The six bodies
        # are independent -- face t reads only its own slices and geopk
        # output, and writes only its own returned arrays; no face
        # reads what another wrote -- so batching would be legal but is
        # an optimisation, not a translation.
        # hs follows the run's storage dtype (fp32/mixed increment 2):
        # ctx.hs6 is already cast by the context builder, so DON'T force
        # f64 here (that would promote the f32 geopk column back to f64
        # and trip the uniformity gate). fp64 default: hs6 is f64 ->
        # byte-identical. The zeros fallback follows a field (delp).
        hs = (jnp.asarray(hs6[t]) if hs6 is not None
              else jnp.zeros((m, m), dtype=dsw_outs["delp"].dtype))
        got = geopk(dsw_outs["delp"][t], dsw_outs["pt"][t], hs, bd,
                    km=km, ptop=ptop, akap=akap, cp_air=cp_air,
                    cg=False, duogrid=True, computehalo=False,
                    npx=bd.ie + 1, npy=bd.je + 1, a2b_ord=a2b_ord,
                    bounded_domain=False, sw_dynamics=False)
        # S16 snapshot.  The spec's np.array(..., copy=True) exists only
        # because one_grad_p scratches pk/gz in place; on this lane pk/gz
        # are immutable values and one_grad_p RETURNS fresh ones, so
        # binding the pre-call references IS the copy (nothing to alias).
        pk_pre, gz_pre = got["pk"], got["gz"]
        if remap_step:  # static config value: stays a Python if (rule 1)
            # dyn_core.F90:1511-1519, taken BEFORE :1531 one_grad_p.
            pk_remap = pk_pre
        # divg2 freshly zero per face, exactly the spec's loop body.
        # dtype follows storage (fp32/fp64), from dsw_outs["delp"]
        divg2 = jnp.zeros((m + 1, m + 1), dtype=dsw_outs["delp"].dtype)
        u_t, v_t, pk_t, gz_t = one_grad_p(
            tail_outs["u"][t], tail_outs["v"][t], got["pk"], got["gz"],
            divg2, dsw_outs["delp"][t], ctx.gs6[t], bd,
            npx=bd.ie + 1, npy=bd.je + 1, npz=km, dt=dt, ptop=ptop,
            akap=akap, hydrostatic=True, a2b_ord=a2b_ord, d_ext=d_ext,
            ng=ng, duogrid=True)
        face = {"pk": pk_t, "gz": gz_t,           # post-one_grad_p (scratched)
                "pe": got["pe"], "peln": got["peln"], "pkz": got["pkz"],
                "u": u_t, "v": v_t,               # C4: S17, caller threads
                "pk_pre_onegradp": pk_pre,        # C6: S16_geopkD payload
                "gz_pre_onegradp": gz_pre}
        if remap_step:
            face["pk_remap"] = pk_remap           # spec's gating, kept exact
        per_face.append(face)
    return stack_faces(fname, per_face)


def _dgrid_pressure_phase_3d_batched(ctx, dsw_outs, tail_outs, km, *,
                                     dt, ptop, akap, cp_air, a2b_ord,
                                     d_ext, remap_step) -> dict:
    """The vmap-over-faces arm of :func:`dgrid_pressure_phase_3d` (C2a).

    Entry gates already ran in the caller.  One ``jax.vmap`` replaces
    the face loop; the per-face body is IDENTICAL to the loop path's --
    ``geopk`` (which takes no gridstruct at all), the ``remap_step``
    snapshot, then ``one_grad_p``.  No GridFlags is involved on this
    chain, so there is no da_min threading here.

    in_axes: ``delp``/``pt``/``hs``/``u``/``v`` and the gridstruct dict
    are 0 (per face).  Closed over (face-invariant): ``bd`` and every
    static keyword (``ptop``/``akap``/``cp_air``/``a2b_ord``/``d_ext``/
    ``remap_step`` -- all Python values, exactly the loop path's static
    split), plus ``dt`` (traced but shared).  ``divg2`` is a fresh zero
    inside the body, the spec's per-face allocation.
    """
    bview = build_batched_gs(ctx)
    bd = ctx.bd
    n, ng = ctx.n, ctx.ng
    m = n + 2 * ng
    hs6 = getattr(ctx, "hs6", None)
    # follow storage dtype (ctx.hs6 already cast); zeros fallback -> field
    hs_stack = (jnp.asarray(hs6) if hs6 is not None
                else jnp.zeros((6, m, m), dtype=dsw_outs["delp"].dtype))

    def one_face(delp_t, pt_t, hs_t, u_t, v_t, gs_t):
        got = geopk(delp_t, pt_t, hs_t, bd, km=km, ptop=ptop,
                    akap=akap, cp_air=cp_air, cg=False, duogrid=True,
                    computehalo=False, npx=bd.ie + 1, npy=bd.je + 1,
                    a2b_ord=a2b_ord, bounded_domain=False,
                    sw_dynamics=False)
        pk_pre, gz_pre = got["pk"], got["gz"]
        # dtype follows storage (fp32/fp64), from delp_t
        divg2 = jnp.zeros((m + 1, m + 1), dtype=delp_t.dtype)
        u_o, v_o, pk_t, gz_t = one_grad_p(
            u_t, v_t, got["pk"], got["gz"], divg2, delp_t, gs_t, bd,
            npx=bd.ie + 1, npy=bd.je + 1, npz=km, dt=dt, ptop=ptop,
            akap=akap, hydrostatic=True, a2b_ord=a2b_ord, d_ext=d_ext,
            ng=ng, duogrid=True)
        face = {"pk": pk_t, "gz": gz_t,
                "pe": got["pe"], "peln": got["peln"], "pkz": got["pkz"],
                "u": u_o, "v": v_o,
                "pk_pre_onegradp": pk_pre,
                "gz_pre_onegradp": gz_pre}
        if remap_step:   # static config value: stays a Python if
            face["pk_remap"] = pk_pre
        return face

    return jax.vmap(one_face, in_axes=(0,) * 6)(
        dsw_outs["delp"], dsw_outs["pt"], hs_stack,
        tail_outs["u"], tail_outs["v"], bview["gs"])


def make_dgrid_pressure_phase_3d_jit(ctx, km, *, ptop, akap, cp_air,
                                     a2b_ord=4, d_ext=0.0, remap_step=False,
                                     remap_follows=False, batched=False):
    """Build the jit-compiled :func:`dgrid_pressure_phase_3d`.

    Every STATIC operand is baked in here, so an invalid static value
    raises at MAKE time (rule 5), not at first trace; the entry gates
    run again inside the trace on the same static values.  The
    returned function takes the DYNAMIC operands only::

        f(dsw_outs, tail_outs, dt) -> face-stacked dict

    ``dt`` is an ARGUMENT, not a closure constant, so a new time step
    reuses the compiled phase (C3).  No ``donate_argnums`` (rule 3).
    """
    fname = "dgrid_pressure_phase_3d"
    km = require_km(fname, km)
    require_bool(fname, "remap_step", remap_step)
    require_bool(fname, "remap_follows", remap_follows)
    require_bool(fname, "batched", batched)
    a2b_ord = require_nord(fname, "a2b_ord", a2b_ord)
    if a2b_ord not in (2, 4):
        raise ValueError(f"{fname}: a2b_ord must be 2 or 4, got {a2b_ord!r}")
    for _nm, _v in (("ptop", ptop), ("akap", akap),
                    ("cp_air", cp_air), ("d_ext", d_ext)):
        if isinstance(_v, jax.Array):
            raise TypeError(f"{fname}: {_nm} is STATIC; got a traced array")
    require_no_remap_needed(km, remap_follows=remap_follows)

    def _phase(dsw_outs, tail_outs, dt):
        return dgrid_pressure_phase_3d(
            ctx, dsw_outs, tail_outs, km, dt=dt, ptop=ptop, akap=akap,
            cp_air=cp_air, a2b_ord=a2b_ord, d_ext=d_ext,
            remap_step=remap_step, remap_follows=remap_follows,
            batched=batched)

    return jax.jit(_phase)


# ---------------------------------------------------------------------------
# PART C -- the non-hydrostatic D-grid tail.  Spec:
# fv3_native_dsw_tail_3d.dgrid_nh_pressure_phase_3d (dyn_core.F90:1403-1543):
# update_dz_d -> riem_solver3 -> pe_halo / (pln_halo | pk3_halo) -> zh+pkc
# duo exchanges -> gz = zh*grav -> (square_domain: second pkc exchange) ->
# nh_p_grad.  This chain REPLACES part B's hydrostatic geopk + one_grad_p
# at beta = 0; everything it writes comes back as a functional carry.
# ---------------------------------------------------------------------------


def _ext_scalar_planes_6(planes6, ctx):
    """Functional twin of the spec's in-place ``_ext_scalar_planes_6``.

    The NH D-stage sites (dyn_core.F90:1482-1483: ``ext_scalar(zh)``,
    ``ext_scalar(pkc)``) exchange ONE 2-D plane per face at A
    staggering.  The spec's ext-bundle / single-tile-fallback dispatch
    collapses on this lane to the ONE ported exchange -- ``ctx.tab`` is
    the duo halo table the barrier module itself takes -- so there is no
    substitution to declare and nothing to fail closed against.

    Returns the exchanged (6, i, j) stack; nothing is mutated.
    """
    return ext_scalar_sixface(planes6, ctx.tab, "A")


def nh_exchanged_area6(ctx):
    """The gridstruct's ``area``, face-stacked, EXCHANGING NOTHING.

    Twin of the spec's ``nh_exchanged_area6`` (which returns a per-face
    LIST and caches on the ctx): ONE ``(6, i, j)`` stack comes back here
    (convention C1).

    The NAME is historical and the behaviour it names is gone.  This
    routine used to run an A-grid six-face exchange over the area planes,
    on the stated grounds that the gridstruct leaves BIG_NUMBER sentinels
    in the corner-diagonal halo.  Measured 2026-08-22: on the contexts
    this path is built with, it does not -- and the exchanged copy was
    12% wrong in the halo edge strips, which was the whole non-hydrostatic
    parity gap (6.6116e-04 -> 1.4778e-06 on its removal).  The authority
    lane's docstring carries the numbers.

    The precondition that deletion rests on -- a real area everywhere --
    is CHECKED, by the same ``require_real_area`` the authority lane
    calls, on the static gridstruct arrays before any tracing.  A raw
    plain builder result does carry the sentinel and is refused here.

    Deviations from the spec, each with its reason:
    - No ``ctx["nh_area6"]`` cache: a value cached during one trace is a
      TRACER, and handing it to the next trace is exactly the defect the
      cache would cause.  The metric is a pure function of static inputs,
      so XLA constant-folds it -- the same work the cache saved.
    - No per-face ``copy``: the spec's ``np.array(..., copy=True)``
      guards a buffer alias that cannot exist on a functional lane
      (sibling deviation D2's rule).
    - ``rarea`` is not returned: the caller takes ``1.0 / area`` (the
      spec's ``ctx["nh_rarea6"]``), so no second cached value exists.
    """
    # The gridstruct arrays are STATIC (concrete numpy on the ctx), so
    # this is a build-time check on the same values the authority lane
    # checks, not a data-dependent branch on a tracer.
    for t in range(batch_size(ctx)):
        require_real_area(ctx.gs6[t]["area"], t)
    return jnp.stack(
        # follow storage dtype: ctx.gs6 area is already cast by the
        # context builder (fp64 default -> byte-identical); DON'T re-force
        # f64 or it promotes the f32 divergence back to f64.
        [jnp.asarray(ctx.gs6[t]["area"])
         for t in range(batch_size(ctx))], axis=0)


def _nh_tail_cfg(fname, cfg, hord_tm, nord_w, damp_w):
    """Effective tail deck -> ``(hord_tm, nord_v, damp_v)``.

    The SWConfig is BUILT from the imported ``DUO_TAIL_CFG`` (never
    restated here).  ``hord_tm`` / ``nord_w`` / ``damp_w`` are not
    SWConfig fields, so they stay explicit keywords whose ``None``
    default reproduces the spec's own ``c.get("nord_w", c["nord_v"])``
    and ``c.get("damp_w", c["damp_v"])`` fallbacks exactly (sibling
    deviation D3's rule); hord_tm rides the deck dict because
    update_dz_d's transport order is flagstruct%hord_tm
    (dyn_core.F90:1406).
    """
    if cfg is None:
        c = SWConfig.from_mapping(DUO_TAIL_CFG)
    elif isinstance(cfg, SWConfig):
        c = cfg
    else:
        raise TypeError(
            f"{fname}: cfg must be an SWConfig or None (a dict is not "
            f"jit-static); got {type(cfg).__name__}")
    h = hord_tm if hord_tm is not None else getattr(c, "hord_tm", None)
    if h is None:
        h = DUO_TAIL_CFG["hord_tm"]
    # Integral by construction (require_nord's rule); the order dispatch
    # itself is update_dz_d's, which raises statically at trace time on
    # an order it does not implement.
    if isinstance(h, bool) or not isinstance(h, int):
        raise TypeError(
            f"{fname}: hord_tm is a transport-order flag and must be an "
            f"int; got {h!r}")
    if h < 0:
        raise ValueError(f"{fname}: unknown transport order hord_tm={h}")
    if nord_w is not None:
        nord_v = require_nord(fname, "nord_w", nord_w)
    else:
        nord_v = require_nord(fname, "nord_v",
                              getattr(c, "nord_w", c.nord_v))
    damp_v = float(damp_w) if damp_w is not None \
        else float(getattr(c, "damp_w", c.damp_v))
    return int(h), nord_v, damp_v


def dgrid_nh_pressure_phase_3d(ctx, csw_press, dsw_outs, tail_outs, nh, km,
                               *, dt, ptop, akap, cp_air, p_fac, a_imp,
                               dp0, delz, remap_step=False,
                               use_logp=False, square_domain=True,
                               cfg=None, hord_tm=None, nord_w=None,
                               damp_w=None, remap_follows=False,
                               return_stages: bool = False,
                               batched: bool = False):
    """The NH D-grid tail (``dyn_core.F90:1403-1543``): ``update_dz_d`` ->
    ``Riem_Solver3`` -> ``pe_halo``/``pk3_halo`` (``pln_halo`` under
    ``use_logp``) -> zh/pkc duo exchanges -> ``gz = zh*grav`` ->
    (``square_domain``: second pkc exchange) -> ``nh_p_grad``.  Consumes
    the post-d_sw2 ``delp``/``pt`` (whose halos the caller has already
    refreshed -- the :1336-1337 exchange precedes this block) and the
    d_sw5-final ``w`` in ``tail_outs``.

    Face-stacked inputs (convention C1).  ``csw_press``: needs ``pkc``.
    ``dsw_outs``: the transport phase's ONE stacked dict (keys
    DSW1_OUT_2D) read at ``[..., k]`` -- there is no "levels" key on
    this lane; this phase reads ``delp``/``pt`` and
    ``crx_adv``/``cry_adv``/``xfx_adv``/``yfx_adv``.  ``tail_outs``:
    part A's dict, read for ``w``/``u``/``v``.  ``delz``: stacked twin
    of the spec's ``delz6`` list, gated against ``nh["zh"]`` with the
    interface axis shortened.  ``nh``: zh/gz/pk3/pe/pk/peln are km+1
    interface fields, zs/ws are 2-D -- these shapes are NOT trusted
    from this prose: ``validate_stacked`` gates them through
    ``field_shape`` (CSW_OUT_LIKE-aware), the lane's single shape
    authority, cross-checked against the kernel contracts
    (update_dz_d: ``zh (isd:ied, jsd:jed, km+1)``, ``ws`` 2-D;
    pe_halo: the oracle allocates ``pe`` with k as its middle axis, and
    this phase never k-indexes pe, mirroring the spec).

    FUNCTIONAL (convention C4): everything the spec mutates comes back
    -- ``nh`` (zh, gz, zs, pk3, pe, pk, peln, ws), ``delz``, ``pkc``
    (the csw_press storage Riem_Solver3 overwrites with the D-stage
    PERTURBATION and nh_p_grad then B-grid-scratches -- trap #6), the
    D winds ``u``/``v`` and the Riemann-updated ``w``.  ``press``
    aliases the returned ``nh["pe"|"pk"|"peln"|"ws"]`` exactly as the
    spec's press list aliases the nh carry.  The CALLER threads them.

    Stage payloads (convention C6): this phase ADDS NO observation
    returns.  Its spec has no stage_hook sites; the only values lost to
    overwriting are the zh/pkc halos the exchanges themselves consume,
    which the module's stage policy (part A) deliberately does not copy.

    STATIC vs DYNAMIC (convention C3): ``ctx``, ``km``, ``remap_step``,
    ``use_logp``, ``square_domain``, ``cfg``/``hord_tm``/``nord_w``/
    ``damp_w`` and ``remap_follows`` are static.  ``dt``, ``ptop``,
    ``akap``, ``cp_air``, ``p_fac`` and ``a_imp`` stay DYNAMIC: the only
    Python branches any callee takes are on ``use_logp``/``last_call``
    (riem_solver3) and ``use_logp`` (nh_p_grad), which are static
    keywords here, and pk3_halo/pln_halo/pe_halo use ``ptop``/``akap``
    arithmetically only -- so a new time step does not recompile.
    """
    fname = "dgrid_nh_pressure_phase_3d"
    km = require_km(fname, km)
    for nm, v in (("remap_step", remap_step), ("use_logp", use_logp),
                  ("square_domain", square_domain),
                  ("remap_follows", remap_follows),
                  ("batched", batched)):
        require_bool(fname, nm, v)
    require_no_remap_needed(km, remap_follows=remap_follows)
    hord_v, nord_v, damp_vt = _nh_tail_cfg(fname, cfg, hord_tm, nord_w,
                                           damp_w)
    require_f64_jax(fname, {
        "csw_press.pkc": csw_press["pkc"],
        "dsw_outs.delp": dsw_outs["delp"],
        "dsw_outs.pt": dsw_outs["pt"],
        "dsw_outs.crx_adv": dsw_outs["crx_adv"],
        "dsw_outs.cry_adv": dsw_outs["cry_adv"],
        "dsw_outs.xfx_adv": dsw_outs["xfx_adv"],
        "dsw_outs.yfx_adv": dsw_outs["yfx_adv"],
        "tail_outs.u": tail_outs["u"],
        "tail_outs.v": tail_outs["v"],
        "tail_outs.w": tail_outs["w"],
        "nh.zh": nh["zh"], "nh.gz": nh["gz"], "nh.zs": nh["zs"],
        "nh.pk3": nh["pk3"], "nh.pe": nh["pe"], "nh.pk": nh["pk"],
        "nh.peln": nh["peln"], "nh.ws": nh["ws"],
        "delz": delz, "dp0": dp0,
    })
    validate_stacked(fname, csw_press, ctx, km, ("pkc",),
                     what="csw_press (C-stage pressure)")
    # Declared fields through the shape table; the d_sw1 TRANSPORT
    # stacks structurally -- the same split part A already makes, and
    # for the same reason: `field_shape` has no entry for `crx_adv` and
    # should not, because that stagger is the kernel's own (job 9417540
    # raised "unknown field 'crx_adv'" here, one gate after the same
    # thing was fixed on the hydrostatic path).
    validate_stacked(fname, dsw_outs, ctx, km, ("delp", "pt"),
                     what="dsw_outs (transport phase)")
    for _nm in ("crx_adv", "cry_adv", "xfx_adv", "yfx_adv"):
        if _nm not in dsw_outs:
            raise KeyError(
                f"{fname}: dsw_outs is missing {_nm!r}; keys are "
                f"{sorted(dsw_outs)}")
        _a = jnp.asarray(dsw_outs[_nm])
        if (_a.ndim != 4 or _a.shape[0] != batch_size(ctx)
                or _a.shape[3] != km):
            raise ValueError(
                f"{fname}: dsw_outs[{_nm!r}] has shape {_a.shape}; "
                f"expected (6, i, j, {km}) -- the face and level axes "
                f"are fixed by convention C1 even where the stagger is "
                f"the kernel's own")
    validate_stacked(fname, tail_outs, ctx, km, ("u", "v", "w"),
                     what="tail_outs (part A d_sw3..d_sw6 chain)")
    # THE NH CARRY IS NOT A field_shape CONTAINER, and one of its
    # members is a TRAP: `field_shape("pk")` is the padded hydrostatic
    # geopk layout, while dyn_core's NH `pk` is COMPUTE-WINDOW
    # (is:ie, js:je, npz+1) -- the spec says so where it allocates the
    # carry, and Riem_Solver3 writes `pk[:, jc, k]` with `ni` rows.
    # Validating the carry through the shape table would therefore have
    # accepted the wrong array under the right name, so the shapes come
    # from `build_nh_carry`'s own allocation instead (jobs 9417540 /
    # 9417598 raised "unknown field 'crx_adv'" and "unknown field 'zh'"
    # on the way to this).
    _m_a, _npx_c = ctx.n + 2 * ctx.ng, ctx.n
    _nb = batch_size(ctx)
    _nh_shapes = {
        "zh": (_nb, _m_a, _m_a, km + 1),
        "gz": (_nb, _m_a, _m_a, km + 1),
        "pk3": (_nb, _m_a, _m_a, km + 1),
        "zs": (_nb, _m_a, _m_a),
        "ws": (_nb, _npx_c, _npx_c),
        "pk": (_nb, _npx_c, _npx_c, km + 1),
        "pe": (_nb,) + field_shape("pe", ctx.n, ctx.ng, km),
        "peln": (_nb,) + field_shape("peln", ctx.n, ctx.ng, km),
    }
    _missing = [k for k in _nh_shapes if k not in nh]
    if _missing:
        raise KeyError(
            f"{fname}: the nh carry is missing {_missing}; keys are "
            f"{sorted(nh)} (build it with build_nh_carry)")
    for _nm, _want in _nh_shapes.items():
        _got = jnp.asarray(nh[_nm]).shape
        if _got != _want:
            raise ValueError(
                f"{fname}: nh[{_nm!r}] has shape {_got}, expected "
                f"{_want} -- these come from build_nh_carry's own "
                f"allocation, NOT from field_shape (dyn_core's NH pk is "
                f"compute-window where the hydrostatic pk is padded)")
    # `delz` is COMPUTE-ONLY, `(6, n, n, km)`.  fv_arrays allocates it
    # `(is:ie, js:je, npz)` and the NumPy lane's shape table says so at
    # the entry, with the reason: padding it would invite a non-oracle
    # halo read.  So it is NOT `nh['zh']`'s padded footprint with the
    # interface axis shortened, which is what this check demanded --
    # measured (6, 12, 12, 3) against a required (6, 18, 18, 3) on the
    # first NH parity run (job 9417599).
    _want_delz = (batch_size(ctx),) + field_shape("delz", ctx.n, ctx.ng,
                                                  km)
    if delz.shape != _want_delz:
        raise ValueError(
            f"{fname}: delz {delz.shape} must be {_want_delz} -- "
            f"fv_arrays allocates delz COMPUTE-ONLY (is:ie, js:je, npz), "
            f"NOT on nh['zh']'s padded footprint {nh['zh'].shape}")
    if dp0.shape != (km,):
        raise ValueError(f"{fname}: dp0 must be ({km},), got {dp0.shape}")

    if batched:
        return _dgrid_nh_pressure_phase_3d_batched(
            ctx, csw_press, dsw_outs, tail_outs, nh, km, dt=dt,
            ptop=ptop, akap=akap, cp_air=cp_air, p_fac=p_fac,
            a_imp=a_imp, dp0=dp0, delz=delz, remap_step=remap_step,
            use_logp=use_logp, square_domain=square_domain,
            hord_v=hord_v, nord_v=nord_v, damp_vt=damp_vt,
            return_stages=return_stages)

    # Pure Python constant; imported at the call site exactly as the spec
    # does, because the JAX callee list carries no grav of its own.
    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV

    bd = ctx.bd
    n, ng = ctx.n, ctx.ng
    npx = ctx.npx
    # update_dz_d / riem_solver3 take the STATIC (is_, ie, js, je, ng)
    # bounds tuple, not the bd object.
    bounds = (bd.is_, bd.ie, bd.js, bd.je, int(ng))
    # The gridstruct's own area/rarea: the spec's nh_exchanged_area6
    # (which exchanges nothing) plus its ctx["nh_rarea6"], computed here
    # instead of cached (see that twin).
    area6 = nh_exchanged_area6(ctx)
    rarea6 = 1.0 / area6

    rdt = 1.0 / dt
    delp6 = dsw_outs["delp"]
    pt6 = dsw_outs["pt"]
    gs6 = ctx.gs6
    faces = []
    stage_zh: list = []
    for t in range(batch_size(ctx)):
        # The DSW1 advective fluxes are ALREADY (i, j, km) with km at
        # axis 2 on this lane -- the spec's np.stack(..., axis=2) is the
        # transport phase's own stacked layout, so this is a plain slice.
        crx_t = dsw_outs["crx_adv"][t]
        cry_t = dsw_outs["cry_adv"][t]
        xfx_t = dsw_outs["xfx_adv"][t]
        yfx_t = dsw_outs["yfx_adv"][t]
        # nord_v/damp_vt are level-invariant on this deck (n_sponge=-1);
        # slot km is free for update_dz_d's :231-232 mutation.  The twin
        # takes km+1 PYTHON-number sequences (its static-by-necessity
        # contract), not the spec's np.full arrays.
        ndif = (float(nord_v),) * (km + 1)
        damp = (damp_vt,) * (km + 1)
        zs_t = nh["zs"][t]
        zh_t = nh["zh"][t]
        ws_t = nh["ws"][t]
        w_t = tail_outs["w"][t]
        delz_t = delz[t]
        pkc_t = csw_press["pkc"][t]
        pe_t = nh["pe"][t]
        pk3_t = nh["pk3"][t]
        pk_t = nh["pk"][t]
        peln_t = nh["peln"][t]
        gs_t = gs6[t]
        # The spec's gs_nh (gridstruct with area/rarea swapped for the
        # exchanged planes) unpacks to the twin's explicit dummies:
        # area/rarea from the exchange, dxa/dya/del6_* from ctx.gs6[t],
        # AND THE GRID FLAGS.
        #
        # ⛔ THE FLAGS ARE NOT OPTIONAL, and the comment that used to
        # stand here -- "no duogrid/corner selector is passed; the
        # twin's defaults reproduce the oracle's copy_corners path" --
        # was wrong.  The NumPy kernel hands `gridstruct` straight to
        # `fv_tp_2d` and `del6_vt_flux`, which read `bounded_domain`,
        # `grid_type` and the four corner flags out of it; the JAX twin
        # takes them as keywords defaulting to bounded_domain=False with
        # all four corners TRUE.  On this lane `oracle_conventions=True`
        # makes bounded_domain TRUE and every corner flag FALSE, so the
        # defaults ran corner fills the oracle skips.  MEASURED (job
        # 9419255): the two twins differ by 1.155e-04 on zh, 80 of 1296
        # cells, on identical operands -- which is what sent the NH
        # parity gate red at 2.465e-02 once it was composed.
        _fl = ctx.flags6[t]
        zh_t, ws_t = update_dz_d(
            ndif, damp, hord_v, bounds, km, npx, npx,
            area6[t], rarea6[t], dp0, zs_t, zh_t,
            crx_t, cry_t, xfx_t, yfx_t, ws_t, rdt,
            gs_t["dxa"], gs_t["dya"], gs_t["del6_u"], gs_t["del6_v"],
            lim_fac=1.0,
            bounded_domain=_fl.bounded_domain, grid_type=_fl.grid_type,
            sw_corner=_fl.sw_corner, se_corner=_fl.se_corner,
            nw_corner=_fl.nw_corner, ne_corner=_fl.ne_corner)
        zh_after_dz = zh_t if return_stages else None
        # trap #6: pkc -- FULL pressure out of the C stage -- is the
        # SAME storage Riem_Solver3 overwrites with the D-stage
        # PERTURBATION (the ppe slot) and nh_p_grad then B-grid
        # scratches; the returned ppe is threaded forward as pkc.
        (w_t, delz_t, zh_t, pe_t, pkc_t, pk3_t, pk_t,
         peln_t) = riem_solver3(
            0, dt, bounds, km, akap, cp_air, ptop, zs_t, w_t, delz_t,
            pt6[t], delp6[t], zh_t, pe_t, pkc_t, pk3_t, pk_t, peln_t,
            ws_t, p_fac, a_imp, use_logp=use_logp,
            last_call=remap_step, fp_out=False)
        if remap_step:
            pe_t = pe_halo(pe_t, delp6[t], bd, npz=km, ptop=ptop)
        # dyn_core.F90:1444-1448: pln_halo under use_logp, pk3_halo
        # otherwise (an unconditional pk3_halo would overwrite log(p)
        # halos with p**akap on a use_logp deck).
        if use_logp:
            pk3_t = pln_halo(pk3_t, delp6[t], bd, npz=km, ptop=ptop)
        else:
            pk3_t = pk3_halo(pk3_t, delp6[t], bd, npz=km, ptop=ptop,
                             akap=akap)
        if return_stages:
            # C6: stage observation by RETURN. The two kernels that
            # write zh, captured BETWEEN them, so the phase can be
            # bisected against the spec's stage_hook on ONE shared
            # bundle. The alternative -- rebuilding the chain by hand
            # outside -- is a second implementation that can differ for
            # its own reasons, and did.
            stage_zh.append({"after_update_dz_d": zh_after_dz,
                             "after_riem_solver3": zh_t})
        faces.append({"zh": zh_t, "ws": ws_t, "w": w_t, "delz": delz_t,
                      "pkc": pkc_t, "pe": pe_t, "pk3": pk3_t,
                      "pk": pk_t, "peln": peln_t})
    upd = stack_faces(fname, faces)
    zh6 = upd["zh"]
    pkc6 = upd["pkc"]

    # zh + pkc duo exchanges (:1482-1483).  Batched over the km+1
    # interface levels via the allk public (M3 v2a; codex MAJOR -- these
    # two loops were the un-batched stragglers behind the "60 -> 6"
    # claim): the certified (ring_comm=None) path inside the allk fn is
    # the SAME ascending-k per-level loop relocated verbatim, so
    # statement order and per-level independence are unchanged; the ring
    # path collapses 2*(km+1) collectives into 2.
    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_allk
    zh6 = ext_scalar_sixface_allk(zh6, ctx.tab, "A")
    pkc6 = ext_scalar_sixface_allk(pkc6, ctx.tab, "A")

    # gz = zh*grav over the two-cell halo box (:1487-1494).  The spec's
    # face loop vectorises over the leading axis of the IDENTICAL
    # window: every face writes only its own slab and reads only that
    # slab's zh -- no face reads another face's write.
    i0 = bd.is_ - bd.isd
    j0 = bd.js - bd.jsd
    sl_i = slice(i0 - 2, i0 + n + 2)
    sl_j = slice(j0 - 2, j0 + n + 2)
    gz6 = nh["gz"].at[:, sl_i, sl_j, :].set(
        zh6[:, sl_i, sl_j, :] * FV3_GRAV)

    # Second pkc exchange, square_domain only (:1496-1502): between the
    # first exchange and it only gz changes, so it is idempotent -- kept
    # for statement-order fidelity, exactly as the spec gates it.
    if square_domain:
        pkc6 = ext_scalar_sixface_allk(pkc6, ctx.tab, "A")

    # nh_p_grad (:1534-1543): beta = 0 on the deck.  It takes the
    # ORIGINAL ctx.gs6[t] (rdx/rdy metrics), not the exchanged-area
    # gridstruct, and returns (u, v, pp, pk3, gz) -- the B-grid corner
    # scratch of pp(=pkc)/pk3/gz rides the returned functional carry.
    pg_faces = []
    for t in range(batch_size(ctx)):
        u_t, v_t, pkc_t, pk3_t, gz_t = nh_p_grad(
            tail_outs["u"][t], tail_outs["v"][t], pkc6[t], gz6[t],
            delp6[t], upd["pk3"][t], gs6[t], bd,
            npx=npx, npy=npx, npz=km, dt=dt, ptop=ptop, akap=akap,
            use_logp=use_logp, ng=ng, duogrid=True)
        pg_faces.append({"u": u_t, "v": v_t, "pkc": pkc_t,
                         "pk3": pk3_t, "gz": gz_t})
    pg = stack_faces(fname, pg_faces)

    nh_out = {"zh": zh6, "gz": pg["gz"], "zs": nh["zs"],
              "pk3": pg["pk3"], "pe": upd["pe"], "pk": upd["pk"],
              "peln": upd["peln"], "ws": upd["ws"]}
    return {
        "nh": nh_out,
        "delz": upd["delz"],
        "w": upd["w"],
        "pkc": pg["pkc"],
        "u": pg["u"],
        "v": pg["v"],
        # The spec's press list aliases the nh arrays post-mutation; the
        # aliasing is preserved here (same values, zero copies).
        "press": {"pe": nh_out["pe"], "pk": nh_out["pk"],
                  "peln": nh_out["peln"], "ws": nh_out["ws"]},
        **({"stages": {"zh": stage_zh}} if return_stages else {}),
    }


def _dgrid_nh_pressure_phase_3d_batched(ctx, csw_press, dsw_outs,
                                        tail_outs, nh, km, *, dt, ptop,
                                        akap, cp_air, p_fac, a_imp, dp0,
                                        delz, remap_step, use_logp,
                                        square_domain, hord_v, nord_v,
                                        damp_vt,
                                        return_stages) -> dict:
    """The vmap-over-faces arm of :func:`dgrid_nh_pressure_phase_3d`.

    Entry gates already ran in the caller (``hord_v``/``nord_v``/
    ``damp_vt`` arrive resolved from ``_nh_tail_cfg``).  The two FACE
    loops -- ``update_dz_d`` -> ``riem_solver3`` -> the pe/pk3 halos,
    and ``nh_p_grad`` -- become one ``jax.vmap`` each; everything
    BETWEEN them (the zh/pkc allk duo exchanges, the ``gz = zh*grav``
    box, the ``square_domain`` second pkc exchange) is the loop path's
    own six-face-stack code, repeated verbatim -- those are cross-face
    collectives and are not face-batched.

    No da_min threading here: the NH chain's kernels take no GridFlags
    (``fv3_nh_core.py:1390`` records ``da_min`` UNREAD on this path);
    the domain/corner selectors ``update_dz_d`` does take are
    face-invariant by ``build_batched_gs``'s common-mode gate and close
    over as the shared Python values, exactly the loop path's per-face
    reads of an all-equal ``flags6``.

    in_axes: the four advective flux planes, the nh-carry members, the
    ``delz``/``pkc``/``pt``/``delp``/``w`` planes, the exchanged
    ``area``/``rarea`` and the gridstruct dict are 0 (per face).
    Closed over: ``bounds``/``km``/``npx``/the static branch selectors,
    plus ``dt``/``ptop``/``akap``/``cp_air``/``p_fac``/``a_imp``/
    ``dp0``/``rdt`` (traced but shared -- one column of reference
    thicknesses and one time step for all six faces, exactly as on the
    loop path).

    ``return_stages`` rides through: the two zh stages come back as
    extra vmapped outputs and are unstacked into the loop path's
    list-of-six-dicts layout.
    """
    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface_allk
    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV

    bview = build_batched_gs(ctx)
    flags = bview["flags"]
    bd = ctx.bd
    n, ng = ctx.n, ctx.ng
    npx = ctx.npx
    bounds = (bd.is_, bd.ie, bd.js, bd.je, int(ng))
    area6 = nh_exchanged_area6(ctx)
    rarea6 = 1.0 / area6
    rdt = 1.0 / dt
    ndif = (float(nord_v),) * (km + 1)
    damp = (damp_vt,) * (km + 1)

    # --- update_dz_d -> riem_solver3 -> halos, faces vmapped ----------
    def one_face(crx_t, cry_t, xfx_t, yfx_t, zs_t, zh_t, ws_t, w_t,
                 delz_t, pkc_t, pe_t, pk3_t, pk_t, peln_t, pt_t,
                 delp_t, area_t, rarea_t, gs_t):
        zh1, ws1 = update_dz_d(
            ndif, damp, hord_v, bounds, km, npx, npx,
            area_t, rarea_t, dp0, zs_t, zh_t,
            crx_t, cry_t, xfx_t, yfx_t, ws_t, rdt,
            gs_t["dxa"], gs_t["dya"], gs_t["del6_u"], gs_t["del6_v"],
            lim_fac=1.0,
            bounded_domain=flags["bounded_domain"],
            grid_type=flags["grid_type"],
            sw_corner=flags["sw_corner"], se_corner=flags["se_corner"],
            nw_corner=flags["nw_corner"], ne_corner=flags["ne_corner"])
        zh_after_dz = zh1 if return_stages else None
        (w1, delz1, zh2, pe1, pkc1, pk3_1, pk1,
         peln1) = riem_solver3(
            0, dt, bounds, km, akap, cp_air, ptop, zs_t, w_t, delz_t,
            pt_t, delp_t, zh1, pe_t, pkc_t, pk3_t, pk_t, peln_t,
            ws1, p_fac, a_imp, use_logp=use_logp,
            last_call=remap_step, fp_out=False)
        if remap_step:
            pe1 = pe_halo(pe1, delp_t, bd, npz=km, ptop=ptop)
        if use_logp:
            pk3_1 = pln_halo(pk3_1, delp_t, bd, npz=km, ptop=ptop)
        else:
            pk3_1 = pk3_halo(pk3_1, delp_t, bd, npz=km, ptop=ptop,
                             akap=akap)
        out = {"zh": zh2, "ws": ws1, "w": w1, "delz": delz1,
               "pkc": pkc1, "pe": pe1, "pk3": pk3_1,
               "pk": pk1, "peln": peln1}
        if return_stages:
            out["_stage_zh_after_update_dz_d"] = zh_after_dz
            out["_stage_zh_after_riem_solver3"] = zh2
        return out

    upd = jax.vmap(one_face, in_axes=(0,) * 19)(
        dsw_outs["crx_adv"], dsw_outs["cry_adv"],
        dsw_outs["xfx_adv"], dsw_outs["yfx_adv"],
        nh["zs"], nh["zh"], nh["ws"], tail_outs["w"], delz,
        csw_press["pkc"], nh["pe"], nh["pk3"], nh["pk"], nh["peln"],
        dsw_outs["pt"], dsw_outs["delp"], area6, rarea6, bview["gs"])

    stage_zh: list = []
    if return_stages:
        a = upd.pop("_stage_zh_after_update_dz_d")
        b = upd.pop("_stage_zh_after_riem_solver3")
        stage_zh = [{"after_update_dz_d": a[t],
                     "after_riem_solver3": b[t]}
                    for t in range(batch_size(ctx))]

    # --- zh/pkc duo exchanges, gz box, second pkc exchange: verbatim --
    # (:1482-1483, :1487-1494, :1496-1502 -- the loop path's own
    # six-face-stack code; cross-face collectives, not batched.)
    zh6 = ext_scalar_sixface_allk(upd["zh"], ctx.tab, "A")
    pkc6 = ext_scalar_sixface_allk(upd["pkc"], ctx.tab, "A")

    i0 = bd.is_ - bd.isd
    j0 = bd.js - bd.jsd
    sl_i = slice(i0 - 2, i0 + n + 2)
    sl_j = slice(j0 - 2, j0 + n + 2)
    gz6 = nh["gz"].at[:, sl_i, sl_j, :].set(
        zh6[:, sl_i, sl_j, :] * FV3_GRAV)

    if square_domain:
        pkc6 = ext_scalar_sixface_allk(pkc6, ctx.tab, "A")

    # --- nh_p_grad (:1534-1543), faces vmapped ------------------------
    def one_face_pg(u_t, v_t, pkc_t, gz_t, delp_t, pk3_t, gs_t):
        return nh_p_grad(u_t, v_t, pkc_t, gz_t, delp_t, pk3_t, gs_t,
                         bd, npx=npx, npy=npx, npz=km, dt=dt,
                         ptop=ptop, akap=akap, use_logp=use_logp,
                         ng=ng, duogrid=True)

    u6, v6, pg_pkc6, pg_pk3_6, pg_gz6 = jax.vmap(
        one_face_pg, in_axes=(0,) * 7)(
        tail_outs["u"], tail_outs["v"], pkc6, gz6, dsw_outs["delp"],
        upd["pk3"], bview["gs"])

    nh_out = {"zh": zh6, "gz": pg_gz6, "zs": nh["zs"],
              "pk3": pg_pk3_6, "pe": upd["pe"], "pk": upd["pk"],
              "peln": upd["peln"], "ws": upd["ws"]}
    return {
        "nh": nh_out,
        "delz": upd["delz"],
        "w": upd["w"],
        "pkc": pg_pkc6,
        "u": u6,
        "v": v6,
        "press": {"pe": nh_out["pe"], "pk": nh_out["pk"],
                  "peln": nh_out["peln"], "ws": nh_out["ws"]},
        **({"stages": {"zh": stage_zh}} if return_stages else {}),
    }


def make_dgrid_nh_pressure_phase_3d_jit(ctx, km, *, remap_step=False,
                                        use_logp=False,
                                        square_domain=True, cfg=None,
                                        hord_tm=None, nord_w=None,
                                        damp_w=None,
                                        remap_follows=False,
                                        batched=False):
    """jit factory for :func:`dgrid_nh_pressure_phase_3d`.

    Every STATIC selector is baked and validated ONCE here, so a bad
    deck knob raises at build time, before any tracing: ``ctx`` is
    closed over (static metrics/halo tables, per the sibling
    convention), and ``km`` plus the branch keywords are Python-only.
    The returned callable takes the DYNAMIC operands only::

        run(csw_press, dsw_outs, tail_outs, nh, delz, dp0,
            dt, ptop, akap, cp_air, p_fac, a_imp)

    ``dt`` stays a traced argument, so a new time step does not
    recompile the phase (convention C3).  No ``donate_argnums`` (this
    lane is differentiable); every return is a fresh value the caller
    threads forward as the functional NH carry.
    """
    fname = "make_dgrid_nh_pressure_phase_3d_jit"
    km = require_km(fname, km)
    for nm, v in (("remap_step", remap_step), ("use_logp", use_logp),
                  ("square_domain", square_domain),
                  ("remap_follows", remap_follows),
                  ("batched", batched)):
        require_bool(fname, nm, v)
    _nh_tail_cfg(fname, cfg, hord_tm, nord_w, damp_w)
    require_no_remap_needed(km, remap_follows=remap_follows)

    def run(csw_press, dsw_outs, tail_outs, nh, delz, dp0,
            dt, ptop, akap, cp_air, p_fac, a_imp):
        return dgrid_nh_pressure_phase_3d(
            ctx, csw_press, dsw_outs, tail_outs, nh, km,
            dt=dt, ptop=ptop, akap=akap, cp_air=cp_air,
            p_fac=p_fac, a_imp=a_imp, dp0=dp0, delz=delz,
            remap_step=remap_step, use_logp=use_logp,
            square_domain=square_domain, cfg=cfg, hord_tm=hord_tm,
            nord_w=nord_w, damp_w=damp_w, remap_follows=remap_follows,
            batched=batched)

    return jax.jit(run)
