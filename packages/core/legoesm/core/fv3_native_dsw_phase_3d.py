"""D-grid transport phase with the two duo barriers, km-general, six faces.

STAGE 1 unit 4 of the staged 3-D port. This is the piece a MONOLITHIC
``d_sw`` structurally cannot express: the oracle splits the D-grid step
into six routines and interleaves two inter-panel averaging barriers
between them, so there has to be a stage boundary for the barrier to sit
in.

ORACLE CADENCE (published, checksum-pinned tree):

    dyn_core.F90:831   call d_sw1                (inside do k=1,npz)
                :872   mpp_get_boundary CGRID_NE  <-- BARRIER 1
                :950   call d_sw2
                :961   call d_sw3
                :984   mpp_get_boundary BGRID_NE  <-- BARRIER 2

``d_sw1`` computes FLUXES ONLY; the delp/pt updates happen in ``d_sw2``
AFTER the averaging. That ordering is the whole point -- averaging the
seam fluxes before they are applied is what makes the panel-edge flux
single-valued, hence conservative and imprint-free across the seam.

THE TWO BARRIERS HAVE DIFFERENT EXTENTS. This is the trap in this unit:

    barrier 1 (CGRID_NE)  fyy at j in {js, je+1} over i = is..ie
                          fxx at i in {is, ie+1} over j = js..je
    barrier 2 (BGRID_NE)  i = is..ie+1, j = js..je+1  (B-grid corners
                          included)

``is..ie`` for one and ``is..ie+1`` for the other, in adjacent routines.
Both are already implemented and certified at km=1 in
``fv3_native_gridstruct`` -- this module does NOT reimplement them, it
applies them per level, which is exactly what the oracle's barrier block
does with its internal ``do k=1,npz``.

SLOT SELECTION (barrier 1). The oracle averages only ``iq==1`` (delp),
``iq==4`` (temp) and ``iq>4`` (tracers). Slots 2 (``w``) and 3
(``q_con``) are left byte-untouched. ``average_allflux_shared_edges``
already encodes that; a test here proves the untouched slots really are
untouched after the 3-D lift.
"""
from __future__ import annotations

import numpy as np

from legoesm.core.fv3_native_state_3d import level_slice, require_no_remap_needed

# The shipped duo decks: nord=2, vtdm4=0.12, hord*=6, d_ext=0, dddmp=0,
# n_sponge=-1 (so the k=1/2/3 sponge branch at dyn_core.F90:1067 is SKIPPED
# and the coefficients are level-invariant -- Lane A's single flat config
# is structurally correct for this deck, it is not a missing feature).
DUO_DECK_CFG = {
    "hord_tr": 6, "hord_vt": 6, "hord_tm": 6, "hord_dp": 6,
    # ``nord`` is the DIVERGENCE-damping order (d_sw5); ``nord_v`` is the
    # VORTICITY-damping order (d_sw1/d_sw6). They coincide at 2 on this
    # deck, which is exactly why gating the divgd exchange on the wrong
    # one of the two was invisible -- dyn_core.F90:652 gates on nord.
    "nord": 2, "nord_v": 2, "damp_v": 0.12,
}


def _exchange_post_pgrad(ctx: dict, csw_outs: list, km: int, *,
                         nord: int) -> None:
    """Apply the shared post-``p_grad_c`` duo exchanges at every level.

    The oracle's exchanges are 3-D calls over the whole column; this lane
    holds one 2-D array per level, so it drives the shared six-face helper
    once per k. Level-independent by construction -- ``ext_scalar`` /
    ``ext_vector`` are horizontal operators.

    The per-level arrays are handed over as UNCONDITIONAL copies and
    written back afterwards. ``csw_outs[t][name][:, :, k]`` is a strided
    view, and the exchange chain (mpp analog -> k2e ring remap -> Lagrange
    corner fill) is only certified against ordinary 2-D arrays.

    ``np.copy`` here, NOT ``np.ascontiguousarray``: the latter is
    copy-IF-NEEDED, and for a C-order ``(i, j, 1)`` field ``[:, :, 0]`` is
    already contiguous, so it would alias the parent at km=1 and copy at
    km>1 -- the same code taking two different aliasing paths depending on
    the level count, silently.
    """
    from legoesm.core.fv3_native_duo_stepper import (
        exchange_post_pgrad_sixface,
    )

    for k in range(km):
        uc6 = [np.array(csw_outs[t]["uc"][:, :, k], copy=True)
               for t in range(6)]
        vc6 = [np.array(csw_outs[t]["vc"][:, :, k], copy=True)
               for t in range(6)]
        dg6 = [np.array(csw_outs[t]["divg_d"][:, :, k], copy=True)
               for t in range(6)]
        exchange_post_pgrad_sixface(ctx, dg6, uc6, vc6, nord=nord)
        for t in range(6):
            csw_outs[t]["uc"][:, :, k] = uc6[t]
            csw_outs[t]["vc"][:, :, k] = vc6[t]
            csw_outs[t]["divg_d"][:, :, k] = dg6[t]


def dsw_transport_phase_3d(ctx: dict, state: list, csw_outs: list,
                           dt: float, km: int, *,
                           cfg: dict | None = None,
                           nq: int = 1,
                           hydrostatic: bool = True,
                           remap_follows: bool = False) -> list:
    """``d_sw1`` (per k) -> BARRIER 1 (per k) -> ``d_sw2`` (per k).

    Returns per-face dicts carrying the averaged allflux stacks and the
    ``d_sw2``-updated ``delp``/``pt`` at every level.

    ``state`` is read for the D winds; ``csw_outs`` supplies ``uc``/``vc``
    as updated in place by ``p_grad_c`` in the C-grid phase.

    ``hydrostatic=False`` threads the NH w path: d_sw1's fv_tp_2d(w)
    into allflux slot 2 (which the barrier deliberately does NOT
    average -- dyn_core.F90:853-900 skips iq==2, NH-spec trap #12),
    then d_sw2's del6 ``dw`` increment and the mass-weighted
    ``w = delp*w + fluxdiv`` on the OLD delp.  The updated ``w`` and
    ``dw`` ride the returned per-face dicts (``w`` stacked, ``dw`` in
    the per-level stage dicts) for d_sw5's finalisation.
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    from legoesm.core.fv3_native_duo_sw_core import d_sw1_duo, d_sw2_duo
    from legoesm.grids.fv3_native_gridstruct import (
        average_allflux_shared_edges,
    )

    c = dict(DUO_DECK_CFG)
    c.update(cfg or {})
    n, ng, bd = ctx["n"], ctx["ng"], ctx["bd"]
    npx = n + 1
    m_a = n + 2 * ng

    # --- POST-p_grad_c duo exchanges, BEFORE d_sw1 ------------------------
    # dyn_core.F90:652  if (duogrid .and. nord > 0) ext_scalar(divgd,...,1,1)
    # dyn_core.F90:655  if (duogrid)                ext_vector(uc,vc,1,0,0,1)
    # (:653 is blank, :654 is the .not.duogrid group-halo completion.
    # :706/:709 are the REGIONAL branch -- regional_boundary_update -- and
    # were the wrong anchor: they are not on the duo lane at all.)
    #
    # Omitting these leaves uc/vc/divgd halos stale, so d_sw1 transports
    # garbage: the first sub-step produced delp ~ -1.5e39 and 267 non-finite
    # u values before any exchange was added.
    #
    # WHICH exchange matters as much as whether. The mpp-analog index-copy
    # helpers fill the four edge STRIPS and leave the CORNER-DIAGONAL halo
    # untouched (fv3_native_gridstruct.exchange_bgrid_scalar_halos says so
    # in its own docstring), whereas upstream ext_scalar is the k2e Lagrange
    # fill that covers those regions. d_sw5's divergence-damping n-loop at
    # nord=2 reads i,j = is-2..ie+3 (sw_core.F90:1748-1758; the i+1/j+1
    # operands push the high side to ie+3/je+3) -- INTO the corner
    # diagonal -- so with
    # the interim helper divg_d(0,0) stayed at the c_sw halo value -9.1e7
    # while the compute window was 1e-8, and `ke += dd8*divg_d` with
    # dd8 = (da_min_c*d4_bg)**(nord+1) = 9.9e31 turned that into
    # ke(1,1) = -4.97e16, hence a 1e11 D wind out of d_sw6.
    # The km=1 lane (fv3_native_duo_stepper.dsw12_step_sixface) already
    # routed through the authoritative path; this is the same dispatch,
    # including the ext_exclude opt-outs, so the two lanes cannot drift.
    _exchange_post_pgrad(ctx, csw_outs, km, nord=int(c["nord"]))

    # --- d_sw1 at every level, on every face -------------------------------
    # Held as [face][k] rather than merged: the barrier consumes one level
    # at a time and the oracle's own barrier block is a do k=1,npz loop.
    per_face_levels: list[list[dict]] = []
    for t in range(6):
        levels = []
        for k in range(km):
            lev = level_slice(state[t], k, km)
            levels.append(d_sw1_duo(
                lev["delp"], lev["pt"], lev["w"],
                csw_outs[t]["uc"][:, :, k], csw_outs[t]["vc"][:, :, k],
                np.zeros((npx, m_a)), np.zeros((m_a, npx)),
                np.zeros((npx, m_a)), np.zeros((m_a, npx)),
                ctx["gs6"][t], bd, npx, npx, dt=dt,
                hord_tr=c["hord_tr"], hord_vt=c["hord_vt"],
                hord_tm=c["hord_tm"], hord_dp=c["hord_dp"],
                nord_v=c["nord_v"], nord_t=0,
                damp_v=c["damp_v"], damp_t=0.0,
                hydrostatic=hydrostatic,
                workspace_sentinel=0.0))
        per_face_levels.append(levels)

    # --- BARRIER 1: inter-panel flux average, one level at a time ----------
    # dyn_core.F90:872. The certified km=1 routine is applied per level,
    # which is what the Fortran's internal do k=1,npz does. Averaging must
    # happen with ALL SIX FACES present for that level -- doing it inside
    # the per-face loop above would average a face against a stale
    # neighbour.
    for k in range(km):
        afx6 = [per_face_levels[t][k]["allflux_x"] for t in range(6)]
        afy6 = [per_face_levels[t][k]["allflux_y"] for t in range(6)]
        average_allflux_shared_edges(afx6, afy6, nq, n, ng)

    # --- d_sw2 at every level, on the AVERAGED fluxes ----------------------
    outs = []
    for t in range(6):
        acc = {"delp": np.zeros((m_a, m_a, km), dtype=np.float64),
               "pt": np.zeros((m_a, m_a, km), dtype=np.float64)}
        if not hydrostatic:
            acc["w"] = np.zeros((m_a, m_a, km), dtype=np.float64)
        for k in range(km):
            s1 = per_face_levels[t][k]
            s2 = d_sw2_duo(s1["delp"], s1["pt"],
                           s1["allflux_x"], s1["allflux_y"],
                           ctx["gs6"][t], bd,
                           w=(None if hydrostatic else s1["w"]),
                           npx=npx, npy=npx, dt=dt,
                           kgb=float(c.get("kgb", 0.0)),
                           nord_w=int(c.get("nord_w", c["nord_v"])),
                           damp_w=float(c.get("damp_w", c["damp_v"])) if
                           not hydrostatic else 0.0,
                           hydrostatic=hydrostatic)
            for name in ("delp", "pt"):
                if name not in s2:
                    raise KeyError(
                        f"d_sw2 returned no {name!r}; keys {sorted(s2)}")
                acc[name][:, :, k] = s2[name]
            if not hydrostatic:
                acc["w"][:, :, k] = s2["w"]
                # d_sw5 needs the del6 increment of THIS level.
                s1["dw"] = s2["dw"]
        acc["allflux_x"] = np.stack(
            [per_face_levels[t][k]["allflux_x"] for k in range(km)], axis=2)
        acc["allflux_y"] = np.stack(
            [per_face_levels[t][k]["allflux_y"] for k in range(km)], axis=2)
        # d_sw3..d_sw6 consume many more d_sw1 outputs (ut, vt, divg_d,
        # crx_adv, cry_adv, xfx_adv, yfx_adv, ra_x, ra_y, uc, vc). Carry the
        # per-level stage dicts forward rather than stacking each field:
        # the downstream stages are themselves per-level, so a stack would
        # only have to be re-sliced.
        acc["levels"] = per_face_levels[t]
        outs.append(acc)
    return outs
