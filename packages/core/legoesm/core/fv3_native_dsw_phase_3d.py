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
# n_sponge=-1 (so the k=1/2/3 sponge branch at dyn_core.F90:833 is SKIPPED
# and the coefficients are level-invariant -- Lane A's single flat config
# is structurally correct for this deck, it is not a missing feature).
DUO_DECK_CFG = {
    "hord_tr": 6, "hord_vt": 6, "hord_tm": 6, "hord_dp": 6,
    "nord_v": 2, "damp_v": 0.12,
}


def dsw_transport_phase_3d(ctx: dict, state: list, csw_outs: list,
                           dt: float, km: int, *,
                           cfg: dict | None = None,
                           nq: int = 1) -> list:
    """``d_sw1`` (per k) -> BARRIER 1 (per k) -> ``d_sw2`` (per k).

    Returns per-face dicts carrying the averaged allflux stacks and the
    ``d_sw2``-updated ``delp``/``pt`` at every level.

    ``state`` is read for the D winds; ``csw_outs`` supplies ``uc``/``vc``
    as updated in place by ``p_grad_c`` in the C-grid phase.
    """
    require_no_remap_needed(km)
    from legoesm.core.fv3_native_duo_sw_core import d_sw1_duo, d_sw2_duo
    from legoesm.grids.fv3_native_gridstruct import (
        average_allflux_shared_edges,
    )

    from legoesm.grids.fv3_native_gridstruct import (
        exchange_bgrid_scalar_halos, exchange_cgrid_vector_halos,
    )

    c = dict(DUO_DECK_CFG)
    c.update(cfg or {})
    n, ng, bd = ctx["n"], ctx["ng"], ctx["bd"]
    npx = n + 1
    m_a = n + 2 * ng

    # --- POST-p_grad_c duo exchanges, BEFORE d_sw1 ------------------------
    # dyn_core.F90:706  if (duogrid .and. nord > 0) ext_scalar(divgd,...,1,1)
    # dyn_core.F90:709  if (duogrid)                ext_vector(uc,vc,1,0,0,1)
    # Omitting these leaves uc/vc/divgd halos stale, so d_sw1 transports
    # garbage: the first sub-step produced delp ~ -1.5e39 and 267 non-finite
    # u values before this was added. The km=1 lane does the same exchanges
    # at fv3_native_duo_stepper.dsw12_step_sixface.
    for k in range(km):
        uc6 = [csw_outs[t]["uc"][:, :, k] for t in range(6)]
        vc6 = [csw_outs[t]["vc"][:, :, k] for t in range(6)]
        dg6 = [csw_outs[t]["divg_d"][:, :, k] for t in range(6)]
        for t in range(1, 7):
            if c["nord_v"] > 0:
                exchange_bgrid_scalar_halos(dg6, t, n, ng)
            exchange_cgrid_vector_halos(uc6, vc6, t, n, ng)

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
        for k in range(km):
            s1 = per_face_levels[t][k]
            s2 = d_sw2_duo(s1["delp"], s1["pt"],
                           s1["allflux_x"], s1["allflux_y"],
                           ctx["gs6"][t], bd)
            for name in ("delp", "pt"):
                if name not in s2:
                    raise KeyError(
                        f"d_sw2 returned no {name!r}; keys {sorted(s2)}")
                acc[name][:, :, k] = s2[name]
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
