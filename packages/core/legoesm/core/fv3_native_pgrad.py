"""FV3-native multilevel hydrostatic pressure chain (km-general port).

Index-exact python transcription of the three ``model/dyn_core.F90``
routines that make up the acoustic-loop hydrostatic pressure chain of
the authoritative Zenodo 8327578 *symmetryclean* tree:

- :func:`geopk`      — ``dyn_core.F90:2660-2790``
- :func:`p_grad_c`   — ``dyn_core.F90:2073-2132``
- :func:`one_grad_p` — ``dyn_core.F90:2347-2480``

This is the ONE implementation.  The four km=1 SW entry points in
``fv3_native_duo_stepper`` (``geopk_sw_1lev``, ``p_grad_c_1lev``,
``geopk_sw_1lev_d``, ``one_grad_p_1lev``) are thin adapters over these
functions — they fold the ``-DSW_DYNAMICS`` convention (akap=1, ptop=0,
no ``cp_air``, no ``peln``) and the 2-D<->3-D staging, and carry NO
numerics of their own.

Vertical convention
-------------------
``k=1`` (python index 0) is the model TOP, ``k=km+1`` the SURFACE.
Pressure integrates DOWNWARD from ``ptop`` (a running accumulator, top
to bottom — the sum ORDER is part of the bit-exact contract);
``gz`` integrates UPWARD from ``hs``.

Array conventions (Fortran storage, origins named per axis)
-----------------------------------------------------------
- ``delp``/``pt``/``q_con`` : ``(m_a, m_a, km)`` origin ``(isd, jsd, 1)``
- ``hs``                    : ``(m_a, m_a)``    origin ``(isd, jsd)``
- ``pk``/``gz``             : ``(m_a, m_a, km+1)`` origin ``(isd, jsd, 1)``
- ``pe``                    : ``(n+2, km+1, n+2)`` origin ``(is-1, 1, js-1)``
- ``peln``                  : ``(n, km+1, n)``     origin ``(is, 1, js)``
- ``pkz``                   : ``(n, n, km)``       origin ``(is, js, 1)``
- ``uc``/``v``              : ``(m_b, m_a, km)``   origin ``(isd, jsd, 1)``
- ``vc``/``u``              : ``(m_a, m_b, km)``   origin ``(isd, jsd, 1)``
- ``divg2``                 : ``(n+1, n+1)``       origin ``(is, js)``

with ``m_a = n + 2*ng``, ``m_b = m_a + 1``.

``pe``/``peln`` keep the upstream ``(i, k, j)`` axis order.  That order
is LOAD-BEARING for the bit-exact oracle compare (UNCERTAIN U7): any
later restaging to ``(i, j, k)`` must be a separate, tested change with
its own equivalence gate.  ``packages/atmosphere/.../_fv3_lin_pgf.py``
is a REFERENCE for the km>1 formulas, NOT a drop-in (different grid
class, ``(face, i, j, lev)`` layout, tendency sign convention, JAX).

Certification
-------------
``tests/grids/test_fv3_native_geopk_pgrad.py`` compares every stage
token bit-for-bit (uint64 words, zero tolerance) against the verbatim
Fortran chain at C12/km=2 and C12/km=3 on ONE face
(``scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch``).
``tests/grids/test_fv3_native_pgrad.py`` is this module's direct unit
test (structure, dispatch hardening, SW-adapter equivalence) and does
not need the fixtures.

SCOPE / EXCLUSIONS carried from the brick spec: ONE face, translation
certificate only — no six-face exchange or averaging cadence, no
``-DUSE_COND`` condensate loading, no ``-DSW_DYNAMICS`` build of the
oracle lane, hydrostatic only, ``beta <= 0``, ``a2b_ord = 4``.

TRANSLATION vs FIDELITY (UNCERTAIN U1).  ``ptop``, ``akap`` and
``cp_air`` are ARGUMENTS here, never repo constants: FMS
``constants_mod`` is absent from the Zenodo tree, ``akap`` is an INPUT to
this chain, and substituting ``legoesm.constants.kappa``
(``R_d/c_pd`` != 2/7) would silently change what the oracle certifies.
Codex r20 CONFIRMED that choice as correct for a routine-translation
oracle, and listed what a later FIDELITY claim (as opposed to a
translation claim) must ADDITIONALLY pin, recorded here verbatim:

    - exact Zenodo archive/file hashes and preprocess defines;
    - FMS ``constants_mod`` source/object hash and resolved ``cp_air``,
      ``R_d``, and real kind;
    - runtime ``ptop``, ``akap``, and thermodynamic configuration;
    - compiler, version, flags, libm/platform, and linked extract
      dependencies;
    - committed fixtures plus the run manifest above.

Until every one of those is pinned, this module and its fixtures support
a TRANSLATION claim only.
"""

from __future__ import annotations

import numpy as np
from legoesm.core.fv3_native_sw_core import BIG_NUMBER
from legoesm.grids.fv3_native_gridstruct import fort

__all__ = ["geopk", "p_grad_c", "one_grad_p", "a2b_gridstruct_view"]


def _w(lo: int, ia: int, ib: int) -> slice:
    """Slice for the Fortran window ``ia..ib`` on an axis whose python
    index 0 holds Fortran index ``lo``."""
    if ib < ia:
        return slice(0, 0)
    return slice(ia - lo, ib - lo + 1)


def a2b_gridstruct_view(gs: dict, bd) -> dict:
    """The Fortran-indexed gridstruct dict ``a2b_ord4`` consumes.

    Mirrors the ``one_grad_p_1lev`` staging exactly (fort views over the
    data-domain metrics + the raw 1-D edge factors + the corner/domain
    flags).  Factored here so the km=1 adapter and the km-general
    :func:`one_grad_p` share ONE construction.

    NOTE (lane fact, verified against ``a2b_edge.F90`` gates 98/185/241
    and the python port's duo branches): on the DUO branch a2b_ord4
    takes the interior-everywhere arms and reads NONE of
    ``dxa``/``dya``/``grid``/``agrid``/``edge_*``.  They are staged
    because the Fortran unconditionally associates the pointers (and
    the plain lane does read them) — not because this lane consumes
    them.
    """
    isd, jsd = bd.isd, bd.jsd
    return {
        "grid_lon": fort(gs["grid_lon"], isd, jsd),
        "grid_lat": fort(gs["grid_lat"], isd, jsd),
        "agrid_lon": fort(gs["agrid_lon"], isd, jsd),
        "agrid_lat": fort(gs["agrid_lat"], isd, jsd),
        "dxa": fort(gs["dxa"], isd, jsd),
        "dya": fort(gs["dya"], isd, jsd),
        "edge_w": gs["edge_w"], "edge_e": gs["edge_e"],
        "edge_s": gs["edge_s"], "edge_n": gs["edge_n"],
        "bounded_domain": bool(gs.get("bounded_domain", False)),
        "grid_type": int(gs.get("grid_type", 0)),
        "sw_corner": bool(gs.get("sw_corner", True)),
        "se_corner": bool(gs.get("se_corner", True)),
        "nw_corner": bool(gs.get("nw_corner", True)),
        "ne_corner": bool(gs.get("ne_corner", True)),
    }


def geopk(delp, pt, hs, bd, *, km, ptop, akap, cp_air, cg, duogrid,
          computehalo, npx, npy, a2b_ord, bounded_domain=False,
          sw_dynamics=False, q_con=None, use_cond=False,
          unwritten_fill: float = BIG_NUMBER) -> dict:
    """``dyn_core.F90:2660-2790``, verbatim port.

    Parameters mirror the Fortran dummies one-for-one.  ``duogrid`` is
    geopk's own ``duogrid`` dummy: dyn_core feeds it from
    ``gridstruct%dg%is_initialized`` at the C-grid site (:534) and from
    ``flagstruct%duogrid`` at the D-grid site (:1402) — TWO DIFFERENT
    structure members.  They are not unified here (UNCERTAIN U5).

    Returns a dict with ``pk``, ``gz``, ``pe``, ``peln``, ``pkz``.

    ``unwritten_fill`` is the value left in every slot the Fortran never
    writes.  Upstream declares ``pk``/``gz``/``pe``/``peln``/``pkz``
    ``intent(OUT)`` and writes only sub-boxes, so those slots are
    UNDEFINED per the standard and any fill is equally faithful.  The
    oracle lane uses ``1e30`` so the write WINDOW is itself certified;
    the km=1 SW adapters pass ``0.0`` to stay byte-identical to the
    pre-refactor stepper (UNCERTAIN U10).
    """
    if use_cond:
        raise ValueError(
            "geopk: use_cond=True selects the -DUSE_COND peg/pkg branch "
            "(dyn_core.F90:2721-2724, 2747-2750, 2772-2773) which is NOT "
            "ported and NOT certified by any fixture; refusing to run a "
            "condensate-loaded column through the dry formulas")
    if a2b_ord not in (2, 4):
        raise ValueError(f"geopk: unknown a2b_ord={a2b_ord!r} (expected 2 or 4)")
    if km < 1:
        raise ValueError(f"geopk: km must be >= 1, got {km!r}")
    del q_con  # unreferenced without -DUSE_COND; kept for interface fidelity

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    m_i = ied - isd + 1
    m_j = jed - jsd + 1
    n_i = ie - is_ + 1
    n_j = je - js + 1

    delp = np.asarray(delp, dtype=np.float64)
    pt = np.asarray(pt, dtype=np.float64)
    hs = np.asarray(hs, dtype=np.float64)

    # --- :2697-2703 range predicate.  A cg=True call ALWAYS takes the
    # 1-halo width (both disjuncts carry `.not. CG`).
    if ((not cg) and a2b_ord == 4) or ((bounded_domain or duogrid) and not cg):
        ifirst, ilast = is_ - 2, ie + 2
        jfirst, jlast = js - 2, je + 2
    else:
        ifirst, ilast = is_ - 1, ie + 1
        jfirst, jlast = js - 1, je + 1

    # --- :2705-2710 computehalo extension (bounded/duo only)
    if (bounded_domain or duogrid) and computehalo:
        if is_ == 1:
            ifirst = isd
        if ie == npx - 1:
            ilast = ied
        if js == 1:
            jfirst = jsd
        if je == npy - 1:
            jlast = jed

    box_i = _w(isd, ifirst, ilast)
    box_j = _w(jsd, jfirst, jlast)

    pk = np.full((m_i, m_j, km + 1), unwritten_fill, dtype=np.float64)
    gz = np.full((m_i, m_j, km + 1), unwritten_fill, dtype=np.float64)
    pe = np.full((n_i + 2, km + 1, n_j + 2), unwritten_fill, dtype=np.float64)
    peln = np.full((n_i, km + 1, n_j), unwritten_fill, dtype=np.float64)
    pkz = np.full((n_i, n_j, km), unwritten_fill, dtype=np.float64)

    # --- seeds :2717-2739.  ptk uses the `**` OPERATOR (dyn_core.F90:248)
    # while the k loop below uses exp(akap*log(p)) (:2746) — DIFFERENT
    # operations that may differ in the last bit.  Mirror each at its site.
    ptk = ptop ** akap
    pk[box_i, box_j, 0] = ptk
    gz[box_i, box_j, km] = hs[box_i, box_j]

    # peln seed: `#ifndef SW_DYNAMICS` (:2727-2733), j in js..je, i in is..ie
    j_pl0, j_pl1 = max(jfirst, js), min(jlast, je)
    if not sw_dynamics:
        if ifirst > is_ or ilast < ie:
            raise ValueError(
                "geopk: the peln window (is..ie) escapes the ifirst..ilast "
                "box — upstream would read undefined logp; refusing")
        peln[:, 0, _w(js, j_pl0, j_pl1)] = np.log(ptop)

    # pe seed (:2735-2739): j in (js-2, je+2) EXCLUSIVE, i clipped to the box
    i_pe0, i_pe1 = max(ifirst, is_ - 1), min(ilast, ie + 1)
    j_pe0, j_pe1 = max(jfirst, js - 1), min(jlast, je + 1)
    pe_i = _w(is_ - 1, i_pe0, i_pe1)
    pe_j = _w(js - 1, j_pe0, j_pe1)
    pe[pe_i, 0, pe_j] = ptop

    # box-relative sub-windows of p1d/logp for the pe/peln writes
    src_pe_i = _w(ifirst, i_pe0, i_pe1)
    src_pe_j = _w(jfirst, j_pe0, j_pe1)
    src_pl_i = _w(ifirst, is_, ie)
    src_pl_j = _w(jfirst, j_pl0, j_pl1)
    pl_j = _w(js, j_pl0, j_pl1)

    # --- top-down recursion :2742-2764.  p1d is a RUNNING accumulator
    # carried across k (strictly top to bottom); writing p1d**akap or
    # summing bottom-up changes the last bit.
    p1d = np.full((ilast - ifirst + 1, jlast - jfirst + 1), ptop,
                  dtype=np.float64)
    for k in range(1, km + 1):
        p1d = p1d + delp[box_i, box_j, k - 1]
        logp = np.log(p1d)
        pk[box_i, box_j, k] = np.exp(akap * logp)
        pe[pe_i, k, pe_j] = p1d[src_pe_i, src_pe_j]
        if not sw_dynamics:
            peln[:, k, pl_j] = logp[src_pl_i, src_pl_j]

    # --- bottom-up recursion :2766-2779 over the FULL box (wider than
    # the pe/peln window).  `cp_air` is DROPPED under -DSW_DYNAMICS (:2770).
    for k in range(km - 1, -1, -1):
        dpk = pk[box_i, box_j, k + 1] - pk[box_i, box_j, k]
        if sw_dynamics:
            gz[box_i, box_j, k] = gz[box_i, box_j, k + 1] + \
                pt[box_i, box_j, k] * dpk
        else:
            gz[box_i, box_j, k] = gz[box_i, box_j, k + 1] + \
                cp_air * pt[box_i, box_j, k] * dpk

    # --- pkz :2781-2787: only when `.not. CG`, only on [is,ie]x[js,je],
    # and it CONSUMES peln — so it is invalid wherever peln was not
    # written.  Under -DSW_DYNAMICS peln is never written at all
    # (:2727-2733), so upstream's pkz would consume undefined memory;
    # the port leaves the sentinel instead of manufacturing a value.
    # DOCUMENTED DEVIATION, SW lane only — dyn_core's SW build has no
    # pkz consumer, and the certified oracle lane is non-SW.
    if (not cg) and not sw_dynamics:
        ki = _w(isd, is_, ie)
        kj = _w(jsd, js, je)
        for k in range(km):
            pkz[:, :, k] = (pk[ki, kj, k + 1] - pk[ki, kj, k]) / (
                akap * (peln[:, k + 1, :] - peln[:, k, :]))

    return {"pk": pk, "gz": gz, "pe": pe, "peln": peln, "pkz": pkz}


def p_grad_c(dt2, delpc, pkc, gz, uc, vc, gs: dict, bd, *, npz,
             hydrostatic: bool = True) -> None:
    """``dyn_core.F90:2073-2132``, verbatim port.  Mutates ``uc``/``vc``
    IN PLACE.

    ``delpc`` is UNREAD on the hydrostatic branch (:2111) — it is kept in
    the signature for interface fidelity, and the oracle proves it is
    never read by re-running the Fortran with ``delpc = -9.e9`` and
    requiring a BITWISE identical result.

    NON-HYDROSTATIC branch (:2109-2113): the ONLY difference is the
    denominator weight — ``wk = delpc(:,:,k)`` instead of the ``pkc``
    interface difference; the two momentum expressions are shared
    verbatim.  On that branch ``pkc`` is FULL interface pressure (the
    header comment at :2079-2081 and ``Riem_Solver_c``'s
    ``pef = pe2 + pem``), not ``pe**cappa`` — the CALLER owns handing the
    right quantity; this routine cannot tell them apart.

    There is NO inter-k coupling (:2100): the only vertical reads are the
    bracketing interfaces ``k`` and ``k+1``.
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd
    if hydrostatic:
        del delpc  # hydrostatic branch never reads it (see docstring)
    else:
        delpc = np.asarray(delpc, dtype=np.float64)

    rdxc = np.asarray(gs["rdxc"], dtype=np.float64)
    rdyc = np.asarray(gs["rdyc"], dtype=np.float64)

    # wk window :2103-2113 == the CG geopk write box
    wlo_i, wlo_j = is_ - 1, js - 1
    wk_i = _w(isd, is_ - 1, ie + 1)
    wk_j = _w(jsd, js - 1, je + 1)

    # uc: i in is..ie+1, j in js..je   (:2116-2122)
    u_i = _w(isd, is_, ie + 1)
    u_im1 = _w(isd, is_ - 1, ie)
    u_j = _w(jsd, js, je)
    uw_i = _w(wlo_i, is_, ie + 1)
    uw_im1 = _w(wlo_i, is_ - 1, ie)
    uw_j = _w(wlo_j, js, je)

    # vc: i in is..ie, j in js..je+1   (:2123-2129)
    v_i = _w(isd, is_, ie)
    v_j = _w(jsd, js, je + 1)
    v_jm1 = _w(jsd, js - 1, je)
    vw_i = _w(wlo_i, is_, ie)
    vw_j = _w(wlo_j, js, je + 1)
    vw_jm1 = _w(wlo_j, js - 1, je)

    for k in range(npz):
        if hydrostatic:
            wk = pkc[wk_i, wk_j, k + 1] - pkc[wk_i, wk_j, k]     # :2104-2108
        else:
            wk = delpc[wk_i, wk_j, k]                            # :2109-2113
        # Grouping copied EXACTLY (:2118-2120): dt2*rdxc/(wk+wk)*(...).
        # Regrouping to dt2*(rdxc/(wk+wk))*(...) changes the last bit.
        uc[u_i, u_j, k] = uc[u_i, u_j, k] + dt2 * rdxc[u_i, u_j] / (
            wk[uw_im1, uw_j] + wk[uw_i, uw_j]) * (
            (gz[u_im1, u_j, k + 1] - gz[u_i, u_j, k])
            * (pkc[u_i, u_j, k + 1] - pkc[u_im1, u_j, k])
            + (gz[u_im1, u_j, k] - gz[u_i, u_j, k + 1])
            * (pkc[u_im1, u_j, k + 1] - pkc[u_i, u_j, k]))
        # :2125-2127 — note the asymmetric second gz term, verbatim
        vc[v_i, v_j, k] = vc[v_i, v_j, k] + dt2 * rdyc[v_i, v_j] / (
            wk[vw_i, vw_jm1] + wk[vw_i, vw_j]) * (
            (gz[v_i, v_jm1, k + 1] - gz[v_i, v_j, k])
            * (pkc[v_i, v_j, k + 1] - pkc[v_i, v_jm1, k])
            + (gz[v_i, v_jm1, k] - gz[v_i, v_j, k + 1])
            * (pkc[v_i, v_jm1, k + 1] - pkc[v_i, v_j, k]))


def one_grad_p(u, v, pk, gz, divg2, delp, gs: dict, bd, *, npx, npy, npz,
               dt, ptop, akap, hydrostatic: bool = True, a2b_ord: int = 4,
               d_ext: float = 0.0, ng: int | None = None,
               duogrid: bool = True, bvertex_mean2: bool = False) -> None:
    """``dyn_core.F90:2347-2480``, verbatim port.

    Mutates ``u``, ``v``, AND ``pk``, ``gz`` IN PLACE: ``a2b_ord4`` is
    called with ``replace=.true.`` at :2399/:2409, so on return ``pk``
    and ``gz`` hold B-GRID CORNER values on ``[is,ie+1] x [js,je+1]`` and
    A-grid values everywhere else.  ``delp`` is NOT modified (:2458/:2460
    omit ``replace``) and, on the hydrostatic branch, is not even read —
    the oracle proves that with a ``delp = -9.e9`` re-run.

    Storage convention (kept deliberately): ``pk``/``gz`` stay
    ``(m_a, m_a, npz+1)`` CELL-shaped planes reused as B-node storage
    after ``replace=True``.  That is valid because a2b writes only
    ``is..ie+1`` / ``js..je+1`` and ``ng >= 1`` leaves the slots.  Do NOT
    restage to ``(m_b, m_b, npz+1)``: it would silently change which halo
    cells survive and break the ``PK_OGP``/``GZ_OGP`` oracle compare.

    ``duogrid`` selects a2b_ord4's duo (interior-everywhere) branch.  It
    is an ARGUMENT here because this repo's ``a2b_ord4`` takes it as one;
    upstream reads ``gridstruct%dg%is_initialized`` inside a2b itself.

    ``bvertex_mean2`` is a NON-FAITHFUL diagnostic screen (codex
    vertex-kill C3) carried over from the km=1 stepper: after each a2b
    replace, the four projected B vertices are replaced by the mean of
    their two edge neighbours.  Default OFF = faithful.
    """
    from legoesm.core.fv3_native_d_sw import a2b_ord4

    if a2b_ord != 4:
        raise NotImplementedError(
            f"one_grad_p: a2b_ord={a2b_ord!r}; only the a2b_ord==4 arm is "
            "ported.  The a2b_ord2 else-arms (dyn_core.F90:2401/2411/2460) "
            "are linked in the Fortran extract purely so they resolve and "
            "are DEAD on this lane (UNCERTAIN U8) — there is no python "
            "a2b_ord2 and no fixture that would certify one")
    if not hydrostatic:
        raise NotImplementedError(
            "one_grad_p: the non-hydrostatic branch needs top_value=ptop "
            "(:2385) and an a2b_ord4(delp) wk (:2456-2461, no `replace`); "
            "neither is certified by any fixture")

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd, ied, jed = bd.isd, bd.jsd, bd.ied, bd.jed
    if ng is None:
        ng = bd.ng

    divg2 = np.asarray(divg2, dtype=np.float64)
    del delp  # hydrostatic branch never reads it (see docstring)

    # :2380-2386 — `ptk` (the `**` operator, :248/:2382), NOT exp(akap*log)
    top_value = ptop ** akap

    b_i = _w(isd, is_, ie + 1)
    b_j = _w(jsd, js, je + 1)

    # :2388-2393 — B box seed at k=1 ONLY
    pk[b_i, b_j, 0] = top_value

    gsf = a2b_gridstruct_view(gs, bd)
    # a2b's `qout` scratch: a single local reused across all three loops
    # upstream.  NaN outside the B box so an out-of-window read is LOUD
    # (the duo branch writes exactly [is,ie+1]x[js,je+1] and the final
    # k loop overwrites exactly that box before reading it).
    wk = np.full((ied - isd + 1, jed - jsd + 1), np.nan, dtype=np.float64)

    def _a2b(plane) -> None:
        a2b_ord4(fort(plane, isd, jsd), fort(wk, isd, jsd), gsf, npx, npy,
                 is_, ie, js, je, ng, replace=True, duogrid=duogrid)
        if bvertex_mean2:
            b = fort(plane, isd, jsd)
            snap = {(i, j): b[i, j]
                    for i in (is_, is_ + 1, ie, ie + 1)
                    for j in (js, js + 1, je, je + 1)}
            b[is_, js] = 0.5 * (snap[(is_ + 1, js)] + snap[(is_, js + 1)])
            b[ie + 1, js] = 0.5 * (snap[(ie, js)] + snap[(ie + 1, js + 1)])
            b[ie + 1, je + 1] = 0.5 * (snap[(ie, je + 1)]
                                       + snap[(ie + 1, je)])
            b[is_, je + 1] = 0.5 * (snap[(is_ + 1, je + 1)]
                                    + snap[(is_, je)])

    # a2b k-bounds DIFFER between the two fields and that is LOAD-BEARING:
    # pk over k=2..npz+1 (:2397; k=1 is already the seeded top_value),
    # gz over k=1..npz+1 (:2407).
    for k in range(1, npz + 1):
        _a2b(pk[:, :, k])
    for k in range(0, npz + 1):
        _a2b(gz[:, :, k])

    # :2415-2443 — external-mode filter increments; both zero at d_ext<=0.
    # The km=1 stepper hard-coded origin 1 for divg2; here the windows are
    # taken off is_/js so any bounds are representable.
    if d_ext > 0.0:
        wk2 = divg2[_w(is_, is_, ie), _w(js, js, je + 1)] \
            - divg2[_w(is_, is_ + 1, ie + 1), _w(js, js, je + 1)]
        wk1 = divg2[_w(is_, is_, ie + 1), _w(js, js, je)] \
            - divg2[_w(is_, is_, ie + 1), _w(js, js + 1, je + 1)]
    else:
        wk2 = np.zeros((ie - is_ + 1, je - js + 2), dtype=np.float64)
        wk1 = np.zeros((ie - is_ + 2, je - js + 1), dtype=np.float64)

    rdx = np.asarray(gs["rdx"], dtype=np.float64)
    rdy = np.asarray(gs["rdy"], dtype=np.float64)

    # u: i in is..ie, j in js..je+1   (:2464-2470)
    ui = _w(isd, is_, ie)
    uip1 = _w(isd, is_ + 1, ie + 1)
    uj = _w(jsd, js, je + 1)
    # v: i in is..ie+1, j in js..je   (:2471-2477)
    vi = _w(isd, is_, ie + 1)
    vj = _w(jsd, js, je)
    vjp1 = _w(jsd, js + 1, je + 1)

    for k in range(npz):
        # :2450-2455 — wk recomputed on the B box INSIDE the k loop
        wk[b_i, b_j] = pk[b_i, b_j, k + 1] - pk[b_i, b_j, k]
        # ASSIGNMENT (not an increment, unlike p_grad_c): the whole
        # bracket is multiplied by rdx/rdy.
        u[ui, uj, k] = rdx[ui, uj] * (
            wk2 + u[ui, uj, k] + dt / (wk[ui, uj] + wk[uip1, uj]) * (
                (gz[ui, uj, k + 1] - gz[uip1, uj, k])
                * (pk[uip1, uj, k + 1] - pk[ui, uj, k])
                + (gz[ui, uj, k] - gz[uip1, uj, k + 1])
                * (pk[ui, uj, k + 1] - pk[uip1, uj, k])))
        v[vi, vj, k] = rdy[vi, vj] * (
            wk1 + v[vi, vj, k] + dt / (wk[vi, vj] + wk[vi, vjp1]) * (
                (gz[vi, vj, k + 1] - gz[vi, vjp1, k])
                * (pk[vi, vjp1, k + 1] - pk[vi, vj, k])
                + (gz[vi, vj, k] - gz[vi, vjp1, k + 1])
                * (pk[vi, vj, k + 1] - pk[vi, vjp1, k])))


def nh_p_grad(u, v, pp, gz, delp, pk3, gs: dict, bd, *, npx, npy, npz,
              dt, ptop, akap, use_logp: bool = False,
              ng: int | None = None, duogrid: bool = True) -> None:
    """``dyn_core.F90:2135-2230`` (``nh_p_grad``), verbatim port.

    The NH D-stage pressure update that replaces ``one_grad_p`` at
    ``beta = 0`` (the dispatch is ``beta < -0.1`` for one_grad_p, so 0
    lands HERE — dyn_core.F90:1536-1543, trap #4 of the NH spec).

    Mutates ``u``, ``v`` and — via ``a2b_ord4(replace=.true.)`` —
    ``pp``, ``pk3``, ``gz`` IN PLACE: on return those three hold B-GRID
    CORNER values on ``[is,ie+1] x [js,je+1]`` (trap #6: pkc is
    perturbation pressure INTO this call and B-grid scratch AFTER it).
    ``delp`` is read through a NO-replace a2b into a scratch and left
    unmodified.

    Windows (:2172-2181): the k=1 seed writes ``pp = 0`` and
    ``pk3 = top_value`` over the B box only; ``top_value`` is ``peln1 =
    log(ptop)`` under ``use_logp`` else ``ptk = ptop**akap`` (the ``**``
    operator, :246-248, matching one_grad_p).

    Per level (:2190-2230): ``wk`` = B-grid ``pk3`` interface
    difference (the hydrostatic weight), ``wk1`` = B-grid ``delp`` (the
    NH weight); u adds ``du1`` (hydrostatic form) plus the NH term in
    ``pp``, then multiplies by ``rdx``; v likewise with ``rdy``.  The
    grouping is copied exactly — see the p_grad_c note on why
    regrouping changes the last bit.
    """
    from legoesm.core.fv3_native_d_sw import a2b_ord4

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd, ied, jed = bd.isd, bd.jsd, bd.ied, bd.jed
    if ng is None:
        ng = bd.ng

    top_value = float(np.log(ptop)) if use_logp else ptop ** akap

    b_i = _w(isd, is_, ie + 1)
    b_j = _w(jsd, js, je + 1)

    gsf = a2b_gridstruct_view(gs, bd)
    wk_scratch = np.full((ied - isd + 1, jed - jsd + 1), np.nan,
                         dtype=np.float64)

    def _a2b_replace(plane) -> None:
        a2b_ord4(fort(plane, isd, jsd), fort(wk_scratch, isd, jsd), gsf,
                 npx, npy, is_, ie, js, je, ng, replace=True,
                 duogrid=duogrid)

    # :2172-2185 — k=1 B-box seed for pp/pk3; a2b for k>=2; gz EVERY k.
    for k in range(npz + 1):
        if k == 0:
            pp[b_i, b_j, 0] = 0.0
            pk3[b_i, b_j, 0] = top_value
        else:
            _a2b_replace(pp[:, :, k])
            _a2b_replace(pk3[:, :, k])
        _a2b_replace(gz[:, :, k])

    rdx = np.asarray(gs["rdx"], dtype=np.float64)
    rdy = np.asarray(gs["rdy"], dtype=np.float64)

    # u: i in is..ie, j in js..je+1 (:2196-2211)
    ui = _w(isd, is_, ie)
    uip1 = _w(isd, is_ + 1, ie + 1)
    uj = _w(jsd, js, je + 1)
    # v: i in is..ie+1, j in js..je (:2213-2228)
    vi = _w(isd, is_, ie + 1)
    vj = _w(jsd, js, je)
    vjp1 = _w(jsd, js + 1, je + 1)

    wk = np.full_like(wk_scratch, np.nan)
    wk1 = np.full_like(wk_scratch, np.nan)
    for k in range(npz):
        # :2191 — B-grid delp into wk1 (NO replace; delp unmodified)
        a2b_ord4(fort(delp[:, :, k], isd, jsd), fort(wk1, isd, jsd), gsf,
                 npx, npy, is_, ie, js, je, ng, replace=False,
                 duogrid=duogrid)
        # :2192-2195 — hydrostatic weight from pk3 interface differences
        wk[b_i, b_j] = pk3[b_i, b_j, k + 1] - pk3[b_i, b_j, k]

        # :2196-2211 — u: INCREMENT-then-scale, hydrostatic du1 + NH pp
        du1 = dt / (wk[ui, uj] + wk[uip1, uj]) * (
            (gz[ui, uj, k + 1] - gz[uip1, uj, k])
            * (pk3[uip1, uj, k + 1] - pk3[ui, uj, k])
            + (gz[ui, uj, k] - gz[uip1, uj, k + 1])
            * (pk3[ui, uj, k + 1] - pk3[uip1, uj, k]))
        u[ui, uj, k] = (u[ui, uj, k] + du1
                        + dt / (wk1[ui, uj] + wk1[uip1, uj]) * (
                            (gz[ui, uj, k + 1] - gz[uip1, uj, k])
                            * (pp[uip1, uj, k + 1] - pp[ui, uj, k])
                            + (gz[ui, uj, k] - gz[uip1, uj, k + 1])
                            * (pp[ui, uj, k + 1] - pp[uip1, uj, k]))
                        ) * rdx[ui, uj]

        # :2213-2228 — v
        dv1 = dt / (wk[vi, vj] + wk[vi, vjp1]) * (
            (gz[vi, vj, k + 1] - gz[vi, vjp1, k])
            * (pk3[vi, vjp1, k + 1] - pk3[vi, vj, k])
            + (gz[vi, vj, k] - gz[vi, vjp1, k + 1])
            * (pk3[vi, vj, k + 1] - pk3[vi, vjp1, k]))
        v[vi, vj, k] = (v[vi, vj, k] + dv1
                        + dt / (wk1[vi, vj] + wk1[vi, vjp1]) * (
                            (gz[vi, vj, k + 1] - gz[vi, vjp1, k])
                            * (pp[vi, vjp1, k + 1] - pp[vi, vj, k])
                            + (gz[vi, vj, k] - gz[vi, vjp1, k + 1])
                            * (pp[vi, vj, k + 1] - pp[vi, vjp1, k]))
                        ) * rdy[vi, vj]


def pk3_halo(pk3, delp, bd, *, npz, ptop, akap) -> None:
    """``dyn_core.F90:1832-1884`` (``pk3_halo``), verbatim port.

    Locally rebuilds the TWO x-rings (i in {is-2, is-1, ie+1, ie+2},
    j = js..je) and TWO y-rings (j in {js-2, js-1, je+1, je+2},
    i = is-2..ie+2) of ``pk3`` from halo ``delp`` — this is a local
    recomputation, NOT a halo exchange (trap #10: replacing it with an
    exchange changes both arithmetic and corner coverage).  Level 1
    (the top interface) is NEVER written here.
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd

    def _col(i, j):
        pei = ptop
        for k in range(npz):
            pei = pei + delp[i - isd, j - jsd, k]
            pk3[i - isd, j - jsd, k + 1] = np.exp(akap * np.log(pei))

    for j in range(js, je + 1):
        for i in (is_ - 2, is_ - 1, ie + 1, ie + 2):
            _col(i, j)
    for i in range(is_ - 2, ie + 2 + 1):
        for j in (js - 2, js - 1, je + 1, je + 2):
            _col(i, j)


def pln_halo(pk3, delp, bd, *, npz, ptop) -> None:
    """``dyn_core.F90:1886-1933`` (``pln_halo``) — the ``use_logp``
    sibling of :func:`pk3_halo` (log(p) rings instead of p**kappa).
    Dead on the pinned deck (USE_LOGP=F) but ten lines away."""
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd

    def _col(i, j):
        pet = ptop
        for k in range(npz):
            pet = pet + delp[i - isd, j - jsd, k]
            pk3[i - isd, j - jsd, k + 1] = np.log(pet)

    for j in range(js, je + 1):
        for i in (is_ - 2, is_ - 1, ie + 1, ie + 2):
            _col(i, j)
    for i in range(is_ - 2, ie + 2 + 1):
        for j in (js - 2, js - 1, je + 1, je + 2):
            _col(i, j)


def pe_halo(pe, delp, bd, *, npz, ptop) -> None:
    """``dyn_core.F90:1935-1963`` (``pe_halo``), verbatim port.

    Fills the ONE-ring edges of ``pe`` — i in {is-1, ie+1} for
    j = js..je, then j in {js-1, je+1} for i = is-1..ie+1 — by local
    hydrostatic integration of halo ``delp``.  ``pe`` is the oracle's
    ``(is-1:ie+1, npz+1, js-1:je+1)`` (i, k, j) array; ``delp`` is the
    padded (i, j, k) A-grid field.  Runs only on remap substeps
    (:1441-1442).
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd

    def _col(i, j):
        pe[i - (is_ - 1), 0, j - (js - 1)] = ptop
        for k in range(npz):
            pe[i - (is_ - 1), k + 1, j - (js - 1)] = (
                pe[i - (is_ - 1), k, j - (js - 1)]
                + delp[i - isd, j - jsd, k])

    for j in range(js, je + 1):
        for i in (is_ - 1, ie + 1):
            _col(i, j)
    for i in range(is_ - 1, ie + 1 + 1):
        for j in (js - 1, je + 1):
            _col(i, j)
