"""C-grid half-step of the duo acoustic cadence, km-general, six faces.

STAGE 1 (loop-faithful NumPy IR) unit 3 of the staged 3-D port. Composes
kernels that are ALREADY bit-exact against the Fortran; it adds the
vertical cadence and the six-face assembly, nothing else.

ORACLE CADENCE (published tree, checksum-pinned at
``/burg-archive/glab/users/pg2328/fv3_oracle_pinned``; the built copy under
``fv3_recon`` carries +102/+66 lines of debug instrumentation and its line
numbers are shifted, so do not cite it):

    dyn_core.F90:339   do it=1,n_split
                :489   call c_sw          <- inside do k=1,npz
                :533   call geopk         <- CG=.true., hydrostatic
                :629   call p_grad_c

WHY c_sw IS LOOPED AND THE PRESSURE CHAIN IS NOT
------------------------------------------------
``c_sw`` takes ``dimension(bd%isd:bd%ied, bd%jsd:bd%jed)`` dummies
(``sw_core.F90:84-87``) and the oracle wraps it in ``do k=1,npz``, so the
faithful lift is to call the certified 2-D kernel once per level -- NOT to
write new stage math.

``geopk``/``p_grad_c`` in ``fv3_native_pgrad`` are already km-general and
take the whole column, so they are called ONCE per face with ``km``.
``geopk`` integrates vertically inside itself and its running accumulator's
SUM ORDER is part of the bit-exact contract; a per-level loop here would
break that. ``p_grad_c`` has no inter-k coupling at all
(``dyn_core.F90:2100``: the only vertical reads are the bracketing
interfaces k and k+1).

WHAT THIS UNIT DELIBERATELY DOES NOT DO
---------------------------------------
No d_sw stages, no flux-average barriers, no D-grid pressure. Those are
units 4 and 5. Keeping the C-grid phase separately testable is the point:
the C grid is DIAGNOSTIC (``fv_arrays.F90``: "The C grid component is
'diagnostic' in that it is predicted every time step from the D grid
variables"), so it must be rebuilt every step and must never be persisted
alongside the prognostic D winds.
"""
from __future__ import annotations

import numpy as np

from legoesm.core.fv3_native_state_3d import (
    field_shape,
    level_slice,
    require_no_remap_needed,
)

# c_sw outputs we assemble into 3-D. Names are the Fortran dummies.
CSW_OUT_2D = ("delpc", "ptc", "uc", "vc", "ua", "va", "ut", "vt", "divg_d")


def csw_phase_3d(ctx: dict, state: list, dt2: float, km: int, *,
                 nord: int = 2, duogrid: bool = True,
                 hydrostatic: bool = True,
                 remap_follows: bool = False) -> list:
    """Per-level ``c_sw`` on all six faces; returns 3-D C-grid outputs.

    ``dt2`` is the half step (``dyn_core.F90`` passes ``dt2`` to c_sw).
    ``nord`` comes from the deck: the shipped duo decks run ``nord = 2``,
    which is why the ``divgd`` exchange downstream is required at all
    (``dyn_core.F90:652`` gates it on ``nord > 0``; :706 is the
    ``flagstruct%regional`` branch and never runs on the duo lane).

    ``hydrostatic=False`` selects c_sw's NH arm (``sw_core.F90:216-291``):
    ``w`` rides the same upwind fluxes as delp/pt and the advected
    cell-centre ``wc`` is collected -- it is the ``omga`` argument
    ``Riem_Solver_c`` consumes (``dyn_core.F90:589-594``; intent(in)
    there, so the C stage never writes it back).
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    from legoesm.core.fv3_native_sw_core import c_sw

    n, ng, bd = ctx["n"], ctx["ng"], ctx["bd"]
    npx = n + 1
    names = CSW_OUT_2D + (() if hydrostatic else ("wc",))
    outs = []
    for t in range(6):
        face = state[t]
        acc = {name: np.zeros(field_shape(_out_like(name), n, ng, km),
                              dtype=np.float64)
               for name in names}
        for k in range(km):
            lev = level_slice(face, k, km)
            got = c_sw(lev["delp"], lev["pt"], lev["w"],
                       lev["u"], lev["v"], ctx["gs6"][t], bd,
                       npx, npx, dt2, duogrid=duogrid, nord=nord,
                       hydrostatic=hydrostatic)
            for name in names:
                if name not in got:
                    raise KeyError(
                        f"c_sw returned no {name!r}; keys are "
                        f"{sorted(got)}. The 3-D assembler must not "
                        f"silently drop a stage output.")
                arr = np.asarray(got[name])
                want = acc[name][:, :, k].shape
                if arr.shape != want:
                    raise ValueError(
                        f"face {t + 1} level {k} output {name!r}: c_sw gave "
                        f"{arr.shape}, the 3-D container expects {want}. A "
                        f"stagger mismatch here would broadcast, not raise.")
                acc[name][:, :, k] = arr
        outs.append(acc)
    return outs


def _out_like(name: str) -> str:
    """Map a c_sw output name onto the field whose shape it shares."""
    return {"divg_d": "divgd", "uc": "uc", "vc": "vc",
            "delpc": "delp", "ptc": "pt", "ua": "ua", "va": "va",
            "ut": "ut", "vt": "vt", "wc": "w"}[name]


def cgrid_pressure_phase_3d(ctx: dict, csw_outs: list, km: int, *,
                            dt2: float, ptop: float, akap: float,
                            cp_air: float, a2b_ord: int = 4,
                            hydrostatic: bool = True,
                            remap_follows: bool = False,
                            stage_hook=None) -> list:
    """C-grid ``geopk`` then ``p_grad_c``, once per face over the column.

    ``stage_hook(name, payload)`` observes the oracle stage boundaries
    S04 (post C-grid geopk) and S05 (post ``p_grad_c``); ``None`` is
    byte-identical to the pre-hook behaviour.

    ``dyn_core.F90:533`` calls geopk with ``CG = .true.``; ``:629`` then
    calls ``p_grad_c``, which MUTATES ``uc``/``vc`` IN PLACE -- so the
    arrays handed in here are the ones the D-grid phase will read.

    Returns the per-face geopk dict (``pk``, ``gz``, ``pe``, ``peln``,
    ``pkz``) so later stages and the trace can consume it.

    ``sw_dynamics=False`` and real ``ptop``/``akap``/``cp_air`` -- the km=1
    SW adapters pass ``sw_dynamics=True, ptop=0, akap=1, cp_air=1``, which
    is the shallow-water degenerate case and NOT what a 3-D hydrostatic
    column wants.
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    if not hydrostatic:
        raise NotImplementedError(
            "cgrid_pressure_phase_3d: only the hydrostatic branch is "
            "certified (fv3_native_pgrad refuses the NH p_grad_c branch); "
            "refusing to silently run hydrostatic formulas for NH")
    from legoesm.core.fv3_native_pgrad import geopk, p_grad_c

    bd = ctx["bd"]
    n, ng = ctx["n"], ctx["ng"]
    hs6 = ctx.get("hs6")
    press = []
    for t in range(6):
        out = csw_outs[t]
        hs = (np.asarray(hs6[t], dtype=np.float64) if hs6 is not None
              else np.zeros(field_shape("delp", n, ng, 1)[:2],
                            dtype=np.float64))
        # geopk integrates p1d = ptop + cumsum(delpc) and takes log(p1d)
        # (fv3_native_pgrad.py:245). A NON-POSITIVE delpc is perfectly
        # FINITE, so it sails through every isfinite() check and only turns
        # into NaN one stage later inside the log -- which is exactly how a
        # sub-step-2 blow-up got mis-attributed twice. Fail at the source,
        # naming the face and level.
        _dc = np.asarray(out["delpc"])
        _win = _dc[bd.is_ - bd.isd:bd.ie - bd.isd + 1,
                   bd.js - bd.jsd:bd.je - bd.jsd + 1, :]
        if not np.all(np.isfinite(_win)) or _win.min() <= 0.0:
            _k = int(np.argmin(_win.min(axis=(0, 1))))
            raise ValueError(
                f"face {t + 1}: delpc entering geopk has min "
                f"{_win.min():.6g} at level {_k} (must be > 0). geopk will "
                f"take log(ptop + cumsum(delpc)); a non-positive layer mass "
                f"is finite and would surface as NaN one stage later.")
        got = geopk(out["delpc"], out["ptc"], hs, bd,
                    km=km, ptop=ptop, akap=akap, cp_air=cp_air,
                    cg=True, duogrid=True, computehalo=False,
                    npx=bd.ie + 1, npy=bd.je + 1, a2b_ord=a2b_ord,
                    bounded_domain=False, sw_dynamics=False)
        # p_grad_c mutates uc/vc in place -- that is the oracle's contract
        # (dyn_core.F90:2073-2132), so csw_outs carries the update forward.
        p_grad_c(dt2, out["delpc"], got["pk"], got["gz"],
                 out["uc"], out["vc"], ctx["gs6"][t], bd,
                 npz=km, hydrostatic=True)
        press.append(got)
    if stage_hook is not None:
        # S04: geopk's pkc/gz (p_grad_c reads but never writes them, so
        # observing after the loop is the same state as at :534).
        stage_hook("S04_geopkC", [{"pkc": press[t]["pk"],
                                   "gz": press[t]["gz"]}
                                  for t in range(6)])
        # S05: uc/vc as mutated in place by p_grad_c (:629).
        stage_hook("S05_pgradc", [{"uc": csw_outs[t]["uc"],
                                   "vc": csw_outs[t]["vc"]}
                                  for t in range(6)])
    return press


def cgrid_nh_pressure_phase_3d(ctx: dict, csw_outs: list, gz6: list,
                               ws3_6: list, km: int, *, dt2: float,
                               ptop: float, akap: float, cp_air: float,
                               p_fac: float, a_imp: float,
                               dp0: np.ndarray, hs6: list, zs6: list,
                               remap_follows: bool = False) -> list:
    """NH C-grid pressure stage: ``update_dz_c`` -> ``Riem_Solver_c`` ->
    NH ``p_grad_c``, per face (``dyn_core.F90:583-594`` then ``:629``).

    Replaces the hydrostatic ``geopk`` chain of
    :func:`cgrid_pressure_phase_3d`.  The CALLER (the acoustic driver)
    owns the gz/zh cadence around this: first substep seeds
    ``gz[..,km] = zs`` padded, rebuilds ``gz`` from ``delz``, duo-
    exchanges ``gz`` and copies it into ``zh``; later substeps restore
    ``gz = zh`` (``dyn_core.F90:384-416, 535-581``).  ``gz6`` arrives
    here in HEIGHT form and leaves as GEOPOTENTIAL (Riem_Solver_c's
    rebuild from ``hs = phis`` -- NH-spec trap #7), which is exactly
    what the NH ``p_grad_c`` consumes.

    Corner flags are FALSE: duogrid forces ``bounded_domain=.true.`` and
    ``fv_grid_utils.F90:219-229`` then leaves all four flags false
    (trap #17) -- ``update_dz_c``'s ``fill_4corners`` calls are live but
    no-ops, faithfully.

    ``ws3_6`` are per-face PADDED 2-D arrays ``update_dz_c`` fills and
    ``Riem_Solver_c`` reads (``ws3`` in dyn_core; intent(in) there).
    ``wc`` (the c_sw advected w, dyn_core's ``omga``) is intent(in) to
    the Riemann solve and never written back.

    Returns per-face dicts with ``pkc`` (FULL C-stage interface
    pressure, ``pe2 + pem``) -- ``delpc``/``ptc``/``uc``/``vc`` live on
    in ``csw_outs`` (p_grad_c mutates uc/vc in place, the oracle's
    contract).
    """
    require_no_remap_needed(km, remap_follows=remap_follows)
    from legoesm.core.fv3_native_nh_core import riem_solver_c, update_dz_c
    from legoesm.core.fv3_native_pgrad import p_grad_c

    bd = ctx["bd"]
    n, ng = ctx["n"], ctx["ng"]
    m_a = n + 2 * ng
    press = []
    for t in range(6):
        out = csw_outs[t]
        if "wc" not in out:
            raise KeyError(
                "cgrid_nh_pressure_phase_3d: csw_outs has no 'wc' -- "
                "csw_phase_3d must run with hydrostatic=False")
        gs = ctx["gs6"][t]
        update_dz_c(bd, km, dt2, dp0, zs6[t],
                    np.asarray(gs["area"], dtype=np.float64),
                    out["ut"], out["vt"], gz6[t], ws3_6[t],
                    n + 1, n + 1,
                    sw_corner=False, se_corner=False,
                    ne_corner=False, nw_corner=False,
                    grid_type=int(gs.get("grid_type", 0)))
        pkc = np.zeros((m_a, m_a, km + 1), dtype=np.float64)
        riem_solver_c(1, dt2, bd, km, akap, cp_air, ptop, hs6[t],
                      out["wc"], out["ptc"], out["delpc"], gz6[t],
                      pkc, ws3_6[t], p_fac, a_imp)
        p_grad_c(dt2, out["delpc"], pkc, gz6[t],
                 out["uc"], out["vc"], gs, bd,
                 npz=km, hydrostatic=False)
        press.append({"pkc": pkc})
    return press
