"""One full FV3 outer time step: ``fv_dynamics`` for the pinned duo lane.

STAGE 1 unit 7 -- the last assembly. ``fv3_native_acoustic_3d`` already
runs ``dyn_core``'s ``do it=1,n_split`` loop and ``fv3_native_mapz`` runs
``Lagrangian_to_Eulerian``; this module is the ``do n_map=1,k_split``
shell that joins them, plus the two conversions that live only in
``fv_dynamics`` itself.

ORACLE ORDER (published, checksum-pinned tree; ``model/fv_dynamics.F90``)::

    :271-279  pfull(k) from ak/bk at p_ref
    :281-294  hydrostatic branch: dp1(i,j,k) = zvir*q(sphum)
    :396-408  pt -> virtual POTENTIAL temperature:
              pt = pt*(1 + dp1)/pkz          <-- COMPUTE WINDOW ONLY
    :451      do n_map = 1, k_split          (mdt = bdt/k_split, :413)
    :472-478    dp1 = delp
    :482        last_step = (n_map == k_split)
    :502        dyn_core(mdt, n_split, ...)
    :534/538    tracer_2d                    <-- NOT PORTED, see below
    :568      if (npz > 4) then
    :618        Lagrangian_to_Eulerian(last_step, ...)
    :674      endif

``pfull`` IS COMPUTED AND NEVER USED ON THIS LANE, so it is not returned.
Its only consumers are the sponge/damping profile at ``dyn_core.F90:772``
and ``Rayleigh_Super``.  The pinned deck has ``n_sponge = -1``, which
takes ``dyn_core.F90:772``'s ``.or.`` arm and sets ``d2_divg = d2_bg``
uniformly, leaving ``nord_v(k) = min(2, nord) = 2`` and
``damp_vt(k) = vtdm4 = 0.12`` at EVERY level -- which is exactly the
uniform ``DUO_TAIL_CFG`` the tail phase already carries.  And ``tau =
-1`` fails ``tau > 0.`` at ``:378``, so Rayleigh never runs.  Both are
asserted by :func:`require_uniform_damping_lane` rather than left as
prose, because a deck with ``n_sponge >= 0`` would need per-level
``nord_v``/``damp_vt`` that the tail phase cannot express.

WHY THE pt CONVERSION TOUCHES ONLY THE COMPUTE WINDOW
-----------------------------------------------------
``fv_dynamics.F90:398-400`` loops ``i=is,ie`` / ``j=js,je``.  The halo
rows therefore still hold TEMPERATURE when ``dyn_core`` is entered, and
are only made consistent by the ``it == 1`` scalar exchange
(``dyn_core.F90:470``), which overwrites them with neighbours' theta.
Converting the padded array instead would put theta into rows that the
exchange then overwrites -- harmless here, but it would also convert the
corner-diagonal sentinels, and those are NOT overwritten.  Mirror the
Fortran window.

TRACERS ARE NOT ADVECTED, AND THAT IS A DECLARED SCOPE LIMIT
------------------------------------------------------------
``fv_tracer2d`` is not ported.  On the pinned deck that changes NOTHING
in ``u``/``v``/``pt``/``delp``, and the reason is checkable rather than
asserted:

* ``adiabatic = .true.`` => ``zvir = 0`` (``driver/solo/atmosphere.F90``),
  so ``dp1 = zvir*q = 0`` at ``:291`` and the closing
  ``pt/(1 + r_vir*q(sphum))`` at ``fv_mapz.F90:975`` is an identity;
* ``consv_te = 0.`` gates out the total-energy fixer
  (``fv_mapz.F90:628``), the only other place ``q`` reaches ``pt``;
* ``fill = .F.`` and ``do_sat_adj = .F.`` gate out ``fillz`` and the
  saturation adjustment, which are the only places ``q`` reaches
  ``delp``.

So ``q`` is a passenger.  :func:`fv_dynamics_step` still REQUIRES the
tracer list, because the remap must make the same ``nr`` passes through
``fv_mapz.F90:330-342`` the oracle makes, and it refuses to run with a
non-zero ``zvir`` or ``consv`` where the passenger argument would stop
being true.

OMEGA IS AN OUTPUT-ONLY PASSENGER TOO
-------------------------------------
``omga`` is read at ``fv_mapz.F90:434-436`` and written at ``:504-523``
and reaches no other field (verified against the port:
``fv3_native_mapz.py:959`` reads it into ``pe3_om`` and ``:979`` writes
it back).  ``dyn_core``'s own ``omga`` fill (``:1636-1660``,
``use_old_omega = .T.``) is therefore not needed for ``u``/``v``/``pt``/
``delp`` parity and is NOT ported.  The array this module allocates is
returned so a caller can see it, but its values are meaningless and
:func:`fv_dynamics_step` says so in the returned bundle.
"""
from __future__ import annotations

import numpy as np

from legoesm.core.fv3_native_acoustic_3d import acoustic_loop_3d
from legoesm.core.fv3_native_mapz import lagrangian_to_eulerian
from legoesm.core.fv3_native_state_3d import field_shape

# fv_grid_utils.F90:56 -- `real, parameter:: ptop_min = 1.d-8`.
PTOP_MIN = 1.0e-8

# fv_dynamics.F90:568. Below this the oracle itself never remaps, so the
# driver must not either -- skipping is faithful, not a shortcut.
REMAP_MIN_NPZ = 4


def require_uniform_damping_lane(*, n_sponge: int, tau: float,
                                 npz: int) -> None:
    """Refuse a deck whose damping is NOT uniform in the vertical.

    ``dyn_core.F90:772`` --
    ``if ( npz==1 .or. flagstruct%n_sponge<0 ) then d2_divg = d2_bg``
    -- is the arm the pinned deck takes (``n_sponge = -1``).  The other
    arm (``:775-804``) gives ``k = 1, 2, 3`` their own ``nord_k``,
    ``d2_divg``, ``nord_w``, ``damp_w``, ``nord_v(k)`` and
    ``damp_vt(k)``.  ``fv3_native_dsw_tail_3d`` applies ONE ``nord_v`` /
    ``damp_v`` to every level, so on a sponge deck it would silently run
    level 1 with interior damping.

    ``tau > 0.`` at ``:378`` switches on ``Rayleigh_Super``, which is not
    ported at all.
    """
    if not (npz == 1 or n_sponge < 0):
        raise NotImplementedError(
            f"n_sponge={n_sponge} >= 0 with npz={npz} selects the sponge arm "
            f"at dyn_core.F90:775-804, which gives k=1,2,3 their own nord_k/"
            f"d2_divg/nord_w/damp_w/nord_v/damp_vt. fv3_native_dsw_tail_3d "
            f"carries a single uniform (nord_v, damp_v) pair, so it would run "
            f"the sponge levels with interior damping. The pinned deck has "
            f"n_sponge = -1.")
    if tau > 0.0:
        raise NotImplementedError(
            f"tau={tau} > 0 switches on Rayleigh_Super at fv_dynamics.F90:384, "
            f"which is not ported. The pinned deck has tau = -1.")


def p_var_hydrostatic(delp: np.ndarray, *, ptop: float, akap: float,
                      n: int, ng: int, km: int) -> dict:
    """``p_var``'s hydrostatic branch (``tools/init_hydro.F90:41-133``).

    Given ``(ptop, delp)`` returns ``ps, pe, peln, pk, pkz`` on the
    layouts of ``fv3_native_state_3d.field_shape``.  This is what
    ``fv_restart.F90:526-533`` calls once at initialisation, and it is
    the ONLY producer of the ``pkz`` that ``fv_dynamics.F90:402`` divides
    by on the first step -- the hydrostatic branch of ``:281-294`` writes
    ``dp1`` and nothing else, so a caller that skips this converts ``pt``
    with an uninitialised ``pkz``.

    ``adjust_dry_mass = .F.`` on the pinned deck, so ``drymadj`` and the
    ``ratio`` rescale (``:76-88``) are absent by configuration, not by
    omission.
    """
    delp = np.asarray(delp, dtype=np.float64)
    want = field_shape("delp", n, ng, km)
    if delp.shape != want:
        raise ValueError(f"p_var_hydrostatic: delp must be {want}, got "
                         f"{delp.shape}")
    if ptop <= 0.0:
        raise ValueError(f"p_var_hydrostatic: ptop must be > 0, got {ptop}")

    ia = ng                                   # Fortran i=is=1 sits at [ng]
    win = delp[ia:ia + n, ia:ia + n, :]       # (i, j, k) over is..ie, js..je

    pk = np.zeros(field_shape("pk", n, ng, km), dtype=np.float64)
    pe = np.zeros(field_shape("pe", n, ng, km), dtype=np.float64)
    peln = np.zeros(field_shape("peln", n, ng, km), dtype=np.float64)
    pkz = np.zeros(field_shape("pkz", n, ng, km), dtype=np.float64)
    ps = np.zeros((n + 2 * ng, n + 2 * ng), dtype=np.float64)

    # :80-83  pe(i,1,j) = ptop ; pk(i,j,1) = ptop**cappa
    pek = ptop ** akap
    pe[1:n + 1, 0, 1:n + 1] = ptop
    pk[ia:ia + n, ia:ia + n, 0] = pek

    # :102-108  running sum down the column, then peln and pk from it.
    # pe's i/j origin is is-1 = 0, so Fortran i=is sits at pe[1].
    acc = np.full((n, n), ptop, dtype=np.float64)
    for k in range(1, km + 1):
        acc = acc + win[:, :, k - 1]
        lnp = np.log(acc)
        pe[1:n + 1, k, 1:n + 1] = acc
        peln[:, k, :] = lnp
        pk[ia:ia + n, ia:ia + n, k] = np.exp(akap * lnp)

    # :110-112  ps = pe(i,km+1,j)
    ps[ia:ia + n, ia:ia + n] = pe[1:n + 1, km, 1:n + 1]

    # :114-125  peln(:,1,:).  ptop > ptop_min on every deck we run, so the
    # `ak1` small-ptop correction is the DEAD arm -- implemented anyway so
    # a low-top deck is refused rather than silently mis-seeded.
    if ptop < PTOP_MIN:
        ak1 = (akap + 1.0) / akap
        peln[:, 0, :] = peln[:, 1, :] - ak1
    else:
        peln[:, 0, :] = np.log(ptop)

    # :127-133  hydrostatic pkz
    pkz[:, :, :] = ((pk[ia:ia + n, ia:ia + n, 1:]
                     - pk[ia:ia + n, ia:ia + n, :-1])
                    / (akap * (peln[:, 1:, :] - peln[:, :-1, :])
                       ).transpose(0, 2, 1))
    return {"ps": ps, "pe": pe, "peln": peln, "pk": pk, "pkz": pkz}


def pt_to_theta_v(pt: np.ndarray, pkz: np.ndarray, *, n: int, ng: int,
                  dp1: np.ndarray | None = None) -> None:
    """``fv_dynamics.F90:396-408``, in place, COMPUTE WINDOW ONLY.

    ``pt = pt*(1 + dp1)/pkz`` with ``dp1 = zvir*q(sphum)``.  ``dp1=None``
    means the adiabatic ``zvir = 0`` lane, where the factor is exactly
    ``1`` -- written as a separate branch rather than multiplying by a
    zeros array so the adiabatic lane is bit-identical to ``pt/pkz``.
    """
    ia = ng
    if dp1 is None:
        pt[ia:ia + n, ia:ia + n, :] /= pkz
    else:
        pt[ia:ia + n, ia:ia + n, :] *= (1.0 + dp1) / pkz


def fv_dynamics_step(ctx: dict, state: list, press: list, *,
                     bdt: float, km: int, k_split: int, n_split: int,
                     ptop: float, ak, bk, akap: float, cp_air: float,
                     kord_mt: int, kord_tm: int, kord_tr,
                     q: list, omga: list | None = None,
                     zvir: float = 0.0, consv_te: float = 0.0,
                     sphum_index: int | None = None,
                     n_sponge: int = -1, tau: float = -1.0,
                     cfg: dict | None = None, a2b_ord: int = 4,
                     validate: bool = True) -> dict:
    """One ``fv_dynamics`` call: ``bdt`` of model time (``:451-674``).

    ``state`` is the six-face prognostic bundle from
    ``fv3_native_state_3d`` (``delp``, ``pt``, ``w``, ``u``, ``v``);
    ``press`` is six per-face dicts of ``ps``/``pe``/``peln``/``pk``/
    ``pkz`` as :func:`p_var_hydrostatic` builds them.  BOTH are mutated
    in place, exactly as the Fortran's ``intent(inout)`` dummies are.

    ``pt`` enters as TEMPERATURE and leaves as TEMPERATURE: this routine
    owns the round trip (``:396-408`` in, ``fv_mapz.F90:209-217`` out).
    A caller that stops between the two gets ``theta_v``.

    ``q`` is six lists of tracer arrays.  It is required -- see the module
    docstring for why the pinned deck's tracers are passengers and why
    that must stay a visible choice rather than a default.
    """
    if k_split < 1:
        raise ValueError(f"k_split must be >= 1, got {k_split}")
    if n_split < 1:
        raise ValueError(f"n_split must be >= 1, got {n_split}")
    if len(state) != 6 or len(press) != 6 or len(q) != 6:
        raise ValueError(
            f"state/press/q must each have 6 faces, got "
            f"{len(state)}/{len(press)}/{len(q)}")
    require_uniform_damping_lane(n_sponge=n_sponge, tau=tau, npz=km)

    if zvir != 0.0:
        raise NotImplementedError(
            "zvir != 0 makes the tracers stop being passengers: dp1 = "
            "zvir*q(sphum) enters the pt->theta_v conversion "
            "(fv_dynamics.F90:291, :402) and the closing pt/(1+r_vir*q) "
            "(fv_mapz.F90:975), and fv_tracer2d -- which would have advected "
            "that q across the step -- is NOT ported. The pinned deck is "
            "adiabatic, so zvir = 0.")
    if consv_te != 0.0:
        raise NotImplementedError(
            "consv_te != 0 activates the total-energy fixer "
            "(fv_mapz.F90:628-747), which is not ported. The pinned deck has "
            "consv_te = 0.")

    n, ng = ctx["n"], ctx["ng"]
    if omga is None:
        omga = [np.zeros(field_shape("delp", n, ng, km), dtype=np.float64)
                for _ in range(6)]

    # :413  mdt = bdt / k_split
    mdt = bdt / float(k_split)

    # :396-408  T -> theta_v, once per fv_dynamics call, on every face.
    for t in range(6):
        pt_to_theta_v(state[t]["pt"], press[t]["pkz"], n=n, ng=ng)

    remapped = km > REMAP_MIN_NPZ
    for n_map in range(1, k_split + 1):
        last_step = (n_map == k_split)

        # :502  dyn_core.  press_out receives the remap-step geopk bundle.
        press_out: list = []
        acoustic_loop_3d(ctx, state, mdt, km, n_split=n_split, ptop=ptop,
                         akap=akap, cp_air=cp_air, cfg=cfg,
                         validate=validate, remap_follows=remapped,
                         press_out=press_out)
        if len(press_out) != 6:
            raise RuntimeError(
                "dyn_core did not return the remap-step pressure bundle; "
                "acoustic_loop_3d must fill press_out on it == n_split")

        # :528-540  tracer_2d -- see the module docstring. Nothing to do.

        if not remapped:
            # :568 gates the remap on npz > 4 and the oracle leaves pt in
            # theta_v below it. Say so instead of returning a state whose
            # pt silently means something else than the docstring claims.
            continue

        for t in range(6):
            g = press_out[t]
            if "pk_remap" not in g:
                raise RuntimeError(
                    f"face {t + 1}: the remap-step geopk bundle has no "
                    f"'pk_remap'. dyn_core.F90:1511-1519 saves pk BEFORE "
                    f"one_grad_p overwrites pkc with B-grid corner values, "
                    f"so remapping against g['pk'] would use the wrong "
                    f"staggering.")
            # dyn_core writes pe/peln/pkz through its dummies and copies
            # pk over the compute window; mirror both onto the carried
            # bundle so the next step's p_var-equivalent state is current.
            press[t]["pe"][:] = g["pe"]
            press[t]["peln"][:] = g["peln"]
            press[t]["pkz"][:] = g["pkz"]
            ia = ng
            press[t]["pk"][ia:ia + n, ia:ia + n, :] = \
                g["pk_remap"][ia:ia + n, ia:ia + n, :]

            lagrangian_to_eulerian(
                pe=press[t]["pe"], peln=press[t]["peln"],
                pk=press[t]["pk"], pkz=press[t]["pkz"],
                delp=state[t]["delp"], pt=state[t]["pt"],
                u=state[t]["u"], v=state[t]["v"], ps=press[t]["ps"],
                ak=ak, bk=bk, ptop=ptop, akap=akap, cp=cp_air,
                r_vir=zvir, km=km, n=n, ng=ng,
                kord_mt=kord_mt, kord_tm=kord_tm, kord_tr=kord_tr,
                q=q[t], omga=omga[t], sphum_index=sphum_index,
                last_step=last_step, hydrostatic=True, adiabatic=True,
                consv=consv_te, fill=False, do_sat_adj=False,
                do_inline_mp=False, do_adiabatic_init=False)

    return {"state": state, "press": press, "q": q,
            "omga": omga, "omga_is_meaningless": True,
            "pt_units": "K" if remapped else "theta_v"}
