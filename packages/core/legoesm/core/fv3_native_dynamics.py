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

TRACERS ARE ADVECTED (tracer_2d_1L) AND REMAPPED
------------------------------------------------
``fv_tracer2d``'s live arm for this deck -- ``tracer_2d_1L``
(``z_tracer = .T.``, duo) -- is ported in
``fv3_native_tracer2d`` and called at the oracle's site
(``fv_dynamics.F90:534``, between ``dyn_core`` and the remap) on the
mfx/mfy/cx/cy flux capacitors the acoustic loop accumulates
(``dyn_core.F90:313`` zeroing + ``sw_core.F90:903-920`` per-sub-step
accumulation, threaded through ``dsw_transport_phase_3d``).  ``dp1``
is the pre-dyn_core ``delp`` (``:472-478``), copied per ``n_map``.
The remap then makes its ``nr`` tracer passes through
``fv_mapz.F90:330-342`` as before.

MOIST FEEDBACK IS ENABLED ON BOTH ARMS (hydrostatic 2026-08-19,
non-hydrostatic 2026-08-20).  ``zvir != 0`` forms
``dp1 = zvir*q(sphum)`` (``:291``) once before the k_split loop and
couples it into ``pt*(1+dp1)/pkz`` (``:402``), with the closing
``pt/(1 + r_vir*q)`` at ``fv_mapz.F90:975`` consuming the POST-transport
humidity.  On the NON-hydrostatic arm it ALSO enters the ``pkz`` this
module recomputes at ``:299-322`` --
``exp(kappa*log(rdg*delp*pt*(1.+dp1)/delz))``, the factor INSIDE the log
-- which is why that pkz is recomputed from the step-entry state rather
than taken from the caller.  On the PINNED (adiabatic) deck zvir is 0
and the tracers remain passengers, which is why the certified 1.1866e-09
parity is unaffected.  Still refused, and gated by a behavioural test:
``consv_te != 0`` (the total-energy fixer, ``fv_mapz.F90:628-747``).

The oracle's ``moist_phys = .false.`` NH arm (``:335``) forces
``dp1 = 0`` and takes the DRY pkz, so it is not a third lane: it is
exactly the ``zvir = 0`` lane, and needs no flag here.  ``moist_phys``
reaches nothing else this lane models -- its only other uses are the
refused ``consv_te`` energy path (``:364``), a ``fv_mapz`` argument the
oracle itself marks ``not used`` (``fv_mapz.F90:129``), and the
``#ifdef FILL2D`` block at ``:546``, which touches only the condensate
species (liq_wat/rainwat/ice_wat/snowwat/graupel), each guarded on
``> 0`` and all absent from this sphum-only lane.

``adiabatic`` likewise reaches exactly ONE branch, ``fv_mapz.F90:985``
(``flagstruct%adiabatic`` is passed at ``:627`` and used nowhere else),
which is why threading it honestly on the moist NH arm changes nothing
else -- and why the moist HYDROSTATIC arm can keep passing
``adiabatic=True`` where the oracle deck resolves ``.false.``: ``:975``
is ungated, which the mapz gate's hydrostatic control checks.

THAT SINGLE-BRANCH CLAIM IS ENFORCED BY NOTHING, so re-run the grep
before adding any ``adiabatic``-keyed branch to either lane, or before
reading a new one out of the oracle::

    grep -n adiabatic <oracle>/model/fv_mapz.F90   # :985 only, besides
                                                   # the declarations
    grep -rn adiabatic packages/core/legoesm/core/fv3*mapz.py

A second gated branch would make ``adiabatic_flag = hydrostatic or
zvir == 0.0`` silently wrong on the moist hydrostatic arm.

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

import operator as _operator

import numpy as np

from legoesm.core.fv3_native_acoustic_3d import acoustic_loop_3d
from legoesm.core.fv3_native_mapz import lagrangian_to_eulerian
from legoesm.core.fv3_native_state_3d import field_shape
from legoesm.grids.fv3_native_gridstruct import (
    FV3_GRAV as _FV3_GRAV,
    FV3_RDGAS as _FV3_RDGAS,
)

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
    ps = np.zeros(field_shape("ps", n, ng, km), dtype=np.float64)

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
    ``1``.  It is a separate branch because the ORACLE forms no ``dp1``
    at all when ``zvir = 0`` (``:281-294`` is the moist branch), not
    because a zeros array would round differently: since the
    association was corrected to ``(pt*(1+dp1))/pkz`` a zeros array is
    bit-identical (``win*1.0`` is exact).  Gated by
    ``test_dry_lane_is_bitwise_under_the_moist_patch``.
    """
    ia = ng
    if dp1 is None:
        pt[ia:ia + n, ia:ia + n, :] /= pkz
    else:
        # ASSOCIATION IS THE ORACLE'S, not convenience: Fortran evaluates
        # `pt*(1.+dp1)/pkz` (:402) left to right as (pt*(1+dp1))/pkz.
        # `pt *= (1.0 + dp1) / pkz` instead forms the quotient FIRST and
        # multiplies -- a different rounding, and the moist arm had never
        # been exercised to catch it (the adiabatic lane takes the branch
        # above). Found by test_moist_arm_matches_the_oracle_expression.
        win = pt[ia:ia + n, ia:ia + n, :]
        pt[ia:ia + n, ia:ia + n, :] = win * (1.0 + dp1) / pkz


def p_var_nonhydrostatic(delp: np.ndarray, delz: np.ndarray,
                         pt: np.ndarray, *, ptop: float, akap: float,
                         n: int, ng: int, km: int,
                         dp1: np.ndarray | None = None) -> dict:
    """``p_var``'s NON-hydrostatic pkz on top of the hydrostatic column.

    ``init_hydro.F90:95-133`` builds ps/pe/peln/pk identically on both
    lanes; only ``pkz`` differs -- the NH branch (:178-184, dry) is

        pkz = exp( cappa * log( rdg*delp*pt/delz ) ),  rdg = -rdgas/grav

    with ``pt`` still TEMPERATURE at this stage (the theta conversion
    happens later in fv_dynamics).  Feeding the HYDROSTATIC kappa-mean
    pkz into an NH run's pt -> theta_v conversion puts a uniform
    ~kappa(1-kappa)/24 * dlnp^2 error on theta (largest in the thickest
    log-layer), which surfaced in the first NH parity as a 0.408 m delz
    residual on every column of every face.

    ``dp1`` selects the MOIST arm (``fv_dynamics.F90:307-309``, the
    ``moist_phys`` branch), where the log argument carries an extra
    ``(1.+dp1)`` -- INSIDE the log, in the Fortran's own left-to-right
    association.  Pre-scaling ``pt`` at the call site instead would
    reassociate the product.  A separate branch, matching the oracle's
    own ``moist_phys`` split, so the certified dry NH lane keeps its
    exact expression.  The
    oracle's ``moist_phys = .false.`` arm (``:335``) forces ``dp1 = 0``
    and takes the dry form, so it IS the ``dp1=None`` branch here and
    needs no flag of its own.
    """
    out = p_var_hydrostatic(delp, ptop=ptop, akap=akap, n=n, ng=ng, km=km)
    rdg = -_FV3_RDGAS / _FV3_GRAV
    ia = ng
    dpw = delp[ia:ia + n, ia:ia + n, :]
    ptw = pt[ia:ia + n, ia:ia + n, :]
    out["pkz"][:] = np.exp(akap * np.log(
        nh_pkz_log_arg(rdg, dpw, ptw, delz, dp1)))
    return out


def nh_pkz_log_arg(rdg, dpw, ptw, delz, dp1=None):
    """The NH ``pkz`` log argument: ``rdg*delp*pt*(1.+dp1)/delz``.

    FACTORED OUT SO THE ASSOCIATION CAN BE GATED BITWISE.  ``pkz``
    itself ends in ``exp(kappa*log(...))``, and XLA's ``exp`` differs
    from libm's by ~1 ulp, so a cross-lane bitwise check on ``pkz``
    is not available -- and a tolerance loose enough to survive that
    also admits the very mistake the association is guarding against:
    pre-scaling ``pt`` at the CALL SITE (``rdg*delp*(pt*(1+dp1))/delz``)
    is algebraically identical and differs only by multiplication
    rounding, well inside any exp/log-sized window (codex MAJOR, job
    9442717).  Below the transcendentals the two trees differ by real
    bits, so the gate can be exact.

    Association is the Fortran's, left to right
    (``fv_dynamics.F90:314-315``): ``((rdg*delp)*pt)*(1+dp1)/delz`` on
    the moist arm, ``((rdg*delp)*pt)/delz`` on the dry one.
    """
    if dp1 is None:
        return rdg * dpw * ptw / delz
    return rdg * dpw * ptw * (1.0 + dp1) / delz


def fv_dynamics_step(ctx: dict, state: list, press: list, *,
                     bdt: float, km: int, k_split: int, n_split: int,
                     ptop: float, ak, bk, akap: float, cp_air: float,
                     kord_mt: int, kord_tm: int, kord_tr,
                     q: list, omga: list | None = None,
                     zvir: float = 0.0, consv_te: float = 0.0,
                     sphum_index: int | None = None,
                     n_sponge: int = -1, tau: float = -1.0,
                     hydrostatic: bool = True,
                     p_fac: float = 0.05, a_imp: float = 1.0,
                     use_logp: bool = False, kord_wz: int = 9,
                     w_limiter: bool | None = None,
                     cfg: dict | None = None, a2b_ord: int = 4,
                     validate: bool = True,
                     hord_tr: int = 6, tracer_q_split: int = 0,
                     nord_tr: int = 0, trdm2: float = 0.0,
                     lim_fac: float = 1.0, z_tracer: bool = True,
                     inline_q: bool = False) -> dict:
    """One ``fv_dynamics`` call: ``bdt`` of model time (``:451-674``).

    ``state`` is the six-face prognostic bundle from
    ``fv3_native_state_3d`` (``delp``, ``pt``, ``w``, ``u``, ``v``);
    ``press`` is six per-face dicts of ``ps``/``pe``/``peln``/``pk``/
    ``pkz`` as :func:`p_var_hydrostatic` builds them.  BOTH are mutated
    in place, exactly as the Fortran's ``intent(inout)`` dummies are.

    THE RETURNED ``press["pkz"]`` IS DRY, ALWAYS.  The remap writes it
    from the post-remap state with no ``(1+dp1)`` (``fv_mapz.F90:
    479-481``), and on the moist NH arm the NEXT call overwrites it at
    ``fv_dynamics.F90:299-322``.  That is exactly what the oracle
    exposes between calls, so it is not a divergence -- but a consumer
    that stops BETWEEN steps and assumes "moist run implies moist pkz"
    would be wrong, which is why it is stated here rather than only in
    a test (GLM MINOR, job 9442483).

    ``pt`` enters as TEMPERATURE and leaves as TEMPERATURE: this routine
    owns the round trip (``:396-408`` in, ``fv_mapz.F90:209-217`` out).
    A caller that stops between the two gets ``theta_v``.

    ``q`` is six lists of tracer arrays (each shaped like ``delp``).
    They are ADVECTED by ``tracer_2d_1L`` between the acoustic loop and
    the remap, then remapped -- see the module docstring.  The tracer
    flags default to the resolved parity deck (``HORD_TR=6 Q_SPLIT=0
    NORD_TR=0 TRDM2=0 LIM_FAC=1 Z_TRACER=T INLINE_Q=F``, from the
    logfile echo); any unported arm raises in
    ``fv3_native_tracer2d.require_tracer_2d_1l_lane``.
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
        # HYDROSTATIC moist coupling is enabled; consv_te stays refused
        # (separate phase). dp1 = zvir*q(i,j,k,sphum) (fv_dynamics.F90:291;
        # USE_COND is NOT defined in this build, so no q_con term) feeds
        # pt = pt*(1.+dp1)/pkz (:402, ported as pt_to_theta_v) and the
        # closing pt/(1+r_vir*q) (fv_mapz.F90:975, already ported in
        # lagrangian_to_eulerian).  Validate, never default: a guessed
        # tracer index would silently couple an arbitrary species.
        if q is None:
            raise ValueError(
                "zvir != 0 requires tracer arrays, but q is None: "
                "dp1 = zvir*q(sphum) (fv_dynamics.F90:291) has no specific "
                "humidity to read.")
        # operator.index NORMALISES; hasattr alone does not. A object
        # that implements only __index__ passed the old check and then
        # died on the bounds COMPARISON with an incidental TypeError
        # (codex MINOR, job 9442482). bool is excluded by name because
        # it has __index__ too, and True would select tracer 1.
        if isinstance(sphum_index, bool):
            raise ValueError(
                f"zvir != 0 requires sphum_index to be an integer index "
                f"into q; got the bool {sphum_index!r}, which would "
                f"silently select tracer {int(sphum_index)}.")
        try:
            sphum_index = _operator.index(sphum_index)
        except TypeError:
            raise ValueError(
                f"zvir != 0 requires sphum_index to be an integer "
                f"indexing the specific-humidity tracer in q; got "
                f"{sphum_index!r}. A guessed index would silently couple "
                f"the wrong species into theta_v.") from None
        for _t, _qf in enumerate(q):
            if len(_qf) <= 0:
                raise ValueError(
                    f"zvir != 0 requires nq > 0, but face {_t} carries no "
                    f"tracers (fv_dynamics.F90:291 needs sphum).")
            if not 0 <= sphum_index < len(_qf):
                raise ValueError(
                    f"sphum_index={sphum_index} out of range "
                    f"[0, {len(_qf)}) for face {_t}; a negative index would "
                    f"silently select another tracer by Python wrap-around.")
        if not hydrostatic and any("delz" not in f for f in state):
            raise ValueError(
                "zvir != 0 with non-hydrostatic dynamics needs delz on "
                "every face: the moist NH pkz is recomputed here from "
                "delp/pt/delz (fv_dynamics.F90:299-322).")
    if consv_te != 0.0:
        raise NotImplementedError(
            "consv_te != 0 activates the total-energy fixer "
            "(fv_mapz.F90:628-747), which is not ported. The pinned deck has "
            "consv_te = 0.")

    n, ng = ctx["n"], ctx["ng"]
    nq = len(q[0])
    if any(len(qt) != nq for qt in q):
        raise ValueError(
            f"every face must carry the same tracer count; got "
            f"{[len(qt) for qt in q]}")
    from legoesm.core.fv3_native_tracer2d import (
        alloc_flux_capacitors,
        require_tracer_2d_1l_lane,
        tracer_2d_1l_sixface,
    )
    if nq > 0:
        # Fail on an unported tracer arm BEFORE the acoustic loop, not
        # after k_split*n_split sub-steps of work.
        require_tracer_2d_1l_lane(z_tracer=z_tracer,
                                  q_split=tracer_q_split,
                                  nord_tr=nord_tr, trdm=trdm2,
                                  inline_q=inline_q)
    if omga is None:
        omga = [np.zeros(field_shape("delp", n, ng, km), dtype=np.float64)
                for _ in range(6)]

    ak = np.asarray(ak, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    # dyn_core.F90:269 -- dp_ref(k) = ak(k+1)-ak(k) + (bk(k+1)-bk(k))*1.E5
    # (1.E5 is the literal reference surface pressure of the hybrid
    # coordinate, not a tunable).
    dp0 = ((ak[1:] - ak[:-1])
           + (bk[1:] - bk[:-1]) * 1.0e5)  # const-ok: dyn_core.F90:269 literal
    if not hydrostatic:
        for t, face in enumerate(state, start=1):
            if "delz" not in face:
                raise ValueError(
                    f"face {t}: hydrostatic=False needs delz in the state "
                    f"(build_state_3d(hydrostatic=False))")
        if ctx.get("hs6") is None:
            raise ValueError(
                "hydrostatic=False needs ctx['hs6'] (phis) for the NH "
                "carry (zs = phis/grav, dyn_core.F90:262-278)")
        if w_limiter is None:
            # codex NH r3 #3: a silent w_limiter default on the NH lane
            # contradicts the resolved deck (W_LIMITER=T) without an
            # error -- an unclamped 200 m/s column is a different model.
            raise ValueError(
                "hydrostatic=False needs an explicit w_limiter (the "
                "resolved NH deck runs W_LIMITER=T; fv_mapz.F90:368)")
    w_limiter = bool(w_limiter) if w_limiter is not None else False

    # :413  mdt = bdt / k_split
    mdt = bdt / float(k_split)

    # :396-408  T -> theta_v, once per fv_dynamics call, on every face.
    for t in range(6):
        if zvir != 0.0:
            # dp1 = zvir*q(i,j,k,sphum) (fv_dynamics.F90:291), formed once
            # per face BEFORE the k_split loop -- from the step-initial q,
            # exactly as :281-294 precedes :451. Sliced to the COMPUTE
            # WINDOW because pkz is (n, n, km) and pt_to_theta_v expects
            # dp1 already matching it.
            dp1 = zvir * q[t][sphum_index][ng:ng + n, ng:ng + n, :]
            if not hydrostatic:
                # :299-322 is INSIDE fv_dynamics and runs on every call
                # on BOTH NH arms -- but only the MOIST one is recomputed
                # here. The dry NH arm still trusts the caller's pkz,
                # which is covered by the certified dry parity and is
                # deliberately left alone; do not read this comment as
                # licence to recompute there (GLM, job 9442483). Moist:
                # on the NH moist arm it OVERWRITES pkz with
                # exp(kappa*log(rdg*delp*pt*(1.+dp1)/delz)) from the
                # step-entry state, while pt is still TEMPERATURE. The
                # caller's pkz -- built dry by p_var_nonhydrostatic, or
                # carried over from the previous step's remap -- is
                # missing the virtual factor, so it is recomputed here
                # rather than trusted. The dry NH lane never enters this
                # branch and keeps its certified expression untouched.
                press[t]["pkz"][:] = p_var_nonhydrostatic(
                    state[t]["delp"], state[t]["delz"], state[t]["pt"],
                    ptop=ptop, akap=akap, n=n, ng=ng, km=km,
                    dp1=dp1)["pkz"]
            pt_to_theta_v(state[t]["pt"], press[t]["pkz"], n=n, ng=ng,
                          dp1=dp1)
        else:
            # dp1=None is the ORACLE's shape: fv_dynamics.F90:281-294
            # forms no dp1 at all when zvir = 0, so there is nothing to
            # multiply by. NOT a rounding argument -- an earlier version
            # of this comment claimed a zeros array 'rounds twice', which
            # was true of the OLD association `(1.+dp1)/pkz` and became
            # FALSE when it was corrected: 1.0+0.0 is exactly 1.0 and
            # win*1.0 is exact, so zeros is now bit-identical. Measured,
            # job 9442478. The branch stays because it is the oracle's
            # structure and skips a whole-field multiply.
            pt_to_theta_v(state[t]["pt"], press[t]["pkz"], n=n, ng=ng)

    remapped = km > REMAP_MIN_NPZ
    # fv_mapz.F90:985: on the NH arm the closing T_v -> T conversion is
    # inside `if (.not. adiabatic)`.  The certified deck is adiabatic AND
    # dry, so True was unconditionally right; with moist NH it is not --
    # adiabatic=True there would make the oracle skip the conversion and
    # leave pt virtual.  With consv = 0, dtmp is identically 0 and the
    # oracle's :987 expression is exactly the one this lane computes, so
    # False is the faithful flag for the moist NH arm.  The remap refuses
    # the other combination rather than trusting this line.
    adiabatic_flag = hydrostatic or zvir == 0.0
    for n_map in range(1, k_split + 1):
        last_step = (n_map == k_split)

        # :472-478  dp1 = delp, full padded box, BEFORE dyn_core.
        # :313-316 (dyn_core)  fresh zeroed flux capacitors per call.
        dp1_6 = [np.array(state[t]["delp"], copy=True) for t in range(6)]
        flux_cap = alloc_flux_capacitors(n, ng, km) if nq > 0 else None

        # :502  dyn_core.  press_out receives the remap-step geopk bundle.
        press_out: list = []
        acoustic_loop_3d(ctx, state, mdt, km, n_split=n_split, ptop=ptop,
                         akap=akap, cp_air=cp_air, cfg=cfg,
                         validate=validate, remap_follows=remapped,
                         hydrostatic=hydrostatic,
                         p_fac=p_fac, a_imp=a_imp, dp0=dp0,
                         use_logp=use_logp,
                         press_out=press_out,
                         flux_cap=flux_cap)
        if len(press_out) != 6:
            raise RuntimeError(
                "dyn_core did not return the remap-step pressure bundle; "
                "acoustic_loop_3d must fill press_out on it == n_split")

        # :517/:528-540  tracer_2d_1L on the accumulated capacitors
        # (`if (.not. inline_q .and. nq /= 0)`; inline_q is refused
        # above, so the gate is just nq).
        if nq > 0:
            tracer_2d_1l_sixface(ctx, q, dp1_6, flux_cap,
                                 km=km, nq=nq, hord_tr=hord_tr, dt=mdt,
                                 q_split=tracer_q_split, nord_tr=nord_tr,
                                 trdm=trdm2, lim_fac=lim_fac,
                                 z_tracer=z_tracer, inline_q=inline_q)

        if not remapped:
            # :568 gates the remap on npz > 4 and the oracle leaves pt in
            # theta_v below it. Say so instead of returning a state whose
            # pt silently means something else than the docstring claims.
            #
            # BUT dyn_core still WROTE pe/peln/pkz through its dummies and
            # copied pk over the compute window (:1401, :1511-1519), and
            # the oracle keeps all of that. Discarding it here left the
            # carried bundle byte-identical to entry -- so a +1 Pa change
            # in level-1 delp would not move pe(:,:,2), and the NEXT
            # call's pt -> theta_v would divide by a stale pkz.
            for t in range(6):
                g = press_out[t]
                press[t]["pe"][:] = g["pe"]
                press[t]["peln"][:] = g["peln"]
                if hydrostatic:
                    press[t]["pkz"][:] = g["pkz"]
                    if "pk_remap" in g:
                        press[t]["pk"][ng:ng + n, ng:ng + n, :] = \
                            g["pk_remap"][ng:ng + n, ng:ng + n, :]
                else:
                    # The NH tail carries no pkz (mapz owns the NH pkz,
                    # :479-481, and below the remap gate it never runs);
                    # pk comes from Riem_Solver3's last_call copy.
                    press[t]["pk"][ng:ng + n, ng:ng + n, :] = g["pk"]
            continue

        for t in range(6):
            g = press_out[t]
            ia = ng
            if hydrostatic:
                if "pk_remap" not in g:
                    raise RuntimeError(
                        f"face {t + 1}: the remap-step geopk bundle has no "
                        f"'pk_remap'. dyn_core.F90:1511-1519 saves pk BEFORE "
                        f"one_grad_p overwrites pkc with B-grid corner "
                        f"values, so remapping against g['pk'] would use "
                        f"the wrong staggering.")
                # dyn_core writes pe/peln/pkz through its dummies and copies
                # pk over the compute window; mirror both onto the carried
                # bundle so the next step's p_var-equivalent state is
                # current.
                press[t]["pe"][:] = g["pe"]
                press[t]["peln"][:] = g["peln"]
                press[t]["pkz"][:] = g["pkz"]
                press[t]["pk"][ia:ia + n, ia:ia + n, :] = \
                    g["pk_remap"][ia:ia + n, ia:ia + n, :]
            else:
                # NH: Riem_Solver3's last_call wrote pe (+pe_halo ring),
                # pk and peln directly (nh_core.F90:164-172); there is no
                # pk_remap ambiguity because nh_p_grad scratches PKC, not
                # this pk.
                press[t]["pe"][:] = g["pe"]
                press[t]["peln"][:] = g["peln"]
                press[t]["pk"][ia:ia + n, ia:ia + n, :] = g["pk"]

            lagrangian_to_eulerian(
                pe=press[t]["pe"], peln=press[t]["peln"],
                pk=press[t]["pk"], pkz=press[t]["pkz"],
                delp=state[t]["delp"], pt=state[t]["pt"],
                u=state[t]["u"], v=state[t]["v"], ps=press[t]["ps"],
                ak=ak, bk=bk, ptop=ptop, akap=akap, cp=cp_air,
                r_vir=zvir, km=km, n=n, ng=ng,
                kord_mt=kord_mt, kord_tm=kord_tm, kord_tr=kord_tr,
                q=q[t], omga=omga[t], sphum_index=sphum_index,
                last_step=last_step, hydrostatic=hydrostatic,
                w=(None if hydrostatic else state[t]["w"]),
                delz=(None if hydrostatic else state[t]["delz"]),
                ws=(None if hydrostatic else g["ws"]),
                kord_wz=kord_wz, w_limiter=w_limiter,
                rdgas=(None if hydrostatic else _FV3_RDGAS),
                grav=(None if hydrostatic else _FV3_GRAV),
                adiabatic=adiabatic_flag,
                consv=consv_te, fill=False, do_sat_adj=False,
                do_inline_mp=False, do_adiabatic_init=False)

    return {"state": state, "press": press, "q": q,
            "omga": omga, "omga_is_meaningless": True,
            "pt_units": "K" if remapped else "theta_v"}
