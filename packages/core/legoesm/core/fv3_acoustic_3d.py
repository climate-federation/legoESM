"""3-D acoustic sub-step driver, km-general, six faces -- JAX lane (part A).

Functional, jit-compilable twin of ``legoesm.core.fv3_native_acoustic_3d``.
Part A: the entry exchanges, the NH carry builder, one acoustic sub-step
and their ``make_*_jit`` factories.  ``acoustic_loop_3d`` (the
``do it=1,n_split`` driver) is appended to this same file as part B.

ORACLE CADENCE (the spec's block, checksum-pinned tree; dyn_core.F90):

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
    :1102 d_sw4      :1107  d_sw5      :1256  d_sw6
    :1401 geopk   (D-grid)
    :1531 one_grad_p

The two duo BARRIERS act one level at a time with all six faces of that
level present; their extents (barrier 1 over i = is..ie / j = js..je,
barrier 2 over i = is..ie+1 / j = js..je+1, tile corners included) are
``legoesm.grids.fv3_duo_halos``'s -- this module never re-derives an
extent or an exchange, it calls the named halo routines.

CONVENTIONS inherited from the sibling 3-D phases: C1 ONE dict of
face-stacked arrays (face t = Fortran tile t+1, Fortran index 1-ng at
numpy 0, km at axis 2, so a stacked field is (6, i, j, km)); C3 ctx, km
and every deck constant or Python-branch selector are STATIC, dt is
DYNAMIC; C4 every array the spec mutates comes back in the returned dict
(the NH carry, delz, pkc, the D winds); C5 no data-dependent guard
survives jit -- the spec's per-substep validate becomes the static
``check_state`` keyword, which RAISES on a tracer rather than silently
skipping; C6 stage observation is by return, never a callback; C7 the
deck comes from the spec's own duo configuration, never restated here.

WHAT THIS MODULE DOES NOT DO: it introduces no numerics of its own --
every arithmetic operation lives in an already-ported phase or halo
module, and this driver owns only WHICH phase runs at WHICH sub-step,
what the entry exchanges refresh, and how the state and NH carry are
threaded; uc/vc are never carried (the C grid is diagnostic, rebuilt
from the D grid every sub-step, per fv_arrays.F90, so only delp, pt,
u, v -- and w/delz on NH -- survive a sub-step); d_con = 0.0 throughout
(the shipped duo decks set it so), hence the KE-to-heat pathway and its
heat_source allocation (dyn_core.F90:322-325, gated on d_con > 1.0E-5)
are inactive and d_con is not a parameter of this module; and the moist
/ tracer path is dead on this cadence -- nothing here reads moisture.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import Tracer
from legoesm.core.fv3_cgrid_phase_3d import (
    cgrid_nh_pressure_phase_3d,
    cgrid_pressure_phase_3d,
    csw_phase_3d,
)
from legoesm.core.fv3_dsw_phase_3d import dsw_transport_phase_3d
from legoesm.core.fv3_dsw_tail_3d import (
    dgrid_nh_pressure_phase_3d,
    dgrid_pressure_phase_3d,
    dsw_tail_phase_3d,
)

# ⛔ THESE THREE ARE NOT IN fv3_phase3d_common, and the first draft of
# this module imported them from there (it would have failed at import).
# `field_shape` and `require_no_remap_needed` are the NumPy lane's shape
# authority and remap guard; FV3_GRAV is the gridstruct's constant.  The
# repo rule that a shared helper has ONE home cuts both ways: it also
# means guessing the home is a defect, not a detail.
from legoesm.core.fv3_native_state_3d import (
    STATE_FIELDS,
    field_shape,
    require_no_remap_needed,
)
from legoesm.core.fv3_phase3d_common import (
    require_f64_jax,  # f64 entry gate, reads only static dtypes
    require_km,
    validate_stacked,
)
from legoesm.grids.fv3_duo_halos import (
    ext_scalar_sixface,
    ext_vector_dgrid_sixface,
)
from legoesm.grids.fv3_native_gridstruct import FV3_GRAV

__all__ = [
    "exchange_state_halos_3d",
    "build_nh_carry",
    "acoustic_substep_3d",
    "make_exchange_state_halos_3d_jit",
    "make_build_nh_carry_jit",
    "make_acoustic_substep_3d_jit",
]

# Orders of magnitude above anything a dry hydrostatic column can hold:
# tripping one means the integration diverged, not that a coefficient
# needs adjusting (the spec's own words for _SANE_MAX).
_SANE_MAX = {"delp": 1.0e6, "pt": 1.0e4, "u": 1.0e4, "v": 1.0e4, "w": 1.0e3}


def _duo_tables(ctx):
    """The DuoHaloTables the JAX ctx carries (csw_phase_3d's contract:
    ctx already carries the halo tables its callees need)."""
    # `ctx.tab`, which is what build_jax_duo_stepper_context names it and
    # what every sibling phase passes to the halo routines.  The first
    # draft looked for `ctx.duo_halos` -- a name nothing sets -- so this
    # guard fired on a perfectly good context (job 9417502).  The guard
    # itself is right and stays: it is the reason the wrong name was a
    # loud failure instead of a silent stale-corner exchange.
    tab = getattr(ctx, "tab", None)
    if tab is None:
        # Fail closed, mirroring the spec: the D-wind exchange at
        # dyn_core.F90:504 is ext_vector; this lane has NO interim
        # substitute to fall back to, so a context without the tables
        # must say so loudly, not get a stale-corner exchange silently.
        raise ValueError(
            "fv3_acoustic_3d: this lane exchanges the D winds with "
            "ext_vector (dyn_core.F90:504) and has no interim helper to "
            "substitute; the context must carry its DuoHaloTables.")
    return tab


def _exchange_scalar_stack(f6, tab, km):
    """Per-level A-grid scalar halo refresh on a (6, m, m, km) stack.

    Each level's exchange reads and writes only that level, so no
    iteration reads what an earlier one writes; the loop stays a Python
    loop over static km because the halo routine's contract -- and the
    spec's per-k call -- is per level.  The extent is the halo
    module's, never re-derived here.

    ⛔ ``ext_scalar_sixface``, NOT ``exchange_agrid_scalar_halos``.  The
    spec routes every one of these sites through ``_pad_scalars_6``,
    which uses the EXT bundle and falls back to the interim
    index-copy helper only when the context explicitly declares the
    substitution.  The two are not interchangeable: the interim helper
    leaves the CORNER DIAGONALS untouched, and that exact substitution
    is what produced the ``u = 1.17534e+11`` acoustic blow-up in this
    campaign's NumPy phase -- the 3-D lane had used the index-copy
    helper where the km=1 lane already used ext_scalar.  Measured here
    as a 37 % delp disagreement against the spec on the first composed
    run (job 9417519).
    """
    for k in range(km):
        f6 = f6.at[..., k].set(ext_scalar_sixface(f6[..., k], tab, "A"))
    return f6


def _exchange_dgrid_winds_stack(u6, v6, tab, km):
    """Per-level ``ext_vector`` duo exchange of the D winds (the k2e
    Lagrange exchange that replaces the WHOLE padded array, corner
    diagonals included -- dyn_core.F90:504).  Same independence argument
    as the scalar stack: levels do not interact."""
    for k in range(km):
        uk, vk = ext_vector_dgrid_sixface(u6[..., k], v6[..., k], tab)
        u6 = u6.at[..., k].set(uk)
        v6 = v6.at[..., k].set(vk)
    return u6, v6


def _check_sane_state(state: dict, label: str) -> None:
    """C5 port of the spec's post-substep validate: a data-dependent
    check cannot run under jit, so it RAISES on a tracer instead of
    silently skipping -- silent skipping is the failure mode this repo
    has been bitten by most often."""
    for name, hi in _SANE_MAX.items():
        if name not in state:
            continue
        for leaf in jax.tree_util.tree_leaves(state[name]):
            if isinstance(leaf, Tracer):
                raise ValueError(
                    f"acoustic_3d[{label}]: check_state=True but '{name}' "
                    "is a tracer -- the spec's per-substep validate is "
                    "data-dependent and cannot run under jit (C5). Run it "
                    "unjitted or pass check_state=False on purpose.")
            a = np.asarray(leaf)
            if not np.isfinite(a).all() or (np.abs(a) > hi).any():
                raise ValueError(
                    f"acoustic_3d[{label}]: '{name}' left the sane "
                    f"envelope (non-finite or |x| > {hi:g}) -- the "
                    "sub-step has diverged, this is not a tuning issue.")


def exchange_state_halos_3d(ctx, state: dict, km: int, *, scalars: bool,
                            winds: bool, w_field: bool = False) -> dict:
    """Entry duo halo exchanges, per level; returns a NEW state (C4).

    dyn_core.F90:470/471 exchanges delp and pt (``scalars`` -- true only
    on it == 1, the :465 gate); :504 exchanges the D winds every
    sub-step; :479-480 exchanges w (NH only, ``w_field``) every
    sub-step, AFTER the D winds, in that order.

    The D-wind exchange is ``ext_vector``, never an index-copy helper:
    on this lane ``ext_vector_dgrid_sixface`` is the only
    implementation, so the NumPy lane's fail-closed guard (against
    silently substituting the interim mpp-analog whose corner diagonals
    stay stale) has no analogue -- there is nothing to substitute, and
    a context without the halo tables fails closed in ``_duo_tables``.
    """
    require_f64_jax("exchange_state_halos_3d", state)
    tab = _duo_tables(ctx)
    out = dict(state)
    if scalars:
        for _nm in ("delp", "pt"):
            out[_nm] = _exchange_scalar_stack(out[_nm], tab, km)
    if winds:
        out["u"], out["v"] = _exchange_dgrid_winds_stack(
            out["u"], out["v"], tab, km)
    if w_field:
        out["w"] = _exchange_scalar_stack(out["w"], tab, km)
    return out


def build_nh_carry(ctx, km: int, hs6) -> dict:
    """Build the persistent NH carry one acoustic loop threads across
    sub-steps: ONE face-stacked float64 dict (C1), keys and shapes each
    traced to its origin (the spec's build_nh_carry plus the kernel
    contracts quoted in dgrid_nh_pressure_phase_3d's docstring):

    zh, gz   (6, m_a, m_a, km+1) padded -- update_dz_d's contract
             "zh (isd:ied, jsd:jed, km+1)"; the first-substep seed
             writes gz[..., km] = zs over the PADDED box, so zs is
             padded too.
    zs       (6, m_a, m_a) = hs6 / FV3_GRAV (the spec: zs = phis/grav,
             dyn_core.F90:262-278).
    ws3      (6, m_a, m_a) padded 2-D, the C-stage surface velocity
             update_dz_c fills and Riem_Solver_C reads.
    ws       (6, n, n) compute-window 2-D, the D-stage one.
    pk3      (6, m_a, m_a, km+1) padded interfaces.
    pe, peln (6,) + field_shape(name, n, ng, km) -- the oracle layouts
             (the oracle allocates pe with k as its middle axis and the
             D-stage never k-indexes it).
    pk       (6, n, n, km+1) -- dyn_core's pk is COMPUTE-window
             (is:ie, js:je, npz+1); Riem_Solver3 writes pk[:, jc, k]
             with ni rows, a different array from the padded hydro-geopk
             pk that field_shape("pk") describes.
    """
    require_f64_jax("build_nh_carry",
                    {f"hs6[{t}]": h for t, h in enumerate(hs6)})
    n, ng = ctx.n, ctx.ng
    m_a = n + 2 * ng
    hs6 = jnp.asarray(hs6, dtype=jnp.float64)   # list of faces OR stacked
    return {
        "zs": hs6 / FV3_GRAV,
        "gz": jnp.zeros((6, m_a, m_a, km + 1), dtype=jnp.float64),
        "zh": jnp.zeros((6, m_a, m_a, km + 1), dtype=jnp.float64),
        "ws3": jnp.zeros((6, m_a, m_a), dtype=jnp.float64),
        "ws": jnp.zeros((6, n, n), dtype=jnp.float64),
        "pk3": jnp.zeros((6, m_a, m_a, km + 1), dtype=jnp.float64),
        "pe": jnp.zeros((6,) + tuple(field_shape("pe", n, ng, km)),
                        dtype=jnp.float64),
        "pk": jnp.zeros((6, n, n, km + 1), dtype=jnp.float64),
        "peln": jnp.zeros((6,) + tuple(field_shape("peln", n, ng, km)),
                          dtype=jnp.float64),
    }


def acoustic_substep_3d(ctx, state: dict, dt, km: int, *,
                        first_substep: bool,
                        ptop: float, akap: float, cp_air: float,
                        cfg=None,
                        a2b_ord: int = 4,
                        exchange: bool = True,
                        remap_step: bool = False,
                        remap_follows: bool = False,
                        hydrostatic: bool = True,
                        nh: dict | None = None,
                        p_fac: float = 0.05, a_imp: float = 1.0,
                        dp0=None,
                        use_logp: bool = False,
                        flux_cap: dict | None = None,
                        check_state: bool = False,
                        substep: int | None = None) -> dict:
    """One ``it`` of ``do it=1,n_split`` (dyn_core.F90:339).  Returns:

    state     delp/pt/u/v updated -- delp/pt from d_sw2 AFTER the
              :1336-1337 exchange, u/v from one_grad_p (hydrostatic) or
              nh_p_grad (NH); on NH, w (Riemann_Solver3's update) and
              delz too.  Every other key passes through untouched.
    nh        the threaded NH carry (None on the hydrostatic arm);
              everything the spec mutates in the nh dict, in delz6 and
              in csw_press["pkc"] comes back here / in press / in
              state (C4).
    flux_cap  the accumulated tracer capacitor bundle (None when none
              was given).  The CALLER zeroes it once per acoustic loop:
              the "Empty the flux capacitors" block runs at dyn_core
              entry, not per sub-step.
    press     the D-grid pressure bundle; ``pk_remap`` is inside it IFF
              remap_step, exactly as the spec gates it.
    stages    the payloads NOT recoverable from the other returns (C6 --
              the spec's stage_hook is not ported, it would be handed
              tracers): S10 = d_sw3's PRE-barrier-2
              ubb/vbb/ubbtemp/vbbtemp stacks; S11 = the POST-barrier
              ubb/vbbtemp stacks; S16 = geopk's pk/gz as they stood
              BEFORE one_grad_p (hydrostatic arm only -- one_grad_p
              scratches both in place, a2b_ord4 replace=.true., and the
              NH tail has no one_grad_p at all), lifted from the
              pressure phase's own S16 return.  S12 (corner KE) and
              S13/S14 (d_sw5 diagnostics, d_sw6 winds) are NOT
              re-returned: they are recoverable from the above plus the
              returned fields, and dsw_tail_phase_3d already returns
              them in ``tail``-derived quantities.  S02/S03/S15 are
              recoverable from the returned state.

    STATIC/DYNAMIC (C3): ``dt`` is dynamic -- a new time step must not
    recompile the phase; ctx, km, first_substep, exchange, remap_step,
    remap_follows, hydrostatic, use_logp, cfg, a2b_ord, check_state and
    substep are static.  ptop/akap/cp_air/p_fac/a_imp are STATIC: every
    callee taking them type-annotates them as float deck constants and
    dgrid_pressure_phase_3d's contract states them static (geopk's own
    contract); they never vary within a run.  state, nh, dp0 and
    flux_cap are dynamic arrays.

    ``first_substep`` is the :465/:470 gate (scalars are exchanged once
    per acoustic loop, the winds every sub-step -- an explicit argument,
    never inferred) and, on NH, seeds gz[..., km] = zs, rebuilds gz
    from delz, duo-exchanges gz and saves zh = gz; every LATER substep
    RESTORES gz = zh (:535-581) -- get that backwards and the run is
    finite, plausible and wrong.  ``remap_step`` is the it == n_split
    gate (:344-348).  ``check_state`` (C5) RAISES on a tracer instead
    of silently skipping.  The post-d_sw delp/pt exchange runs BEFORE
    the D-grid pressure phase every sub-step: without it the spec
    recorded sub-step 2 producing 278 non-finite delp values.
    """
    km = require_km("acoustic_substep_3d", km)
    require_no_remap_needed(km, remap_follows=remap_follows)
    require_f64_jax("acoustic_substep_3d", state)
    if a2b_ord not in (2, 4):
        raise ValueError(
            f"acoustic_substep_3d: unknown a2b_ord {a2b_ord!r}")
    dt2 = 0.5 * dt
    if not hydrostatic and (nh is None or dp0 is None):
        raise ValueError(
            "acoustic_substep_3d: hydrostatic=False needs the persistent "
            "nh carry (build_nh_carry) and dp0 (dp_ref)")
    if not hydrostatic:
        require_f64_jax("acoustic_substep_3d[nh]", nh)

    tab = _duo_tables(ctx)
    bd = ctx.bd
    i0 = bd.is_ - bd.isd
    j0 = bd.js - bd.jsd
    n = ctx.n

    if not hydrostatic and first_substep:
        # :384-416 -- gz[..., km] = zs over the PADDED box, then the
        # downward recurrence gz(k) = gz(k+1) - delz(k) over the COMPUTE
        # window only (halo values below km+1 keep their old contents,
        # exactly as the spec's windowed assignment leaves them).
        gz = nh["gz"].at[:, :, :, km].set(nh["zs"])
        wi = slice(i0, i0 + n)
        wj = slice(j0, j0 + n)
        # The spec's per-face loop is the stacked axis 0 (identical
        # window every face); only delz's compute window is read.
        dz_cw = state["delz"][:, wi, wj, :]                # (6,n,n,km)

        def _dz_down(gzw, dz):
            gzw = gzw - dz
            return gzw, gzw

        # k-recurrence -> lax.scan (never cumsum).  reverse=True walks
        # k = km-1 .. 0 from the zs carry; ys stack in xs order, so
        # ys[k] is gz at level k.
        gzcw, _ = jax.lax.scan(_dz_down, nh["zs"][:, wi, wj],
                               jnp.moveaxis(dz_cw, -1, 0), reverse=True)
        gz = gz.at[:, wi, wj, :km].set(jnp.moveaxis(gzcw, 0, -1))
        nh = {**nh, "gz": gz}

    if exchange:
        # :465 -- delp/pt only on it == 1; winds every sub-step (:504);
        # w on NH, after the winds (:479-480).
        state = exchange_state_halos_3d(ctx, state, km,
                                        scalars=first_substep, winds=True,
                                        w_field=not hydrostatic)

    csw = csw_phase_3d(ctx, state, dt2=dt2, km=km, nord=2,
                       hydrostatic=hydrostatic,
                       remap_follows=remap_follows)

    if hydrostatic:
        # `check_delpc` is a DATA-DEPENDENT check and this sub-step is
        # traced inside `acoustic_loop_3d`'s scan, where the C-grid
        # phase REFUSES it rather than skipping silently (its C5
        # contract, and it fired at n_split=3 in job 9417652).  It is
        # threaded from this function's own `check_state` flag, so the
        # eager entry can still ask for it and the traced lane cannot
        # accidentally get it.
        csw_press = cgrid_pressure_phase_3d(ctx, csw, km, dt2=dt2,
                                            ptop=ptop, akap=akap,
                                            cp_air=cp_air,
                                            a2b_ord=a2b_ord,
                                            remap_follows=remap_follows,
                                            check_delpc=check_state)
    else:
        if first_substep:
            # :535-557 -- duo-exchange gz, then save zh = gz (padded).
            if exchange:
                gz = nh["gz"]
                for k in range(km + 1):
                    # stag "A" (0,0): gz is a cell-centred scalar plane;
                    # per level, all six faces -- the halo module's extent.
                    gz = gz.at[..., k].set(
                        ext_scalar_sixface(gz[..., k], tab, "A"))
                nh = {**nh, "gz": gz}
            nh = {**nh, "zh": nh["gz"]}
        else:
            # :559-581 -- restore gz = zh (padded).
            nh = {**nh, "gz": nh["zh"]}
        csw_press = cgrid_nh_pressure_phase_3d(
            ctx, csw, nh["gz"], nh["ws3"], km, dt2=dt2, ptop=ptop,
            akap=akap, cp_air=cp_air, p_fac=p_fac, a_imp=a_imp,
            dp0=dp0, zs6=nh["zs"], remap_follows=remap_follows)
        # C4: the C stage comes back with the REBUILT geopotential gz
        # and the refilled ws3 -- thread both into the carry (the
        # in-place twin of the spec's mutation).
        nh = {**nh, "gz": csw_press["gz"], "ws3": csw_press["ws3"]}

    # p_grad_c mutates uc/vc in place in the oracle; here they come back
    # in the pressure dict, so merge them into the c_sw bundle the
    # D-grid phases read (both arms return "uc"/"vc").
    csw = {**csw, "uc": csw_press["uc"], "vc": csw_press["vc"]}

    dsw = dsw_transport_phase_3d(ctx, state, csw, dt=dt, km=km, cfg=cfg,
                                 hydrostatic=hydrostatic,
                                 remap_follows=remap_follows,
                                 flux_cap=flux_cap)
    if flux_cap is not None:
        # C4: d_sw1 accumulates into the capacitors every sub-step (the
        # UN-averaged fluxes, before barrier 1); the transport phase
        # returns the accumulated bundle only when one was given.
        flux_cap = dsw["flux_cap"]

    # ⛔ THE TAIL READS THE EXCHANGED uc/vc, NOT THE ONES THE C STAGE
    # PRODUCED.  The spec hands the SAME `csw` object to the transport
    # phase and then to the tail, and the transport phase MUTATES
    # uc/vc/divg_d in place with the post-p_grad_c duo exchanges
    # (dyn_core.F90:652/:655).  On this functional lane those come back
    # in the transport phase's return as stage S07, so they have to be
    # threaded in here -- exactly convention C4, at a seam that a
    # phase-by-phase reading does not show.
    #
    # MEASURED (job 9417625): composing the NUMPY phases by hand with a
    # fresh csw for the tail reproduces this lane's answer and differs
    # from the spec's own sub-step by 2.106e-02 on u, 1512 of 6156
    # cells -- the identical number this module was showing.  d_sw3 and
    # d_sw5 both read uc/vc, so stale halos there move the winds.
    csw = {**csw, "uc": dsw["uc"], "vc": dsw["vc"],
           "divg_d": dsw["divg_d"]}

    tail = dsw_tail_phase_3d(ctx, state, csw, dsw, dt=dt, km=km, cfg=cfg,
                             hydrostatic=hydrostatic,
                             remap_follows=remap_follows)

    # POST-d_sw scalar exchange, BEFORE the D-grid geopk (:1336-1337,
    # every sub-step, ahead of :1401): d_sw2 has just rewritten
    # delp/pt, so without this their halos are stale into the pressure
    # chain and the NEXT sub-step.  Static `if` on `exchange` (C3).
    if exchange:
        for _nm in ("delp", "pt"):
            dsw = {**dsw, _nm: _exchange_scalar_stack(dsw[_nm], tab, km)}

    if hydrostatic:
        press = dgrid_pressure_phase_3d(ctx, dsw, tail, km, dt=dt,
                                        ptop=ptop, akap=akap,
                                        cp_air=cp_air, a2b_ord=a2b_ord,
                                        remap_step=remap_step,
                                        remap_follows=remap_follows)
        # u/v are one_grad_p's updated winds (the S17 payload); the
        # press dict also carries pk_remap IFF remap_step (the spec's
        # press contract: "plus pk_remap on a remap step").
        u_new, v_new = press["u"], press["v"]
    else:
        res = dgrid_nh_pressure_phase_3d(
            ctx, csw_press, dsw, tail, nh, km, dt=dt, ptop=ptop,
            akap=akap, cp_air=cp_air, p_fac=p_fac, a_imp=a_imp,
            dp0=dp0, delz=state["delz"], remap_step=remap_step,
            use_logp=use_logp, cfg=cfg, remap_follows=remap_follows)
        # C4 / rule: everything the spec mutates comes back.  nh
        # members, delz, pkc (the csw_press storage Riemann_Solver3
        # overwrites), the D winds and the Riemann-updated w.
        nh = {**nh,
              "zh": res["zh"], "gz": res["gz"], "zs": res["zs"],
              "pk3": res["pk3"], "pe": res["pe"], "pk": res["pk"],
              "peln": res["peln"], "ws": res["ws"]}
        # press aliases the returned nh members exactly as the spec's
        # press list aliases the nh carry; pkc rides along (C4).
        press = {"pe": res["pe"], "pk": res["pk"], "peln": res["peln"],
                 "ws": res["ws"], "pkc": res["pkc"]}
        if remap_step:
            press["pk_remap"] = res["pk_remap"]
        state = {**state, "delz": res["delz"]}
        u_new, v_new = res["u"], res["v"]

    # Prognostic write-back: delp/pt from the EXCHANGED d_sw2 outputs,
    # u/v from the pressure tail, w (NH) from the Riemann solver.
    state = {**state, "delp": dsw["delp"], "pt": dsw["pt"],
             "u": u_new, "v": v_new}
    if not hydrostatic:
        state = {**state, "w": res["w"]}

    if check_state:
        _check_sane_state(state, f"it={substep}" if substep else "?")

    # C6 stage payloads -- only the ones not recoverable elsewhere.
    stages = {
        "S10": {nm: tail[nm] for nm in
                ("ubb_prebarrier", "vbb_prebarrier",
                 "ubbtemp_prebarrier", "vbbtemp_prebarrier")},
        "S11": {"ubb": tail["ubb_postbarrier"],
                "vbbtemp": tail["vbbtemp_postbarrier"]},
    }
    if hydrostatic:
        # S16 keys follow the dsw lane's <field>_pre<stage> convention;
        # dgrid_pressure_phase_3d returns the S16 payload itself.
        # The producing phase names these `pk_pre_onegradp` /
        # `gz_pre_onegradp`; this part was authored against a guessed
        # spelling and raised KeyError on the first composed run (job
        # 9417516). Read them by the names the phase returns.
        stages["S16"] = {"pk": press["pk_pre_onegradp"],
                         "gz": press["gz_pre_onegradp"]}

    return {"state": state, "nh": nh, "flux_cap": flux_cap,
            "press": press, "stages": stages}


def make_exchange_state_halos_3d_jit(ctx, km: int, *, scalars: bool,
                                     winds: bool, w_field: bool = False):
    """jit closure with ctx/km and the static exchange selectors bound;
    the returned callable takes the face-stacked state (dynamic, C3)."""
    return jax.jit(partial(exchange_state_halos_3d, ctx, km=km,
                           scalars=scalars, winds=winds,
                           w_field=w_field))


def make_build_nh_carry_jit(ctx, km: int):
    """jit closure over the static (ctx, km); the callable takes hs6."""
    return jax.jit(partial(build_nh_carry, ctx, km))


def make_acoustic_substep_3d_jit(ctx, km: int, *, first_substep: bool,
                                 ptop: float, akap: float, cp_air: float,
                                 cfg=None, a2b_ord: int = 4,
                                 exchange: bool = True,
                                 remap_step: bool = False,
                                 remap_follows: bool = False,
                                 hydrostatic: bool = True,
                                 p_fac: float = 0.05, a_imp: float = 1.0,
                                 use_logp: bool = False,
                                 check_state: bool = False,
                                 substep: int | None = None):
    """Bind every STATIC selector (C3); the jitted callable is
    ``(state, dt, nh=None, dp0=None, flux_cap=None)`` -- dt, the state,
    the NH carry, dp0 and the flux capacitors are the only dynamic
    inputs, so a new time step does not recompile.  With
    ``check_state=True`` the closure raises on the first tracer, which
    is the D3/C5 contract, not a bug."""
    def _run(state, dt, nh=None, dp0=None, flux_cap=None):
        return acoustic_substep_3d(
            ctx, state, dt, km, first_substep=first_substep,
            ptop=ptop, akap=akap, cp_air=cp_air, cfg=cfg,
            a2b_ord=a2b_ord, exchange=exchange, remap_step=remap_step,
            remap_follows=remap_follows, hydrostatic=hydrostatic, nh=nh,
            p_fac=p_fac, a_imp=a_imp, dp0=dp0, use_logp=use_logp,
            flux_cap=flux_cap, check_state=check_state, substep=substep)
    return jax.jit(_run)


# =====================================================================
# Part B -- the outer driver: `do it = 1, n_split` (dyn_core.F90:339),
# written as peel / lax.scan / peel.  This routine introduces NO
# numerics: every arithmetic operation lives in acoustic_substep_3d
# (Part A) and the phase modules it assembles.
#
# Part A contract relied on below:
#   acoustic_substep_3d(...) -> (state, nh, flux_cap, press, stages)
#     state     face-stacked dict (C1), rebuilt functionally;
#     nh        the NH carry dict (None when hydrostatic);
#     flux_cap  the six-face tracer flux capacitor this sub-step
#               accumulated into (dyn_core.F90:313-316 zeroes it once
#               per dyn_core call, i.e. in the CALLER of this loop);
#     press     the D-grid pressure bundle when remap_step=True, else
#               None;
#     stages    S10/S11/S16 payloads by return (C6/R8), plus pk_remap
#               only when remap_step=True -- exactly as the spec gates.
# =====================================================================


def _check_substep_state(state, ctx, km, it, n_split, hydrostatic,
                         remap_follows):
    """Eager twin of the spec's per-sub-step ``validate`` block.

    Only ever runs on concrete arrays: acoustic_loop_3d raises on
    check_state=True under a tracer (C5/D3), so nothing here can be
    handed a tracer.  Raises naming the sub-step that broke.
    """
    try:
        # NOT `validate_stacked`, which is the shared ENTRY gate
        # (fname, container, ctx, km, required, *, what).  What the
        # spec calls here is the NumPy lane's state validator, and its
        # JAX-side equivalent is the shape/keys check plus the finite
        # check below.
        validate_stacked("acoustic_loop_3d", state, ctx, km,
                         STATE_FIELDS, what=f"state after sub-step {it}")
    except Exception as exc:
        raise RuntimeError(
            f"state invalid after acoustic sub-step {it}/{n_split}: "
            f"{exc}") from None
    bd = ctx.bd
    i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
    ni, nj = bd.ie - bd.is_ + 1, bd.je - bd.js + 1
    fields = (("delp", "pt", "u", "v") if hydrostatic
              else ("delp", "pt", "u", "v", "w"))
    for t in range(6):
        for name in fields:
            # Compute window only (C1: state[name] is (6, i, j, km)); the
            # corner-diagonal halo carries `sentinel` by construction, so
            # scoring the padded array would flag every healthy step, and
            # scoring a[isfinite(a)] is a tautology that cannot fail.
            win = state[name][t, i0:i0 + ni, j0:j0 + nj, :]
            bad = int(jnp.count_nonzero(~jnp.isfinite(win)))
            if bad:
                raise RuntimeError(
                    f"sub-step {it}/{n_split}: face {t + 1} {name} has "
                    f"{bad} non-finite value(s) inside the compute window")
            if name == "delp":
                dmin = float(win.min())
                if dmin <= 0.0:
                    # FINITENESS IS NOT ENOUGH: a non-positive layer mass
                    # is finite, drives delpc negative one sub-step later
                    # and only turns NaN inside geopk's log.  Name the
                    # sub-step that actually diverged.
                    raise RuntimeError(
                        f"sub-step {it}/{n_split}: face {t + 1} delp min "
                        f"{dmin:.6g} <= 0 -- non-positive layer mass")
            lim = _SANE_MAX.get(name)
            if lim is not None:
                peak = float(jnp.abs(win).max())
                if peak > lim:
                    raise RuntimeError(
                        f"sub-step {it}/{n_split}: face {t + 1} {name} "
                        f"peaked at {peak:.6g}, above the sanity bound "
                        f"{lim:g}. Finite but non-physical -- the "
                        f"divergence starts HERE, not where the NaN "
                        f"appears.")


def acoustic_loop_3d(ctx, state, dt_atmos, km, *, n_split, ptop, akap,
                     cp_air, cfg=None, check_state=False,
                     remap_follows=False, hydrostatic=True, nh=None,
                     p_fac=0.05, a_imp=1.0, dp0=None, use_logp=False,
                     flux_cap=None):
    """``do it = 1, n_split`` -- one outer dynamics step, JAX lane.

    ``dt = bdt/n_split`` (dyn_core.F90:249); the shipped duo decks run
    ``k_split = 1``, so one call of this is one ``dt_atmos``.  dt is
    DYNAMIC (C3/D4): ``dt_atmos`` is traced, ``n_split`` is a Python
    trip count and scan length and can never be traced.

    Structure (D1): the sub-steps are NOT all the same program --
    sub-step 1 has first_substep=True (the entry exchange refreshes
    delp/pt as well as the winds, and the NH arm seeds ``gz[..., km] =
    zs``, rebuilds gz from delz and saves ``zh = gz``), sub-step n_split
    has remap_step=True (the pk snapshot the vertical remap reads; the
    NH tail calls pe_halo), and every sub-step between is the SAME
    program.  A lax.scan body must be one program, so sub-steps 1 and
    n_split are peeled and only the identical middle runs under scan.

    Returns the C4 dict ``{"state", "nh", "flux_cap", "press",
    "stages"}``:

      state / nh / flux_cap
            everything the spec mutates; the caller threads ``nh`` and
            ``flux_cap`` forward (D5 -- the NH carry is functional).
      press
            the FINAL sub-step's D-grid pressure bundle -- the spec's
            ``press_out`` receives only the remap-step bundle, so the
            middle bundles are dropped; the twin always returns it (the
            spec's ``press_out=None`` merely declined to capture it).
      stages
            observation by return, never a callback (C6/R8); the spec's
            stage_hook is not ported.  ``"first"`` and ``"last"`` carry
            that sub-step's payload from acoustic_substep_3d: S10 (the
            PRE-barrier ubb/vbb/ubbtemp/vbbtemp stacks, d_sw3's output),
            S11 (the POST-barrier ubb/vbbtemp stacks) and S16 (geopk's
            pk/gz as they stood BEFORE one_grad_p scratched them in
            place, hence unrecoverable afterwards).  ``"last"``
            additionally carries pk_remap, remap step only, exactly as
            the spec gates it.  Middle sub-steps' payloads are not
            returned; run the eager entry per sub-step to observe them.
            For n_split == 1 both keys alias the single sub-step.

    check_state (C5/D3): the spec's ``validate`` block re-checks the
    state after every sub-step; that is data-dependent and cannot exist
    inside a scan.  With check_state=True the checks run EAGERLY after
    every sub-step (the middle sub-steps then run as a Python loop over
    the identical program instead of a scan) and the routine RAISES if
    asked for while jitted, rather than silently skipping.

    ``hydrostatic=False`` with ``nh=None`` builds the carry from
    ``ctx.hs6`` via build_nh_carry (``zs = phis / grav``, D5).
    """
    # The f64 gate belongs on the ARRAYS, and it runs inside
    # acoustic_substep_3d on the state and the carry.  The first draft
    # called `require_f64_jax(ctx)` here -- ctx is the context OBJECT,
    # not a dict of arrays, and the helper's signature is
    # (fname, arrays), so this raised TypeError on the first call.
    km = require_km("acoustic_loop_3d", km)
    require_no_remap_needed(km, remap_follows=remap_follows)
    if n_split < 1:
        raise ValueError(f"n_split must be >= 1, got {n_split}")
    if check_state and isinstance(dt_atmos, jax.core.Tracer):
        raise ValueError(
            "acoustic_loop_3d: check_state=True is a data-dependent "
            "per-sub-step validation (the spec's `validate`) and cannot "
            "run under jit; call the eager entry with check_state=True "
            "outside jit instead (C5/D3)")
    if not hydrostatic and nh is None:
        hs6 = ctx.get("hs6")
        if hs6 is None:
            raise ValueError(
                "acoustic_loop_3d: hydrostatic=False needs ctx.hs6 "
                "(phis) to build the NH carry")
        nh = build_nh_carry(ctx, km, hs6)
    dt = dt_atmos / float(n_split)

    def _sub(state_, nh_, flux_cap_, first, remap):
        # Single call site for all three positions.  `first` / `remap`
        # are static Python bools -- a loop-index predicate is a static
        # partition here, never jnp.where (R2).
        out = acoustic_substep_3d(
            ctx, state_, dt, km, first_substep=first, ptop=ptop,
            akap=akap, cp_air=cp_air, cfg=cfg, remap_step=remap,
            remap_follows=remap_follows, hydrostatic=hydrostatic, nh=nh_,
            p_fac=p_fac, a_imp=a_imp, dp0=dp0, use_logp=use_logp,
            flux_cap=flux_cap_)
        # Part A returns a DICT, not the 5-tuple this part was authored
        # against: the two halves were written in separate calls and the
        # seam is exactly where a contract goes missing.  Unpacked ONCE,
        # here, so the three call sites below stay unchanged.
        return (out["state"], out["nh"], out["flux_cap"],
                out["press"], out["stages"])

    # --- sub-step 1 (it = 1) is peeled: first_substep=True is a
    # different program from the middle.  When n_split == 1 it is ALSO
    # the remap step, so the flag is this static Python bool.
    state, nh, flux_cap, press, stages_first = _sub(
        state, nh, flux_cap, True, n_split == 1)
    stages_last = stages_first
    if check_state:
        _check_substep_state(state, ctx, km, 1, n_split, hydrostatic,
                             remap_follows)

    if n_split > 2:
        if check_state:
            # C5: the eager path must hold each middle state in hand to
            # validate it, so it runs the IDENTICAL middle program in a
            # Python loop instead of a scan.
            for it in range(2, n_split):
                state, nh, flux_cap, _press, _stages = _sub(
                    state, nh, flux_cap, False, False)
                _check_substep_state(state, ctx, km, it, n_split,
                                     hydrostatic, remap_follows)
        else:
            # D2: the carry is (state, nh, flux_cap) -- the same pytree
            # every iteration (first_substep changes the entry exchange,
            # not shapes, dtypes or keys; nothing may appear or
            # disappear).  press is NOT in the carry: only the FINAL
            # sub-step's bundle is returned, exactly as the spec's
            # press_out receives only the remap-step bundle.  Middle
            # stage payloads are dropped (see docstring).
            def _middle(carry, _):
                st, nhc, capc = carry
                st, nhc, capc, _press, _stages = _sub(
                    st, nhc, capc, False, False)
                return (st, nhc, capc), None

            (state, nh, flux_cap), _ = lax.scan(
                _middle, (state, nh, flux_cap), None,
                length=n_split - 2)

    if n_split > 1:
        # --- sub-step n_split is peeled: remap_step=True is a different
        # program from the middle (pk snapshot; pe_halo on the NH tail).
        state, nh, flux_cap, press, stages_last = _sub(
            state, nh, flux_cap, False, True)
        if check_state:
            _check_substep_state(state, ctx, km, n_split, n_split,
                                 hydrostatic, remap_follows)

    return {"state": state, "nh": nh, "flux_cap": flux_cap,
            "press": press,
            "stages": {"first": stages_first, "last": stages_last}}


def make_acoustic_loop_3d_jit(ctx, km, *, n_split, cfg=None,
                              check_state=False, remap_follows=False,
                              hydrostatic=True, use_logp=False):
    """jit factory for acoustic_loop_3d (one per public routine).

    STATIC, captured here (a change recompiles): ctx, km, n_split (a
    Python trip count and scan length -- never traceable), cfg, and the
    Python-branch selectors remap_follows / hydrostatic / use_logp.
    The hydrostatic=False carry is built inside acoustic_loop_3d from
    the closed-over ctx when nh is None, or threaded in as a traced
    dict when it is given.

    DYNAMIC, keyword arguments of the returned callable (C3/D4):
    state, dt_atmos (a new time step must not recompile this phase),
    nh, flux_cap, dp0, ptop, akap, cp_air, p_fac, a_imp.  The five
    physics scalars stay dynamic because no landed callee branches on
    them in Python (R12).  No donate_argnums (R3); the f64 gate runs
    once here on static dtypes, not per call.
    """
    # (see the note in acoustic_loop_3d: the f64 gate runs on the
    # arrays inside the sub-step, not on the context object.)
    require_no_remap_needed(km, remap_follows=remap_follows)
    if n_split < 1:
        raise ValueError(f"n_split must be >= 1, got {n_split}")
    if check_state:
        # The loop is jitted by construction here, so asking for the
        # data-dependent per-sub-step validation must RAISE, not
        # silently skip (C5/D3).
        raise ValueError(
            "make_acoustic_loop_3d_jit: check_state=True cannot run "
            "jitted (data-dependent, the spec's `validate`); call the "
            "eager acoustic_loop_3d with check_state=True instead")

    def _loop(state, dt_atmos, *, ptop, akap, cp_air, nh=None,
              flux_cap=None, dp0=None, p_fac=0.05, a_imp=1.0):
        return acoustic_loop_3d(
            ctx, state, dt_atmos, km, n_split=n_split, ptop=ptop,
            akap=akap, cp_air=cp_air, cfg=cfg, check_state=False,
            remap_follows=remap_follows, hydrostatic=hydrostatic, nh=nh,
            p_fac=p_fac, a_imp=a_imp, dp0=dp0, use_logp=use_logp,
            flux_cap=flux_cap)

    return jax.jit(_loop)
