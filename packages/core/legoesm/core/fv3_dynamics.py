"""One full FV3 outer time step: ``fv_dynamics`` for the pinned duo lane, JAX twin.

Functional, jit-compilable twin of ``legoesm.core.fv3_native_dynamics``
(STAGE 1 unit 7, the last assembly): the ``do n_map=1,k_split`` shell that
joins the already-ported acoustic loop, tracer transport and vertical remap,
plus the conversions that live only in ``fv_dynamics`` itself.

ORACLE CADENCE (the spec's block, checksum-pinned; model/fv_dynamics.F90):

    :271-279  pfull(k) from ak/bk at p_ref
    :281-294  hydrostatic branch: dp1(i,j,k) = zvir*q(sphum)
    :396-408  pt -> virtual POTENTIAL temperature:
              pt = pt*(1 + dp1)/pkz          <-- COMPUTE WINDOW ONLY
    :451      do n_map = 1, k_split          (mdt = bdt/k_split, :413)
    :472-478    dp1 = delp
    :482        last_step = (n_map == k_split)
    :502        dyn_core(mdt, n_split, ...)
    :534/538    tracer_2d
    :568      if (npz > 4) then
    :618        Lagrangian_to_Eulerian(last_step, ...)
    :674      endif

BARRIERS: both duo barriers (extent 1 over i = is..ie / j = js..je; extent 2
over i = is..ie+1 / j = js..je+1, tile corners included, applied ONE LEVEL
AT A TIME with all six faces of that level present) live inside the acoustic
callee and are ``legoesm.grids.fv3_duo_halos``'s; this module never applies,
re-derives or re-states an extent or an exchange.

CONVENTIONS inherited from the sibling 3-D phases: C1 ONE dict of
face-stacked arrays (face t = Fortran tile t+1, Fortran index 1-ng at numpy
0, shapes are (6,) + field_shape(name, n, ng, km); the per-face remap call
slices ``[t]``); C3 ctx, km, k_split, n_split and every deck constant or
Python-branch selector are STATIC, bdt and the fields are DYNAMIC; C4 every
array the spec mutates (state, press, q, omga, the NH carry) is RETURNED and
the caller threads it -- nothing is written through an argument; C5 a
data-dependent guard becomes a static check_* keyword that RAISES on a
tracer rather than silently skipping; C6 stage observation is by return,
never a callback; C7 the deck is the spec's own duo configuration.

WHAT THIS MODULE DOES NOT DO: it introduces no numerics beyond its own
pressure diagnostics (p_var_hydrostatic / p_var_nonhydrostatic /
pt_to_theta_v, ported literally with the oracle's operation order); the
barrier and exchange extents are the halo module's; d_con = 0.0 throughout
and is NOT a parameter of this module -- the KE-to-heat pathway and its
heat_source allocation (dyn_core.F90:322-325, gated on d_con > 1.0E-5) are
inactive on the shipped duo decks; the moist path is supported on BOTH arms
(zvir != 0 couples dp1 into pt_to_theta_v and r_vir into the remap, and
on the NH arm also into the pkz this module recomputes at :299-322;
consv_te > CONSV_MIN runs the total-energy fixer through the
two-phase defer/reduce/apply protocol, while NEGATIVE consv_te and
moist x consv are refused); pfull is computed-and-unused on this lane and is
not ported; omga is an output-only passenger whose values are meaningless;
and dyn_core's use_old_omega fill is not ported (no u/v/pt/delp parity).
"""
from __future__ import annotations

import operator as _operator

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.core import Tracer as _Tracer
from legoesm.core.fv3_acoustic_3d import acoustic_loop_3d, build_nh_carry
from legoesm.core.fv3_mapz import (
    CONSV_MIN as _CONSV_MIN,
    close_out_pt,
    lagrangian_to_eulerian,
)

# NOT `fv3_state_3d` -- the shape authority is the NumPy lane's
# `fv3_native_state_3d`, and the guessed home would have failed
# at import (third time in this campaign).
from legoesm.core.fv3_native_state_3d import (
    field_shape,
)
from legoesm.core.fv3_phase3d_common import batch_size, require_f64_jax
from legoesm.core.fv3_tracer2d import (
    alloc_flux_capacitors,
    require_tracer_2d_1l_lane,
    tracer_2d_1l_sixface,
)
from legoesm.grids.fv3_native_gridstruct import (
    FV3_GRAV as _FV3_GRAV,
)
from legoesm.grids.fv3_native_gridstruct import (
    FV3_RDGAS as _FV3_RDGAS,
)

# fv_grid_utils.F90:56 -- `real, parameter:: ptop_min = 1.d-8`.
PTOP_MIN = 1.0e-8

# fv_dynamics.F90:568. Below this the oracle itself never remaps, so the
# driver must not either -- skipping is faithful, not a shortcut (D6:
# fail-closed on the lane, never silently run a different one).
REMAP_MIN_NPZ = 4


def require_uniform_damping_lane(*, n_sponge: int, tau: float,
                                 npz: int) -> None:
    """Refuse a deck whose damping is NOT uniform in the vertical.

    Static and fail-closed (C5/D6): reads deck constants only.  The pinned
    deck takes the ``dyn_core.F90:772`` arm (``npz==1 .or. n_sponge<0``),
    which sets ``d2_divg = d2_bg`` uniformly with ``nord_v = 2`` /
    ``damp_vt = 0.12`` at every level -- exactly the uniform tail config.
    ``tau > 0.`` at :378 switches on ``Rayleigh_Super``, not ported.
    """
    if not (npz == 1 or n_sponge < 0):
        raise NotImplementedError(
            f"n_sponge={n_sponge} >= 0 with npz={npz} selects the sponge arm "
            f"at dyn_core.F90:775-804 (per-level nord_k/d2_divg/nord_w/"
            f"damp_w/nord_v/damp_vt); the tail phase carries a single "
            f"uniform (nord_v, damp_v) pair. The pinned deck has "
            f"n_sponge = -1.")
    if tau > 0.0:
        raise NotImplementedError(
            f"tau={tau} > 0 switches on Rayleigh_Super, which is not "
            f"ported. The pinned deck has tau = -1.")


def p_var_hydrostatic(delp, *, ptop, akap, n: int, ng: int, km: int,
                      check_args: bool = False) -> dict:
    """``p_var``'s hydrostatic branch (tools/init_hydro.F90:41-133), stacked.

    Face-stacked (C1) twin: ``delp`` is (6, m_a, m_a, km) and the returned
    dict holds ps/pk as (6,) + field_shape, pe as (6, n+2, km+1, n+2) and
    peln as (6, n, km+1, n) -- the oracle's own layouts, so the per-face
    remap call can slice ``[t]`` straight out of them.  This is the ONLY
    producer of the pkz that :396-408 divides by on the first step.
    ``adjust_dry_mass = .F.`` on the pinned deck, so ``drymadj`` and the
    ratio rescale (:76-88) are absent by configuration.
    """
    delp = jnp.asarray(delp)
    require_f64_jax("p_var_hydrostatic", {"delp": delp})
    nb = delp.shape[0]            # 6 faces, or 6*kt*kt windows
    want = (nb,) + tuple(field_shape("delp", n, ng, km))
    if delp.shape != want:
        raise ValueError(f"p_var_hydrostatic: delp must be {want}, got "
                         f"{delp.shape}")
    if check_args:  # C5: the spec's `if (ptop <= 0)` raise, static value
        if isinstance(ptop, _Tracer) or not (ptop > 0.0):
            raise ValueError(f"p_var_hydrostatic: ptop must be > 0, got "
                             f"{ptop}")
    ia = ng
    dt = delp.dtype
    # compute window only; the six faces ride the trailing axes (no face
    # reads another face's write, so the per-face loop vectorises)
    win = delp[:, ia:ia + n, ia:ia + n, :]           # (6, n, n, km)

    # :102-108 k-recurrence -> lax.scan (never cumsum); the body is the
    # oracle's own per-k arithmetic in the oracle's order (D7)
    # (3, 0, 1, 2), NOT (0, 3, 1, 2): lax.scan consumes the LEADING axis,
    # so k must lead -- the old transpose put the FACE axis first and
    # scanned the six faces as if they were levels, accumulating one
    # face's delp onto the next. It failed loudly here only because
    # km != 6 makes the shapes disagree; at km = 6 it would have run and
    # returned a plausible pressure field.
    wink = jnp.transpose(win, (3, 0, 1, 2))          # (km, 6, n, n)
    def _col(acc, x):
        acc = acc + x
        lnp = jnp.log(acc)
        return acc, (acc, lnp, jnp.exp(akap * lnp))
    acc0 = jnp.zeros((nb, n, n), dtype=dt) + ptop
    # lax.scan returns (final_carry, stacked_ys) -- TWO values. The
    # three stage outputs come back inside ys, so they unpack from
    # the second element, never from the call.
    _, (pe_c, lnp_c, pk_c) = lax.scan(_col, acc0, wink)  # (km,6,n,n)

    # :80-83  pe(i,1,j) = ptop ; pk(i,j,1) = ptop**cappa
    pek = ptop ** akap
    pe = jnp.zeros((nb,) + tuple(field_shape("pe", n, ng, km)), dtype=dt)
    pe = pe.at[:, 1:n + 1, 0, 1:n + 1].set(ptop)
    pe = pe.at[:, 1:n + 1, 1:km + 1, 1:n + 1].set(
        jnp.transpose(pe_c, (1, 2, 0, 3)))           # (6,i,k,j)
    pk = jnp.zeros((nb,) + tuple(field_shape("pk", n, ng, km)), dtype=dt)
    pk = pk.at[:, ia:ia + n, ia:ia + n, 0].set(pek)
    pk = pk.at[:, ia:ia + n, ia:ia + n, 1:km + 1].set(
        jnp.transpose(pk_c, (1, 2, 3, 0)))           # (6,i,j,k)
    peln = jnp.zeros((nb,) + tuple(field_shape("peln", n, ng, km)),
                     dtype=dt)
    peln = peln.at[:, :, 1:km + 1, :].set(jnp.transpose(lnp_c, (1, 2, 0, 3)))
    # :110-112  ps = pe(i,km+1,j)
    ps = jnp.zeros((nb,) + tuple(field_shape("ps", n, ng, km)), dtype=dt)
    ps = ps.at[:, ia:ia + n, ia:ia + n].set(pe[:, 1:n + 1, km, 1:n + 1])
    # :114-125 peln(:,1,:).  jnp.where: both arms total and finite -- the
    # dead log arm is clamped at PTOP_MIN, which cannot alter the live
    # (ptop >= PTOP_MIN) branch, so the small-ptop deck is still refused
    # (via check_args) rather than silently mis-seeded.
    ak1 = (akap + 1.0) / akap
    peln0 = jnp.where(ptop < PTOP_MIN, peln[:, :, 1, :] - ak1,
                      jnp.log(jnp.maximum(ptop, PTOP_MIN)))
    peln = peln.at[:, :, 0, :].set(peln0)
    # :127-133 hydrostatic pkz, the oracle's own expression order
    pkz = ((pk[:, ia:ia + n, ia:ia + n, 1:] - pk[:, ia:ia + n, ia:ia + n, :-1])
           / (akap * (peln[:, :, 1:, :] - peln[:, :, :-1, :]))
           .transpose(0, 1, 3, 2))
    return {"ps": ps, "pe": pe, "peln": peln, "pk": pk, "pkz": pkz}


# --------------------------------------------------------------------
# The consv_te energy integrals -- JAX twins of the spec lane's
# --------------------------------------------------------------------
# WRITTEN IN jnp, NOT REUSED FROM THE SPEC VIA np.asarray. The first
# version did that on the argument that these run "outside any traced
# loop"; codex showed the argument is false for the public jit entry
# point (BLOCKER, job 9446299): make_fv_dynamics_step_jit wraps the
# WHOLE step, so the peeled final n_map iteration is still inside
# jax.jit and np.asarray(tracer) raises TracerArrayConversionError.
# jax.grad, vmap or any enclosing scan have the same problem.
#
# So these are twins in the repo's established two-lane pattern, and
# they are FACE-STACKED where the spec's are per-face -- the six faces
# reduce together anyway.


def _ke_column_jax(delpw, u, v, rsin2, cosa_s, *, cp_times, n, ng):
    """``sum_k delp*(<cp term> + 0.25*rsin2*KE)``, the term both
    integrals share (fv_mapz.F90:650-656 and :1145-1150).

    ``u`` is (6, m_a, m_a+1) and ``v`` is (6, m_a+1, m_a), so the j+1
    and i+1 neighbours slice DIFFERENT axes.
    """
    ia = ng
    r = rsin2[:, ia:ia + n, ia:ia + n][:, :, :, None]
    c = cosa_s[:, ia:ia + n, ia:ia + n][:, :, :, None]
    u0 = u[:, ia:ia + n, ia:ia + n, :]
    u1 = u[:, ia:ia + n, ia + 1:ia + 1 + n, :]
    v0 = v[:, ia:ia + n, ia:ia + n, :]
    v1 = v[:, ia + 1:ia + 1 + n, ia:ia + n, :]
    ke = 0.25 * r * (u0 ** 2 + u1 ** 2 + v0 ** 2 + v1 ** 2
                     - (u0 + u1) * (v0 + v1) * c)
    return jnp.sum(delpw * (cp_times + ke), axis=3)


def total_energy_2d_hydrostatic_jax(pt, delp, u, v, pe, peln, hs, rsin2,
                                    cosa_s, *, qc=None, cp, rg, n, ng, km):
    """``te0_2d`` (compute_total_energy, :1128-1151), face-stacked.

    ``pt`` is TEMPERATURE and ``qc = zvir*q(sphum)``, so ``tv`` is the
    virtual temperature this routine forms itself. ``phiz`` accumulates
    DOWNWARD, k = km..1 -- the OTHER integral goes upward, and they are
    deliberately not shared.
    """
    ia = ng
    ptw = pt[:, ia:ia + n, ia:ia + n, :]
    tv = ptw if qc is None else ptw * (1.0 + qc)
    dpe = peln[:, :, 1:, :] - peln[:, :, :-1, :]      # (6, n, km, n)
    dpe = jnp.transpose(dpe, (0, 1, 3, 2))            # -> (6, n, n, km)
    lay = rg * tv * dpe
    # phiz[k] = hs + sum_{k' >= k} lay[k'] : a reverse cumulative sum,
    # which IS the k = km..1 recurrence, without a Python loop.
    rev = jnp.cumsum(lay[:, :, :, ::-1], axis=3)[:, :, :, ::-1]
    hs_w = hs[:, ia:ia + n, ia:ia + n]
    phiz_top = hs_w + rev[:, :, :, 0]
    pe_w = pe[:, 1:n + 1, :, 1:n + 1]
    te = (pe_w[:, :, km, :] * hs_w) - (pe_w[:, :, 0, :] * phiz_top)
    return te + _ke_column_jax(delp[:, ia:ia + n, ia:ia + n, :], u, v,
                               rsin2, cosa_s, cp_times=cp * tv, n=n, ng=ng)


def fixer_energy_2d_hydrostatic_jax(pt, delp, u, v, pe, peln, hs, rsin2,
                                    cosa_s, *, cp, rg, n, ng, km):
    """``te_2d`` (the fixer, :638-658), face-stacked.

    ``pt`` is POST-remap ``T_v`` so this uses ``cp*pt`` with NO virtual
    factor, and ``gz`` accumulates UPWARD from ``hs``.
    """
    ia = ng
    ptw = pt[:, ia:ia + n, ia:ia + n, :]
    dpe = peln[:, :, 1:, :] - peln[:, :, :-1, :]
    dpe = jnp.transpose(dpe, (0, 1, 3, 2))
    hs_w = hs[:, ia:ia + n, ia:ia + n]
    gz = hs_w + jnp.sum(rg * ptw * dpe, axis=3)       # k = 1..km, upward
    pe_w = pe[:, 1:n + 1, :, 1:n + 1]
    te = (pe_w[:, :, km, :] * hs_w) - (pe_w[:, :, 0, :] * gz)
    return te + _ke_column_jax(delp[:, ia:ia + n, ia:ia + n, :], u, v,
                               rsin2, cosa_s, cp_times=cp * ptw, n=n, ng=ng)


def energy_fixer_zsum0_hydrostatic_jax(pkz, delp, pk, *, ptop, n, ng, km):
    """``zsum0`` (:692-703), face-stacked."""
    ia = ng
    zsum1 = jnp.sum(pkz * delp[:, ia:ia + n, ia:ia + n, :], axis=3)
    return ptop * (pk[:, ia:ia + n, ia:ia + n, 0]
                   - pk[:, ia:ia + n, ia:ia + n, km]) + zsum1


def energy_fixer_dtmp_jax(te0, te, zsum0, area, *, consv, n, ng):
    """``dtmp`` (:708-714) -- a TRACED scalar, not a Python float.

    See the spec twin's docstring for why the numerator's cancellation
    means the summation choice is a measured question rather than an
    obvious one.
    """
    ia = ng
    a = area[:, ia:ia + n, ia:ia + n]
    return consv * jnp.sum((te0 - te) * a) / jnp.sum(zsum0 * a)


def _hs_face_jax(ctx, t: int, n: int, ng: int):
    """Surface geopotential for face ``t`` as NUMPY, or zeros.

    The energy integrals are the spec lane's and take numpy; every duo
    deck resolves ``mountain = .F.`` and the parity harness asserts the
    oracle's phis is identically zero, so this is zeros in practice --
    written against a real array because the integrals reference ``hs``
    twice each and a non-zero-orography deck would need it.
    """
    hs6 = getattr(ctx, "hs6", None)
    if hs6 is None:
        return np.zeros((n + 2 * ng, n + 2 * ng), dtype=np.float64)
    return np.asarray(hs6[t], dtype=np.float64)


def pt_to_theta_v(pt, pkz, *, n: int, ng: int, dp1=None):
    """fv_dynamics.F90:396-408, COMPUTE WINDOW ONLY -- functional (C4):
    returns the converted pt instead of mutating it.

    The halo rows keep TEMPERATURE until the it==1 scalar exchange
    (dyn_core.F90:470) overwrites them; converting the padded array would
    also convert the corner-diagonal sentinels, which are NOT overwritten.
    ``dp1=None`` is the adiabatic ``zvir = 0`` lane. It is a
    separate branch because the oracle forms no ``dp1`` there at
    all (:281-294 is the moist branch), NOT because a zeros array
    would round differently -- since the association was corrected
    to ``(pt*(1+dp1))/pkz`` a zeros array is bit-identical
    (``win*1.0`` is exact; codex confirmed this holds for every finite
    input including subnormals and both signed zeros, job 9442717).
    Gated in THIS lane by
    ``tests/grids/test_fv3_dynamics.py::test_dry_branch_is_not_a_multiply_by_one``
    and in the NumPy lane by
    ``test_fv3_moist_dynamics.py::test_dry_lane_is_bitwise_under_the_moist_patch``.
    """
    pt = jnp.asarray(pt)
    pkz = jnp.asarray(pkz)
    require_f64_jax("pt_to_theta_v",
                    {"pt": pt, "pkz": pkz, "dp1": dp1})
    ia = ng
    win = pt[:, ia:ia + n, ia:ia + n, :]
    if dp1 is None:  # static None-ness: stays a Python if
        new = win / pkz
    else:
        # ASSOCIATION IS THE ORACLE'S. Fortran evaluates `pt*(1.+dp1)/pkz`
        # (:402) left to right as (pt*(1+dp1))/pkz. Forming the quotient
        # first and multiplying rounds differently; the spec lane carries
        # the same comment for the same reason.
        new = win * (1.0 + dp1) / pkz
    # the six per-face compute windows are disjoint; no face reads
    # another's write, so the spec's face loop is one stacked slice
    return pt.at[:, ia:ia + n, ia:ia + n, :].set(new)


def p_var_nonhydrostatic(delp, delz, pt, *, ptop, akap, n: int, ng: int,
                         km: int, check_args: bool = False,
                         dp1=None) -> dict:
    """``p_var``'s NON-hydrostatic pkz on top of the hydrostatic column.

    init_hydro.F90:178-184 (dry): ``pkz = exp(cappa*log(rdg*delp*pt/delz))``
    with ``rdg = -rdgas/grav`` and pt still TEMPERATURE.  Feeding the
    hydrostatic kappa-mean pkz into an NH run's conversion puts a uniform
    second-order error on theta (a 0.408 m delz residual in first parity).
    ``delz`` broadcasts to the compute window exactly as in the spec's own
    expression.  pkz's layout IS the compute window, so the whole array is
    replaced.

    ``dp1`` is the MOIST arm (``fv_dynamics.F90:307-309``, the
    ``moist_phys`` branch), where the log argument carries an extra
    ``(1.+dp1)``.  It is a separate branch, matching the oracle's own
    ``moist_phys`` split, so the certified dry NH lane keeps its exact
    expression;
    the factor sits INSIDE the log in the Fortran's own left-to-right
    association, not applied to ``pt`` at the call site, which would
    reassociate the product.  The oracle's ``moist_phys = .false.`` arm
    (``:335``) forces ``dp1 = 0`` and computes the DRY form, so it is
    reached here by passing ``dp1=None`` -- i.e. it is exactly the
    ``zvir = 0`` lane and needs no flag of its own.
    """
    delp = jnp.asarray(delp)
    delz = jnp.asarray(delz)
    pt = jnp.asarray(pt)
    require_f64_jax("p_var_nonhydrostatic",
                    {"delp": delp, "delz": delz, "pt": pt})
    out = p_var_hydrostatic(delp, ptop=ptop, akap=akap, n=n, ng=ng, km=km,
                            check_args=check_args)
    rdg = -_FV3_RDGAS / _FV3_GRAV
    ia = ng
    dpw = delp[:, ia:ia + n, ia:ia + n, :]
    ptw = pt[:, ia:ia + n, ia:ia + n, :]
    if dp1 is not None:
        dp1 = jnp.asarray(dp1)
        require_f64_jax("p_var_nonhydrostatic", {"dp1": dp1})
    out["pkz"] = jnp.exp(akap * jnp.log(
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
    if dp1 is None:  # static None-ness: stays a Python if
        return rdg * dpw * ptw / delz
    return rdg * dpw * ptw * (1.0 + dp1) / delz


def fv_dynamics_step(ctx: dict, state: dict, press: dict, *,
                     bdt, km: int, k_split: int, n_split: int,
                     ptop, ak, bk, akap, cp_air,
                     kord_mt: int, kord_tm: int, kord_tr,
                     q, omga=None, nh=None,
                     zvir: float = 0.0, consv_te: float = 0.0,
                     batched: bool = False,
                     sphum_index: int | None = None,
                     n_sponge: int = -1, tau: float = -1.0,
                     hydrostatic: bool = True,
                     p_fac=0.05, a_imp=1.0, use_logp: bool = False,
                     kord_wz: int = 9, w_limiter=None,
                     cfg: dict | None = None, a2b_ord: int = 4,
                     check_state: bool = False,
                     hord_tr: int = 6, tracer_q_split: int = 0,
                     nord_tr: int = 0, trdm2=0.0, lim_fac=1.0,
                     z_tracer: bool = True, inline_q: bool = False,
                     fill: bool = False) -> dict:
    """One ``fv_dynamics`` call: ``bdt`` of model time (:451-674), functional.

    C4/D4 returns ``{"state", "press", "q", "omga", "nh", "stages",
    "omga_is_meaningless", "pt_units"}`` and the CALLER threads everything
    into the next call; nothing is written through an argument.
    ``state`` is ONE face-stacked dict (delp, pt, u, v; w and delz on NH);
    ``press`` is ONE face-stacked dict (ps, pe, peln, pk, pkz) as
    :func:`p_var_hydrostatic` builds them; ``q`` is a list of nq stacked
    tracer arrays each shaped like stacked delp (6, m_a, m_a, km).
    ``nh`` (None on the hydrostatic lane) is the NH carry threaded through
    fv3_acoustic_3d.build_nh_carry's contract -- its shapes are THAT
    docstring's, traced to the kernel contracts there: zh/gz/pk3 padded
    (6,m_a,m_a,km+1), zs/ws3 padded (6,m_a,m_a), ws compute (6,n,n),
    pe/peln (6,)+field_shape, pk compute (6,n,n,km+1).  This module
    re-derives none of them (R9).

    ``stages`` is the LAST n_map iteration's acoustic payload, by return
    not callback (C6/R8; the spec's stage_hook is not ported): S10 the
    pre-barrier ubb/vbb/ubbtemp/vbbtemp stacks, S11 the post-barrier
    ubb/vbbtemp, S16 geopk's pk/gz before one_grad_p scratched them.  The
    remap reads ``pk_remap`` (the final sub-step's pre-scratch snapshot),
    never ``pk`` (D2).  S12 (corner KE) and S13/S14 are recoverable from
    those payloads plus the returned fields, so they are NOT re-returned.

    Static/dynamic (C3/D5): ctx, km, k_split, n_split, kords, flags and
    deck selectors are STATIC; bdt and the fields are DYNAMIC, with
    ``mdt = bdt/k_split`` (:413).  ptop/akap/cp_air/p_fac/a_imp are kept
    DYNAMIC (R12): no callee's given signature branches on them in Python
    (acoustic_loop_3d and lagrangian_to_eulerian take them as traced
    operands), and this module's only ptop branch -- p_var's PTOP_MIN arm
    -- is a jnp.where with both arms finite.  ``check_state`` is the
    spec's ``validate`` renamed (C5); it raises on a tracer, so run it
    eagerly outside jit.

    THE RETURNED ``press["pkz"]`` IS DRY, ALWAYS.  The remap writes it
    from the post-remap state with no ``(1+dp1)`` (``fv_mapz.F90:
    479-481``), and on the moist NH arm the NEXT call overwrites it at
    ``fv_dynamics.F90:299-322``.  That is exactly what the oracle
    exposes between calls, so it is not a divergence -- but a consumer
    that stops BETWEEN steps and assumes "moist run implies moist pkz"
    would be wrong, which is why it is stated here rather than only in
    a test (GLM MINOR, job 9442483).

    pt enters as TEMPERATURE and leaves as TEMPERATURE (theta_v in
    between: :396-408 in, fv_mapz.F90:209-217 out); a caller stopping in
    between gets theta_v.  omga is returned but meaningless (use_old_omega
    fill not ported; it touches no parity field).
    """
    # ---- static entry guards (C3/C5/D6): fail-closed, never a silent lane
    if k_split < 1:
        raise ValueError(f"k_split must be >= 1, got {k_split}")
    if n_split < 1:
        raise ValueError(f"n_split must be >= 1, got {n_split}")
    require_uniform_damping_lane(n_sponge=n_sponge, tau=tau, npz=km)
    if zvir != 0.0:  # static deck constant, stays a Python if
        # HYDROSTATIC moist coupling is enabled, mirroring the spec lane
        # (fv3_native_dynamics.py). dp1 = zvir*q(sphum)
        # (fv_dynamics.F90:291; USE_COND is not defined in the pinned
        # build, so no q_con term) feeds pt = pt*(1.+dp1)/pkz (:402) and
        # the closing pt/(1+r_vir*q) (fv_mapz.F90:975), which
        # lagrangian_to_eulerian already carries via r_vir=zvir below.
        # Validate, never default: a guessed tracer index would silently
        # couple an arbitrary species into theta_v.
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
        if len(q) <= 0:
            raise ValueError(
                "zvir != 0 requires nq > 0, but q carries no tracers "
                "(fv_dynamics.F90:291 needs sphum).")
        if not 0 <= sphum_index < len(q):
            raise ValueError(
                f"sphum_index={sphum_index} out of range [0, {len(q)}); a "
                f"negative index would silently select another tracer by "
                f"Python wrap-around.")
        if not hydrostatic and "delz" not in state:
            raise ValueError(
                "zvir != 0 with non-hydrostatic dynamics needs state["
                "'delz']: the moist NH pkz is recomputed here from "
                "delp/pt/delz (fv_dynamics.F90:299-322).")
    if abs(consv_te) > _CONSV_MIN:
        # Fortran's DEAD BAND (fv_mapz.F90:630): 0 < |consv| <= consv_min
        # is ACCEPTED and leaves dtmp exactly 0, so it is fixer-OFF here
        # rather than an error (codex MINOR, job 9446299). Raising on it
        # was stricter than the oracle.
        if not hydrostatic:
            raise NotImplementedError(
                "consv_te != 0 with non-hydrostatic dynamics is not "
                "enabled: compute_total_energy and the fixer both take "
                "their NON-hydrostatic branches (fv_mapz.F90:1155-1190 "
                "and :659-687), which integrate phiz from delz and carry "
                "the w**2 term. Only the hydrostatic pair is ported.")
        if consv_te < 0.0:
            # NEGATIVE consv IS A DIFFERENT PROGRAM, not a sign choice
            # (codex MAJOR, job 9446299). fv_mapz.F90:738-741 treats it
            # as a PRESCRIBED energy flux --
            # dtmp = consv*(grav*pdt*4*pi*radius**2)/g_sum(zsum0) -- and
            # never forms te0_2d - te_2d at all. Accepting it here would
            # run the positive branch's physics under the negative
            # branch's flag.
            raise NotImplementedError(
                f"consv_te={consv_te} < 0: fv_mapz.F90:738-741 is the "
                f"PRESCRIBED-FLUX branch, which needs pdt, grav and the "
                f"planetary radius and does not use te0_2d - te_2d. Only "
                f"the positive branch (:630-715) is ported.")
        if zvir != 0.0:
            # MOIST x CONSV IS UNSCORED (GLM MAJOR, job 9446300). The
            # harness refuses the combination, but a refusal that lives
            # only in the harness is not a refusal: the qc path of
            # total_energy_2d_hydrostatic (forms tv itself, integrates
            # DOWN) and the no-virtual path of
            # fixer_energy_2d_hydrostatic (takes T_v, integrates UP) are
            # exactly the distinction this port advertises, and under
            # every existing gate they are dead code. Build a moist
            # consv deck before enabling this.
            raise NotImplementedError(
                "zvir != 0 with consv_te != 0 has no oracle deck, so the "
                "two energy integrals' virtual-temperature conventions "
                "are unscored -- the one thing about this port most "
                "likely to be wrong. Build a moist consv_te deck "
                "(build_consv_te_oracle.sbatch on the moist deck) "
                "first.")
        if getattr(ctx, "gs6", None) is None:
            raise ValueError(
                "consv_te != 0 needs ctx.gs6: the energy integrals are "
                "area-weighted and use rsin2/cosa_s "
                "(fv_mapz.F90:650-656).")
    n, ng = ctx.n, ctx.ng
    nb = batch_size(ctx)          # 6 faces, or the window ctx's 6*kt*kt
    want_delp = (nb,) + tuple(field_shape("delp", n, ng, km))
    for nm in ("delp", "pt", "u", "v"):
        if nm not in state:
            raise ValueError(f"state is missing '{nm}'")
        if state[nm].shape[0] != nb:
            raise ValueError(f"state['{nm}'] must lead with {nb} faces/"
                             f"windows, got {state[nm].shape}")
    for nm in ("delp", "pt"):
        if state[nm].shape != want_delp:
            raise ValueError(f"state['{nm}'] must be {want_delp}, got "
                             f"{state[nm].shape}")
    for nm in ("ps", "pe", "peln", "pk", "pkz"):
        if nm not in press:
            raise ValueError(f"press is missing '{nm}'")
        if press[nm].shape[0] != nb:
            raise ValueError(f"press['{nm}'] must lead with {nb} faces/"
                             f"windows, got {press[nm].shape}")
    nq = len(q)
    for i, qt in enumerate(q):
        if qt.shape != want_delp:
            raise ValueError(f"tracer {i} must be {want_delp}, got "
                             f"{qt.shape}")
    if nq > 0:
        # refuse an unported tracer arm BEFORE k_split*n_split sub-steps
        require_tracer_2d_1l_lane(z_tracer=z_tracer, q_split=tracer_q_split,
                                  nord_tr=nord_tr, trdm=trdm2,
                                  inline_q=inline_q)
    if not hydrostatic:
        for nm in ("delz", "w"):
            if nm not in state:
                raise ValueError(f"hydrostatic=False needs '{nm}' in the "
                                 f"state")
        # attribute access, not dict .get -- see fv3_acoustic_3d
        if getattr(ctx, "hs6", None) is None:
            raise ValueError("hydrostatic=False needs ctx['hs6'] (phis) for "
                             "the NH carry (zs = phis/grav, "
                             "dyn_core.F90:262-278)")
        if w_limiter is None:
            raise ValueError(
                "hydrostatic=False needs an explicit w_limiter (the "
                "resolved NH deck runs W_LIMITER=T; fv_mapz.F90:368)")
    w_limiter = bool(w_limiter) if w_limiter is not None else False

    if omga is None:
        # dtype follows storage (fp32/fp64), from state["delp"]
        omga = jnp.zeros(want_delp, dtype=state["delp"].dtype)
    ak = jnp.asarray(ak)
    bk = jnp.asarray(bk)
    require_f64_jax("fv_dynamics_step",
                    {f"leaf[{i}]": v for i, v in
                     enumerate(jax.tree_util.tree_leaves(
                         (state, press, q, omga, nh)))})

    # dyn_core.F90:269 -- dp_ref(k) = ak(k+1)-ak(k) + (bk(k+1)-bk(k))*1.E5
    # (1.E5 is the literal reference surface pressure, not a tunable)
    dp0 = (ak[1:] - ak[:-1]) + (bk[1:] - bk[:-1]) * 1.0e5

    if not hydrostatic and nh is None:
        # R9/D5: the NH carry is functional; build the seed once here so
        # every n_map iteration is the same program (see D1 below)
        nh = build_nh_carry(ctx, km, ctx.hs6)

    # :413  mdt = bdt / k_split  (dynamic: a new bdt must not recompile)
    mdt = bdt / float(k_split)
    ia = ng
    remapped = km > REMAP_MIN_NPZ  # :568, static
    # fv_mapz.F90:985: on the NH arm the closing T_v -> T conversion is
    # inside `if (.not. adiabatic)`.  The certified deck is adiabatic AND
    # dry, so True was unconditionally right; with moist NH it is not --
    # adiabatic=True there would make the oracle skip the conversion and
    # leave pt virtual.  With consv = 0, dtmp is identically 0 and the
    # oracle's :987 expression is exactly the one this lane computes, so
    # False is the faithful flag for the moist NH arm.  The remap refuses
    # the other combination rather than trusting this line.
    adiabatic_flag = hydrostatic or zvir == 0.0

    # :396-408  T -> theta_v once per call; the six per-face compute
    # windows are disjoint, so the spec's face loop is one stacked call
    state = dict(state)
    # :355-365  te0_2d, BEFORE the theta conversion below, because
    # compute_total_energy runs at :359 while pt is still TEMPERATURE.
    # THE SPEC'S INTEGRALS ARE REUSED, not re-written in jnp: they are
    # pure per-face numpy, run ONCE per step outside any traced loop,
    # and a second implementation of numerics this repo already has is
    # exactly what the no-duplicate-numerics rule forbids.
    te0_2d = None
    if abs(consv_te) > _CONSV_MIN:
        _gs = ctx.gs6
        _stack = lambda k: jnp.asarray(  # noqa: E731
            np.stack([np.asarray(_gs[t][k]) for t in range(nb)]))
        _rsin2, _cosa_s = _stack("rsin2"), _stack("cosa_s")
        _area = _stack("area")
        _hs = jnp.asarray(np.stack([_hs_face_jax(ctx, t, n, ng)
                                    for t in range(nb)]))
        qc = (zvir * q[sphum_index][:, ia:ia + n, ia:ia + n, :]
              if zvir != 0.0 else None)
        te0_2d = total_energy_2d_hydrostatic_jax(
            state["pt"], state["delp"], state["u"], state["v"],
            press["pe"], press["peln"], _hs, _rsin2, _cosa_s,
            qc=qc, cp=cp_air, rg=_FV3_RDGAS, n=n, ng=ng, km=km)
    if zvir != 0.0:
        # dp1 = zvir*q(i,j,k,sphum) (fv_dynamics.F90:291), formed ONCE
        # from the step-initial q before the k_split loop, exactly as
        # :281-294 precedes :451. Sliced to the COMPUTE WINDOW because
        # pkz is (6, n, n, km) and pt_to_theta_v expects dp1 matching it.
        dp1_theta = zvir * q[sphum_index][:, ia:ia + n, ia:ia + n, :]
        if not hydrostatic:
            # :299-322 is INSIDE fv_dynamics and runs on every call on
            # BOTH NH arms -- but only the MOIST one is recomputed here.
            # The dry NH arm still trusts the caller's pkz, which is
            # covered by the certified dry parity and is deliberately
            # left alone; do not read this comment as licence to
            # recompute there (GLM, job 9442483). On the moist arm:
            # the NH moist arm it OVERWRITES pkz with
            # exp(kappa*log(rdg*delp*pt*(1.+dp1)/delz)) from the
            # step-entry state, while pt is still TEMPERATURE. The
            # caller's pkz -- built dry by p_var_nonhydrostatic, or
            # carried over from the previous step's remap -- is missing
            # the virtual factor, so it is recomputed here rather than
            # trusted. The dry NH lane never enters this branch and
            # keeps its certified expression untouched.
            press = dict(press)
            press["pkz"] = p_var_nonhydrostatic(
                state["delp"], state["delz"], state["pt"], ptop=ptop,
                akap=akap, n=n, ng=ng, km=km, dp1=dp1_theta)["pkz"]
        state["pt"] = pt_to_theta_v(state["pt"], press["pkz"], n=n, ng=ng,
                                    dp1=dp1_theta)
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
        state["pt"] = pt_to_theta_v(state["pt"], press["pkz"], n=n, ng=ng)

    def _n_map(carry, last_step: bool):
        st, pr, qq, om, nhc, nspl, nexc = carry
        # :472-478 dp1 = delp, full padded box, BEFORE dyn_core; arrays are
        # immutable here so the spec's anti-alias copy is a plain binding.
        # NAMED dp1_delp, not dp1: the Fortran reuses the name `dp1` for
        # two unrelated things (the moist zvir*q above and this delp
        # snapshot), and one of them is now live on this lane.
        dp1_delp = st["delp"]
        # dyn_core.F90:313-316 sits INSIDE dyn_core: one zeroing per n_map
        # call (never per acoustic sub-step); k_split=1 makes it once per
        # fv_dynamics_step.  This module owns the zeroing (D3).
        fc = (alloc_flux_capacitors(n, ng, km, dtype=st["delp"].dtype,
                                    nb=nb)
              if nq > 0 else None)
        # :502 dyn_core; press_out's role is taken by ac["press"] (the
        # callee contract supplies it, replacing the spec's length check)
        ac = acoustic_loop_3d(ctx, st, mdt, km, batched=batched,
                              n_split=n_split, ptop=ptop,
                              akap=akap, cp_air=cp_air, cfg=cfg,
                              check_state=check_state,
                              remap_follows=remapped,
                              hydrostatic=hydrostatic, nh=nhc,
                              p_fac=p_fac, a_imp=a_imp, dp0=dp0,
                              use_logp=use_logp, flux_cap=fc)
        st = ac["state"]
        nhc = ac["nh"]
        g = ac["press"]
        if nq > 0:
            # :517/:528-540 tracer_2d_1L on the accumulated capacitors
            # (inline_q is refused above, so the gate is just nq)
            # LAYOUT BRIDGE, not a reshape for convenience: this module
            # carries q tracer-major (a list of nq face-stacked arrays,
            # which is what the per-face remap at :480 indexes), while
            # tracer_2d_1l_sixface takes ONE (6, nq, m_a, m_a, km) stack
            # -- the spec's own [face][iq] order. Passing the list
            # straight through made the callee see nq as the face axis.
            _q_in = jnp.stack(qq, axis=1)
            # WINDOW lane: the tracer step is a phase of its own between
            # the acoustic loop and the remap, and like every acoustic
            # substep it needs its seam pads rebuilt from the owners at
            # ENTRY (acoustic_loop_3d does this for state/nh/flux_cap).
            # Without it the tracer's pad NaN (the outer stencil-reach
            # cells, expected) survived into the next step and ate ~2
            # cells inward per step, reaching owned cells at step 3
            # (C24 kt=2 pad=5 n_split=8; gate jobs 9910440/1, probe
            # 9912744; forced refresh before OR after the step confines
            # it, jobs 9912821/2).  No-op on the six-face lane.
            # One tracer per key (each (nb, W, W, km), the layout pt already
            # relies on) in ONE firing -- NOT km*nq merged into the trailing
            # axis, which can collide with a horizontal extent and trip the
            # layout classifier (codex: nq=4, km=5 at C24 gives 20 = n_w+1).
            _wc = getattr(getattr(ctx, "tab", None), "window_comm", None)
            if _wc is not None:
                _ref = _wc.refresh({f"q{i}": _q_in[:, i] for i in range(nq)})
                _q_in = jnp.stack([_ref[f"q{i}"] for i in range(nq)], axis=1)
            _tr = tracer_2d_1l_sixface(ctx, _q_in, dp1_delp, ac["flux_cap"],
                                       km=km, nq=nq, hord_tr=hord_tr,
                                       dt=mdt, q_split=tracer_q_split,
                                       nord_tr=nord_tr, trdm=trdm2,
                                       lim_fac=lim_fac, z_tracer=z_tracer,
                                       inline_q=inline_q, batched=batched)
            # The routine returns a DICT (C4: everything the spec
            # mutates). Binding the whole dict to qq made the remap's
            # `qq[i]` a KeyError on any run with tracers.
            qq = [_tr["q"][:, i] for i in range(nq)]
            # ...and its loud-failure pair has to travel with it. The
            # tracer factory's default caller runs check_nsplt_schedule;
            # this is the RAW routine, so an nsplt above NSPLT_MAX would
            # otherwise run NSPLT_MAX passes and under-advect SILENTLY --
            # the exact failure C5 exists to prevent. The flag rides the
            # carry (a Python bool on it here would be a tracer error,
            # and impossible inside the scan) and is checked once, by the
            # caller, where it is concrete.
            nspl = _tr["nsplt"]
            nexc = jnp.logical_or(nexc, _tr["nsplt_exceeded"])
        if not remapped:
            # :568 gates the remap on npz > 4 and the oracle leaves pt in
            # theta_v below it.  BUT dyn_core still wrote pe/peln/pkz and
            # copied pk over the compute window, and the oracle keeps all
            # of that -- a stale pkz would poison the next call's
            # pt -> theta_v, so mirror the refresh exactly.
            newpr = {"ps": pr["ps"], "pe": g["pe"], "peln": g["peln"],
                     "pk": pr["pk"], "pkz": pr["pkz"]}
            if hydrostatic:
                newpr["pkz"] = g["pkz"]
                if "pk_remap" in g:  # static dict structure
                    newpr["pk"] = pr["pk"].at[:, ia:ia + n, ia:ia + n, :] \
                        .set(g["pk_remap"][:, ia:ia + n, ia:ia + n, :])
            else:
                # NH pk is Riem_Solver3's compute-window last_call copy;
                # the NH tail carries no pkz (mapz owns it)
                newpr["pk"] = pr["pk"].at[:, ia:ia + n, ia:ia + n, :] \
                    .set(g["pk"])
            # SEVEN elements, like the other return: the carry gained
            # nsplt/nsplt_exceeded and this early branch was missed.
            return (st, newpr, qq, om, nhc, nspl, nexc), ac["stages"]

        # press refresh hoisted out of the spec's per-face loop: each
        # face's update touches only that face's slices and no face reads
        # another's write, so hoisting is identity-preserving
        pr = dict(pr)
        pr["pe"] = g["pe"]
        pr["peln"] = g["peln"]
        if hydrostatic:
            if "pk_remap" not in g:
                raise RuntimeError(
                    "the remap-step geopk bundle has no 'pk_remap': "
                    "dyn_core.F90:1511-1519 saves pk BEFORE one_grad_p "
                    "overwrites pkc with B-grid corner values, so "
                    "remapping against g['pk'] would use the wrong "
                    "staggering")
            pr["pkz"] = g["pkz"]
            # D2/R8: read pk_remap (the pre-scratch snapshot), never pk
            pr["pk"] = pr["pk"].at[:, ia:ia + n, ia:ia + n, :].set(
                g["pk_remap"][:, ia:ia + n, ia:ia + n, :])
        else:
            pr["pk"] = pr["pk"].at[:, ia:ia + n, ia:ia + n, :].set(g["pk"])

        # :618 the remap, ONE FACE PER CALL -- the callee's contract is
        # per-face with the numpy-lane layouts.  Two arms: the certified
        # default is the 6-deep static (unrolled) face loop below (no
        # face reads another's writes); `batched` (face-batching ladder
        # step 6, pattern 4c7198bd1) vmaps the SAME callee over the face
        # axis instead -- each traced `x[t]` read in the loop is a
        # masked-select + all-reduce under SPMD, and the remap was one
        # of the two remaining per-face `[t]` sites.
        if batched:
            # in_axes: every per-face operand at axis 0 -- the press
            # bundle (pe/peln/pk/pkz/ps), the state (delp/pt/u/v),
            # each tracer leaf of the q list, omga, and on the NH arm
            # w/delz/ws (vmap's default in_axes=0 over the positional
            # args of the wrapper).  Closed over (face-invariant at
            # trace time): ak/bk/ptop/akap/cp_air/zvir/km/n/ng, the
            # three kords, kord_wz/w_limiter, hydrostatic/adiabatic/
            # consv_te/sphum_index, and last_step/defer_close (Python
            # bools per k_split iteration).  The callee Python-branches
            # only on those statics, never on a per-face VALUE (read,
            # fv3_mapz.py:1319-1720), so the six faces are one program.
            _defer = bool(last_step) and abs(consv_te) > _CONSV_MIN

            def _remap_face(pe_t, peln_t, pk_t, pkz_t, delp_t, pt_t,
                            u_t, v_t, ps_t, q_t, om_t, *nh_t):
                w_t, delz_t, ws_t = nh_t if nh_t else (None, None, None)
                return lagrangian_to_eulerian(
                    pe=pe_t, peln=peln_t, pk=pk_t, pkz=pkz_t,
                    delp=delp_t, pt=pt_t, u=u_t, v=v_t, ps=ps_t,
                    ak=ak, bk=bk, ptop=ptop, akap=akap, cp=cp_air,
                    r_vir=zvir, km=km, n=n, ng=ng,
                    kord_mt=kord_mt, kord_tm=kord_tm, kord_tr=kord_tr,
                    q=list(q_t), omga=om_t, sphum_index=sphum_index,
                    last_step=last_step, hydrostatic=hydrostatic,
                    adiabatic=adiabatic_flag, consv=consv_te,
                    w=w_t, delz=delz_t, ws=ws_t,
                    kord_wz=kord_wz, w_limiter=w_limiter,
                    rdgas=(None if hydrostatic else _FV3_RDGAS),
                    grav=(None if hydrostatic else _FV3_GRAV),
                    fill=fill, do_sat_adj=False, do_inline_mp=False,
                    do_adiabatic_init=False, defer_close=_defer)

            _nh_ops = (() if hydrostatic
                       else (st["w"], st["delz"], g["ws"]))
            o = jax.vmap(_remap_face)(
                pr["pe"], pr["peln"], pr["pk"], pr["pkz"],
                st["delp"], st["pt"], st["u"], st["v"], pr["ps"],
                list(qq), om, *_nh_ops)
            # Assembly mirrors the loop arm KEY-FOR-KEY: vmap stacks
            # every output leaf at axis 0, which is exactly what the
            # loop arm's jnp.stack builds, and the same carry-contract
            # rules apply (w/delz only on NH; unowned keys carried
            # through; q back to a LIST -- the callee returns a tuple).
            _st_in = st
            st = {"delp": o.delp, "pt": o.pt, "u": o.u, "v": o.v}
            if not hydrostatic:
                st["w"] = o.w
                st["delz"] = o.delz
            for _k in _st_in:
                if _k not in st:
                    st[_k] = _st_in[_k]
            pr = {"ps": o.ps, "pe": o.pe, "peln": o.peln,
                  "pk": o.pk, "pkz": o.pkz}
            if nq > 0:
                qq = list(o.q)
            om = o.omga
        else:
            fs = []
            for t in range(nb):
                fs.append(lagrangian_to_eulerian(
                    pe=pr["pe"][t], peln=pr["peln"][t], pk=pr["pk"][t],
                    pkz=pr["pkz"][t], delp=st["delp"][t], pt=st["pt"][t],
                    u=st["u"][t], v=st["v"][t], ps=pr["ps"][t],
                    ak=ak, bk=bk, ptop=ptop, akap=akap, cp=cp_air,
                    r_vir=zvir, km=km, n=n, ng=ng,
                    kord_mt=kord_mt, kord_tm=kord_tm, kord_tr=kord_tr,
                    q=[qq[i][t] for i in range(nq)],
                    omga=om[t], sphum_index=sphum_index,
                    last_step=last_step, hydrostatic=hydrostatic,
                    adiabatic=adiabatic_flag, consv=consv_te,
                    w=(None if hydrostatic else st["w"][t]),
                    delz=(None if hydrostatic else st["delz"][t]),
                    ws=(None if hydrostatic else g["ws"][t]),
                    kord_wz=kord_wz, w_limiter=w_limiter,
                    rdgas=(None if hydrostatic else _FV3_RDGAS),
                    grav=(None if hydrostatic else _FV3_GRAV),
                    fill=fill, do_sat_adj=False, do_inline_mp=False,
                    do_adiabatic_init=False,
                    defer_close=(bool(last_step) and abs(consv_te) > _CONSV_MIN)))
            # PYTREE STRUCTURE IS PART OF THE CARRY CONTRACT. The remap owns
            # delp/pt/u/v always and w/delz only on the NH arm, so rebuilding
            # from scratch DROPS a hydrostatic run's `w` -- which the state
            # still carries (build_state_3d allocates it zeroed either way).
            # lax.scan then rejects the k_split loop with "carry input and
            # carry output must have the same pytree structure ... symmetric
            # difference {'w'}". Any key the remap does not own is carried
            # through untouched rather than recreated, so the structure is
            # whatever the caller handed in.
            _st_in = st
            st = {"delp": jnp.stack([o.delp for o in fs]),
                  "pt": jnp.stack([o.pt for o in fs]),
                  "u": jnp.stack([o.u for o in fs]),
                  "v": jnp.stack([o.v for o in fs])}
            if not hydrostatic:
                st["w"] = jnp.stack([o.w for o in fs])
                st["delz"] = jnp.stack([o.delz for o in fs])
            for _k in _st_in:
                if _k not in st:
                    st[_k] = _st_in[_k]
            pr = {"ps": jnp.stack([o.ps for o in fs]),
                  "pe": jnp.stack([o.pe for o in fs]),
                  "peln": jnp.stack([o.peln for o in fs]),
                  "pk": jnp.stack([o.pk for o in fs]),
                  "pkz": jnp.stack([o.pkz for o in fs])}
            if nq > 0:
                # rebuild the list of stacked tracers from the per-face lists
                # list(), not the tree_map result as-is: the callee returns
                # its tracers as a tuple, and a carry that goes in a list and
                # comes out a tuple is a pytree-structure mismatch under
                # lax.scan even though every leaf matches.
                qq = list(jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                                 *[o.q for o in fs]))
            om = jnp.stack([o.omga for o in fs])

        if last_step and abs(consv_te) > _CONSV_MIN:
            # THE REDUCTION, and the reason the remap was split. fv_mapz
            # does te_2d -> g_sum -> apply inside ONE call because its
            # "domain" is every tile; a call here is one face, so all
            # six are collected, summed with area weights, and only then
            # is :975 applied. pt is T_v on every face at this point --
            # defer_close skipped exactly that conversion.
            #
            # pr["pe"]/["peln"]/["pkz"] here are the POST-remap bundle
            # rebuilt from `fs` a few lines above, not the caller's --
            # GLM asked for that in writing, because a stale pe would
            # move dtmp at O(1) relative and pt by ~1e-8, over the gate.
            te_2d = fixer_energy_2d_hydrostatic_jax(
                st["pt"], st["delp"], st["u"], st["v"],
                pr["pe"], pr["peln"], _hs, _rsin2, _cosa_s,
                cp=cp_air, rg=_FV3_RDGAS, n=n, ng=ng, km=km)
            zsum0 = energy_fixer_zsum0_hydrostatic_jax(
                pr["pkz"], st["delp"], pr["pk"], ptop=ptop, n=n, ng=ng,
                km=km)
            dtmp = energy_fixer_dtmp_jax(te0_2d, te_2d, zsum0, _area,
                                         consv=consv_te, n=n, ng=ng)
            st = dict(st)
            if batched:
                # Face-batch the energy fixer's close_out_pt (the last
                # dormant per-face x[t] read; consv_te-only).  dtmp is a
                # GLOBAL scalar reduction (energy_fixer_dtmp_jax sums over
                # every face) so it is face-invariant -> closed over, NOT
                # mapped; pt/pkz and each q leaf are per-face -> in_axes 0.
                # A positional-only wrapper keeps the kwargs (static +
                # the closed dtmp) out of vmap's in_axes.
                def _close_one(pt_t, pkz_t, q_t):
                    return close_out_pt(
                        pt_t, pkz_t, q_t, sphum_index=sphum_index,
                        r_vir=zvir, dtmp=dtmp, cp=cp_air, n=n, ng=ng,
                        fixer_on=True)
                st["pt"] = jax.vmap(_close_one, in_axes=(0, 0, 0))(
                    st["pt"], pr["pkz"], [qq[i] for i in range(nq)])
            else:
                st["pt"] = jnp.stack([
                    close_out_pt(st["pt"][t], pr["pkz"][t],
                                 [qq[i][t] for i in range(nq)],
                                 sphum_index=sphum_index, r_vir=zvir,
                                 dtmp=dtmp, cp=cp_air, n=n, ng=ng,
                                 fixer_on=True)
                    for t in range(nb)])

        return (st, pr, qq, om, nhc, nspl, nexc), ac["stages"]

    # nsplt seeds at 1 (a schedule of all-ones is "no sub-cycling", the
    # correct answer when nq == 0 and the tracer routine never runs) and
    # the exceeded flag at False.
    carry0 = (state, press, q, omga, nh,
              jnp.ones((km,), dtype=jnp.int32), jnp.asarray(False))
    # D1: with the NH carry prebuilt at entry, iterations 1..k_split-1 are
    # ONE program (dp1_delp and the capacitors are re-derived identically each
    # time); only the LAST differs -- last_step=True reaches the remap
    # (:482) -- so the middle scans and the last is peeled.  k_split == 1
    # is first AND last.  Middle iterations' stage payloads are dropped,
    # exactly as the spec drops the middle press bundles.
    if k_split == 1:
        carry, stages = _n_map(carry0, True)
    else:
        def _mid(c, _):
            c2, _s = _n_map(c, False)
            return c2, None
        carry, _ = lax.scan(_mid, carry0, None, length=k_split - 1)
        carry, stages = _n_map(carry, True)
    state, press, q, omga, nh, nsplt, nsplt_exceeded = carry
    return {"state": state, "press": press, "q": q, "omga": omga, "nh": nh,
            # C5, propagated: check_nsplt_schedule(out) is the caller's
            # gate and works on this dict unchanged.
            "nsplt": nsplt, "nsplt_exceeded": nsplt_exceeded,
            "omga_is_meaningless": True,
            "pt_units": "K" if remapped else "theta_v",
            "stages": stages}


def make_require_uniform_damping_lane_jit():
    """Static deck guard (C5): the check runs on Python values at trace
    time and cannot exist under jit, so the factory hands back the plain
    function -- it raises before any tracer exists."""
    return require_uniform_damping_lane


def make_p_var_hydrostatic_jit(*, n: int, ng: int, km: int,
                               check_args: bool = False):
    """Static: n, ng, km, check_args.  Dynamic: delp, ptop, akap."""
    def run(delp, ptop, akap):
        return p_var_hydrostatic(delp, ptop=ptop, akap=akap, n=n, ng=ng,
                                 km=km, check_args=check_args)
    return jax.jit(run)


def make_pt_to_theta_v_jit(*, n: int, ng: int):
    """Static: n, ng (dp1's None-ness is fixed per compiled closure)."""
    def run(pt, pkz, dp1=None):
        return pt_to_theta_v(pt, pkz, n=n, ng=ng, dp1=dp1)
    return jax.jit(run)


def make_p_var_nonhydrostatic_jit(*, n: int, ng: int, km: int,
                                  check_args: bool = False):
    """Static: n, ng, km, check_args.  Dynamic: delp, delz, pt, ptop, akap."""
    def run(delp, delz, pt, ptop, akap):
        return p_var_nonhydrostatic(delp, delz, pt, ptop=ptop, akap=akap,
                                    n=n, ng=ng, km=km,
                                    check_args=check_args)
    return jax.jit(run)


def make_fv_dynamics_step_jit(ctx: dict, km: int, *, k_split: int,
                              n_split: int, ptop, ak, bk, akap, cp_air,
                              kord_mt: int, kord_tm: int,
                              kord_tr, p_fac=0.05, a_imp=1.0,
                              sphum_index=None, n_sponge: int = -1,
                              tau: float = -1.0, hydrostatic: bool = True,
                              use_logp: bool = False, kord_wz: int = 9,
                              w_limiter=None, cfg=None, a2b_ord: int = 4,
                              check_state: bool = False, hord_tr: int = 6,
                              tracer_q_split: int = 0, nord_tr: int = 0,
                              fill: bool = False,
                              trdm2=0.0, lim_fac=1.0, z_tracer: bool = True,
                              inline_q: bool = False, zvir: float = 0.0,
                              consv_te: float = 0.0, out_shardings=None,
                              batched: bool = False):
    """Static (C3/D5): ctx, km and every DECK constant.  Dynamic: only
    state, press, q, bdt, omga and nh.

    ptop/ak/bk/akap/cp_air/p_fac/a_imp are BAKED, matching the module-4
    factory. They were dynamic here, and that could not work: geopk
    computes `float(np.log(ptop))` at trace time (fv3_pgrad.py:1001), so
    a traced ptop raises TracerArrayConversionError. A deck constant is
    static in this lane by contract, and this factory was the one place
    that disagreed."""
    def run(state, press, q, bdt, omga=None, nh=None):
        return fv_dynamics_step(
            ctx, state, press, bdt=bdt, km=km, k_split=k_split,
            n_split=n_split, ptop=ptop, ak=ak, bk=bk, akap=akap,
            cp_air=cp_air, kord_mt=kord_mt, kord_tm=kord_tm,
            kord_tr=kord_tr, q=q, omga=omga, nh=nh, zvir=zvir,
            consv_te=consv_te, sphum_index=sphum_index, n_sponge=n_sponge,
            tau=tau, hydrostatic=hydrostatic, p_fac=p_fac, a_imp=a_imp,
            use_logp=use_logp, kord_wz=kord_wz, w_limiter=w_limiter,
            cfg=cfg, a2b_ord=a2b_ord, check_state=check_state,
            hord_tr=hord_tr, tracer_q_split=tracer_q_split,
            nord_tr=nord_tr, trdm2=trdm2, lim_fac=lim_fac,
            fill=fill,
            z_tracer=z_tracer, inline_q=inline_q, batched=batched)

    # THE RETURN CARRIES TWO NON-ARRAY LEAVES -- `pt_units` (a str) and
    # `omga_is_meaningless` (a bool) -- and jit refuses to return either
    # ("returned a value of type <class 'str'>, which is not a valid JAX
    # type"), so jitting `run` directly could never have worked. Both
    # are decided by STATIC arguments alone: pt_units by km vs
    # REMAP_MIN_NPZ, omga_is_meaningless unconditionally. So they are
    # computed here, outside the trace, and re-attached to the compiled
    # call's result -- the caller sees the same dict either way, which
    # is what makes eager-vs-jit comparable at all.
    # C5/D3, matching make_acoustic_loop_3d_jit: check_state reaches
    # data-dependent Python conversions in the acoustic loop, so a
    # compiled step can never honour it. Rejecting HERE means the caller
    # finds out at construction; leaving it to the callee's tracer guard
    # means a callable that builds fine and dies on first invocation.
    if check_state:
        raise ValueError(
            "make_fv_dynamics_step_jit: check_state=True cannot be "
            "compiled -- the state sanity check converts traced values "
            "to Python. Use the eager fv_dynamics_step for checked runs.")

    _meta = {"omga_is_meaningless": True,
             "pt_units": "K" if km > REMAP_MIN_NPZ else "theta_v"}

    def _arrays_only(*a, **kw):
        out = run(*a, **kw)
        out = {k: v for k, v in out.items() if k not in _meta}
        if out_shardings is not None:
            # SPMD boundary pin (measured 2026-08-24, job 9483159): with
            # face-sharded inputs and an UNCONSTRAINED jit, GSPMD keeps
            # the interior distributed (halo transfers lower to face/
            # strip-sized collective-permutes) but resolves the OUTPUTS
            # fully replicated, so a stepping loop decays to replication
            # after one step.  Constrain by OUTPUT ROOT, not by extent
            # (codex MAJOR: nsplt is (km,), so a shape[0]==6 test would
            # shard a VERTICAL schedule over faces at km=6): every leaf
            # under the face-stacked roots is face-leading by
            # construction, and the schedule leaves (nsplt,
            # nsplt_exceeded, stages) stay untouched.  A jit-level
            # out_shardings prefix is not usable here -- it would try to
            # tile those non-face leaves and raise IndivisibleError.
            # stages IS face-stacked (S10/S11/S16 stack six faces --
            # codex: sharding them is layout-correct) and leaving it
            # unconstrained makes GSPMD resolve it REPLICATED, i.e. a
            # full gather of every stage array at each step's boundary.
            _face_roots = ("state", "press", "q", "omga", "nh", "stages")
            out = {**out,
                   **{k: jax.tree.map(
                          lambda x: jax.lax.with_sharding_constraint(
                              x, out_shardings), out[k])
                      for k in _face_roots
                      if out.get(k) is not None}}
        return out

    _compiled = jax.jit(_arrays_only)

    def _call(*a, **kw):
        return {**_compiled(*a, **kw), **_meta}

    # the lowering of the compiled body, for instruments that measure
    # the traced program (arm-selection and program-size gates)
    _call.lower = _compiled.lower
    return _call
