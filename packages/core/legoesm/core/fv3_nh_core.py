"""FV3 non-hydrostatic core -- JAX lane (SIM1_solver, Riem_Solver_c,
Riem_Solver3, edge_profile, update_dz_c, update_dz_d).

Functional, jit-compatible mirror of the certified NumPy fp64 lane
(``fv3_native_nh_core.py``, itself a loop-faithful port of the pinned
oracle: ``nh_utils.F90:1193-1324`` SIM1, ``nh_utils.F90:313-420``
Riem_Solver_c, ``nh_core.F90:42-206`` Riem_Solver3,
``nh_utils.F90:1535-1641`` edge_profile, ``nh_utils.F90:49-191``
update_dz_c, ``nh_utils.F90:194-311`` update_dz_d — non-MOIST_CAPPA /
non-USE_COND branches, ``dz_min = 2.``).  The NumPy lane stays the
oracle-parity reference; this module is the production/JAX twin and is
certified AGAINST the NumPy lane, never against the Fortran directly
(one authority per hop).

``update_dz_d`` drags in the transport call tree the oracle reuses:
``fv_tp_2d`` (-> ``copy_corners``/``xppm``/``yppm``/``pert_ppm``) and
``del6_vt_flux``.  The tp_core half is IMPORTED from
``legoesm.core.fv3_tp_core`` — no tp_core numerics are re-derived here.
``del6_vt_flux`` is a **sw_core** routine with no JAX twin, so it alone
is mirrored privately at the bottom of this module against
``fv3_native_d_sw.del6_vt_flux``.  It is NOT the same operator as
``fv3_del6_vt_flux.py`` / ``fv3_sw_core._del6_vt_flux``, which are the
repo's research cubed-sphere solvers on ``(6, n, n)`` face arrays with
cdgrid objects and halo exchange (strategy doc R3: the research solver
is not the target of this port).

Mirror doctrine established here for the remaining NH kernels:

* **Functional**: the NumPy lane mutates ``pe``/``w2``/``dz2`` in place;
  the JAX twin takes the same operands and RETURNS ``(pe, w2, dz2)``.
  No aliasing, no output parameters.
* **k-recurrences via ``lax.scan``** -- NOT ``associative_scan``.  The
  Thomas forward/backward sweeps are Moebius (rational) recurrences in
  ``bet``; they can be rewritten as 2x2 matrix chains and evaluated with
  an associative scan, but that changes the floating-point association
  order (and divides by intermediate matrix entries), which both breaks
  the <=1e-15 parity contract with the sequential NumPy lane and is less
  numerically robust.  ``km`` is small (5..~130); a sequential scan over
  k with everything vectorised over i is the right shape.  EXCEPTION:
  ``edge_profile`` unrolls its recurrences in python instead — its
  consumer ``update_dz_d``'s EAGER path (the one the free-stream gates
  measure) needs BITWISE parity with the NumPy lane, and a scan-compiled
  body's fused multiply-adds broke it (job 9421844; see
  ``edge_profile``).  Under ``jax.jit`` the unrolled graph is compiled
  like everything else and only the ~1e-14 jit-parity contract holds.
* **Sequential integrals too**: the pe rebuild is a cumulative sum, but
  ``jnp.cumsum`` may lower to a log-depth associative scan whose
  association order differs from the NumPy loop -- it is a ``lax.scan``
  here for the same parity reason.
* **Explicit x64**: every operand must arrive float64 (the oracle build
  is ``-fdefault-real-8``); a float32 operand raises ``TypeError`` at
  entry exactly like the NumPy lane's ``_require_f64`` (the check reads
  only static dtypes, so it is jit-safe).
* **No ``donate_argnums``** anywhere in this lane (grad-path doctrine,
  CLAUDE.md).
* **Differentiable** end-to-end (exp/log/div + scans), with one caveat:
  the ``p_fac`` pressure floor is a ``jnp.maximum`` -- at the floor
  boundary the kernel is only C^0 and order-2 ``check_grads`` is not
  expected to hold THERE; away from the floor (the generic and the
  physical case) it holds.

COST -- ``update_dz_d``'s STATIC ``damp``/``ndif`` and its UNROLLED k
loop
-----------------------------------------------------------------

Stated so the trade-off is costed rather than discovered.  NONE of the
numbers below is measured; each is an op-count argument or an explicit
unknown, and the measurement that would settle it is named at the end.

*Program size vs km.*  ``update_dz_d``'s k loop is a PYTHON loop over
``km + 1`` interfaces, unrolled at trace time, because ``damp[k] > 1e-5``
selects between two different operator chains and ``ndif[k]`` is the
del-nord LOOP COUNT -- neither can be traced.  Each unrolled body holds
one ``fv_tp_2d`` (4 PPM sweeps + 2 ``copy_corners`` + 2 flux-form nests)
and, on a damped level, one ``_del6_vt_flux`` (``2 + 2*nord``
``copy_corners`` and as many flux nests).  Traced op count is therefore
LINEAR in ``km + 1`` with a large constant.  The pinned decks resolve
``npz = 3`` or ``5`` (every ``input.nml`` under ``fv3_oracle_pinned/``),
i.e. 4 or 6 unrolled bodies.  At the coordinator's stated production
target of ``km = 79`` -- which NO pinned deck resolves -- that is 80
bodies, ~13x the km=5 program, and ``update_dz_d`` sits inside an
acoustic substep inside ``k_split``, so trace+compile time could become
the binding constraint on the lane.  Runtime FLOPs are unaffected: the
oracle does the same work per level either way.  The ``edge_profile``
recurrences are ALSO python-unrolled over k (they were a ``lax.scan``
until the free-stream measurement, job 9421844 — see the
``edge_profile`` docstring), adding roughly four array recurrence
chains of length km plus scalar coefficient/bottom-row bookkeeping —
expected (unmeasured) to be small next to the km+1 ``fv_tp_2d``
bodies.

*How many distinct compilations a run produces.*  The jit cache key is
``(ndif, damp, hord, bounds, km, npx, npy, rdt)`` plus the flag
keywords.  On the shipped duo deck ``nord_v``/``damp_vt`` are
LEVEL-INVARIANT (``fv3_native_dsw_tail_3d.py:406-408`` builds them with
``np.full(km + 1, ...)``), and ``rdt = 1/dt`` is fixed within a run, so
a whole run is ONE compilation, reused across all six faces, every
acoustic substep and every ``k_split`` step.  A deck with a sponge
(``n_sponge >= 0``) still has ONE fixed pattern, hence still one
compilation.  Only a pattern that CHANGES between substeps would
multiply compilations, and no deck in the tree does that.  So the cost
is one big compile, not many.

*The alternative, if compile time binds.*  A ``lax.scan`` over levels
needs a UNIFORM per-level program, which costs two things:

1. ``damp[k] > 1e-5`` must become a masked blend instead of a static
   branch.  That IS faithful, and for a non-obvious reason worth
   recording: at ``damp = 0`` the del-nord chain starts from
   ``d2 = 0 * q = 0``, so every ``fx2``/``fy2`` is exactly ``0`` and the
   damped update ``num/den + del6`` reduces EXACTLY (not approximately)
   to the undamped ``num/den``.  The one piece that does NOT fall out of
   the arithmetic is the GHOST-CELL aliasing: the damped branch runs
   ``fv_tp_2d`` on a copy and preserves ``zh``'s corner ghosts, the
   undamped branch aliases ``zh`` and lets ``copy_corners`` rewrite them
   (nh_utils.F90:264-268 vs :279).  Under a scan that becomes a
   ``jnp.where`` over the whole level between the two candidate ghost
   states -- expressible, and still exact.
2. ``ndif[k]`` must be uniform across levels, because the del-nord
   iteration count is a loop bound, not a coefficient.  It IS uniform on
   every deck in the tree.  For a level-VARYING ``nord`` a scan would
   have to run ``max(nord)`` iterations everywhere and select the right
   INTERMEDIATE per level, which means carrying all intermediates -- so
   that case should keep the unrolled loop rather than grow a second
   numerics path.

FLOP cost of the blend, as an op-count estimate and NOT a measurement:
on the shipped deck every level is damped, so computing both chains adds
nothing.  On a mixed-``damp`` deck the del-nord chain would run on the
undamped levels too, and it is roughly 6 flux nests (nord=2) against
``fv_tp_2d``'s 4 full PPM sweeps, so of order +10-20% per extra level.

*UNKNOWN UNTIL MEASURED, and the measurement.*  Whether trace+compile
actually binds at ``km = 79`` is not derivable from op counts.  The
measurement is: in the lane's SLURM measurement job, for
``km in {5, 79}`` at the deck's ``hord_tm = 6``, ``nord_v = 2``, record
(a) ``jax.jit(update_dz_d).lower(*args).compile()`` wall time and the
lowered HLO instruction count, and (b) the number of distinct jit cache
entries created over one representative ``k_split x n_split`` step.  If
(a) at km=79 is a small multiple of one step's runtime, the unrolled
loop stays; if it dominates, the scan-with-blend above is the change to
make.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np  # STATIC trace-time values only, never traced
from jax import lax
from legoesm.core.fv3_native_nh_core import DZ_MIN  # nh_utils.F90:40-44
from legoesm.core.fv3_native_sw_core import Bounds
from legoesm.core.fv3_tp_core import copy_corners, fv_tp_2d
from legoesm.grids.fv3_native_gridstruct import FV3_GRAV, FV3_RDGAS

# nh_utils.F90:45 (math constant; mirrors the NumPy lane's module constant)
_R3 = 1.0 / 3.0

# ``update_dz_d`` reuses the SHIPPED JAX tp_core twins (``fv_tp_2d`` ->
# xppm/yppm/pert_ppm/copy_corners) exactly as the oracle reuses its own
# — no tp_core numerics are re-derived in this module.  Only
# ``del6_vt_flux`` is mirrored here (privately, below): it is a
# **sw_core** routine, not tp_core, and the only JAX ``del6_vt_flux``
# functions in the tree (``fv3_del6_vt_flux.py``, ``fv3_sw_core.py``)
# are the repo's research cubed-sphere solvers on ``(6, n, n)`` face
# arrays with cdgrid objects, which is a different operator (strategy
# doc R3).


def _require_f64_jax(fname: str, arrays: dict) -> None:
    """dtype-UNIFORMITY gate (2026-08-28): was strict float64. The JAX duo runtime now runs ONE uniform float dtype (FV3DuoConfig.storage_dtype), so this accepts f32 OR f64 provided every operand matches; the anti-silent-downcast guard moved to FV3DuoDynamicsModel.step's boundary check. The rationale below is the ORIGINAL strict-f64 history.

    Reads only ``.dtype`` (static under jit): a float32 operand would
    otherwise be silently upcast -- or worse, with x64 disabled the whole
    solve would silently run in float32 -- and the oracle build is
    ``-fdefault-real-8``.
    """
    # dtype-UNIFORMITY gate (2026-08-28): was strict float64; relaxed for
    # the coarse fv3_duo precision policy (FV3DuoConfig.storage_dtype).
    seen = None
    for name, a in arrays.items():
        if a is None:
            continue
        _arr = jnp.asarray(a)
        if _arr.ndim == 0 and getattr(_arr, "weak_type", False):
            # Skip ONLY a WEAK-typed 0-dim scalar (a python-float
            # timestep/coeff like dt/kgb): it is weak-promoting and not a
            # field, so it is not part of the field uniformity invariant.
            # A STRONG-f64 0-dim (an f64 constant / damping coeff that
            # "went strong") is NOT skipped -> it still trips this gate
            # against f32 fields, closing the silent-promotion blind spot
            # a wholesale 0-dim skip left (codex+GLM+Claude, increment 2).
            continue
        dt = _arr.dtype
        if dt not in (jnp.float32, jnp.float64):
            raise TypeError(
                f"{fname}: {name} must be float32 or float64 (got {dt})")
        if seen is None:
            seen = dt
        elif dt != seen:
            raise TypeError(
                f"{fname}: MIXED float dtypes ({seen} vs {dt} on {name}); "
                f"a phase must be single-precision-uniform "
                f"(FV3DuoConfig.storage_dtype).")


def sim1_solver(dt: float, rgas: float, gama: float, kappa: float,
                dm2: jnp.ndarray, pm2: jnp.ndarray, pem: jnp.ndarray,
                w2: jnp.ndarray, dz2: jnp.ndarray, pt2: jnp.ndarray,
                ws: jnp.ndarray, p_fac: float):
    """JAX twin of ``fv3_native_nh_core.sim1_solver`` (nh_utils.F90:1193-1324).

    Same (i, k) window arrays as the NumPy lane -- i is axis 0 (the
    caller has already selected the i window, so the NumPy lane's
    ``is_``/``ie``/``km`` args are dropped; ``km = dm2.shape[1]``).
    ``gm2``/``cp2`` of the Fortran signature are MOIST_CAPPA-only reads
    and are deliberately not parameters (matches the NumPy lane).

    Returns ``(pe, w2, dz2)`` -- the three arrays the NumPy lane writes
    in place: ``pe`` is (ni, km+1), ``w2``/``dz2`` are (ni, km).
    Requires ``km >= 2`` (km=1 divides by zero in the oracle recurrence).
    """
    _require_f64_jax("sim1_solver", {
        "dm2": dm2, "pm2": pm2, "pem": pem, "w2": w2, "dz2": dz2,
        "pt2": pt2, "ws": ws})
    km = dm2.shape[1]
    if km < 2:
        # km=1 is invalid in the ORACLE algebra too: the NumPy lane's
        # k-loop leaves bb[:, 0] == 0 when km-1 == 0, so pp[:, 1] =
        # dd[:, 0] / bet divides by zero (codex SIM1 r1 #1: km=2 IS
        # valid — the w forward sweep is simply empty — and is covered
        # by a parity test).
        raise ValueError(f"sim1_solver: km={km} < 2 unsupported (km=1 "
                         f"divides by bb[:,0]=0 in the oracle recurrence)")
    t1g = gama * 2.0 * dt * dt          # :1211 (non-moist)
    rdt = 1.0 / dt
    capa1 = kappa - 1.0

    w1 = w2                              # :1224 (functional: no copy needed)
    # :1219-1226  pe = (-dm/dz * rgas * pt)^gama - pm   (k = 0..km-1)
    pe_body = jnp.exp(gama * jnp.log(-dm2 / dz2 * rgas * pt2)) - pm2

    # :1228-1234  k = 0..km-2
    g_rat = dm2[:, :-1] / dm2[:, 1:]                     # (ni, km-1)
    bb = jnp.concatenate(
        [2.0 * (1.0 + g_rat),
         2.0 * jnp.ones((dm2.shape[0], 1), dm2.dtype)], axis=1)   # :1240
    dd = jnp.concatenate(
        [3.0 * (pe_body[:, :-1] + g_rat * pe_body[:, 1:]),
         3.0 * pe_body[:, km - 1:km]], axis=1)                    # :1241

    # --- pp forward Thomas sweep :1237-1250 (scan over k=1..km-1) ---
    bet0 = bb[:, 0]                                      # :1237
    pp1 = dd[:, 0] / bet0

    def _pp_fwd(carry, x):
        bet, pp_prev = carry
        g_km1, bb_k, dd_k = x
        gam_k = g_km1 / bet
        bet_new = bb_k - gam_k
        pp_next = (dd_k - pp_prev) / bet_new
        return (bet_new, pp_next), (gam_k, pp_next)

    xs_pp = (g_rat.T, bb[:, 1:].T, dd[:, 1:].T)          # k = 1..km-1
    (_, _), (gam_tail, pp_tail) = lax.scan(_pp_fwd, (bet0, pp1), xs_pp)
    # gam[:, k] for k=1..km-1 ; pp[:, k+1] for k=1..km-1
    pp = jnp.concatenate(
        [jnp.zeros((dm2.shape[0], 1), dm2.dtype),        # pp[:, 0] = 0
         pp1[:, None], pp_tail.T], axis=1)               # (ni, km+1)

    # --- pp backward substitution :1252-1256 (k = km-1..1) ---
    def _pp_bwd(pp_next, x):
        pp_k_raw, gam_k = x
        pp_k = pp_k_raw - gam_k * pp_next
        return pp_k, pp_k

    xs_bwd = (pp[:, 1:km].T, gam_tail)                   # k = 1..km-1
    _, pp_mid_rev = lax.scan(_pp_bwd, pp[:, km], xs_bwd, reverse=True)
    pp = jnp.concatenate(
        [pp[:, 0:1], pp_mid_rev.T, pp[:, km:km + 1]], axis=1)

    # --- w solver :1259-1291 ---
    # aa[:, k] for k = 1..km-1 (aa[:, 0] unused, kept 0 for alignment)
    aa_tail = (t1g / (dz2[:, :-1] + dz2[:, 1:])
               * (pem[:, 1:km] + pp[:, 1:km]))           # (ni, km-1)
    aa = jnp.concatenate(
        [jnp.zeros((dm2.shape[0], 1), dm2.dtype), aa_tail], axis=1)

    bet_w0 = dm2[:, 0] - aa[:, 1]
    w2_0 = (dm2[:, 0] * w1[:, 0] + dt * pp[:, 1]) / bet_w0

    def _w_fwd(carry, x):
        bet, w_prev = carry
        aa_k, aa_kp1, dm_k, w1_k, dpp_k = x
        gam_k = aa_k / bet
        bet_new = dm_k - (aa_k + aa_kp1 + aa_k * gam_k)
        w_k = (dm_k * w1_k + dt * dpp_k - aa_k * w_prev) / bet_new
        return (bet_new, w_k), (gam_k, w_k)

    xs_w = (aa[:, 1:km - 1].T, aa[:, 2:km].T, dm2[:, 1:km - 1].T,
            w1[:, 1:km - 1].T,
            (pp[:, 2:km] - pp[:, 1:km - 1]).T)           # k = 1..km-2
    (bet_w, _), (gam_w_mid, w_mid) = lax.scan(_w_fwd, (bet_w0, w2_0), xs_w)
    w_raw = jnp.concatenate([w2_0[:, None], w_mid.T], axis=1)  # k = 0..km-2
    w_km2 = w_raw[:, km - 2]

    p1 = t1g / dz2[:, km - 1] * (pem[:, km] + pp[:, km])            # :1282
    gam_km = aa[:, km - 1] / bet_w
    bet_bot = dm2[:, km - 1] - (aa[:, km - 1] + p1 + aa[:, km - 1] * gam_km)
    w_bot = (dm2[:, km - 1] * w1[:, km - 1]
             + dt * (pp[:, km] - pp[:, km - 1])
             - p1 * ws - aa[:, km - 1] * w_km2) / bet_bot

    # gam[:, k] for backsub: k = 1..km-1 (gam[:, km-1] = gam_km)
    gam_w = jnp.concatenate([gam_w_mid, gam_km[None, :]], axis=0)

    # :1288-1291  k = km-2..0: w2[k] -= gam[k+1] * w2[k+1]
    def _w_bwd(w_next, x):
        w_k_raw, gam_kp1 = x
        w_k = w_k_raw - gam_kp1 * w_next
        return w_k, w_k

    _, w_head_rev = lax.scan(_w_bwd, w_bot, (w_raw.T, gam_w), reverse=True)
    w2_out = jnp.concatenate([w_head_rev.T, w_bot[:, None]], axis=1)

    # :1293-1300  pe rebuild (sequential integral; scan for parity, see
    # module docstring on cumsum association order)
    def _pe_step(pe_k, inc_k):
        pe_kp1 = pe_k + inc_k
        return pe_kp1, pe_kp1

    inc = (dm2 * (w2_out - w1) * rdt).T                  # k = 0..km-1
    _, pe_tail2 = lax.scan(_pe_step, jnp.zeros_like(pe_body[:, 0]), inc)
    pe = jnp.concatenate(
        [jnp.zeros((dm2.shape[0], 1), dm2.dtype), pe_tail2.T], axis=1)

    # :1302-1319  dz2 back-out (bottom-up), the p_fac floor lives here
    p1_bot = (pe[:, km - 1] + 2.0 * pe[:, km]) * _R3
    dz_bot = -dm2[:, km - 1] * rgas * pt2[:, km - 1] * jnp.exp(
        capa1 * jnp.log(jnp.maximum(p_fac * pm2[:, km - 1],
                                    p1_bot + pm2[:, km - 1])))

    def _dz_bwd(p1_c, x):
        pe_k, pe_kp1, pe_kp2, bb_k, g_k, dm_k, pt_k, pm_k = x
        p1_new = (pe_k + bb_k * pe_kp1 + g_k * pe_kp2) * _R3 - g_k * p1_c
        dz_k = -dm_k * rgas * pt_k * jnp.exp(
            capa1 * jnp.log(jnp.maximum(p_fac * pm_k, p1_new + pm_k)))
        return p1_new, dz_k

    xs_dz = (pe[:, 0:km - 1].T, pe[:, 1:km].T, pe[:, 2:km + 1].T,
             bb[:, 0:km - 1].T, g_rat[:, 0:km - 1].T,
             dm2[:, 0:km - 1].T, pt2[:, 0:km - 1].T, pm2[:, 0:km - 1].T)
    _, dz_head_rev = lax.scan(_dz_bwd, p1_bot, xs_dz, reverse=True)
    dz2_out = jnp.concatenate([dz_head_rev.T, dz_bot[:, None]], axis=1)

    return pe, w2_out, dz2_out


def _seq_cumsum(x0: jnp.ndarray, inc: jnp.ndarray):
    """Sequential prefix accumulation over k (axis 1), scan for parity.

    Returns the (B, km+1) array ``[x0, x0+inc0, x0+inc0+inc1, ...]`` with
    the NumPy lane's exact left-to-right association (jnp.cumsum may lower
    to a log-depth associative scan — see module docstring)."""

    def _step(carry, inc_k):
        nxt = carry + inc_k
        return nxt, nxt

    _, tail = lax.scan(_step, x0, inc.T)
    return jnp.concatenate([x0[:, None], tail.T], axis=1)


def _seq_backbuild(bottom: jnp.ndarray, dz: jnp.ndarray, scale: float):
    """Bottom-up interface rebuild ``z[k] = z[k+1] - dz[k]*scale``.

    Mirrors the NumPy lanes' reverse k loops (riem_solver_c gz rebuild
    nh_utils.F90:410-418 with scale=grav; riem_solver3 zh rebuild
    nh_core.F90:195-202 with scale=1).  Returns (B, km+1) with ``bottom``
    as the last column (exact copy, no arithmetic on it)."""

    def _step(carry, dz_k):
        z_k = carry - dz_k * scale
        return z_k, z_k

    _, head_rev = lax.scan(_step, bottom, dz.T, reverse=True)
    return jnp.concatenate([head_rev.T, bottom[:, None]], axis=1)


def riem_solver_c(ms: int, dt: float, bounds, km: int, akap: float,
                  cp: float, ptop: float, hs: jnp.ndarray, w3: jnp.ndarray,
                  pt: jnp.ndarray, delp: jnp.ndarray, gz: jnp.ndarray,
                  pef: jnp.ndarray, ws: jnp.ndarray, p_fac: float,
                  a_imp: float, scale_m: float = 0.0):
    """JAX twin of ``fv3_native_nh_core.riem_solver_c`` (nh_utils.F90:313-420).

    Functional: the NumPy lane updates ``gz``/``pef`` in place; this twin
    RETURNS ``(gz, pef)`` with the compute window
    (i = is-1..ie+1, j = js-1..je+1) rewritten and every halo cell
    carried through unchanged (``.at[window].set``), so the oracle's
    write footprint is preserved exactly.

    ``bounds`` replaces the NumPy lane's ``bd`` object: a STATIC tuple
    ``(is_, ie, js, je, ng)`` of Python ints (hashable by value, so the
    jit cache never retraces on a fresh-but-equal bounds object the way a
    by-identity ``bd`` shim would).  ``ms``/``cp``/``scale_m`` are dead
    reads on the non-USE_COND branch, kept only for signature parity.
    ``pef`` here is FULL interface pressure ``pe2 + pem``
    (nh_utils.F90:404) — the D stage's :func:`riem_solver3` writes the
    PERTURBATION instead; do not conflate them.

    The j rows are independent columns in the oracle (one sim1 solve per
    j); they are batched into sim1's i axis here, which is FP-identical
    because every sim1 operation is elementwise over that axis.
    """
    _require_f64_jax("riem_solver_c", {
        "hs": hs, "w3": w3, "pt": pt, "delp": delp, "gz": gz,
        "pef": pef, "ws": ws})
    # Same 0.999 threshold as the NumPy lane: (0.5, 0.999] belongs to the
    # unported SIM_solver arm and must be heard, not absorbed.
    if not (a_imp > 0.999):
        raise NotImplementedError(
            f"riem_solver_c: a_imp={a_imp} selects a dead arm on the "
            f"pinned deck (a_imp=1. -> SIM1); SIM3p0/SIM3/RIM_2D/"
            f"SIM_solver are not ported. nh_utils.F90:392-401.")

    # In/out operands go through .at[...].set below, which needs jax
    # arrays (check_grads' numerical-FD path hands plain NumPy in).
    gz = jnp.asarray(gz)
    pef = jnp.asarray(pef)

    is_, ie, js, je, ng = bounds
    gama = 1.0 / (1.0 - akap)
    rgrav = 1.0 / FV3_GRAV
    ni = ie - is_ + 3                 # i = is-1..ie+1  (:334)
    nj = je - js + 3                  # j = js-1..je+1  (:334)
    o = ng - 1                        # padded index of i = is-1
    iw = slice(o, o + ni)
    jw = slice(o, o + nj)
    nb = ni * nj

    dm_pa = delp[iw, jw, :km].reshape(nb, km)              # :347-351
    pem = _seq_cumsum(jnp.full((nb,), ptop, dm_pa.dtype), dm_pa)  # :354-366

    gzw = gz[iw, jw, :].reshape(nb, km + 1)
    dz2 = gzw[:, 1:] - gzw[:, :-1]                         # :370-385
    pm2 = dm_pa / jnp.log(pem[:, 1:] / pem[:, :-1])
    dm = dm_pa * rgrav
    w2 = w3[iw, jw, :km].reshape(nb, km)
    pt2 = pt[iw, jw, :km].reshape(nb, km)
    ws2 = ws[iw, jw].reshape(nb)

    pe2, _, dz2_new = sim1_solver(dt, FV3_RDGAS, gama, akap, dm, pm2,
                                  pem, w2, dz2, pt2, ws2, p_fac)  # :398-400
    # (w2 is solved but NOT written back — the C stage leaves w3 alone,
    # exactly like the oracle; only dz2 feeds the gz rebuild.)

    pef_win = jnp.concatenate(
        [jnp.full((nb, 1), ptop, dm_pa.dtype),             # :354-356
         pe2[:, 1:] + pem[:, 1:]], axis=1)                 # :403-407
    gz_win = _seq_backbuild(hs[iw, jw].reshape(nb), dz2_new,
                            FV3_GRAV)                      # :410-418

    gz_out = gz.at[iw, jw, :].set(gz_win.reshape(ni, nj, km + 1))
    pef_out = pef.at[iw, jw, :].set(pef_win.reshape(ni, nj, km + 1))
    return gz_out, pef_out


def riem_solver3(ms: int, dt: float, bounds, km: int, akap: float,
                 cp: float, ptop: float, zs: jnp.ndarray, w: jnp.ndarray,
                 delz: jnp.ndarray, pt: jnp.ndarray, delp: jnp.ndarray,
                 zh: jnp.ndarray, pe: jnp.ndarray, ppe: jnp.ndarray,
                 pk3: jnp.ndarray, pk: jnp.ndarray, peln: jnp.ndarray,
                 ws: jnp.ndarray, p_fac: float, a_imp: float, *,
                 scale_m: float = 0.0, use_logp: bool = False,
                 last_call: bool = True, fp_out: bool = False):
    """JAX twin of ``fv3_native_nh_core.riem_solver3`` (nh_core.F90:42-206).

    Functional: returns ``(w, delz, zh, pe, ppe, pk3, pk, peln)`` — the
    eight arrays the NumPy lane writes in place — with each written
    window rewritten and everything else (halos, the pe ring rows/slots,
    and, when ``last_call`` is false, ALL of pe/pk/peln) carried through
    from the operands unchanged.

    ``last_call``/``fp_out``/``use_logp``/``scale_m`` are STATIC Python
    values (Python ``if`` arms, never traced; the jit factory marks them
    static) — on the pinned deck ``last_call=True, fp_out=False,
    use_logp=False``.  ``ppe`` is the PERTURBATION ``pe2`` unless
    ``fp_out`` (nh_core.F90:173-185); ``pk`` copies the EXP form of pk3
    BEFORE any use_logp overwrite (:164-172 precedes :187-193); ``zh``
    rebuilds from ``zs`` WITHOUT a grav factor (:195-202).  ``bounds`` is
    the static ``(is_, ie, js, je, ng)`` tuple as in
    :func:`riem_solver_c`; the j window here is js..je with NO ring
    (:87).  ``ms``/``cp`` are dead reads kept for signature parity.
    """
    _require_f64_jax("riem_solver3", {
        "zs": zs, "w": w, "delz": delz, "pt": pt, "delp": delp,
        "zh": zh, "pe": pe, "ppe": ppe, "pk3": pk3, "pk": pk,
        "peln": peln, "ws": ws})
    if not (a_imp > 0.999):
        raise NotImplementedError(
            f"riem_solver3: a_imp={a_imp} selects a dead arm on the "
            f"pinned deck (a_imp=1. -> SIM1); nh_core.F90:138-154.")

    # In/out operands go through .at[...].set below (needs jax arrays;
    # check_grads' numerical-FD path hands plain NumPy in).
    w = jnp.asarray(w)
    delz = jnp.asarray(delz)
    zh = jnp.asarray(zh)
    pe = jnp.asarray(pe)
    ppe = jnp.asarray(ppe)
    pk3 = jnp.asarray(pk3)
    pk = jnp.asarray(pk)
    peln = jnp.asarray(peln)

    is_, ie, js, je, ng = bounds
    ni = ie - is_ + 1
    nj = je - js + 1
    if ws.shape != (ni, nj):
        raise ValueError(
            f"riem_solver3: ws must be the compute window "
            f"({ni}, {nj}) (nh_core.F90:60 ws(is:ie,js:je)), "
            f"got {ws.shape}")
    gama = 1.0 / (1.0 - akap)
    rgrav = 1.0 / FV3_GRAV
    # Static Python scalars, computed exactly as the NumPy lane does
    # (float(np.log/exp)) so the trace-time constants are bit-identical.
    peln1 = float(np.log(ptop))
    ptk = float(np.exp(akap * peln1))
    o = ng                            # padded index of i = is (no ring)
    iw = slice(o, o + ni)
    jw = slice(o, o + nj)
    nb = ni * nj

    dm_pa = delp[iw, jw, :km].reshape(nb, km)

    # :99-118  pem/peln2 accumulation (sequential; log INSIDE the k loop
    # exactly as the oracle interleaves them — elementwise, so the value
    # order matches the lane's).
    def _pem_step(pem_k, dm_k):
        pem_k1 = pem_k + dm_k
        return pem_k1, (pem_k1, jnp.log(pem_k1))

    pem0 = jnp.full((nb,), ptop, dm_pa.dtype)
    _, (pem_tail, peln_tail) = lax.scan(_pem_step, pem0, dm_pa.T)
    pem = jnp.concatenate([pem0[:, None], pem_tail.T], axis=1)
    peln2 = jnp.concatenate(
        [jnp.full((nb, 1), peln1, dm_pa.dtype), peln_tail.T], axis=1)
    pk3_exp = jnp.concatenate(
        [jnp.full((nb, 1), ptk, dm_pa.dtype),              # :80
         jnp.exp(akap * peln2[:, 1:])], axis=1)            # :91

    # :121-135  pm2 divides by peln2 DIFFERENCES (not log(ratio) — same
    # value, different rounding; kept literal like the NumPy lane).
    pm2 = dm_pa / (peln2[:, 1:] - peln2[:, :-1])
    dm = dm_pa * rgrav
    zhw = zh[iw, jw, :].reshape(nb, km + 1)
    dz2 = zhw[:, 1:] - zhw[:, :-1]
    w2 = w[iw, jw, :km].reshape(nb, km)
    pt2 = pt[iw, jw, :km].reshape(nb, km)
    ws2 = ws.reshape(nb)

    pe2, w2_new, dz2_new = sim1_solver(dt, FV3_RDGAS, gama, akap, dm,
                                       pm2, pem, w2, dz2, pt2, ws2,
                                       p_fac)              # :147-149

    w_out = w.at[iw, jw, :km].set(w2_new.reshape(ni, nj, km))  # :157-162
    delz_out = delz.at[:, :, :].set(dz2_new.reshape(ni, nj, km))

    if last_call:                                          # :164-172
        peln_out = peln.at[:, :, :].set(
            peln2.reshape(ni, nj, km + 1).transpose(0, 2, 1))
        # pk holds the EXP form even under use_logp (:164-172 runs first).
        pk_out = pk.at[:, :, :].set(pk3_exp.reshape(ni, nj, km + 1))
        # pe writes i=is..ie / j=js..je only; ring rows/slots keep the
        # caller's values (upstream pe_halo owns them).
        pe_out = pe.at[1:1 + ni, :, 1:1 + nj].set(
            pem.reshape(ni, nj, km + 1).transpose(0, 2, 1))
    else:
        peln_out, pk_out, pe_out = peln, pk, pe

    if fp_out:                                             # :173-185
        ppe_win = pe2 + pem
    else:
        ppe_win = pe2
    ppe_out = ppe.at[iw, jw, :].set(ppe_win.reshape(ni, nj, km + 1))

    if use_logp:                                           # :187-193
        pk3_win = jnp.concatenate([pk3_exp[:, :1], peln2[:, 1:]], axis=1)
    else:
        pk3_win = pk3_exp
    pk3_out = pk3.at[iw, jw, :].set(pk3_win.reshape(ni, nj, km + 1))

    zh_win = _seq_backbuild(zs[iw, jw].reshape(nb), dz2_new,
                            1.0)                           # :195-202
    zh_out = zh.at[iw, jw, :].set(zh_win.reshape(ni, nj, km + 1))

    return (w_out, delz_out, zh_out, pe_out, ppe_out, pk3_out, pk_out,
            peln_out)


def make_sim1_solver_jit(fn=sim1_solver):
    """The ONE jit policy for SIM1 (codex SIM1 r1 #2: the retrace test
    must exercise the production policy, not a shadow copy — both the
    production entry point below and any instrumented test wrapper are
    built HERE, so a policy drift cannot pass unnoticed).

    dt/rgas/gama/kappa/p_fac static: compile-time deck constants in this
    lane (closure-constant doctrine — they never vary per call within a
    run; varying dt intentionally recompiles).  NOT donating any buffer
    (grad-path doctrine).
    """
    return jax.jit(fn, static_argnums=(0, 1, 2, 3, 11))


sim1_solver_jit = make_sim1_solver_jit()


def make_riem_solver_c_jit(fn=riem_solver_c):
    """The ONE jit policy for the C-stage Riemann driver (same doctrine
    as :func:`make_sim1_solver_jit`: the production entry point and any
    instrumented retrace-test wrapper are built HERE).

    Static: ms/dt/bounds/km/akap/cp/ptop/p_fac/a_imp/scale_m — deck
    constants (bounds is a hashable int tuple, so equal-valued bounds
    share one cache entry).  No donated buffers (grad-path doctrine).
    """
    return jax.jit(fn, static_argnums=(0, 1, 2, 3, 4, 5, 6, 14, 15, 16))


riem_solver_c_jit = make_riem_solver_c_jit()


def edge_profile(q1: jnp.ndarray, q2: jnp.ndarray, j_lo: int, km: int,
                 dp0: jnp.ndarray, uniform_grid: bool, limiter: int):
    """JAX twin of ``fv3_native_nh_core.edge_profile``
    (nh_utils.F90:1535-1641).

    Same row-pair shape as the NumPy lane: ``q1``/``q2`` are (ni, km) for
    one j row, ``dp0`` is (km,), and the return is the pair of (ni, km+1)
    edge profiles.  ``j_lo`` is unused arithmetic-wise (kept for call-site
    readability against the oracle) and, like ``km``/``uniform_grid``/
    ``limiter``, is STATIC — the two grid branches and the limiter are
    Python ``if`` arms, never traced.

    The interface tridiagonal is a sequential Thomas recurrence in k,
    UNROLLED IN PYTHON over the static ``km`` — deliberately NOT a
    ``lax.scan``.  ``lax.scan`` compiles its body, and the compiled
    body's contracted multiply-adds broke bit parity with the NumPy
    loop: the stage-split probe on the face-3 gate bundle (follow-up to
    job 9421844) measured ~1e-14 relative differences in all four
    ``*_adv`` outputs while every OTHER sub-operation of ``update_dz_d``
    (fv_tp_2d, xppm/yppm, del6_vt_flux) was bitwise identical — eager
    op-by-op primitives are bit-exact against NumPy, compiled fusions
    are not.  Those ulp perturbations flipped two smt5/upwind selector
    bits on an interior-constant zh level and broke exact free-stream
    preservation in ``update_dz_d`` (job 9421844: 2 cells at k=0,
    2.9e-6 abs, where NumPy returns the input to all 17 digits).
    SCOPE OF THE GUARANTEE: the unroll makes the NON-jitted call
    op-by-op eager, which is what the free-stream gates measure and
    what the stage-split probe re-verified bitwise after the change.
    Inside ``edge_profile_jit``/``update_dz_d_jit`` the unrolled graph
    is compiled like everything else, fusion/contraction remains
    permitted, and only the documented ~1e-14 jit-parity tolerance is
    established.  The
    k-recurrence coefficients (``gam``/``gak``) depend only on ``dp0``
    and ride the loop as scalars; the NumPy lane stores them as (ni,)
    vectors of identical entries, which is the same elementwise
    arithmetic.

    Differentiability: both grid branches are smooth (rational in dp0 and
    linear in q).  The ``limiter != 0`` zero-crossing clamp is a
    ``jnp.where`` on the sign of ``q*qe`` — C^0 at the crossing; grads
    are exact wherever ``q1[:,0]*qe1[:,0]`` (and the three siblings) are
    bounded away from 0 (the pinned deck runs ``limiter=0``, where the
    branch does not exist in the trace at all).

    Requires ``km >= 2``: at km=1 the top row reads ``q1[:, 1]`` (out of
    bounds — the NumPy lane raises IndexError there) and the bottom row
    reads ``q[:, km-2]``, which would silently WRAP to the last column in
    both lanes.  The explicit guard turns both into one loud error.
    """
    del j_lo
    # GLM review 2026-08-17: the unrolled recurrences do `range(1, km)`,
    # so a TRACED km would fail with an opaque tracer error deep in the
    # loop machinery. km is static by contract (C3); say so at the door.
    if isinstance(km, bool) or not isinstance(km, int):
        raise TypeError(
            f"edge_profile: km must be a static python int (C3; the "
            f"recurrences are unrolled over it), got {type(km).__name__}")
    _require_f64_jax("edge_profile", {"q1": q1, "q2": q2, "dp0": dp0})
    q1 = jnp.asarray(q1)
    q2 = jnp.asarray(q2)
    dp0 = jnp.asarray(dp0)
    if km < 2:
        raise ValueError(
            f"edge_profile: km={km} < 2 unsupported (the top row reads "
            f"q[:, 1] and the bottom row reads q[:, km-2]; the NumPy lane "
            f"raises IndexError at km=1)")
    if q1.shape[1] != km or q2.shape[1] != km:
        raise ValueError(f"edge_profile: q1/q2 must be (ni, {km}), got "
                         f"{q1.shape}, {q2.shape}")
    if dp0.shape != (km,):
        raise ValueError(f"edge_profile: dp0 must be ({km},), got "
                         f"{dp0.shape}")

    # Both recurrences below are UNROLLED python loops over the static
    # ``km`` (never lax.scan) so that on a NON-jitted call every op is
    # an eager elementwise primitive — measured requirement, not style:
    # the scan-compiled body's fused multiply-adds perturbed
    # crx_adv/xfx_adv at ~1e-14 relative and broke update_dz_d's exact
    # free-stream preservation (job 9421844: 2 cells at k=0, 2.9e-6
    # abs; NumPy exact to 17 digits).  Under jax.jit the graph is
    # compiled anyway and only the ~1e-14 jit-parity contract holds.
    # See the docstring paragraph on the parity doctrine.
    if uniform_grid:                                   # :1552-1581
        r2o3 = 2.0 / 3.0
        r4o3 = 4.0 / 3.0
        e1 = r4o3 * q1[:, 0] + r2o3 * q1[:, 1]
        e2 = r4o3 * q2[:, 0] + r2o3 * q2[:, 1]
        gak = 7.0 / 3.0                                # gak(1)
        e1_rows, e2_rows, coefs = [e1], [e2], [gak]
        for k in range(1, km):
            gak = 1.0 / (4.0 - gak)
            e1 = (3.0 * (q1[:, k - 1] + q1[:, k]) - e1) * gak
            e2 = (3.0 * (q2[:, k - 1] + q2[:, k]) - e2) * gak
            e1_rows.append(e1)
            e2_rows.append(e2)
            coefs.append(gak)

        bet = 1.0 / (1.5 - 3.5 * gak)                  # :1571
        e1_bot = (4.0 * q1[:, km - 1] + q1[:, km - 2]
                  - 3.5 * e1) * bet
        e2_bot = (4.0 * q2[:, km - 1] + q2[:, km - 2]
                  - 3.5 * e2) * bet
    else:                                              # :1583-1618
        g0 = dp0[1] / dp0[0]
        xt1 = 2.0 * g0 * (g0 + 1.0)
        bet0 = g0 * (g0 + 0.5)
        e1 = (xt1 * q1[:, 0] + q1[:, 1]) / bet0
        e2 = (xt1 * q2[:, 0] + q2[:, 1]) / bet0
        gam = (1.0 + g0 * (g0 + 1.5)) / bet0           # gam(1)
        e1_rows, e2_rows, coefs = [e1], [e2], [gam]
        # gk is ALWAYS assigned: the km >= 2 entry guard means this
        # loop runs at least once, and the bottom-row formulas below
        # reuse its LAST value exactly as the Fortran does (the same
        # division dp0[km-2]/dp0[km-1], bit-identical). (GLM review
        # flagged the bare sentinel as fragile without this note.)
        gk = None
        for k in range(1, km):
            gk = dp0[k - 1] / dp0[k]
            bet_v = 2.0 + 2.0 * gk - gam
            e1 = (3.0 * (q1[:, k - 1] + gk * q1[:, k]) - e1) / bet_v
            e2 = (3.0 * (q2[:, k - 1] + gk * q2[:, k]) - e2) / bet_v
            gam = gk / bet_v
            e1_rows.append(e1)
            e2_rows.append(e2)
            coefs.append(gam)

        # :1602-1609 — the Fortran reuses the LAST loop gk
        # (dp0(km-1)/dp0(km), 1-based), exactly as the loop above
        # leaves it (km >= 2 is guarded at entry).
        a_bot = 1.0 + gk * (gk + 1.5)
        xt1b = 2.0 * gk * (gk + 1.0)
        xt2 = gk * (gk + 0.5) - a_bot * gam
        e1_bot = (xt1b * q1[:, km - 1] + q1[:, km - 2]
                  - a_bot * e1) / xt2
        e2_bot = (xt1b * q2[:, km - 1] + q2[:, km - 2]
                  - a_bot * e2) / xt2

    # Back-substitution k=km-1..0 (:1577-1580 / :1612-1617): the raw
    # forward rows are e1_rows/e2_rows for k=0..km-1; coefs[k] is
    # gak/gam for k=0..km-1.  Same unrolled-python doctrine as above.
    nxt1, nxt2 = e1_bot, e2_bot
    out1: list = [None] * km
    out2: list = [None] * km
    for k in range(km - 1, -1, -1):
        nxt1 = e1_rows[k] - coefs[k] * nxt1
        nxt2 = e2_rows[k] - coefs[k] * nxt2
        out1[k] = nxt1
        out2[k] = nxt2
    qe1 = jnp.stack(out1 + [e1_bot], axis=1)           # (ni, km+1)
    qe2 = jnp.stack(out2 + [e2_bot], axis=1)

    if limiter != 0:                                   # :1623-1633
        qe1 = qe1.at[:, 0].set(
            jnp.where(q1[:, 0] * qe1[:, 0] < 0.0, 0.0, qe1[:, 0]))
        qe2 = qe2.at[:, 0].set(
            jnp.where(q2[:, 0] * qe2[:, 0] < 0.0, 0.0, qe2[:, 0]))
        qe1 = qe1.at[:, km].set(
            jnp.where(q1[:, km - 1] * qe1[:, km] < 0.0, 0.0,
                      qe1[:, km]))
        qe2 = qe2.at[:, km].set(
            jnp.where(q2[:, km - 1] * qe2[:, km] < 0.0, 0.0,
                      qe2[:, km]))
    return qe1, qe2


def _fill_4corners_3d(q: jnp.ndarray, direction: int, npx: int, npy: int,
                      ilo: int, jlo: int, sw: bool, se: bool, ne: bool,
                      nw: bool) -> jnp.ndarray:
    """Functional twin of ``fv3_native_sw_core.fill_4corners`` on a 3-D
    (i, j, k) array (the fill is k-independent, applied to all levels).

    All corner indices are STATIC Fortran indices mapped to storage via
    the (ilo, jlo) origins, exactly as the NumPy lane's ``fort`` views.
    The assignments run in the oracle's source order (sw_core.F90:
    3876-3893 XDir / :3895-3913 YDir); within one direction no target is
    a later source, so the sequential ``.at`` chain equals the in-place
    loop.
    """
    if direction == 1:
        pairs = []
        if sw:
            pairs += [((-1, 0), (0, 2)), ((0, 0), (0, 1))]
        if se:
            pairs += [((npx + 1, 0), (npx, 2)), ((npx, 0), (npx, 1))]
        if nw:
            pairs += [((0, npy), (0, npy - 1)), ((-1, npy), (0, npy - 2))]
        if ne:
            pairs += [((npx, npy), (npx, npy - 1)),
                      ((npx + 1, npy), (npx, npy - 2))]
    elif direction == 2:
        pairs = []
        if sw:
            pairs += [((0, 0), (1, 0)), ((0, -1), (2, 0))]
        if se:
            pairs += [((npx, 0), (npx - 1, 0)), ((npx, -1), (npx - 2, 0))]
        if nw:
            pairs += [((0, npy), (1, npy)), ((0, npy + 1), (2, npy))]
        if ne:
            pairs += [((npx, npy), (npx - 1, npy)),
                      ((npx, npy + 1), (npx - 2, npy))]
    else:  # pragma: no cover - guard (mirrors the NumPy lane)
        raise ValueError(f"_fill_4corners_3d: dir={direction}")
    for (ti, tj), (si, sj) in pairs:
        q = q.at[ti - ilo, tj - jlo].set(q[si - ilo, sj - jlo])
    return q


def update_dz_c(bounds, km: int, dt: float, dp0: jnp.ndarray,
                zs: jnp.ndarray, area: jnp.ndarray, ut: jnp.ndarray,
                vt: jnp.ndarray, gz: jnp.ndarray, ws: jnp.ndarray,
                npx: int, npy: int, *, sw_corner: bool, se_corner: bool,
                ne_corner: bool, nw_corner: bool, grid_type: int = 0):
    """JAX twin of ``fv3_native_nh_core.update_dz_c``
    (nh_utils.F90:49-191).

    Functional: the NumPy lane updates ``gz``/``ws`` in place; this twin
    RETURNS ``(gz, ws)`` with the write window (i = is-1..ie+1,
    j = js-1..je+1, the oracle's :166-171 loop) rewritten and every halo
    cell carried through unchanged.  ``bounds`` is the STATIC
    ``(is_, ie, js, je, ng)`` int tuple (same doctrine as
    :func:`riem_solver_c`); ``dt``/``npx``/``npy``/corner flags/
    ``grid_type`` are static too (the corner fill and the level branches
    are Python arms).

    The km+1 interface levels are INDEPENDENT in the oracle's k loop
    (each level reads only the input gz), so they are batched over a
    trailing k axis here — FP-identical because every flux/update op is
    elementwise per (i, j, k).  The two sequential pieces stay
    ``lax.scan``: none in the flux update, and the bottom-up ``dz_min``
    monotonicity limiter (:179-189), whose ``jnp.maximum`` floor is C^0
    at its boundary (grads are exact where every level clears the floor
    strictly — same caveat class as sim1's p_fac floor).  The upwind
    flux select (:143-164) is a ``jnp.where`` on the sign of the
    advective wind — C^0 in ut/vt at exactly 0 wind at a flux point.

    Requires ``km >= 2`` (the top/bottom extrapolation ratios read
    dp0[1]/dp0[km-2]; at km=1 the NumPy lane raises IndexError on
    dp0[1]) and ``ng >= 2`` (the upwind stencil reads i = is-2 / j =
    js-2; with ng=1 the NumPy lane's fort views silently WRAP — here it
    is a loud error).
    """
    _require_f64_jax("update_dz_c", {
        "dp0": dp0, "zs": zs, "area": area, "ut": ut, "vt": vt,
        "gz": gz, "ws": ws})
    is_, ie, js, je, ng = bounds
    if km < 2:
        raise ValueError(
            f"update_dz_c: km={km} < 2 unsupported (top/bottom ratios "
            f"read dp0[1] and dp0[km-2]; the NumPy lane raises "
            f"IndexError at km=1)")
    if ng < 2:
        raise ValueError(
            f"update_dz_c: ng={ng} < 2 unsupported (the upwind flux "
            f"stencil reads i = is-2; the NumPy lane's fort views would "
            f"silently wrap)")
    dp0 = jnp.asarray(dp0)
    gz = jnp.asarray(gz)
    ws = jnp.asarray(ws)
    if dp0.shape != (km,):
        raise ValueError(f"update_dz_c: dp0 must be ({km},), got "
                         f"{dp0.shape}")
    if gz.shape[2] != km + 1:
        raise ValueError(f"update_dz_c: gz needs {km + 1} interfaces, "
                         f"got {gz.shape[2]}")

    rdt = 1.0 / dt
    top_ratio = dp0[0] / (dp0[0] + dp0[1])              # :75
    bot_ratio = dp0[km - 1] / (dp0[km - 2] + dp0[km - 1])   # :76
    ilo = is_ - ng
    jlo = js - ng
    a0 = ng - 1                       # storage row of i = is-1
    b0 = ng - 1                       # storage col of j = js-1
    ni_w = ie - is_ + 3               # write window is-1..ie+1
    nj_w = je - js + 3

    # --- advective interface winds, all km+1 levels at once ---
    # :94-133 — top extrapolation / interior dp0-weighted mean / bottom
    # extrapolation; elementwise per (i, j), so batching over k is
    # FP-identical to the oracle's per-level loop.
    w_lo = dp0[:-1]                   # dp0[k-1] for interior k=1..km-1
    w_hi = dp0[1:]                    # dp0[k]
    int_ratio = 1.0 / (w_lo + w_hi)                     # :123
    x_top = ut[:, :, 0] + (ut[:, :, 0] - ut[:, :, 1]) * top_ratio
    x_bot = (ut[:, :, km - 1]
             + (ut[:, :, km - 1] - ut[:, :, km - 2]) * bot_ratio)
    x_int = (w_hi * ut[:, :, :-1] + w_lo * ut[:, :, 1:]) * int_ratio
    xful = jnp.concatenate(
        [x_top[:, :, None], x_int, x_bot[:, :, None]], axis=2)
    y_top = vt[:, :, 0] + (vt[:, :, 0] - vt[:, :, 1]) * top_ratio
    y_bot = (vt[:, :, km - 1]
             + (vt[:, :, km - 1] - vt[:, :, km - 2]) * bot_ratio)
    y_int = (w_hi * vt[:, :, :-1] + w_lo * vt[:, :, 1:]) * int_ratio
    yful = jnp.concatenate(
        [y_top[:, :, None], y_int, y_bot[:, :, None]], axis=2)

    # --- corner-filled upwind source fields (:136-164) ---
    # The oracle fills a per-level COPY (gz itself keeps its halo
    # corners): direction 1 before the x fluxes, direction 2 ON TOP of
    # it before the y fluxes, and the flux-form numerator's center value
    # reads the doubly-filled copy.
    if grid_type < 3:
        g2x = _fill_4corners_3d(gz, 1, npx, npy, ilo, jlo, sw_corner,
                                se_corner, ne_corner, nw_corner)
        g2y = _fill_4corners_3d(g2x, 2, npx, npy, ilo, jlo, sw_corner,
                                se_corner, ne_corner, nw_corner)
    else:
        g2x = gz
        g2y = gz

    # Flux windows (:143-164): x on (is-1..ie+2, js-1..je+1), y on
    # (is-1..ie+1, js-1..je+2).
    xw = xful[a0:a0 + ni_w + 1, b0:b0 + nj_w, :]
    yw = yful[a0:a0 + ni_w, b0:b0 + nj_w + 1, :]
    fx = xw * jnp.where(xw > 0.0,
                        g2x[a0 - 1:a0 + ni_w, b0:b0 + nj_w, :],
                        g2x[a0:a0 + ni_w + 1, b0:b0 + nj_w, :])
    fy = yw * jnp.where(yw > 0.0,
                        g2y[a0:a0 + ni_w, b0 - 1:b0 + nj_w, :],
                        g2y[a0:a0 + ni_w, b0:b0 + nj_w + 1, :])

    # Flux-form update (:166-171).
    aw = area[a0:a0 + ni_w, b0:b0 + nj_w][:, :, None]
    center = g2y[a0:a0 + ni_w, b0:b0 + nj_w, :]
    num = (center * aw + fx[:-1, :, :] - fx[1:, :, :]
           + fy[:, :-1, :] - fy[:, 1:, :])
    den = (aw + xw[:-1, :, :] - xw[1:, :, :]
           + yw[:, :-1, :] - yw[:, 1:, :])
    gz_win = num / den

    # ws diagnosis (:173-178) — from the post-flux bottom interface,
    # which the limiter below never touches (k runs km-1..0).
    ws_win = (zs[a0:a0 + ni_w, b0:b0 + nj_w] - gz_win[:, :, km]) * rdt

    # Bottom-up dz_min monotonicity limiter (:179-189): sequential in k
    # (each level reads the LIMITED level below) — reverse lax.scan.
    def _lim(gz_below, gz_k):
        gz_new = jnp.maximum(gz_k, gz_below + DZ_MIN)
        return gz_new, gz_new

    _, lim_rev = lax.scan(_lim, gz_win[:, :, km],
                          jnp.moveaxis(gz_win[:, :, :km], 2, 0),
                          reverse=True)
    gz_lim = jnp.concatenate(
        [jnp.moveaxis(lim_rev, 0, 2), gz_win[:, :, km:km + 1]], axis=2)

    gz_out = gz.at[a0:a0 + ni_w, b0:b0 + nj_w, :].set(gz_lim)
    ws_out = ws.at[a0:a0 + ni_w, b0:b0 + nj_w].set(ws_win)
    return gz_out, ws_out


def make_riem_solver3_jit(fn=riem_solver3):
    """The ONE jit policy for the D-stage Riemann driver.

    Static: the same deck constants as the C stage plus the four flag
    arms (scale_m/use_logp/last_call/fp_out) as static_argnames — they
    select Python ``if`` branches and MUST never be traced.  No donated
    buffers (grad-path doctrine).
    """
    return jax.jit(
        fn, static_argnums=(0, 1, 2, 3, 4, 5, 6, 19, 20),
        static_argnames=("scale_m", "use_logp", "last_call", "fp_out"))


riem_solver3_jit = make_riem_solver3_jit()


def make_edge_profile_jit(fn=edge_profile):
    """The ONE jit policy for edge_profile (same doctrine as
    :func:`make_sim1_solver_jit`: production entry point and any
    instrumented retrace-test wrapper are built HERE).

    Static: j_lo/km (window labels), uniform_grid/limiter (Python branch
    selectors — never traced).  q1/q2/dp0 dynamic.  No donated buffers
    (grad-path doctrine).
    """
    return jax.jit(fn, static_argnums=(2, 3, 5, 6))


edge_profile_jit = make_edge_profile_jit()


def make_update_dz_c_jit(fn=update_dz_c):
    """The ONE jit policy for the C-stage height update.

    Static: bounds/km/dt/npx/npy positionally plus the corner flags and
    grid_type as static_argnames — all deck constants or Python branch
    selectors.  dp0/zs/area/ut/vt/gz/ws dynamic.  No donated buffers
    (grad-path doctrine).
    """
    return jax.jit(
        fn, static_argnums=(0, 1, 2, 10, 11),
        static_argnames=("sw_corner", "se_corner", "ne_corner",
                         "nw_corner", "grid_type"))


update_dz_c_jit = make_update_dz_c_jit()


# =====================================================================
# update_dz_d (nh_utils.F90:194-311).
#
# The transport half comes from the SHIPPED ``legoesm.core.fv3_tp_core``
# twins (``fv_tp_2d``, and ``copy_corners`` for the del6 chain); only
# ``del6_vt_flux`` — a sw_core routine with no JAX twin — is mirrored
# privately below, against ``fv3_native_d_sw.del6_vt_flux`` (hop B: the
# NumPy lane is the specification, the Fortran is never read directly).
#
# WHICH ARMS ARE LIVE (R6, read what the deck RESOLVES): the shipped duo
# tail config is ``hord_tm = 6``, ``nord_v = 2``, ``damp_v = 0.12``
# (``fv3_native_dsw_tail_3d.DUO_TAIL_CFG``).  The oracle's own call
# (nh_utils.F90:269-271) passes fv_tp_2d WITHOUT mfx/mfy/nord/damp_c/
# damp_smag and del6_vt_flux WITHOUT damp_km, so those optional
# arguments are ported as ABSENT (R5: port the EXECUTION, not the call
# text).  Consequently ``deln_flux``, fv_tp_2d's mass-weighted
# flux-averaging arm, and del6_vt_flux's damp_km rescale are not in this
# tree at all — they are unreachable from update_dz_d, not skipped.
# =====================================================================

def _fw(a, ilo, jlo, i0, i1, j0, j1):
    """Read the Fortran window ``a(i0:i1, j0:j1)`` — INCLUSIVE bounds,
    as a Fortran ``do`` — from a JAX array whose storage ``[0, 0]``
    element is Fortran ``(ilo, jlo)``.

    The read half of the NumPy lane's ``fort`` view, made functional.
    Unlike ``fort``, an out-of-storage index RAISES: ``fort.__getitem__``
    evaluates ``a[i - ilo]``, so a negative result silently WRAPS to the
    far edge and a too-large one silently truncates the slice — both
    return plausible numbers rather than an error, which is the most
    expensive defect class in this port.  Every bound here is a STATIC
    Python int, so the check runs once at trace time.

    ``i1 < i0`` is a legal EMPTY window (a Fortran ``do i=a,b`` with
    ``b < a`` executes zero times) and is not bound-checked.
    """
    lo_i, hi_i = i0 - ilo, i1 - ilo
    lo_j, hi_j = j0 - jlo, j1 - jlo
    if i0 <= i1 and (lo_i < 0 or hi_i >= a.shape[0]):
        raise IndexError(
            f"_fw: i window ({i0}:{i1}) escapes an array holding Fortran "
            f"i = {ilo}..{ilo + a.shape[0] - 1}")
    if j0 <= j1 and (lo_j < 0 or hi_j >= a.shape[1]):
        raise IndexError(
            f"_fw: j window ({j0}:{j1}) escapes an array holding Fortran "
            f"j = {jlo}..{jlo + a.shape[1] - 1}")
    return a[lo_i:hi_i + 1, lo_j:hi_j + 1]


def _fs(a, ilo, jlo, i0, i1, j0, j1, v):
    """Functional write ``a(i0:i1, j0:j1) = v`` (INCLUSIVE bounds).

    Returns a NEW array — the write half of ``fort`` with the same
    static bound checks as :func:`_fw`.
    """
    lo_i, hi_i = i0 - ilo, i1 - ilo
    lo_j, hi_j = j0 - jlo, j1 - jlo
    if i0 <= i1 and (lo_i < 0 or hi_i >= a.shape[0]):
        raise IndexError(
            f"_fs: i window ({i0}:{i1}) escapes an array holding Fortran "
            f"i = {ilo}..{ilo + a.shape[0] - 1}")
    if j0 <= j1 and (lo_j < 0 or hi_j >= a.shape[1]):
        raise IndexError(
            f"_fs: j window ({j0}:{j1}) escapes an array holding Fortran "
            f"j = {jlo}..{jlo + a.shape[1] - 1}")
    return a.at[lo_i:hi_i + 1, lo_j:hi_j + 1].set(v)


def _del6_vt_flux(nord, npx, npy, damp, q, bd, del6_u, del6_v,
                  rarea, bounded_domain, sw_corner, se_corner,
                  nw_corner, ne_corner, duogrid):
    """Functional twin of ``fv3_native_d_sw.del6_vt_flux``
    (sw_core.F90 del6_vt_flux, non-USE_SG branch).

    Returns ``(fx2, fy2)`` — the NumPy lane writes them into caller work
    arrays.  ``q``/``rarea`` have Fortran origin ``(isd, jsd)``;
    ``del6_u`` is ``(isd:ied, jsd:jed+1)`` and ``del6_v`` is
    ``(isd:ied+1, jsd:jed)``; ``fx2`` is ``(isd:ied+1, jsd:jed)`` and
    ``fy2`` is ``(isd:ied, jsd:jed+1)``.

    ``nord`` is a LOOP-ITERATION COUNT, so it is a static Python int by
    construction (CLAUDE.md: loop counts are never traced/trainable),
    and ``damp`` rides with it as a static float — the two together pick
    the operator, not a number in it.

    ``damp_km`` is absent (update_dz_d does not pass it), so the 0.5 *
    damp_km rescale at the end of the NumPy lane is unreachable here.

    ``d2`` is NaN-filled and only the oracle's own windows are written;
    the windows are sized so that no unwritten slot is ever read back
    (a wrongly sized window therefore shows up as a NaN, not as a
    plausible number).
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    nid, njd = ied - isd + 1, jed - jsd + 1
    dtype = q.dtype

    d2 = jnp.full((nid, njd), jnp.nan, dtype)
    fx2 = jnp.full((nid + 1, njd), jnp.nan, dtype)
    fy2 = jnp.full((nid, njd + 1), jnp.nan, dtype)

    i1, i2 = is_ - 1 - nord, ie + 1 + nord                    # :1727-1730
    j1, j2 = js - 1 - nord, je + 1 + nord
    d2 = _fs(d2, isd, jsd, i1, i2, j1, j2,                    # :1732-1734
             damp * _fw(q, isd, jsd, i1, i2, j1, j2))

    if nord > 0 and (not bounded_domain):                     # :1736-1739
        d2 = copy_corners(d2, npx, npy, 1, bounded_domain, bd,
                          sw_corner, se_corner, nw_corner, ne_corner,
                          duogrid=duogrid)
    fx2 = _fs(fx2, isd, jsd, is_ - nord, ie + nord + 1,       # :1740-1742
              js - nord, je + nord,
              _fw(del6_v, isd, jsd, is_ - nord, ie + nord + 1,
                  js - nord, je + nord)
              * (_fw(d2, isd, jsd, is_ - nord - 1, ie + nord,
                     js - nord, je + nord)
                 - _fw(d2, isd, jsd, is_ - nord, ie + nord + 1,
                       js - nord, je + nord)))

    if nord > 0 and (not bounded_domain):                     # :1744-1747
        d2 = copy_corners(d2, npx, npy, 2, bounded_domain, bd,
                          sw_corner, se_corner, nw_corner, ne_corner,
                          duogrid=duogrid)
    fy2 = _fs(fy2, isd, jsd, is_ - nord, ie + nord,           # :1748-1750
              js - nord, je + nord + 1,
              _fw(del6_u, isd, jsd, is_ - nord, ie + nord,
                  js - nord, je + nord + 1)
              * (_fw(d2, isd, jsd, is_ - nord, ie + nord,
                     js - nord - 1, je + nord)
                 - _fw(d2, isd, jsd, is_ - nord, ie + nord,
                       js - nord, je + nord + 1)))

    if nord > 0:                                              # :1752-1780
        for n in range(1, nord + 1):
            nt = nord - n
            d2 = _fs(d2, isd, jsd, is_ - nt - 1, ie + nt + 1,
                     js - nt - 1, je + nt + 1,
                     (_fw(fx2, isd, jsd, is_ - nt - 1, ie + nt + 1,
                          js - nt - 1, je + nt + 1)
                      - _fw(fx2, isd, jsd, is_ - nt, ie + nt + 2,
                            js - nt - 1, je + nt + 1)
                      + _fw(fy2, isd, jsd, is_ - nt - 1, ie + nt + 1,
                            js - nt - 1, je + nt + 1)
                      - _fw(fy2, isd, jsd, is_ - nt - 1, ie + nt + 1,
                            js - nt, je + nt + 2))
                     * _fw(rarea, isd, jsd, is_ - nt - 1, ie + nt + 1,
                           js - nt - 1, je + nt + 1))

            if not bounded_domain:
                d2 = copy_corners(d2, npx, npy, 1, bounded_domain, bd,
                                  sw_corner, se_corner, nw_corner,
                                  ne_corner, duogrid=duogrid)
            fx2 = _fs(fx2, isd, jsd, is_ - nt, ie + nt + 1,
                      js - nt, je + nt,
                      _fw(del6_v, isd, jsd, is_ - nt, ie + nt + 1,
                          js - nt, je + nt)
                      * (_fw(d2, isd, jsd, is_ - nt, ie + nt + 1,
                             js - nt, je + nt)
                         - _fw(d2, isd, jsd, is_ - nt - 1, ie + nt,
                               js - nt, je + nt)))

            if not bounded_domain:
                d2 = copy_corners(d2, npx, npy, 2, bounded_domain, bd,
                                  sw_corner, se_corner, nw_corner,
                                  ne_corner, duogrid=duogrid)
            fy2 = _fs(fy2, isd, jsd, is_ - nt, ie + nt,
                      js - nt, je + nt + 1,
                      _fw(del6_u, isd, jsd, is_ - nt, ie + nt,
                          js - nt, je + nt + 1)
                      * (_fw(d2, isd, jsd, is_ - nt, ie + nt,
                             js - nt, je + nt + 1)
                         - _fw(d2, isd, jsd, is_ - nt, ie + nt,
                               js - nt - 1, je + nt)))
    return fx2, fy2


def update_dz_d(ndif, damp, hord, bounds, km, npx, npy, area, rarea,
                dp0, zs, zh, crx, cry, xfx, yfx, ws, rdt,
                dxa, dya, del6_u, del6_v, *, lim_fac=1.0,
                bounded_domain=False, grid_type=0,
                sw_corner=True, se_corner=True, nw_corner=True,
                ne_corner=True, duogrid=False):
    """JAX twin of ``fv3_native_nh_core.update_dz_d``
    (nh_utils.F90:194-311).

    Functional: the NumPy lane updates ``zh`` and ``ws`` in place; this
    twin RETURNS ``(zh, ws)``.

    ``zh``'s compute window (i = is..ie, j = js..je, all km+1
    interfaces) is rewritten; every halo cell is carried through
    UNCHANGED **except** at levels where ``damp[k] <= 1e-5``, where the
    oracle hands ``zh(isd,jsd,k)`` itself to ``fv_tp_2d`` and
    ``copy_corners`` writes the tile-corner ghost blocks in place
    (nh_utils.F90:279).  Damped levels work on a copy (``z2``,
    :264-268) and their ghosts are untouched.  Both behaviours are
    reproduced exactly, so a full-array comparison against the NumPy
    lane is meaningful.

    ARGUMENT LAYOUT (Fortran dummies -> here; every 2-D operand carries
    Fortran origin ``(isd, jsd)`` unless stated):

      ``zh``            (isd:ied, jsd:jed, km+1)
      ``zs``, ``area``, ``rarea``, ``dxa``, ``dya``   (isd:ied, jsd:jed)
      ``del6_u`` (isd:ied, jsd:jed+1), ``del6_v`` (isd:ied+1, jsd:jed)
      ``crx``, ``xfx``  (is:ie+1, jsd:jed, km)
      ``cry``, ``yfx``  (isd:ied, js:je+1, km)
      ``dp0``           (km,)
      ``ws``            (is:ie, js:je)  — see below
      ``bounds``        STATIC ``(is_, ie, js, je, ng)`` int tuple, as
                        in :func:`riem_solver_c`

    ``ws`` is ``intent(out)`` in the oracle (nh_utils.F90:210) and every
    cell of the compute window is written (:296-298).  The operand is
    taken for signature parity with the NumPy lane and is dtype/shape
    gated, but its VALUE is never read — the returned ``ws`` is built
    fresh, and ``d(ws_out)/d(ws_in)`` is identically zero.

    STATIC-BY-NECESSITY OPERANDS.  ``damp`` and ``ndif`` are sequences
    of length km+1 of PYTHON numbers, not arrays: ``damp[k] > 1e-5``
    selects between two different operator chains (with/without the
    del-nord flux) and ``ndif[k]`` is the del-nord LOOP-ITERATION COUNT.
    Neither can be traced.  The oracle's ``damp(km+1) = damp(km)`` /
    ``ndif(km+1) = ndif(km)`` mutation (:231-232) is applied internally
    to a rebuilt tuple — the functional form of that write.  It is NOT
    returned, because static Python data cannot be a jit output; nothing
    reads it, since the only production caller rebuilds both arrays on
    every call (``fv3_native_dsw_tail_3d.py:406-408``).  A caller that
    needs the mutated value can apply the same one-line copy.

    ``uniform_grid = .false.`` (:229) and the ``edge_profile`` limiter is
    0 (:241-246), exactly as in the NumPy lane.

    PRECONDITION on ``area`` (measured, NumPy-lane probe job 9355001,
    restated because it is inherited, not re-measured here): the
    CORNER-DIAGONAL halo cells must hold REAL areas, as the oracle's
    mpp-exchanged ``gridstruct%area`` does.  The single-tile builder
    leaves BIG_NUMBER sentinels there and ``fv_tp_2d``'s y-intermediates
    read them.  Supplying corner-diagonal areas is owned by the six-face
    NH integration, not by this routine.

    THE k LOOP IS UNROLLED IN PYTHON, not a ``lax.scan``: each level's
    ``damp``/``ndif`` picks a different program, so the levels are not a
    single scanned body.  Compile cost therefore grows with km+1; that
    is inherent to the per-level static branch, not an oversight.

    NON-SMOOTH SITES (all C^0, named for the gradient gate):
      * the PPM ``smt5``/``smt6``/``hi5`` selectors and every
        ``copysign``/``min``/``max`` limiter inside ``fv3_tp_core``'s
        ``xppm``/``yppm`` — absent entirely at ``hord = 2``, the
        perfectly-linear arm the gradient gate uses;
      * the upwind selection ``jnp.where(c > 0)`` at every flux point;
      * ``fv3_tp_core.pert_ppm``'s branches (reachable at |hord| in
        {7, 9, 12, 13} and at the monotone edge fixes);
      * the ``dz_min`` bottom-up ``jnp.maximum`` floor (:305).

    UNSUPPORTED ``hord`` raises: the guard is ``fv3_tp_core``'s
    ``_validate_ord`` against its ``_PPM_ORDS`` (-6..-1, 1..13), reached
    through :func:`fv_tp_2d`, so a typo cannot silently fall through to
    a neighbouring PPM branch.
    """
    _require_f64_jax("update_dz_d", {
        "area": area, "rarea": rarea, "dp0": dp0, "zs": zs, "zh": zh,
        "crx": crx, "cry": cry, "xfx": xfx, "yfx": yfx, "ws": ws,
        "dxa": dxa, "dya": dya, "del6_u": del6_u, "del6_v": del6_v})

    is_, ie, js, je, ng = bounds
    isd, ied = is_ - ng, ie + ng
    jsd, jed = js - ng, je + ng
    ni, nj = ie - is_ + 1, je - js + 1
    ni_x, nj_y = ie + 1 - is_ + 1, je + 1 - js + 1
    nid, njd = ied - isd + 1, jed - jsd + 1
    # fv3_tp_core takes the NumPy lane's Bounds NamedTuple (hashable by
    # VALUE, so a fresh-but-equal one shares a jit cache entry); this
    # module's public surface keeps the 5-tuple used by riem_solver_c
    # and update_dz_c, and derives the rest here rather than asking
    # callers to keep two conventions straight.
    bd = Bounds(is_, ie, js, je, isd, ied, jsd, jed, ng)

    if km < 2:
        raise ValueError(
            f"update_dz_d: km={km} < 2 unsupported (edge_profile's top "
            f"row reads q[:, 1] and its bottom row reads q[:, km-2])")
    if ng < 3:
        raise ValueError(
            f"update_dz_d: ng={ng} < 3 unsupported. xppm/yppm's "
            f"is_==1 / js==1 edge stencils read q at Fortran index -2 "
            f"(tp_core.F90 xppm :299-303), which needs isd <= -2; the "
            f"NumPy lane's fort views would silently WRAP there.")
    if len(damp) != km + 1 or len(ndif) != km + 1:
        raise ValueError(
            f"update_dz_d: damp/ndif must have km+1={km + 1} slots (got "
            f"{len(damp)}, {len(ndif)}) — nh_utils.F90:231-232 writes "
            f"slot km+1")

    # :231-232, functionally: rebuild with slot km (0-based) = slot km-1.
    damp_e = tuple(float(x) for x in damp)[:km] + (float(damp[km - 1]),)
    ndif_e = tuple(int(x) for x in ndif)[:km] + (int(ndif[km - 1]),)
    for k, (dk, nk) in enumerate(zip(damp_e, ndif_e)):
        if dk > 1.0e-5 and not (0 <= nk <= ng - 1):
            raise ValueError(
                f"update_dz_d: ndif[{k}]={nk} out of range for ng={ng}. "
                f"del6_vt_flux reads d2 down to Fortran i = is-1-nord "
                f"(fv3_native_d_sw.py:1727), so 0 <= nord <= ng-1.")

    # Operands go through .at[...].set / slicing below, which needs jax
    # arrays (check_grads' numerical-FD path hands plain NumPy in).
    dp0 = jnp.asarray(dp0)
    zh = jnp.asarray(zh)
    crx = jnp.asarray(crx)
    cry = jnp.asarray(cry)
    xfx = jnp.asarray(xfx)
    yfx = jnp.asarray(yfx)
    area = jnp.asarray(area)
    rarea = jnp.asarray(rarea)
    zs = jnp.asarray(zs)
    dxa = jnp.asarray(dxa)
    dya = jnp.asarray(dya)
    del6_u = jnp.asarray(del6_u)
    del6_v = jnp.asarray(del6_v)
    if dp0.shape != (km,):
        raise ValueError(f"update_dz_d: dp0 must be ({km},), got "
                         f"{dp0.shape}")
    if zh.shape != (nid, njd, km + 1):
        raise ValueError(
            f"update_dz_d: zh must be ({nid}, {njd}, {km + 1}), got "
            f"{zh.shape}")
    if crx.shape != (ni_x, njd, km) or xfx.shape != (ni_x, njd, km):
        raise ValueError(
            f"update_dz_d: crx/xfx must be ({ni_x}, {njd}, {km}), got "
            f"{crx.shape}, {xfx.shape}")
    if cry.shape != (nid, nj_y, km) or yfx.shape != (nid, nj_y, km):
        raise ValueError(
            f"update_dz_d: cry/yfx must be ({nid}, {nj_y}, {km}), got "
            f"{cry.shape}, {yfx.shape}")
    if jnp.asarray(ws).shape != (ni, nj):
        raise ValueError(
            f"update_dz_d: ws must be the compute window ({ni}, {nj}) "
            f"(nh_utils.F90:210 ws(is:ie,js:je)), got "
            f"{jnp.asarray(ws).shape}")
    for nm, arr, want in (("area", area, (nid, njd)),
                          ("rarea", rarea, (nid, njd)),
                          ("zs", zs, (nid, njd)),
                          ("dxa", dxa, (nid, njd)),
                          ("dya", dya, (nid, njd)),
                          ("del6_u", del6_u, (nid, njd + 1)),
                          ("del6_v", del6_v, (nid + 1, njd))):
        if arr.shape != want:
            raise ValueError(
                f"update_dz_d: {nm} must be {want} with Fortran origin "
                f"({isd}, {jsd}), got {arr.shape}")

    # :238-246  vertical edge reconstruction.  The oracle runs one
    # edge_profile per j row; every operation in edge_profile is
    # elementwise over the free (row) axis with the k-recurrence
    # coefficients depending only on dp0, so batching all rows into one
    # call is FP-identical (the same argument riem_solver_c uses to
    # batch its j rows into sim1's i axis).
    crx_adv, xfx_adv = edge_profile(
        crx.reshape(ni_x * njd, km), xfx.reshape(ni_x * njd, km),
        jsd, km, dp0, False, 0)
    crx_adv = crx_adv.reshape(ni_x, njd, km + 1)
    xfx_adv = xfx_adv.reshape(ni_x, njd, km + 1)
    cry_adv, yfx_adv = edge_profile(
        cry.reshape(nid * nj_y, km), yfx.reshape(nid * nj_y, km),
        js, km, dp0, False, 0)
    cry_adv = cry_adv.reshape(nid, nj_y, km + 1)
    yfx_adv = yfx_adv.reshape(nid, nj_y, km + 1)

    zh_out = zh
    for k in range(km + 1):                                   # :254-291
        ra_x = (_fw(area, isd, jsd, is_, ie, jsd, jed)         # :249-253
                + xfx_adv[0:ni_x - 1, :, k] - xfx_adv[1:ni_x, :, k])
        ra_y = (_fw(area, isd, jsd, isd, ied, js, je)          # :255-260
                + yfx_adv[:, 0:nj_y - 1, k] - yfx_adv[:, 1:nj_y, k])

        z_in = zh_out[:, :, k]
        # The SHIPPED tp_core twin (no tp_core numerics re-derived here).
        # mfx/mfy/mass/nord/damp_c/damp_smag/damp_km stay at their None
        # defaults -- the oracle's call passes none of them (:269-270) --
        # so `da_min` is UNREAD: fv3_tp_core only touches it inside the
        # `damp_c > 1e-4` / `damp_smag > 1e-3` deln_flux arms, which
        # those Nones make unreachable.  0.0 is passed to make that
        # non-participation explicit rather than smuggling in a metric.
        q_cc, fx, fy = fv_tp_2d(
            z_in, crx_adv[:, :, k], cry_adv[:, :, k], npx, npy, hord,
            xfx_adv[:, :, k], yfx_adv[:, :, k], dxa, dya, area,
            del6_v, del6_u, rarea, 0.0, bd, ra_x, ra_y, lim_fac,
            bounded_domain, grid_type, sw_corner, se_corner, nw_corner,
            ne_corner, duogrid=duogrid)

        num = (_fw(q_cc, isd, jsd, is_, ie, js, je)
               * _fw(area, isd, jsd, is_, ie, js, je)
               + _fw(fx, is_, js, is_, ie, js, je)
               - _fw(fx, is_, js, is_ + 1, ie + 1, js, je)
               + _fw(fy, is_, js, is_, ie, js, je)
               - _fw(fy, is_, js, is_, ie, js + 1, je + 1))
        den = (_fw(ra_x, is_, jsd, is_, ie, js, je)
               + _fw(ra_y, isd, js, is_, ie, js, je)
               - _fw(area, isd, jsd, is_, ie, js, je))

        if damp_e[k] > 1.0e-5:                                # :263-278
            fx2, fy2 = _del6_vt_flux(
                ndif_e[k], npx, npy, damp_e[k], q_cc, bd,
                del6_u, del6_v, rarea, bounded_domain, sw_corner,
                se_corner, nw_corner, ne_corner, duogrid)
            win = num / den + (
                (_fw(fx2, isd, jsd, is_, ie, js, je)
                 - _fw(fx2, isd, jsd, is_ + 1, ie + 1, js, je)
                 + _fw(fy2, isd, jsd, is_, ie, js, je)
                 - _fw(fy2, isd, jsd, is_, ie, js + 1, je + 1))
                * _fw(rarea, isd, jsd, is_, ie, js, je))
            # The damped branch's fv_tp_2d works on a COPY of the level,
            # so zh's own ghost cells stay as they arrived.
            z_k = z_in
        else:                                                 # :279-289
            win = num / den
            # The undamped branch ALIASES zh, so copy_corners' ghost
            # writes persist into the output level.
            z_k = q_cc

        z_k = _fs(z_k, isd, jsd, is_, ie, js, je, win)
        zh_out = zh_out.at[:, :, k].set(z_k)

    # :294-309  ws diagnosis (BEFORE the limiter) + monotone height floor
    wi = slice(is_ - isd, ie - isd + 1)
    wj = slice(js - jsd, je - jsd + 1)
    ws_out = ((_fw(zs, isd, jsd, is_, ie, js, je)
               - zh_out[wi, wj, km]) * rdt)

    def _lim(z_below, z_k):
        z_new = jnp.maximum(z_k, z_below + DZ_MIN)
        return z_new, z_new

    win3 = zh_out[wi, wj, :]
    _, lim_rev = lax.scan(_lim, win3[:, :, km],
                          jnp.moveaxis(win3[:, :, :km], 2, 0),
                          reverse=True)
    zh_out = zh_out.at[wi, wj, :].set(jnp.concatenate(
        [jnp.moveaxis(lim_rev, 0, 2), win3[:, :, km:km + 1]], axis=2))
    return zh_out, ws_out


def make_update_dz_d_jit(fn=update_dz_d):
    """The ONE jit policy for the D-stage height update (same doctrine as
    :func:`make_sim1_solver_jit`: the production entry point and any
    instrumented retrace-test wrapper are built HERE, so a policy drift
    cannot pass unnoticed).

    Static positionally: ndif/damp (per-level operator selectors — pass
    them as TUPLES so they hash by value), hord, bounds, km, npx, npy,
    rdt.  Static by name: lim_fac plus the grid/corner flags, which
    select Python ``if`` arms and must never be traced.  Dynamic:
    area/rarea/dp0/zs/zh/crx/cry/xfx/yfx/ws/dxa/dya/del6_u/del6_v.
    No donated buffers (grad-path doctrine).
    """
    return jax.jit(
        fn, static_argnums=(0, 1, 2, 3, 4, 5, 6, 17),
        static_argnames=("lim_fac", "bounded_domain", "grid_type",
                         "sw_corner", "se_corner", "nw_corner",
                         "ne_corner", "duogrid"))


update_dz_d_jit = make_update_dz_d_jit()
