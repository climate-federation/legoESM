"""FV3 non-hydrostatic core -- JAX lane (SIM1_solver, Riem_Solver_c,
Riem_Solver3, edge_profile, update_dz_c).

Functional, jit-compatible mirror of the certified NumPy fp64 lane
(``fv3_native_nh_core.py``, itself a loop-faithful port of the pinned
oracle: ``nh_utils.F90:1193-1324`` SIM1, ``nh_utils.F90:313-420``
Riem_Solver_c, ``nh_core.F90:42-206`` Riem_Solver3,
``nh_utils.F90:1535-1641`` edge_profile, ``nh_utils.F90:49-191``
update_dz_c — non-MOIST_CAPPA / non-USE_COND branches, ``dz_min = 2.``).  The NumPy lane stays the oracle-parity
reference; this module is the production/JAX twin and is certified
AGAINST the NumPy lane, never against the Fortran directly (one
authority per hop).

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
  k with everything vectorised over i is the right shape.
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
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np  # STATIC trace-time scalars only (peln1/ptk), never traced
from jax import lax

from legoesm.core.fv3_native_nh_core import DZ_MIN  # nh_utils.F90:40-44
from legoesm.grids.fv3_native_gridstruct import FV3_GRAV, FV3_RDGAS

# nh_utils.F90:45 (math constant; mirrors the NumPy lane's module constant)
_R3 = 1.0 / 3.0


def _require_f64_jax(fname: str, arrays: dict) -> None:
    """Static-dtype gate mirroring the NumPy lane's ``_require_f64``.

    Reads only ``.dtype`` (static under jit): a float32 operand would
    otherwise be silently upcast -- or worse, with x64 disabled the whole
    solve would silently run in float32 -- and the oracle build is
    ``-fdefault-real-8``.
    """
    for name, a in arrays.items():
        if jnp.asarray(a).dtype != jnp.float64:
            raise TypeError(
                f"{fname}: {name} must be float64 (got "
                f"{jnp.asarray(a).dtype}); enable jax_enable_x64 and pass "
                f"f64 operands (oracle build is -fdefault-real-8)")


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

    The interface tridiagonal is a sequential Thomas recurrence in k —
    ``lax.scan``, NOT associative_scan (parity doctrine, module
    docstring).  The k-recurrence coefficients (``gam``/``gak``) depend
    only on ``dp0``, so they ride the scan carry as scalars; the NumPy
    lane stores them as (ni,) vectors of identical entries, which is the
    same elementwise arithmetic.

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

    if uniform_grid:                                   # :1552-1581
        r2o3 = 2.0 / 3.0
        r4o3 = 4.0 / 3.0
        e1_0 = r4o3 * q1[:, 0] + r2o3 * q1[:, 1]
        e2_0 = r4o3 * q2[:, 0] + r2o3 * q2[:, 1]
        coef0 = jnp.asarray(7.0 / 3.0, dp0.dtype)      # gak(1)

        def _fwd_u(carry, x):
            gak_prev, e1p, e2p = carry
            q1m, q1k, q2m, q2k = x
            gak = 1.0 / (4.0 - gak_prev)
            e1 = (3.0 * (q1m + q1k) - e1p) * gak
            e2 = (3.0 * (q2m + q2k) - e2p) * gak
            return (gak, e1, e2), (gak, e1, e2)

        xs = (q1[:, :-1].T, q1[:, 1:].T, q2[:, :-1].T, q2[:, 1:].T)
        ((gak_last, e1_last, e2_last),
         (coef_tail, e1_tail, e2_tail)) = lax.scan(
            _fwd_u, (coef0, e1_0, e2_0), xs)

        bet = 1.0 / (1.5 - 3.5 * gak_last)             # :1571
        e1_bot = (4.0 * q1[:, km - 1] + q1[:, km - 2]
                  - 3.5 * e1_last) * bet
        e2_bot = (4.0 * q2[:, km - 1] + q2[:, km - 2]
                  - 3.5 * e2_last) * bet
    else:                                              # :1583-1618
        g_arr = dp0[:-1] / dp0[1:]                     # gk for k=1..km-1
        g0 = dp0[1] / dp0[0]
        xt1 = 2.0 * g0 * (g0 + 1.0)
        bet0 = g0 * (g0 + 0.5)
        e1_0 = (xt1 * q1[:, 0] + q1[:, 1]) / bet0
        e2_0 = (xt1 * q2[:, 0] + q2[:, 1]) / bet0
        coef0 = (1.0 + g0 * (g0 + 1.5)) / bet0         # gam(1)

        def _fwd_n(carry, x):
            gam_prev, e1p, e2p = carry
            gk, q1m, q1k, q2m, q2k = x
            bet_v = 2.0 + 2.0 * gk - gam_prev
            e1 = (3.0 * (q1m + gk * q1k) - e1p) / bet_v
            e2 = (3.0 * (q2m + gk * q2k) - e2p) / bet_v
            gam = gk / bet_v
            return (gam, e1, e2), (gam, e1, e2)

        xs = (g_arr, q1[:, :-1].T, q1[:, 1:].T, q2[:, :-1].T,
              q2[:, 1:].T)
        ((gam_last, e1_last, e2_last),
         (coef_tail, e1_tail, e2_tail)) = lax.scan(
            _fwd_n, (coef0, e1_0, e2_0), xs)

        # :1602-1609 — the Fortran reuses the LAST loop gk; g_arr[km-2]
        # is the same division dp0(km-1)/dp0(km), bit-identical.
        gk = g_arr[km - 2]
        a_bot = 1.0 + gk * (gk + 1.5)
        xt1b = 2.0 * gk * (gk + 1.0)
        xt2 = gk * (gk + 0.5) - a_bot * gam_last
        e1_bot = (xt1b * q1[:, km - 1] + q1[:, km - 2]
                  - a_bot * e1_last) / xt2
        e2_bot = (xt1b * q2[:, km - 1] + q2[:, km - 2]
                  - a_bot * e2_last) / xt2

    # Back-substitution k=km-1..0 (:1577-1580 / :1612-1617): the raw
    # forward rows are e[k]=e_0..e_last; coef[k] is gak/gam for k=0..km-1.
    e1_raw = jnp.concatenate([e1_0[:, None], e1_tail.T], axis=1)  # (ni,km)
    e2_raw = jnp.concatenate([e2_0[:, None], e2_tail.T], axis=1)
    coef = jnp.concatenate([coef0[None], coef_tail], axis=0)      # (km,)

    def _bwd(carry, x):
        e1n, e2n = carry
        e1r, e2r, ck = x
        e1 = e1r - ck * e1n
        e2 = e2r - ck * e2n
        return (e1, e2), (e1, e2)

    (_, (e1_head_rev, e2_head_rev)) = lax.scan(
        _bwd, (e1_bot, e2_bot), (e1_raw.T, e2_raw.T, coef),
        reverse=True)
    qe1 = jnp.concatenate([e1_head_rev.T, e1_bot[:, None]], axis=1)
    qe2 = jnp.concatenate([e2_head_rev.T, e2_bot[:, None]], axis=1)

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
