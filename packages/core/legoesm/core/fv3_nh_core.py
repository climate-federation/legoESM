"""FV3 non-hydrostatic core -- JAX lane (SIM1_solver, Riem_Solver_c,
Riem_Solver3).

Functional, jit-compatible mirror of the certified NumPy fp64 lane
(``fv3_native_nh_core.py``, itself a loop-faithful port of the pinned
oracle: ``nh_utils.F90:1193-1324`` SIM1, ``nh_utils.F90:313-420``
Riem_Solver_c, ``nh_core.F90:42-206`` Riem_Solver3 — non-MOIST_CAPPA /
non-USE_COND branches).  The NumPy lane stays the oracle-parity
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
