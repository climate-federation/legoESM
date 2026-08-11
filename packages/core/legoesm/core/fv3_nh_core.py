"""FV3 non-hydrostatic core -- JAX lane (pattern-setter: SIM1_solver).

Functional, jit-compatible mirror of the certified NumPy fp64 lane
(``fv3_native_nh_core.py``, itself a loop-faithful port of the pinned
oracle ``nh_utils.F90:1193-1324``, non-MOIST_CAPPA branches).  The NumPy
lane stays the oracle-parity reference; this module is the production/JAX
twin and is certified AGAINST the NumPy lane, never against the Fortran
directly (one authority per hop).

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
from jax import lax

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
