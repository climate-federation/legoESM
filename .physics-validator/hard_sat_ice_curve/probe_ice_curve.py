"""Numerical probe of the mixed-phase (ice-curve) hard-saturation drain.

Runs under JAX_ENABLE_X64=1.  Every block prints PASS/FAIL + the numbers so
the reviewer can see the evidence, not just a verdict.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    hard_saturation_drain,
    mixed_phase_liquid_fraction,
    mixed_phase_saturation_mixing_ratio,
    mixed_phase_l_over_cp,
    _hard_saturation_condensation,
)
from legoesm.driver.model_driver import _mpas_hard_saturation_poststep

DT = 75.0
fails = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        fails.append(name)
    print(f"[{tag}] {name}: {detail}")


# ---------------------------------------------------------------------------
# A. Blend monotonicity d(blend)/dT > 0  (bracket validity precondition)
# ---------------------------------------------------------------------------
Tg = jnp.linspace(190.0, 295.0, 1000)
p = jnp.full_like(Tg, 30000.0)
blend = mixed_phase_saturation_mixing_ratio(Tg, p)
dblend = jax.vmap(lambda t: jax.grad(
    lambda tt: mixed_phase_saturation_mixing_ratio(
        tt.reshape(1), jnp.array([30000.0]))[0])(t))(Tg)
check("A blend d(sat)/dT>0 over 190-295K", bool(jnp.all(dblend > 0)),
      f"min dsat/dT={float(jnp.min(dblend)):.3e}")
# also monotone by finite difference (robust to kink subgradients)
check("A blend monotone (FD)", bool(jnp.all(jnp.diff(blend) > 0)),
      f"min diff={float(jnp.min(jnp.diff(blend))):.3e}")

# ---------------------------------------------------------------------------
# B. Residual f(q_eq)=q_eq - sat(T+lcp*(q_v-q_eq)) strictly increasing in q_eq
#    with the FROZEN lcp(T0) but T-varying blend weight inside the solve.
# ---------------------------------------------------------------------------
T0 = 250.0
p0 = 40000.0
q_v0 = 3.0 * float(saturation_mixing_ratio_ice(jnp.array([T0]), jnp.array([p0]))[0])
lcp = float(mixed_phase_l_over_cp(jnp.array([T0]))[0])
qeq = np.linspace(1e-8, q_v0, 400)


def resid(qe):
    tnew = T0 + lcp * (q_v0 - qe)
    s = float(mixed_phase_saturation_mixing_ratio(
        jnp.array([tnew]), jnp.array([p0]))[0])
    return qe - s


rvals = np.array([resid(q) for q in qeq])
check("B residual strictly increasing in q_eq (frozen lcp, T-varying w)",
      bool(np.all(np.diff(rvals) > 0)),
      f"min d(resid)={np.min(np.diff(rvals)):.3e}, "
      f"f(lo)={rvals[0]:.3e}<=0<=f(hi)={rvals[-1]:.3e}")
check("B bracket signs f(qsat)<=0<=f(qv)", rvals[0] <= 0 <= rvals[-1],
      f"f(lo)={rvals[0]:.3e}, f(hi)={rvals[-1]:.3e}")

# ---------------------------------------------------------------------------
# C. Byte-identical when OFF (warm cells: both modes see pure liquid curve)
#    and across a T sweep spanning warm/mixed/cold for ice_curve=False vs the
#    plain liquid path.
# ---------------------------------------------------------------------------
Tc = jnp.linspace(200.0, 300.0, 50).reshape(-1, 1)
pc = jnp.full_like(Tc, 60000.0)
qvc = 1.4 * saturation_mixing_ratio(Tc, pc)
r_off = hard_saturation_drain(Tc, qvc, pc, DT, ice_curve=False)
# recompute the "old" liquid path directly (sat_fn default, lcp default)
q_sat_ref = saturation_mixing_ratio(Tc, pc)
cond_ref = _hard_saturation_condensation(Tc, qvc, pc, DT, q_sat_ref)
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    _hard_saturation_rate_limit, _DEFAULT_HARD_SAT_THRESHOLD,
    _HARD_SAT_QSAT_FLOOR)
drain_ref = jnp.where(
    qvc > _DEFAULT_HARD_SAT_THRESHOLD * jnp.maximum(q_sat_ref,
                                                    _HARD_SAT_QSAT_FLOOR),
    cond_ref, 0.0)
r_ref = _hard_saturation_rate_limit(drain_ref, qvc, DT, 5.0)
check("C ice_curve=False == explicit liquid path (bit)",
      bool(jnp.array_equal(r_off, r_ref)),
      f"max|diff|={float(jnp.max(jnp.abs(r_off-r_ref))):.3e}")
# warm cells: ON==OFF
Tw = jnp.full((4,), 290.0)
pw = jnp.full((4,), 85000.0)
qvw = 1.3 * saturation_mixing_ratio(Tw, pw)
check("C warm-cell ON==OFF (bit)",
      bool(jnp.array_equal(hard_saturation_drain(Tw, qvw, pw, DT),
                           hard_saturation_drain(Tw, qvw, pw, DT,
                                                 ice_curve=True))),
      "")

# ---------------------------------------------------------------------------
# D. fp32 parity for ice_curve=True
# ---------------------------------------------------------------------------
Td = jnp.full((5,), 210.0)
pd = jnp.full((5,), 12000.0)
qvd = 1.3 * saturation_mixing_ratio_ice(Td, pd)
r64 = hard_saturation_drain(Td, qvd, pd, DT, ice_curve=True)
r32 = hard_saturation_drain(Td.astype(jnp.float32),
                            qvd.astype(jnp.float32),
                            pd.astype(jnp.float32), DT, ice_curve=True)
rel = float(jnp.max(jnp.abs(r32 - r64) / jnp.maximum(jnp.abs(r64), 1e-30)))
check("D fp32 vs fp64 ice_curve parity (rel<2e-3)", rel < 2e-3,
      f"max rel diff={rel:.3e}")

# ---------------------------------------------------------------------------
# E. Gradient vs centered finite difference, ice_curve=True, at TTL + mixed.
# ---------------------------------------------------------------------------
def drain_scalar(qv, T, pp):
    return hard_saturation_drain(
        T.reshape(1), qv.reshape(1), pp.reshape(1), DT, ice_curve=True)[0]


for ( Te, pe, mult, label) in [(195.0, 10000.0, 1.4, "TTL 195K"),
                               (250.0, 40000.0, 1.5, "mixed 250K")]:
    qsat_i = float(saturation_mixing_ratio_ice(
        jnp.array([Te]), jnp.array([pe]))[0])
    qv = jnp.array(mult * qsat_i)
    T = jnp.array(Te)
    pp = jnp.array(pe)
    gqv = float(jax.grad(drain_scalar, 0)(qv, T, pp))
    gT = float(jax.grad(drain_scalar, 1)(qv, T, pp))
    gp = float(jax.grad(drain_scalar, 2)(qv, T, pp))

    def fd(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    fd_qv = fd(lambda x: float(drain_scalar(jnp.array(x), T, pp)),
               float(qv), 1e-9)
    fd_T = fd(lambda x: float(drain_scalar(qv, jnp.array(x), pp)),
              float(T), 1e-4)
    fd_p = fd(lambda x: float(drain_scalar(qv, T, jnp.array(x))),
              float(pp), 1e-1)
    for nm, a, n in [("d/dqv", gqv, fd_qv), ("d/dT", gT, fd_T),
                     ("d/dp", gp, fd_p)]:
        rel = abs(a - n) / max(abs(n), 1e-12)
        check(f"E {label} {nm} grad~FD (rel<1e-4)", rel < 1e-4,
              f"AD={a:.6e} FD={n:.6e} rel={rel:.2e}")
    check(f"E {label} grads finite",
          all(np.isfinite([gqv, gT, gp])), f"{gqv:.3e},{gT:.3e},{gp:.3e}")

# ---------------------------------------------------------------------------
# F. Poststep conservation: total water + moist enthalpy with frozen L_eff(T0)
# ---------------------------------------------------------------------------
ncol, nlev = 3, 4
Tp = jnp.full((ncol, nlev), 210.0)
p_s = jnp.full(ncol, 1.0e5)
sig = jnp.linspace(0.08, 0.12, nlev)
pf = p_s[:, None] * sig[None, :]
qvp = 1.4 * saturation_mixing_ratio_ice(Tp, pf)
qcp = jnp.zeros((ncol, nlev))
qip = jnp.zeros((ncol, nlev))
T2, qv2, qc2, qi2, dq = _mpas_hard_saturation_poststep(
    Tp, qvp, qcp, p_s, sig, DT, 1.1, 5.0, ice_curve=True, q_i=qip)
w_tot0 = qvp + qcp + qip
w_tot1 = qv2 + qc2 + qi2
check("F poststep total-water conserved",
      float(jnp.max(jnp.abs(w_tot1 - w_tot0))) < 1e-15,
      f"max|dW|={float(jnp.max(jnp.abs(w_tot1-w_tot0))):.3e}")
lcp_arr = mixed_phase_l_over_cp(Tp)
h0 = Tp + lcp_arr * qvp
h1 = T2 + lcp_arr * qv2
check("F poststep enthalpy c_pd*T+L_eff(T0)*qv conserved",
      float(jnp.max(jnp.abs(h1 - h0))) < 1e-10,
      f"max|dH|={float(jnp.max(jnp.abs(h1-h0))):.3e}")
check("F all-cold routes to q_i (q_c untouched)",
      float(jnp.max(jnp.abs(qc2))) < 1e-18, f"max q_c={float(jnp.max(qc2)):.3e}")

# ---------------------------------------------------------------------------
# G. Rate-limit L_v vs L_eff: does the heating cap bind at the STATED K with
#    the ice branch, or ~L_eff/L_v too high?
# ---------------------------------------------------------------------------
for Tg1, label in [(250.0, "mixed 250K"), (234.0, "near-ice 234K")]:
    Tr = jnp.full((1, 1), Tg1)
    pr = jnp.full((1, 1), 50000.0)
    # huge pool so the on-curve drain exceeds the 2 g/kg heating cap
    qvr = jnp.full((1, 1), 0.05)  # 50 g/kg, absurd supersat pool
    r = hard_saturation_drain(Tr, qvr, pr, DT, 1.1, 5.0, ice_curve=True)
    dq_r = float(r[0, 0]) * DT
    lcp_r = float(mixed_phase_l_over_cp(Tr)[0, 0])
    dT_actual = lcp_r * dq_r
    dqv_cap = 5.0 * constants.c_pd / constants.L_v
    print(f"    G[{label}] dq={dq_r*1e3:.3f} g/kg (cap {dqv_cap*1e3:.3f}), "
          f"actual dT={dT_actual:.3f} K (nominal cap 5.0 K), "
          f"overshoot={100*(dT_actual/5.0-1):.1f}%")
    check(f"G {label} heating cap holds at ~5K (<=5% over)",
          dT_actual <= 5.0 * 1.05,
          f"actual dT={dT_actual:.3f} K")

# ---------------------------------------------------------------------------
# H. Sub-freezing above-ramp (250K) condensate split sanity in poststep.
# ---------------------------------------------------------------------------
Th = jnp.full((1, 1), 250.0)
p_sh = jnp.full(1, 5.0e4)
sigh = jnp.ones(1)
qvh = 2.0 * saturation_mixing_ratio_ice(Th, jnp.full((1, 1), 5.0e4))
_, _, qc_h, qi_h, dq_h = _mpas_hard_saturation_poststep(
    Th, qvh, jnp.zeros((1, 1)), p_sh, sigh, DT, 1.1, 5.0,
    ice_curve=True, q_i=jnp.zeros((1, 1)))
w = float(mixed_phase_liquid_fraction(Th)[0, 0])
ok = (abs(float(qc_h[0, 0]) - w * float(dq_h[0, 0])) < 1e-15 and
      abs(float(qi_h[0, 0]) - (1 - w) * float(dq_h[0, 0])) < 1e-15)
check("H 250K split q_c=w*dq, q_i=(1-w)*dq", ok,
      f"w={w:.3f} q_c/dq={float(qc_h[0,0])/max(float(dq_h[0,0]),1e-30):.3f}")

# ---------------------------------------------------------------------------
# I. Landing: after ice-curve drain, cell sits ON/ABOVE blended curve of the
#    WARMED state (pure drain, no overshoot below).
# ---------------------------------------------------------------------------
Ti = jnp.full((1, 1), 205.0)
pi = jnp.full((1, 1), 15000.0)
qvi = 1.5 * saturation_mixing_ratio_ice(Ti, pi)
ri = hard_saturation_drain(Ti, qvi, pi, DT, ice_curve=True)
dqi = ri * DT
qv_after = qvi - dqi
T_after = Ti + mixed_phase_l_over_cp(Ti) * dqi
q_target = mixed_phase_saturation_mixing_ratio(T_after, pi)
resid_land = float((qv_after - q_target)[0, 0])
check("I lands ON/above warmed blended curve (pure drain)",
      resid_land >= -1e-12, f"qv_after-q_sat(T_after)={resid_land:.3e}")
check("I lands close to curve (|resid| small rel)",
      abs(resid_land) < 1e-6, f"resid={resid_land:.3e}")

# ---------------------------------------------------------------------------
# J. Gate on ice curve at TTL = RHi trigger.  Fires just above 1.1*q_sat_ice,
#    zero just below.
# ---------------------------------------------------------------------------
Tj = jnp.full((1, 1), 195.0)
pj = jnp.full((1, 1), 10000.0)
qsat_i = saturation_mixing_ratio_ice(Tj, pj)
r_below = hard_saturation_drain(Tj, 1.05 * qsat_i, pj, DT, ice_curve=True)
r_above = hard_saturation_drain(Tj, 1.20 * qsat_i, pj, DT, ice_curve=True)
check("J RHi<=1.1 no drain", float(r_below[0, 0]) == 0.0,
      f"r(RHi1.05)={float(r_below[0,0]):.3e}")
check("J RHi=1.20 drains", float(r_above[0, 0]) > 0.0,
      f"r(RHi1.20)={float(r_above[0,0]):.3e}")

print("\n=== SUMMARY ===")
print("FAILURES:", fails if fails else "NONE")
