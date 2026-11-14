"""Analytic stub 'schemes' shared bit-for-bit with oracle_seq.F90 (same directory).

Rates in SI per second, arrays (ncol, nlev) top->bottom.  They are NOT
physics: each is a deterministic, state-dependent tendency so that the order
in which the glue evaluates/applies them changes the answer.
"""
import jax.numpy as jnp

NB = 10                 # turbulence mixes the bottom NB levels
SHF, LHF = 20.0, 100.0  # fixed surface fluxes [W/m2] (coupler-held, CAM-like)
B_SINK = 2.0e-10        # forced qc sink [kg/kg/s] on levels 20..24 (0-based)


def conv(T, q, u, pm, cp, Lv):
    w = (pm / 1.0e5) ** 4
    dT = 0.02 * q * w
    dlf = 0.2 * cp / Lv * dT
    dq = -cp / Lv * dT - dlf
    du = -1.0e-5 * u * w
    return dT, dq, dlf, du


def turb(T, q, qc, u, dt, dp, cp, Lv, g):
    fac = (1.0 - jnp.exp(-dt / 3600.0)) / dt
    dpb = dp[-NB:]
    out = []
    for X in (T, q, qc, u):
        Xb = jnp.sum(X[:, -NB:] * dpb, axis=1, keepdims=True) / jnp.sum(dpb)
        tb = (Xb - X[:, -NB:]) * fac
        out.append(jnp.concatenate([jnp.zeros_like(X[:, :-NB]), tb], axis=1))
    dT, dq, dqc, du = out
    dT = dT.at[:, -1].add(SHF * g / (cp * dp[-1]))
    dq = dq.at[:, -1].add(LHF * g / (Lv * dp[-1]))
    return dT, dq, dqc, du


def micro(T, q, qc, qi, dt, q0ref, dp, cp, Lv, g):
    cond = jnp.maximum((q - q0ref) / 1800.0, -jnp.maximum(qc, 0.0) / dt)
    auto = qc * (1.0 - jnp.exp(-dt / 1200.0)) / dt
    dep = qi * (1.0 - jnp.exp(-dt / 3000.0)) / dt
    k = jnp.arange(T.shape[1])
    b = jnp.where((k >= 20) & (k <= 24), B_SINK, 0.0)[None, :] * jnp.ones_like(T)
    dq = -cond
    dqc = cond - auto - b
    dqi = -dep
    dT = Lv * cond / cp
    precip = jnp.sum((auto + b + dep) * dp[None, :], axis=1) / g
    return dT, dq, dqc, dqi, precip


def rad(T, qc, qi):
    return -(T - 180.0) / (40.0 * 86400.0) + 1.0e-5 * (qc + qi) / 1.0e-4


def gw(T, u, cp):
    k = jnp.arange(T.shape[1])[None, :]
    du = jnp.where(k < 8, -u * (T - 150.0) * 1.0e-8, 0.0)
    dT = -u * du / cp
    return du, dT
