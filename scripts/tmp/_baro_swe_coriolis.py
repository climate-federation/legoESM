"""Minimal standalone C-grid barotropic shallow-water solver to DERIVE the
null-mode-free Coriolis for internal_tide #576, in isolation from the model's
entangled explicit_ab2/F_slow machinery.

Periodic in BOTH x and y (the oracle's Flat-y / re-entrant channel), f-plane.
IC: U=U0 uniform, V=0, eta=0 -> analytic free inertial oscillation
U=U0 cos(ft), V=-U0 sin(ft) (eta stays ~0 since U is x-uniform).

Tests Coriolis discretizations:
  'naive'  : 4-pt average V->u, U->v (the model's form; has the 2Δ null mode)
  'energy' : Sadourny energy-conserving — the U->v average is the EXACT ADJOINT
             (transpose) of the V->u average, so the Coriolis does zero net work
             (no spurious energy source -> no null-mode growth).

Forward-backward (Matsuno) time stepping (neutral for gravity+Coriolis).
"""
from __future__ import annotations
import numpy as np

NY, NX = 16, 8
dx = 4.0e3
H = 2.0e3
g = 9.80616
f = -1.0312e-4
U0 = 0.2
dt = 20.0   # < dx/sqrt(gH)=28s -> explicit gravity-wave CFL-safe (isolate Coriolis)
Tin = 2 * np.pi / abs(f)
c = np.sqrt(g * H)


def avg_v_to_u(V):
    """V (NY+1,NX) -> u-points (NY,NX). 4-pt: average the 4 surrounding v-faces.
    u-point (j, i) [i is x-face between cell i-1 and i] sees v-faces
    (j,i-1),(j,i),(j+1,i-1),(j+1,i). Periodic in x and y."""
    # V on v-faces rows 0..NY (row 0 == row NY periodic). Use cell-row V: Vc(NY,NX)
    Vc = 0.5 * (V[:-1] + V[1:])              # v at cell centers (NY,NX)
    # u-face i sees cells i-1 and i (periodic x)
    return 0.5 * (Vc + np.roll(Vc, 1, axis=1))   # (NY,NX) at u-faces (drop the +1 wrap col)


def avg_u_to_v(U):
    """U (NY,NX) u-faces -> v-points (NY,NX) cell-rows. EXACT ADJOINT of avg_v_to_u
    for energy conservation."""
    Uc = 0.5 * (U + np.roll(U, -1, axis=1))   # u at cell centers (NY,NX)
    return 0.5 * (Uc + np.roll(Uc, -1, axis=0))  # (NY,NX) — adjoint pattern


def run(scheme, nstep):
    # Simplified: keep U,V,eta all at (NY,NX) cell-row staggering with periodic
    # roll operators (a clean C-grid: U on x-faces, V on y-faces, eta centers).
    U = np.full((NY, NX), U0); V = np.zeros((NY, NX)); eta = np.zeros((NY, NX))
    # tiny seed of the 2Δy mode to test whether it grows
    rng = np.arange(NY)[:, None]
    V += 1e-6 * ((-1.0) ** rng)
    hist = []
    for n in range(nstep):
        # gradients (periodic): d/dx at u-face, d/dy at v-face
        detadx = (eta - np.roll(eta, 1, axis=1)) / dx
        detady = (eta - np.roll(eta, 1, axis=0)) / dx
        if scheme == 'naive':
            Vu = 0.5 * (V + np.roll(V, 1, axis=1))      # V->u 2pt (x-avg)
            Vu = 0.5 * (Vu + np.roll(Vu, -1, axis=0))   # ->u (y-avg) ~ 4pt
            Unew = U + dt * (f * Vu - g * detadx)
            Uv = 0.5 * (Unew + np.roll(Unew, -1, axis=1))
            Uv = 0.5 * (Uv + np.roll(Uv, 1, axis=0))
            Vnew = V + dt * (-f * Uv - g * detady)
        elif scheme == 'energy':
            Vu = avg_v_to_u(np.concatenate([V, V[:1]], axis=0))
            Unew = U + dt * (f * Vu - g * detadx)
            Uv = avg_u_to_v(Unew)
            Vnew = V + dt * (-f * Uv - g * detady)
        # continuity (forward-backward: use new U,V)
        divU = (np.roll(Unew, -1, axis=1) - Unew) / dx + (np.roll(Vnew, -1, axis=0) - Vnew) / dx
        eta = eta - dt * H * divU
        U, V = Unew, Vnew
        if n % (nstep // 16) == 0:
            hist.append((n * dt / Tin, U.mean(), np.abs(U).max(), np.abs(V).max()))
    return hist


for scheme in ['naive', 'energy']:
    h = run(scheme, int(round(2 * Tin / dt)))
    print(f'--- {scheme} ---')
    for t, um, umx, vmx in h:
        print(f'  t/Tin={t:.2f}  <U>={um:+.3f}  max|U|={umx:.3f}  max|V|={vmx:.3f}  '
              f'analytic U={U0*np.cos(2*np.pi*t):+.3f}')
