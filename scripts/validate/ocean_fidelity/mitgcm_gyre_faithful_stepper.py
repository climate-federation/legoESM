"""Bit-exact MITgcm gyre stepper (all operators corr~1.0 vs MITgcm diagnostics),
AB2 + implicit free surface (sparse-LU), in MITgcm (ny,nx) convention.
Test: does it reproduce MITgcm laminar |u|max=0.031?"""
import numpy as np, sys, time
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

g = 9.81; rho = 1000.; drF = 5000.; f0 = 1e-4; beta = 1e-11
A_h = 400.; H = 5000.; ABEPS = 0.01; sideDragFactor = 2.0
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
N = int(sys.argv[2]) if len(sys.argv) > 2 else 62
DXF = 20e3 if N == 62 else 10e3
dx = dy = DXF; rAw = dx * dy; dt = 1200. if N == 62 else 600.
ORIG = -20e3 if N == 62 else -10e3

m = np.ones((N, N)); m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0
mW = m * np.roll(m, 1, axis=1)
mS = m * np.roll(m, 1, axis=0)
hFacZ = m * np.roll(m, 1, axis=1) * np.roll(m, 1, axis=0) * np.roll(np.roll(m, 1, axis=1), 1, axis=0)
rx = lambda a, s: np.roll(a, s, axis=1)
ry = lambda a, s: np.roll(a, s, axis=0)
yc = (np.arange(N) + 0.5) * dy + ORIG
fC = (f0 + beta * yc)[:, None] * np.ones((N, N))
jj = np.arange(N); Y = (jj - 0.5) / (N - 2); tau = -0.1 * np.cos(np.pi * Y)
wind = (tau / (rho * drF))[:, None] * np.ones((N, N)) * mW


def G_expl(u, v):
    uT = u * dy; vT = v * dx
    fZon = 0.25 * (uT + rx(uT, -1)) * (u + rx(u, -1))
    fMer = 0.25 * (vT + rx(vT, 1)) * (u + ry(u, 1))
    aU = -(1 / rAw) * ((fZon - rx(fZon, 1)) + (ry(fMer, -1) - fMer)) * mW
    gZon = 0.25 * (vT + ry(vT, -1)) * (v + ry(v, -1))
    gMer = 0.25 * (uT + ry(uT, 1)) * (v + rx(v, 1))
    aV = -(1 / rAw) * ((gZon - ry(gZon, 1)) + (rx(gMer, -1) - gMer)) * mS
    V_at_u = 0.25 * (v + ry(v, -1) + rx(v, 1) + rx(ry(v, -1), 1)); f_u = 0.5 * (rx(fC, 1) + fC)
    cU = f_u * V_at_u * mW
    U_at_v = 0.25 * (u + rx(u, -1) + ry(u, 1) + ry(rx(u, -1), 1)); f_v = 0.5 * (ry(fC, 1) + fC)
    cV = -f_v * U_at_v * mS
    fZx = A_h * (dy / dx) * (rx(u, -1) - u) * m
    fMy = A_h * (dx / dy) * (u - ry(u, 1)) * hFacZ
    vU = (1 / rAw) * ((fZx - rx(fZx, 1)) + (ry(fMy, -1) - fMy)) * mW
    gZx = A_h * (dy / dx) * (rx(v, -1) - v) * m
    gMy = A_h * (dx / dy) * (v - ry(v, 1)) * hFacZ
    vV = (1 / rAw) * ((gZx - rx(gZx, 1)) + (ry(gMy, -1) - gMy)) * mS
    hzcS = (mW - hFacZ); hzcN = (mW - ry(hFacZ, -1))
    sU = -(1 / rAw) * (hzcS * (dx / dy) + hzcN * (dx / dy)) * u * sideDragFactor * A_h * mW
    hzcW = (mS - hFacZ); hzcE = (mS - rx(hFacZ, -1))
    sV = -(1 / rAw) * (hzcW * (dy / dx) + hzcE * (dy / dx)) * v * sideDragFactor * A_h * mS
    return (aU + cU + vU + sU + wind) * mW, (aV + cV + vV + sV) * mS


coef = g * dt * dt * H
rows = []; cols = []; vals = []
idx = lambda i, j: ((i % N) * N + (j % N))
for i in range(N):
    for j in range(N):
        c = i * N + j
        if m[i, j] == 0:
            rows.append(c); cols.append(c); vals.append(1.0); continue
        diag = 1.0
        for (di, dj, face) in [(0, 1, mW[i, (j + 1) % N]), (0, -1, mW[i, j]),
                               (1, 0, mS[(i + 1) % N, j]), (-1, 0, mS[i, j])]:
            if face > 0:
                coup = coef / (dx * dx)
                diag += coup
                rows.append(c); cols.append(idx(i + di, j + dj)); vals.append(-coup)
        rows.append(c); cols.append(c); vals.append(diag)
A = sp.csc_matrix((vals, (rows, cols)), shape=(N * N, N * N))
solve_fs = factorized(A)

u = np.zeros((N, N)); v = np.zeros((N, N)); eta = np.zeros((N, N))
gup = np.zeros((N, N)); gvp = np.zeros((N, N))
t0 = time.time()
for it in range(1, NT + 1):
    Gu, Gv = G_expl(u, v)
    Gu_ab = (1.5 + ABEPS) * Gu - (0.5 + ABEPS) * gup
    Gv_ab = (1.5 + ABEPS) * Gv - (0.5 + ABEPS) * gvp
    gup, gvp = Gu, Gv
    us = (u + dt * Gu_ab) * mW; vs = (v + dt * Gv_ab) * mS
    divHu = H * ((rx(us, -1) - us) / dx + (ry(vs, -1) - vs) / dy) * m
    rhs = (eta - dt * divHu).ravel()
    eta = solve_fs(rhs).reshape(N, N) * m
    u = (us - g * dt * (eta - rx(eta, 1)) / dx) * mW
    v = (vs - g * dt * (eta - ry(eta, 1)) / dy) * mS
    if it % 2500 == 0 or it == 1:
        print(f"  step {it:6d} ({it*dt/86400/365:.2f}yr) |u|max={np.abs(u).max():.5f} "
              f"|v|max={np.abs(v).max():.5f} ({time.time()-t0:.0f}s)", flush=True)
print("MITgcm ref: laminar |u|max=0.031")
