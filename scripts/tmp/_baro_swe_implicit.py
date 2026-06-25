"""Standalone C-grid barotropic SWE with an IMPLICIT free surface (FFT Helmholtz,
doubly-periodic) at the MODEL's dt=300s — to reproduce internal_tide #576's blow-up
and validate the fix.

Reproduces: explicit/split-timescale Coriolis + implicit free surface at large dt
is UNSTABLE (the model's regime). Validates: forward-backward (FB) Coriolis coupled
consistently keeps it stable + tracks the analytic inertial oscillation.

IC: U=U0 uniform + tiny 2Δy seed, V=0, eta=0. Analytic: U=U0 cos(ft), V=-U0 sin(ft).
"""
from __future__ import annotations
import numpy as np

NY, NX = 16, 8
dx = 4.0e3
H = 2.0e3
g = 9.80616
f = -1.0312e-4
U0 = 0.2
dt = 300.0
theta = 1.0   # CN implicitness (1.0 = backward Euler free surface, matches model theta=1)
Tin = 2 * np.pi / abs(f)

# FFT wavenumbers for the periodic Helmholtz operator (I - theta^2 dt^2 g H lap)
kx = 2 * np.pi * np.fft.fftfreq(NX, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(NY, d=dx)
KX, KY = np.meshgrid(kx, ky)
# discrete C-grid Laplacian eigenvalue for the gradient/divergence pair:
# div(grad eta) with forward/backward differences -> -(|1-e^{-ikdx}|^2)/dx^2 etc.
lap_eig = -(np.abs(1 - np.exp(-1j * KX * dx)) ** 2 + np.abs(1 - np.exp(-1j * KY * dx)) ** 2) / dx ** 2


def helmholtz_solve(rhs):
    """(I - theta^2 dt^2 g H lap) eta = rhs, periodic, via FFT."""
    denom = 1.0 - theta ** 2 * dt ** 2 * g * H * lap_eig
    return np.real(np.fft.ifft2(np.fft.fft2(rhs) / denom))


def run(coriolis, nstep):
    U = np.full((NY, NX), U0); V = np.zeros((NY, NX)); eta = np.zeros((NY, NX))
    V += 1e-6 * ((-1.0) ** np.arange(NY)[:, None])
    hist = []
    for n in range(nstep):
        detadx = (eta - np.roll(eta, 1, axis=1)) / dx
        detady = (eta - np.roll(eta, 1, axis=0)) / dx
        # --- predictor velocities (old eta gradient) ---
        if coriolis == 'explicit':
            Vu = 0.25 * (V + np.roll(V, -1, axis=0) + np.roll(V, 1, axis=1) + np.roll(np.roll(V, -1, axis=0), 1, axis=1))
            Up = U + dt * (f * Vu - g * detadx)
            Uv = 0.25 * (U + np.roll(U, 1, axis=0) + np.roll(U, -1, axis=1) + np.roll(np.roll(U, 1, axis=0), -1, axis=1))
            Vp = V + dt * (-f * Uv - g * detady)
        elif coriolis == 'fb':
            # forward-backward: V uses the NEW U (Up)
            Vu = 0.25 * (V + np.roll(V, -1, axis=0) + np.roll(V, 1, axis=1) + np.roll(np.roll(V, -1, axis=0), 1, axis=1))
            Up = U + dt * (f * Vu - g * detadx)
            Uv = 0.25 * (Up + np.roll(Up, 1, axis=0) + np.roll(Up, -1, axis=1) + np.roll(np.roll(Up, 1, axis=0), -1, axis=1))
            Vp = V + dt * (-f * Uv - g * detady)
        # --- implicit free surface: solve for eta^{n+1} ---
        divUp = (np.roll(Up, -1, axis=1) - Up) / dx + (np.roll(Vp, -1, axis=0) - Vp) / dx
        rhs = eta - dt * H * divUp
        eta_new = helmholtz_solve(rhs)
        # --- corrector: add the implicit eta-gradient increment (CN) ---
        dgx = ((eta_new - eta) - np.roll(eta_new - eta, 1, axis=1)) / dx  # delta d/dx
        dgy = ((eta_new - eta) - np.roll(eta_new - eta, 1, axis=0)) / dx
        U = Up - theta * dt * g * dgx
        V = Vp - theta * dt * g * dgy
        eta = eta_new
        if n % max(1, nstep // 12) == 0:
            hist.append((n * dt / Tin, U.mean(), np.abs(U).max(), np.abs(V).max()))
    return hist


for cor in ['explicit', 'fb']:
    h = run(cor, int(round(2 * Tin / dt)))
    print(f'--- Coriolis={cor} (implicit FS, dt={dt:.0f}s) ---')
    for t, um, umx, vmx in h:
        flag = ' BLEW' if not np.isfinite(umx) or umx > 1.0 else ''
        print(f'  t/Tin={t:.2f}  <U>={um:+.3f}  max|U|={umx:.3f}  max|V|={vmx:.3f}  '
              f'analytic U={U0*np.cos(2*np.pi*t):+.3f}{flag}')
