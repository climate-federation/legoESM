"""MITgcm-faithful single-layer barotropic-gyre integrator (the oracle stepper).

A bit-exact numpy reproduction of MITgcm ``tutorial_barotropic_gyre``'s unsplit
single-layer dynamics: explicit momentum tendency ``Gu`` (flux-form advection +
face-f Coriolis + component-Laplacian lateral viscosity + no-slip side-drag +
wind, NO surface pressure gradient) → Adams-Bashforth-2 (``abEps``) → predictor
``u* = u + Δt·Gu`` → implicit free-surface elliptic solve → velocity correction
``u = u* − gΔt·∇η``.

WHY a dedicated stepper (not ``LatLonCGridOceanModel``): the canonical model
operator-splits barotropic/baroclinic and applies Coriolis via a forward-backward
predictor inside ``barotropic_implicit_latlon_cgrid``.  For the marginally-resolved
(Munk δ≈1.7-cell) wind-driven gyre that split machinery sits on the *unstable* side
of the western-boundary-current barotropic instability and runs the gyre turbulent
(``|u|max`` 0.15–0.37), whereas MITgcm's unsplit explicit-Coriolis → ``cg2d``
sequencing is laminar (0.031).  Every operator here is calibrated to ``corr≈1.0``
against MITgcm's ``momU``/``momV`` diagnostics; the stepper reproduces MITgcm's
laminar equilibrium ``|u|max≈0.031, |v|max≈0.084`` at 1× and 2× resolution.  This is
mimicry glue for the gyre oracle (per ``docs/ocean_fidelity/oracle_recipe_strategy.md``),
kept in the fidelity harness rather than the shippable model.

Index convention: MITgcm ``(ny, nx)`` C-grid — ``u`` at western cell faces, ``v`` at
southern cell faces, ``eta`` at cell centres.  Flat bottom, single level.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

# --- MITgcm tutorial_barotropic_gyre fixed parameters (input/data) -----------
_F0 = 1.0e-4            # [s^-1]  Coriolis at y-origin (f0)
_BETA = 1.0e-11        # [m^-1 s^-1]  beta
_VISC_AH = 400.0       # [m^2/s]  viscAh
_RHO = 1000.0          # rhoConst
_H = 5000.0            # flat depth Ho [m]
_G_BARO = 9.81         # const-ok: MITgcm deck gBaro (the oracle's barotropic-g config
#                        value, deliberately NOT the physical legoesm.constants.g);
#                        must equal the oracle's gBaro exactly for a bit-faithful match.
_AB_EPS = 0.01          # MITgcm abEps (AB2 stabilisation)
_SIDE_DRAG_FACTOR = 2.0  # MITgcm sideDragFactor (full no-slip)
_TAU_MAX = 0.1          # [N/m^2]  tauMax


class GyreFaithfulState(NamedTuple):
    """Carry for the MITgcm-faithful gyre stepper."""

    u: np.ndarray        # (ny, nx) western-face zonal velocity
    v: np.ndarray        # (ny, nx) southern-face meridional velocity
    eta: np.ndarray      # (ny, nx) cell-centre free surface
    gu_prev: np.ndarray  # previous explicit u-tendency (AB2 history)
    gv_prev: np.ndarray


class GyreFaithfulModel:
    """Bit-exact MITgcm single-layer barotropic-gyre integrator.

    Parameters
    ----------
    n : int
        Grid size (n x n cells, incl. the one-cell wall ring).  62 → 20 km
        (the tutorial deck), 122 → 10 km, etc.
    dx_m : float
        Cell size [m].
    dt_s : float
        Time step [s].
    """

    def __init__(self, n: int = 62, dx_m: float = 20.0e3, dt_s: float = 1200.0):
        self.n = n
        self.dx = self.dy = float(dx_m)
        self.dt = float(dt_s)
        self.rAw = self.dx * self.dy
        self.g = _G_BARO
        N = n
        m = np.ones((N, N)); m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
        self.m = m
        self.mW = m * np.roll(m, 1, axis=1)
        self.mS = m * np.roll(m, 1, axis=0)
        self.hFacZ = (m * np.roll(m, 1, axis=1) * np.roll(m, 1, axis=0)
                      * np.roll(np.roll(m, 1, axis=1), 1, axis=0))
        yc = (np.arange(N) + 0.5) * self.dy - self.dx
        self.fC = (_F0 + _BETA * yc)[:, None] * np.ones((N, N))
        Y = (np.arange(N) - 0.5) / (N - 2)
        tau = -_TAU_MAX * np.cos(np.pi * Y)
        self.wind = (tau / (_RHO * _H))[:, None] * np.ones((N, N)) * self.mW
        self._build_helmholtz()

    @staticmethod
    def _rx(a, s):
        return np.roll(a, s, axis=1)

    @staticmethod
    def _ry(a, s):
        return np.roll(a, s, axis=0)

    def _G(self, u, v):
        rx, ry = self._rx, self._ry
        dx, dy, rAw, A = self.dx, self.dy, self.rAw, _VISC_AH
        mW, mS, m, hFacZ = self.mW, self.mS, self.m, self.hFacZ
        uT = u * dy; vT = v * dx
        fZon = 0.25 * (uT + rx(uT, -1)) * (u + rx(u, -1))
        fMer = 0.25 * (vT + rx(vT, 1)) * (u + ry(u, 1))
        aU = -(1 / rAw) * ((fZon - rx(fZon, 1)) + (ry(fMer, -1) - fMer)) * mW
        gZon = 0.25 * (vT + ry(vT, -1)) * (v + ry(v, -1))
        gMer = 0.25 * (uT + ry(uT, 1)) * (v + rx(v, 1))
        aV = -(1 / rAw) * ((gZon - ry(gZon, 1)) + (rx(gMer, -1) - gMer)) * mS
        V_at_u = 0.25 * (v + ry(v, -1) + rx(v, 1) + rx(ry(v, -1), 1))
        f_u = 0.5 * (rx(self.fC, 1) + self.fC)
        cU = f_u * V_at_u * mW
        U_at_v = 0.25 * (u + rx(u, -1) + ry(u, 1) + ry(rx(u, -1), 1))
        f_v = 0.5 * (ry(self.fC, 1) + self.fC)
        cV = -f_v * U_at_v * mS
        fZx = A * (dy / dx) * (rx(u, -1) - u) * m
        fMy = A * (dx / dy) * (u - ry(u, 1)) * hFacZ
        vU = (1 / rAw) * ((fZx - rx(fZx, 1)) + (ry(fMy, -1) - fMy)) * mW
        gZx = A * (dy / dx) * (rx(v, -1) - v) * m
        gMy = A * (dx / dy) * (v - ry(v, 1)) * hFacZ
        vV = (1 / rAw) * ((gZx - rx(gZx, 1)) + (ry(gMy, -1) - gMy)) * mS
        hzcS = (mW - hFacZ); hzcN = (mW - ry(hFacZ, -1))
        sU = -(1 / rAw) * (hzcS * (dx / dy) + hzcN * (dx / dy)) * u * _SIDE_DRAG_FACTOR * A * mW
        hzcW = (mS - hFacZ); hzcE = (mS - rx(hFacZ, -1))
        sV = -(1 / rAw) * (hzcW * (dy / dx) + hzcE * (dy / dx)) * v * _SIDE_DRAG_FACTOR * A * mS
        Gu = (aU + cU + vU + sU + self.wind) * mW
        Gv = (aV + cV + vV + sV) * mS
        return Gu, Gv

    def _build_helmholtz(self):
        N = self.n
        coef = self.g * self.dt * self.dt * _H
        mW, mS, m = self.mW, self.mS, self.m
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
                        coup = coef / (self.dx * self.dx)
                        diag += coup
                        rows.append(c); cols.append(idx(i + di, j + dj)); vals.append(-coup)
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csc_matrix((vals, (rows, cols)), shape=(N * N, N * N))
        self._solve_fs = factorized(A)

    def rest_state(self) -> GyreFaithfulState:
        z = np.zeros((self.n, self.n))
        return GyreFaithfulState(z.copy(), z.copy(), z.copy(), z.copy(), z.copy())

    def step(self, s: GyreFaithfulState) -> GyreFaithfulState:
        rx, ry = self._rx, self._ry
        Gu, Gv = self._G(s.u, s.v)
        Gu_ab = (1.5 + _AB_EPS) * Gu - (0.5 + _AB_EPS) * s.gu_prev
        Gv_ab = (1.5 + _AB_EPS) * Gv - (0.5 + _AB_EPS) * s.gv_prev
        us = (s.u + self.dt * Gu_ab) * self.mW
        vs = (s.v + self.dt * Gv_ab) * self.mS
        divHu = _H * ((rx(us, -1) - us) / self.dx + (ry(vs, -1) - vs) / self.dy) * self.m
        rhs = (s.eta - self.dt * divHu).ravel()
        eta = self._solve_fs(rhs).reshape(self.n, self.n) * self.m
        u = (us - self.g * self.dt * (eta - rx(eta, 1)) / self.dx) * self.mW
        v = (vs - self.g * self.dt * (eta - ry(eta, 1)) / self.dy) * self.mS
        return GyreFaithfulState(u, v, eta, Gu, Gv)
