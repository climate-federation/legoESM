"""Unit tests for the semi-implicit (Robin) surface boundary condition of
``solve_soil_thermal`` (the ``surface_conductance`` argument).

The multilayer land step folds a surface conductance ``lambda = -dG/dT_sfc``
into the implicit backward-Euler soil-thermal solve so a large dt with a thin
top layer under a stiff surface (sigma T^4 + bulk SH/LH) stays stable instead of
overshooting and diverging.  These tests pin:

  * ``surface_conductance=None`` is bit-identical to ``=0`` (the conductance
    term is purely additive; existing callers are unaffected),
  * the implementation matches an independent dense tridiagonal solve with the
    Robin modification applied exactly (``diag[0]+=lambda``, ``rhs[0]+=lambda*T_old0``),
  * a larger conductance damps the top-layer response monotonically.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import (
    SoilThermalConfig,
    compute_heat_capacity,
    compute_thermal_conductivity,
    solve_soil_thermal,
)


def _setup():
    grid = make_soil_grid(SoilGridConfig())
    hcfg = SoilHydraulicsConfig()
    tcfg = SoilThermalConfig()
    n = grid.n_layers
    T_old = jnp.linspace(292.0, 285.0, n)[None, :]      # (1, n)
    theta = jnp.full((1, n), 0.2)
    return grid, hcfg, tcfg, T_old, theta


def test_none_equals_zero_conductance() -> None:
    """surface_conductance=None reduces EXACTLY to the explicit Neumann BC."""
    grid, hcfg, tcfg, T_old, theta = _setup()
    G = jnp.array([180.0])
    dt = 1800.0
    t_none = solve_soil_thermal(T_old, theta, grid, hcfg, tcfg, G, dt)
    t_zero = solve_soil_thermal(
        T_old, theta, grid, hcfg, tcfg, G, dt,
        surface_conductance=jnp.zeros(1),
    )
    np.testing.assert_array_equal(np.asarray(t_none), np.asarray(t_zero))


def test_robin_bc_matches_dense_solve() -> None:
    """The semi-implicit solve equals an independent dense linear solve of the
    same tridiagonal system with diag[0]+=lambda and rhs[0]+=lambda*T_old0."""
    grid, hcfg, tcfg, T_old, theta = _setup()
    G = jnp.array([220.0])
    lam = jnp.array([150.0])
    dt = 1800.0
    T_new = solve_soil_thermal(
        T_old, theta, grid, hcfg, tcfg, G, dt, surface_conductance=lam,
    )

    n = grid.n_layers
    C = np.asarray(compute_heat_capacity(theta, hcfg, tcfg))[0]
    k = np.asarray(compute_thermal_conductivity(theta, hcfg, tcfg))[0]
    dz = np.asarray(grid.dz)
    dz_if = np.asarray(grid.dz_interface)
    kh = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:] + 1e-20)
    coeff = kh / dz_if

    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = C[i] * dz[i] / dt
    for i in range(n - 1):
        A[i, i] += coeff[i]
        A[i + 1, i + 1] += coeff[i]
        A[i, i + 1] -= coeff[i]
        A[i + 1, i] -= coeff[i]
    b = C * dz * np.asarray(T_old)[0] / dt
    b[0] += float(G[0])
    b[-1] += float(tcfg.Q_geothermal)
    # Robin modification on the top layer.
    A[0, 0] += float(lam[0])
    b[0] += float(lam[0]) * float(np.asarray(T_old)[0, 0])
    ref = np.linalg.solve(A, b)

    np.testing.assert_allclose(np.asarray(T_new)[0], ref, rtol=1e-9, atol=1e-9)


def test_larger_conductance_damps_top_response() -> None:
    """A bigger surface conductance pulls the top-layer step closer to zero
    (more damping); lambda=0 gives the largest |dT_top|."""
    grid, hcfg, tcfg, T_old, theta = _setup()
    G = jnp.array([5.0e3])           # large flux to provoke a big top-layer step
    dt = 1800.0
    t0 = float(np.asarray(T_old)[0, 0])
    jumps = []
    for lam in (0.0, 1.0e2, 1.0e3, 1.0e4):
        T_new = solve_soil_thermal(
            T_old, theta, grid, hcfg, tcfg, G, dt,
            surface_conductance=jnp.array([lam]),
        )
        jumps.append(abs(float(np.asarray(T_new)[0, 0]) - t0))
    # Strictly monotone-decreasing top-layer excursion with increasing lambda,
    # and every result finite (bounded).
    assert all(np.isfinite(j) for j in jumps)
    assert jumps[0] > jumps[1] > jumps[2] > jumps[3]
