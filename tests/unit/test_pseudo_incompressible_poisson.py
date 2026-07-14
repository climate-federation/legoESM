"""Unit tests for the pseudo-incompressible variable-coefficient pressure Poisson.

Covers: (1) the matrix-free operator reproduces the analytic constant-coefficient
Laplacian on a manufactured periodic-x/y, Neumann-z field (MMS); (2) the operator is
symmetric (the Krylov method needs it); (3) the Jacobi-BiCGSTAB solve recovers a
manufactured variable-coefficient π' and the residual meets tolerance; (4) Jacobi
preconditioning cuts the residual vs unconditioned at fixed iterations; (5) the solve is
JIT- and ``jax.grad``-safe.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.dynamics.les import pseudo_incompressible_poisson as pip


def _grid(ny=16, nx=16, nz=12, Lx=1600.0, Ly=1600.0, Lz=1200.0):
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    x = (jnp.arange(nx) + 0.5) * dx
    y = (jnp.arange(ny) + 0.5) * dy
    zc = (jnp.arange(nz) + 0.5) * dz
    Y, X, Z = jnp.meshgrid(y, x, zc, indexing="ij")
    return dict(ny=ny, nx=nx, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, dx=dx, dy=dy, dz=dz,
                X=X, Y=Y, Z=Z)


def test_operator_matches_analytic_constant_coeff():
    """Constant C: Cp·C·∇²π' for a field periodic in x,y and Neumann (cos) in z.

    π' = cos(kx x)·cos(ky y)·cos(m z),  ∂π'/∂z = 0 at z=0,Lz (Neumann walls).
    ∇²π' = −(kx²+ky²+m²) π'.  Centred 2nd-order differences ⇒ match the modified
    wavenumbers, so compare against the DISCRETE Laplacian eigenvalue (exact to round-off).
    """
    g = _grid()
    kx = 2 * np.pi / g["Lx"]          # one period in x
    ky = 2 * np.pi / g["Ly"]
    m = np.pi / g["Lz"]               # half period in z ⇒ cos has zero slope at both walls
    C0 = 1.7
    pi_f = jnp.cos(kx * g["X"]) * jnp.cos(ky * g["Y"]) * jnp.cos(m * g["Z"])
    c = jnp.full_like(pi_f, C0)
    out = pip.laplace_pi(pi_f, c, g["dx"], g["dy"], g["dz"])
    # Discrete (modified-wavenumber) eigenvalue of the 2nd-order centred Laplacian.
    lam = (2 * (np.cos(kx * g["dx"]) - 1) / g["dx"] ** 2
           + 2 * (np.cos(ky * g["dy"]) - 1) / g["dy"] ** 2
           + 2 * (np.cos(m * g["dz"]) - 1) / g["dz"] ** 2)
    expected = constants.c_pd * C0 * lam * pi_f
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-9, atol=1e-9)


def test_operator_is_symmetric():
    """⟨a, L b⟩ == ⟨L a, b⟩ for the variable-coefficient operator (self-adjoint)."""
    g = _grid(ny=8, nx=8, nz=6)
    key = jax.random.PRNGKey(0)
    ka, kb, kc = jax.random.split(key, 3)
    shape = (g["ny"], g["nx"], g["nz"])
    a = jax.random.normal(ka, shape)
    b = jax.random.normal(kb, shape)
    c = 1.0 + 0.3 * jax.random.uniform(kc, shape)     # strictly positive variable coeff
    La = pip.laplace_pi(a, c, g["dx"], g["dy"], g["dz"])
    Lb = pip.laplace_pi(b, c, g["dx"], g["dy"], g["dz"])
    lhs = float(jnp.sum(a * Lb))
    rhs = float(jnp.sum(La * b))
    assert abs(lhs - rhs) <= 1e-8 * (abs(lhs) + abs(rhs) + 1.0)


def test_diag_matches_operator_diagonal():
    """poisson_diag == L applied to unit impulses (probe a few cells)."""
    g = _grid(ny=8, nx=8, nz=6)
    key = jax.random.PRNGKey(2)
    c = 1.0 + 0.5 * jax.random.uniform(key, (g["ny"], g["nx"], g["nz"]))
    diag = pip.poisson_diag(c, g["dx"], g["dy"], g["dz"])
    for (j, i, k) in [(0, 0, 0), (3, 2, 5), (7, 7, 3), (4, 1, 0)]:
        e = jnp.zeros((g["ny"], g["nx"], g["nz"])).at[j, i, k].set(1.0)
        Le = pip.laplace_pi(e, c, g["dx"], g["dy"], g["dz"])
        np.testing.assert_allclose(float(Le[j, i, k]), float(diag[j, i, k]), rtol=1e-10)


def test_solve_recovers_manufactured_variable_coeff():
    """Manufacture π'_true and C>0, form rhs = L(π'_true), solve, compare (zero-mean)."""
    g = _grid()
    kx, ky = 2 * np.pi / g["Lx"], 2 * np.pi / g["Ly"]
    m = np.pi / g["Lz"]
    pi_true = (jnp.cos(kx * g["X"]) * jnp.cos(ky * g["Y"]) * jnp.cos(m * g["Z"])
               + 0.5 * jnp.cos(2 * kx * g["X"]) * jnp.cos(m * g["Z"]))
    pi_true = pi_true - jnp.mean(pi_true)
    c = 1.0 + 0.4 * jnp.cos(kx * g["X"]) * jnp.cos(m * g["Z"])     # variable, >0
    rhs = pip.laplace_pi(pi_true, c, g["dx"], g["dy"], g["dz"])
    sol, _info = pip.solve_pressure(rhs, c, g["dx"], g["dy"], g["dz"],
                                    tol=1e-10, atol=1e-12, maxiter=500)
    # jax.scipy bicgstab returns info=None; verify convergence via the residual.
    # Residual of the linear system meets tolerance.
    res = pip.laplace_pi(sol, c, g["dx"], g["dy"], g["dz"]) - (rhs - jnp.mean(rhs))
    assert float(jnp.linalg.norm(res)) / float(jnp.linalg.norm(rhs)) < 1e-6
    # And the recovered field matches the manufactured one (both zero-mean).
    np.testing.assert_allclose(np.asarray(sol), np.asarray(pi_true), atol=1e-6)


def test_jacobi_preconditioner_helps():
    """Jacobi BiCGSTAB reaches a smaller residual than unconditioned at fixed iters."""
    g = _grid(ny=24, nx=24, nz=20)
    key = jax.random.PRNGKey(7)
    rhs = jax.random.normal(key, (g["ny"], g["nx"], g["nz"]))
    # Strong coefficient contrast ⇒ poor conditioning that Jacobi scaling improves.
    c = 1.0 + 0.9 * jnp.cos(2 * np.pi * g["X"] / g["Lx"]) * jnp.cos(np.pi * g["Z"] / g["Lz"])

    def resid(precondition):
        sol, _ = pip.solve_pressure(rhs, c, g["dx"], g["dy"], g["dz"],
                                    tol=0.0, atol=0.0, maxiter=30,
                                    precondition=precondition)
        r = pip.laplace_pi(sol, c, g["dx"], g["dy"], g["dz"]) - (rhs - jnp.mean(rhs))
        return float(jnp.linalg.norm(r))

    # Non-vacuous factor bound (measured ratio ≈ 0.63): a no-op preconditioner would
    # give ratio 1.0, so require a clear gap rather than mere strict inequality.
    assert resid(True) < 0.8 * resid(False)


def test_solve_jit_and_grad_safe():
    """The solve compiles under jit and admits a finite gradient through it."""
    g = _grid(ny=8, nx=8, nz=6)
    key = jax.random.PRNGKey(11)
    rhs = jax.random.normal(key, (g["ny"], g["nx"], g["nz"]))
    c0 = 1.0 + 0.2 * jax.random.uniform(jax.random.PRNGKey(12), (g["ny"], g["nx"], g["nz"]))

    @jax.jit
    def loss(scale):
        sol, _ = pip.solve_pressure(rhs, scale * c0, g["dx"], g["dy"], g["dz"],
                                    tol=1e-10, atol=1e-12, maxiter=300)
        return jnp.sum(sol ** 2)

    val = loss(1.0)
    grad = jax.grad(loss)(1.0)
    assert np.isfinite(float(val)) and np.isfinite(float(grad))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
