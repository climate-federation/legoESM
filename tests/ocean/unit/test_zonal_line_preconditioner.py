"""Zonal-line (periodic-tridiagonal) preconditioner for the implicit-CN
Helmholtz PCG (task: ocean barotropic reduction-latency wall; POP
EVP-class lever — comm-free local solves that cut latency-bound
iteration counts).

Pins:
1. cyclic_thomas_batched == dense periodic-tridiagonal solve.
2. Coupling-pieces consistency: ``1 + coeff·(E+W+merid)`` reconstructs
   ``1/_helmholtz_inv_diag`` (the verbatim Jacobi expression) to fp
   tolerance — the mechanical tie that lets the two coexist without
   sharing (sharing would fp-regroup the baked-in Jacobi trajectory).
3. M⁻¹ is W-self-adjoint (single_reduce/Chronopoulos–Gear requirement).
4. Equal-M PCG residual: zonal_line beats jacobi on a pole-stiffened
   coastal problem (the point of the lever).
5. Unknown preconditioner name refuses loudly.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.timestepping.tridiagonal import cyclic_thomas_batched
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    _helmholtz_coupling_pieces,
    _helmholtz_inv_diag,
    _make_helmholtz,
    _make_diag_preconditioner,
    _make_zonal_line_preconditioner,
)
from legoesm.ocean.dynamics.barotropic_common import (
    solve_helmholtz_implicit,
)


def _dense_cyclic(a, b, c):
    n = b.shape[-1]
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = b[i]
        M[i, (i - 1) % n] = a[i]
        M[i, (i + 1) % n] = c[i]
    return M


def test_cyclic_thomas_matches_dense():
    rng = np.random.default_rng(3)
    n, batch = 17, 5
    a = rng.uniform(-0.4, -0.1, (batch, n))
    c = rng.uniform(-0.4, -0.1, (batch, n))
    b = 1.0 + np.abs(a) + np.abs(c) + rng.uniform(0.1, 0.5, (batch, n))
    d = rng.standard_normal((batch, n))
    x = np.asarray(cyclic_thomas_batched(
        jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)))
    for k in range(batch):
        x_ref = np.linalg.solve(_dense_cyclic(a[k], b[k], c[k]), d[k])
        np.testing.assert_allclose(x[k], x_ref, rtol=0, atol=1e-12)


def test_cyclic_thomas_grad_finite():
    n = 12
    a = jnp.full((n,), -0.2)
    c = jnp.full((n,), -0.3)
    b = jnp.full((n,), 1.8)
    d = jnp.linspace(0.0, 1.0, n)

    g = jax.grad(lambda dd: jnp.sum(
        cyclic_thomas_batched(a, b, c, dd) ** 2))(d)
    assert np.all(np.isfinite(np.asarray(g)))


def _setup(n_lat=24, n_lon=48, coastal=True):
    grid = create_latlon_grid(n_lat, n_lon)
    rng = np.random.default_rng(11)
    H = 1000.0 + 500.0 * rng.random((n_lat, n_lon))
    mask = np.ones((n_lat, n_lon))
    if coastal:
        mask[:, 10:14] = 0.0          # meridional land strip
        mask[5:8, 30:40] = 0.0        # island block
    H_u = np.zeros((n_lat, n_lon + 1))
    H_u[:, 1:-1] = 0.5 * (H[:, 1:] + H[:, :-1])
    # periodic u-face wrap + wall masking through land
    H_u[:, 0] = H_u[:, -1] = 0.5 * (H[:, 0] + H[:, -1])
    u_wet = np.zeros_like(H_u)
    u_wet[:, 1:-1] = mask[:, 1:] * mask[:, :-1]
    u_wet[:, 0] = u_wet[:, -1] = mask[:, 0] * mask[:, -1]
    H_u = H_u * u_wet
    H_v = np.zeros((n_lat + 1, n_lon))
    H_v[1:-1, :] = 0.5 * (H[1:, :] + H[:-1, :])
    v_wet = np.zeros_like(H_v)
    v_wet[1:-1, :] = mask[1:, :] * mask[:-1, :]
    H_v = H_v * v_wet
    coeff = jnp.asarray(5.0e7)        # stiff (large dt·g·θ² scale)
    return (grid, jnp.asarray(H_u), jnp.asarray(H_v), coeff,
            jnp.asarray(mask), jnp.asarray(u_wet), jnp.asarray(v_wet))


def test_coupling_pieces_reconstruct_jacobi_diag():
    grid, H_u, H_v, coeff, mask, _, _ = _setup()
    zE, zW, mer = _helmholtz_coupling_pieces(H_u, H_v, grid)
    diag = 1.0 + coeff * (zE + zW + mer)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    wet = np.asarray(mask) > 0.5
    np.testing.assert_allclose(
        np.asarray(diag)[wet], 1.0 / np.asarray(inv_diag)[wet],
        rtol=1e-13,
        err_msg="coupling pieces drifted from the verbatim Jacobi diag",
    )


def test_zonal_line_W_self_adjoint():
    grid, H_u, H_v, coeff, mask, _, _ = _setup()
    M_inv = _make_zonal_line_preconditioner(H_u, H_v, coeff, grid, mask)
    rng = np.random.default_rng(5)
    w = grid.area * mask
    x = jnp.asarray(rng.standard_normal(mask.shape)) * mask
    y = jnp.asarray(rng.standard_normal(mask.shape)) * mask
    lhs = float(jnp.sum(w * M_inv(x) * y))
    rhs = float(jnp.sum(w * x * M_inv(y)))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs)), (
        f"M_inv not W-self-adjoint: <Mx,y>_W={lhs!r} vs <x,My>_W={rhs!r}"
    )


@pytest.mark.parametrize("variant", ["standard", "single_reduce"])
def test_equal_M_residual_zonal_line_beats_jacobi(variant):
    grid, H_u, H_v, coeff, mask, u_wet, v_wet = _setup()
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_wet, v_wet)
    rng = np.random.default_rng(7)
    rhs = jnp.asarray(rng.standard_normal(mask.shape)) * mask
    x0 = jnp.zeros_like(rhs)
    w = grid.area * mask

    def resid(M_inv):
        x, diag = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0, distributed=True, fixed_iters=12,
            residual_tol=1.0e-30, stock_cg_tol=1.0e-12,
            stock_cg_maxiter=200, pcg_variant=variant, dot_weight=w,
        )
        return float(diag.rel_residual)

    r_jac = resid(_make_diag_preconditioner(H_u, H_v, coeff, grid, mask))
    r_line = resid(
        _make_zonal_line_preconditioner(H_u, H_v, coeff, grid, mask))
    assert np.isfinite(r_line) and r_line > 0
    assert r_line < 0.5 * r_jac, (
        f"zonal_line ({r_line:.3e}) should beat jacobi ({r_jac:.3e}) "
        f"by >=2x at equal M=12 ({variant})"
    )


def test_unknown_preconditioner_refuses():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
        _select_preconditioner,
    )
    assert (LatLonCGridOceanConfig.from_flat().barotropic.barotropic_implicit_preconditioner
            == "jacobi")
    grid, H_u, H_v, coeff, mask, _, _ = _setup(coastal=False)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    with pytest.raises(ValueError, match="preconditioner"):
        _select_preconditioner(
            "bogus", inv_diag, H_u, H_v, coeff, grid, mask)
    # And the two valid names dispatch.
    for name in ("jacobi", "zonal_line"):
        M = _select_preconditioner(
            name, inv_diag, H_u, H_v, coeff, grid, mask)
        out = M(jnp.ones_like(mask))
        assert np.all(np.isfinite(np.asarray(out)))


def test_zonal_line_rejects_tiny_n_lon():
    # codex review MINOR: fail at preconditioner construction with
    # ocean-context text, not deep inside cyclic_thomas at apply time.
    grid = create_latlon_grid(8, 2)
    H_u = jnp.ones((8, 3)) * 100.0
    H_v = jnp.ones((9, 2)) * 100.0
    mask = jnp.ones((8, 2))
    with pytest.raises(ValueError, match="n_lon"):
        _make_zonal_line_preconditioner(
            H_u, H_v, jnp.asarray(1.0e6), grid, mask)
