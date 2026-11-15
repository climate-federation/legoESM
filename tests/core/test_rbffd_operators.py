"""RBF-FD collocated operators on the sphere (tangent-plane PHS + polynomials).

Accuracy against analytic spherical-harmonic fields with observed convergence
order over Voronoi levels 3/4/5; check_grads (order 2) of the operators in the
fields and of the weight builder in the node coordinates; jit parity; the
collocated B2 contract.  fp64 throughout.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm import constants
from legoesm.core import neighbour_ops
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.neighbour_connectivity import (
    conservative_column_correction,
    phs_planar_weights,
    polynomial_matrix,
    polynomial_preserving_correction,
    rbffd_tangent_coordinates,
    rbffd_sphere_mesh,
    rbffd_sphere_weights,
    rbffd_stencils,
)
from legoesm.grids.operator_adapters import rbffd_operators
from legoesm.grids.operator_protocol import (
    CollocatedOperators,
    validate_collocated_operators,
)
from legoesm.grids.voronoi import create_voronoi_mesh

# Voted 2026-11-09 (author A, GLM A, codex A): n=31, r^7, degree 4.
N_STENCIL, PHS, POLY = 31, 7, 4
LEVELS = (3, 4, 5)
# Divergence = conservatively corrected operator; KNOWN LIMIT (measured
# max-norm orders 4.4 then 1.5), threshold voted 2026-11-09 (T1 3/3).
MIN_ORDER = {"gradient": 3.0, "divergence": 1.4, "vorticity": 3.0, "laplacian": 2.0}
JIT_RTOL = 1e-13
# Voted 2026-11-09: D25 3/3; W1 2-1 (sparse LU + refinement); X1, X2 2-1
# (measured worst at 642-10242 nodes: residual 4.6e-9, moments 2.5e-8).
MASS_RTOL = 1e-13
REFINE = 3
SOLVE_RTOL = 1e-8
POLY_EXACT_RTOL = 1e-7
# Voted 2026-11-09 (3/3): weight solve cond <= 3.6e5 (measured) -> 1e-10.
WEIGHT_JIT_RTOL = 1e-10
# Voted 2026-11-09 (3/3): jax check_grads compares a scalar inner product whose
# FD truncation at eps 1e-4 is ~1e-4 relative (measured), so 1e-3 there; the
# normwise eps^2 test below is the decisive derivative check.
COORD_GRAD_TOL = 1e-3
FD_EPS_PAIR = (1e-2, 1e-3)
MIN_FD_RATIO = 50.0
# Voted 2026-11-09 (author+codex vs GLM): 1e-8 relative to the summation scale.
REPRO_RTOL = 1e-8
R = constants.R_earth


@pytest.fixture(scope="module", autouse=True)
def _x64():
    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)
    jax.config.update("jax_enable_x64", prev_x64)


def _nodes(level):
    m = create_voronoi_mesh(level, lloyd_iterations=50)
    xyz = jnp.stack([jnp.asarray(m.xCell, jnp.float64), jnp.asarray(m.yCell, jnp.float64),
                     jnp.asarray(m.zCell, jnp.float64)], axis=-1)
    return xyz, jnp.asarray(m.areaCell, jnp.float64)


@pytest.fixture(scope="module")
def meshes(_x64):
    out = {}
    for lev in LEVELS:
        xyz, area = _nodes(lev)
        out[lev] = (xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True),
                    rbffd_sphere_mesh(xyz, R, area, n=N_STENCIL, phs_degree=PHS,
                                      poly_degree=POLY, refine_steps=REFINE), area)
    return out


# Homogeneous harmonic polynomials restricted to the unit sphere are spherical
# harmonics: Lap_s Y_l = -l(l+1) Y_l.  psi ~ Y_4^3, chi ~ Y_2^1 (+ Y_3^2 for f).
def _psi(x):
    return (x[0] ** 3 - 3 * x[0] * x[1] ** 2) * x[2]


def _chi(x):
    return x[0] * x[2]


def _f(x):
    return _psi(x) + _chi(x) + x[0] * x[1] * x[2]


def _lap_f(x):
    return (-20 * _psi(x) - 6 * _chi(x) - 12 * x[0] * x[1] * x[2]) / R**2


def _sgrad(fn, xu):
    """Surface gradient on radius R (projected 3-D gradient), (N, 3)."""
    g = jax.vmap(jax.grad(fn))(xu)
    return (g - jnp.sum(g * xu, -1, keepdims=True) * xu) / R


def _velocity(xu):
    """u = n x grad psi + grad chi: div = Lap chi, radial curl = Lap psi."""
    return jnp.cross(xu, _sgrad(_psi, xu)) + _sgrad(_chi, xu)


def _errors(xu, mesh):
    ops = rbffd_operators(mesh)
    fx = jax.vmap(_f)(xu)
    u = _velocity(xu)
    lap_psi = -20 * jax.vmap(_psi)(xu) / R**2
    lap_chi = -6 * jax.vmap(_chi)(xu) / R**2

    def rel(num, ref):
        return float(jnp.max(jnp.abs(num - ref)) / jnp.max(jnp.abs(ref)))

    return {
        "gradient": rel(jnp.stack(ops.gradient(fx), -1), _sgrad(_f, xu)),
        "divergence": rel(ops.divergence(u[:, 0], u[:, 1], u[:, 2]), lap_chi),
        "vorticity": rel(ops.vorticity(u[:, 0], u[:, 1], u[:, 2]), lap_psi),
        "laplacian": rel(ops.laplacian(fx), jax.vmap(_lap_f)(xu)),
    }


def test_operator_convergence(meshes):
    errs = {lev: _errors(*meshes[lev][:2]) for lev in LEVELS}
    n = {lev: meshes[lev][0].shape[0] for lev in LEVELS}
    for op in MIN_ORDER:
        e = [errs[lev][op] for lev in LEVELS]
        orders = [math.log(e[i] / e[i + 1]) / (0.5 * math.log(n[LEVELS[i + 1]] / n[LEVELS[i]]))
                  for i in range(len(LEVELS) - 1)]
        print(f"RBFFD {op}: rel max err {['%.3e' % v for v in e]} orders {['%.2f' % o for o in orders]}")
        assert e[0] > e[1] > e[2], (op, e)
        assert orders[-1] >= MIN_ORDER[op], (op, orders)


def test_gradient_is_tangent(meshes):
    xu, mesh, _ = meshes[3]
    g = jnp.stack(rbffd_operators(mesh).gradient(jax.vmap(_f)(xu)), -1)
    assert float(jnp.max(jnp.abs(jnp.sum(g * xu, -1)))) <= 1e-12 * float(jnp.max(jnp.abs(g)))


def test_operators_check_grads_in_fields(meshes):
    xu, mesh, _ = meshes[3]
    ops = rbffd_operators(mesh)
    u = _velocity(xu)
    fx = jax.vmap(_f)(xu)
    # Nondimensional (x R, x R^2) so default tolerances cannot pass a zero.
    check_grads(lambda a: R * jnp.stack(ops.gradient(a)), (fx,), order=2)
    check_grads(lambda a: R**2 * ops.laplacian(a), (fx,), order=2)
    check_grads(lambda a, b, c: R * ops.divergence(a, b, c), tuple(u.T), order=2)
    check_grads(lambda a, b, c: R * ops.vorticity(a, b, c), tuple(u.T), order=2)


def test_weights_check_grads_in_coordinates(meshes):
    xu, _, _ = meshes[3]
    stencil = rbffd_stencils(xu, N_STENCIL)   # frozen indices
    fn = functools.partial(rbffd_sphere_weights, stencil=stencil, phs_degree=PHS,
                           poly_degree=POLY)
    for k in (0, 1):
        check_grads(lambda x: fn(x)[k], (xu,), order=2, modes=("fwd", "rev"),
                    atol=COORD_GRAD_TOL, rtol=COORD_GRAD_TOL)


def test_weight_derivatives_converge_like_eps_squared(meshes):
    """Central FD of the first and second directional derivatives converges
    to the AD values as eps^2 (AD correct, not just within tolerance)."""
    xu, _, _ = meshes[3]
    stencil = rbffd_stencils(xu, N_STENCIL)
    v = jax.random.normal(jax.random.PRNGKey(0), xu.shape)
    w = jax.random.normal(jax.random.PRNGKey(1), xu.shape)
    v, w = v / jnp.linalg.norm(v), w / jnp.linalg.norm(w)
    for k in (0, 1):
        def f(x, k=k):
            return rbffd_sphere_weights(x, stencil, phs_degree=PHS, poly_degree=POLY)[k]

        def g(x, f=f):
            return jax.jvp(f, (x,), (v,))[1]

        for fn, d in ((f, v), (g, w)):
            exact = jax.jvp(fn, (xu,), (d,))[1]
            errs = [float(jnp.linalg.norm((fn(xu + e * d) - fn(xu - e * d)) / (2 * e) - exact)
                          / jnp.linalg.norm(exact)) for e in FD_EPS_PAIR]
            assert errs[0] / errs[1] >= MIN_FD_RATIO, (k, errs)


def test_jit_matches_eager(meshes):
    xu, mesh, _ = meshes[3]
    ops = rbffd_operators(mesh)
    u = _velocity(xu)
    fx = jax.vmap(_f)(xu)
    stencil = rbffd_stencils(xu, N_STENCIL)
    cases = [
        (lambda a: jnp.stack(ops.gradient(a)), (fx,), JIT_RTOL),
        (ops.laplacian, (fx,), JIT_RTOL),
        (ops.divergence, tuple(u.T), JIT_RTOL),
        (ops.vorticity, tuple(u.T), JIT_RTOL),
        (lambda x: jnp.concatenate([
            w.reshape(-1) for w in rbffd_sphere_weights(
                x, stencil, phs_degree=PHS, poly_degree=POLY)]), (xu,), WEIGHT_JIT_RTOL),
    ]
    for fn, args, rtol in cases:
        eager, jitted = fn(*args), jax.jit(fn)(*args)
        assert float(jnp.max(jnp.abs(jitted - eager))) <= rtol * float(jnp.max(jnp.abs(eager)))


def test_stencils_nearest_self_first(meshes):
    xu, _, _ = meshes[3]
    idx = rbffd_stencils(xu, N_STENCIL)
    assert idx.shape == (xu.shape[0], N_STENCIL)
    np.testing.assert_array_equal(idx[:, 0], np.arange(xu.shape[0]))


def test_collocated_contract(meshes):
    ops = rbffd_operators(meshes[3][1])
    assert isinstance(ops, CollocatedOperators)
    validate_collocated_operators(ops)

    class TwoArgDivergence:
        def divergence(self, u, v):
            return u

        def gradient(self, phi):
            return phi

        def vorticity(self, ux, uy, uz):
            return ux

    with pytest.raises(TypeError, match="divergence"):
        validate_collocated_operators(TwoArgDivergence())


def test_planar_weights_reproduce_augmented_space(_x64):
    """Weights are exact on the approximation space: every monomial of degree
    <= POLY, and PHS combinations sum_j c_j |xi - xi_j|^m with P^T c = 0 (single
    kernels are not reproduced under polynomial augmentation)."""
    m, b, n = PHS, 8, N_STENCIL
    key = jax.random.PRNGKey(3)
    rad = jnp.sqrt(jax.random.uniform(key, (b, n - 1)))
    ang = 2 * jnp.pi * jax.random.uniform(jax.random.PRNGKey(4), (b, n - 1))
    xs = jnp.concatenate([jnp.zeros((b, 1, 2)),
                          jnp.stack([rad * jnp.cos(ang), rad * jnp.sin(ang)], -1)], 1)
    w = phs_planar_weights(xs, phs_degree=m, poly_degree=POLY)            # (b, n, 3)
    exps = [(i - j, j) for i in range(POLY + 1) for j in range(i + 1)]
    p = jnp.stack([xs[..., 0] ** i * xs[..., 1] ** j for i, j in exps], -1)

    def check(values, target):
        terms = w * values[..., None]                                    # (b, n, 3)
        scale = jnp.maximum(jnp.sum(jnp.abs(terms), 1), jnp.abs(target))
        assert float(jnp.max(jnp.abs(jnp.sum(terms, 1) - target) / scale)) <= REPRO_RTOL

    for k, (i, j) in enumerate(exps):
        target = jnp.asarray([float((i, j) == (1, 0)), float((i, j) == (0, 1)),
                              2.0 * float((i, j) in ((2, 0), (0, 2)))])
        check(p[..., k], jnp.broadcast_to(target, (b, 3)))

    c = jax.random.normal(jax.random.PRNGKey(5), (b, n))
    proj = jnp.linalg.solve(jnp.swapaxes(p, 1, 2) @ p, jnp.swapaxes(p, 1, 2) @ c[..., None])
    c = c - (p @ proj)[..., 0]                                           # P^T c = 0
    r2 = jnp.sum((xs[:, :, None, :] - xs[:, None, :, :]) ** 2, -1)
    f = jnp.sum(c[:, None, :] * r2 ** (m / 2), -1)                     # f(xi_i)
    rj = jnp.sum(xs ** 2, -1) ** ((m - 2) / 2)
    target = jnp.stack([jnp.sum(-m * c * rj * xs[..., 0], -1),
                        jnp.sum(-m * c * rj * xs[..., 1], -1),
                        jnp.sum(m * m * c * rj, -1)], -1)
    check(f, target)


def test_divergence_and_mass_laplacian_conserve(meshes):
    """sum(area * div F) = 0 for tangent F and sum(area * lap_mass phi) = 0 for
    any phi, to round-off; the uncorrected weights do not conserve."""
    for lev in LEVELS:
        xu, mesh, area = meshes[lev]
        ops = rbffd_operators(mesh)
        r = jax.random.normal(jax.random.PRNGKey(lev), (4, xu.shape[0]))
        f = jnp.stack([r[0], r[1], r[2]], -1)
        f = f - jnp.sum(f * xu, -1, keepdims=True) * xu                  # tangent
        for out in (ops.divergence(*f.T), ops.laplacian_mass(r[3])):
            assert abs(float(jnp.sum(area * out))) <= MASS_RTOL * float(jnp.sum(area * jnp.abs(out)))
    raw = sum(neighbour_ops.neighbor_sum(c, x) for c, x in zip(mesh.grad, f.T))
    assert abs(float(jnp.sum(area * raw))) > 1e3 * MASS_RTOL * float(jnp.sum(area * jnp.abs(raw)))


def test_conservative_correction_check_grads(meshes):
    """The residual clean-up is differentiable in the weights and the areas
    (area normalised to O(1); the correction is invariant to its scale)."""
    xu, _, area = meshes[3]
    stencil = rbffd_stencils(xu, N_STENCIL)
    grad, lap = rbffd_sphere_weights(xu, stencil, phs_degree=PHS, poly_degree=POLY)
    a = area / jnp.mean(area)
    check_grads(lambda w, aa: conservative_column_correction(w, stencil, aa, xu),
                (grad, a), order=2, modes=("fwd", "rev"))
    check_grads(lambda w, aa: conservative_column_correction(w, stencil, aa),
                (lap, a), order=2, modes=("fwd", "rev"))


def test_correction_keeps_rows_polynomial_exact(meshes):
    """Corrected rows stay exact on their tangent-plane polynomials: the change
    delta_i is orthogonal to P_i (both tangent components for the divergence,
    and no normal component), at every node count; the solve converged."""
    for lev in LEVELS:
        xu, _, area = meshes[lev]
        stencil = rbffd_stencils(xu, N_STENCIL)
        grad, lap = rbffd_sphere_weights(xu, stencil, phs_degree=PHS, poly_degree=POLY)
        e1, e2, xs, _ = rbffd_tangent_coordinates(xu, stencil)
        p = polynomial_matrix(xs, POLY)
        kw = dict(poly_degree=POLY, refine_steps=REFINE)
        div, res_d = polynomial_preserving_correction(grad, stencil, xu, area, **kw)
        lapm, res_l = polynomial_preserving_correction(lap, stencil, xu, area, **kw)
        print(f"level {lev}: solve rel residual div {res_d:.1e} lap {res_l:.1e}")
        assert res_d <= SOLVE_RTOL and res_l <= SOLVE_RTOL
        d = div - grad
        parts = [jnp.sum(d * e1[:, None], -1), jnp.sum(d * e2[:, None], -1), lapm - lap]
        for delta in parts:
            moments = jnp.einsum("bnm,bn->bm", p, delta)
            assert float(jnp.max(jnp.abs(moments))) <= POLY_EXACT_RTOL * float(jnp.max(jnp.abs(delta)))
        normal = jnp.sum(d * jnp.cross(e1, e2)[:, None], -1)
        assert float(jnp.max(jnp.abs(normal))) <= POLY_EXACT_RTOL * float(jnp.max(jnp.abs(d)))
