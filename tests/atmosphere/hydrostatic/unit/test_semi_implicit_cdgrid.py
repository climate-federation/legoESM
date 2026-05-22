"""Phase 2 scaffolding tests for the Hoskins–Simmons FV3 D-grid port.

Locks in the current behaviour of
``src/legoesm/atmosphere/dynamics/semi_implicit_cdgrid.py``.  Phase 2
work (see module docstring there) lowers the adjoint-residual
threshold once the cubed-sphere FV gradient + divergence are made
into a proper adjoint pair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.semi_implicit_cdgrid import (
    adjoint_residual_norm,
    cdgrid_scalar_laplacian,
    make_helmholtz_op,
)
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


@pytest.fixture(autouse=True)
def _x64_fp64():
    orig_x64 = jax.config.jax_enable_x64
    orig_pol = get_policy()
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig_pol)
    jax.config.update("jax_enable_x64", orig_x64)


@pytest.fixture(scope="module")
def cdgrid():
    grid = create_cubed_sphere(8)
    return create_cubed_sphere_cdgrid(grid)


# ----------------------------------------------------------------------
# cdgrid_scalar_laplacian — basic sanity
# ----------------------------------------------------------------------


class TestScalarLaplacian:
    def test_constant_field_has_zero_laplacian(self, cdgrid):
        n = cdgrid.base.n
        p = jnp.full((6, n, n), 1.0e5)
        lap = cdgrid_scalar_laplacian(p, cdgrid)
        # Allow O(roundoff) residual from area-weighted divergence.
        assert float(jnp.max(jnp.abs(lap))) < 1.0e-8

    def test_shape_preserved(self, cdgrid):
        n = cdgrid.base.n
        np.random.seed(0)
        p = jnp.asarray(np.random.randn(6, n, n))
        lap = cdgrid_scalar_laplacian(p, cdgrid)
        assert lap.shape == p.shape


# ----------------------------------------------------------------------
# adjoint_residual_norm — locks current Phase 2 status
# ----------------------------------------------------------------------


class TestAdjointResidual:
    def test_current_operator_is_non_symmetric(self, cdgrid):
        """Phase 2 baseline: the composed ``div(grad)`` is **not**
        a discrete FV adjoint pair on the cubed-sphere.  Locks the
        finding so a future Phase-2 PR that drives this below
        ``1e-10`` triggers the assertion update and proves the fix."""
        L = lambda p: cdgrid_scalar_laplacian(p, cdgrid)
        residual = adjoint_residual_norm(L, cdgrid, n_samples=4)
        # Empirically ~0.13 on n=8.  Assert the gap is present
        # (residual > 1e-6) so the test stays meaningful.
        assert residual > 1.0e-6, (
            f"adjoint residual {residual:.3e} unexpectedly small — "
            f"either the operator was made symmetric (great! update "
            f"this assertion to assert symmetry) or the metric weight "
            f"is now masking the true asymmetry"
        )
        assert residual < 1.0, (
            f"adjoint residual {residual:.3e} unreasonably large — "
            f"check that area weighting and operator are wired correctly"
        )


# ----------------------------------------------------------------------
# make_helmholtz_op — identity when coeff = 0
# ----------------------------------------------------------------------


class TestHelmholtzOp:
    def test_fails_closed_by_default(self, cdgrid):
        """Production callers must NOT be able to obtain a non-SPD
        operator and pass it to ``cg`` silently."""
        with pytest.raises(NotImplementedError, match="Phase 2"):
            make_helmholtz_op(coeff=0.5, cdgrid=cdgrid)

    def test_identity_at_zero_coeff(self, cdgrid):
        A = make_helmholtz_op(
            coeff=0.0, cdgrid=cdgrid,
            allow_nonsymmetric_for_testing=True,
        )
        n = cdgrid.base.n
        np.random.seed(1)
        p = jnp.asarray(np.random.randn(6, n, n))
        Ap = A(p)
        # A(p) should equal p exactly when coeff=0.
        assert float(jnp.max(jnp.abs(Ap - p))) < 1.0e-14

    def test_linearity(self, cdgrid):
        A = make_helmholtz_op(
            coeff=0.5, cdgrid=cdgrid,
            allow_nonsymmetric_for_testing=True,
        )
        n = cdgrid.base.n
        np.random.seed(2)
        x = jnp.asarray(np.random.randn(6, n, n))
        y = jnp.asarray(np.random.randn(6, n, n))
        Ax = A(x)
        Ay = A(y)
        Axy = A(x + 3.0 * y)
        residual = float(jnp.max(jnp.abs(Axy - (Ax + 3.0 * Ay))))
        assert residual < 1.0e-10
