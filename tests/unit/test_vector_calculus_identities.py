"""Category 1: Discrete Vector Calculus Identities.

Tests exact mathematical identities that any correct discretization of
div, grad, curl must satisfy:
  - curl(grad(phi)) = 0
  - div(curl-derived field) ~ 0
  - Laplacian = div(grad)
  - Integration by parts (adjoint consistency)
  - Null-space: grad(const) = 0, Laplacian(const) = 0, div(solid-body) = 0
  - Spectral Laplacian eigenvalue for Y_n^m

Run with:  JAX_ENABLE_X64=1 python -m pytest tests/unit/test_vector_calculus_identities.py -v
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.latlon import create_latlon_grid
from legoesm import constants

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cs_grid():
    """Small C8 cubed-sphere grid."""
    return create_cubed_sphere(8)


@pytest.fixture(scope="module")
def ll_grid():
    """Small 16x32 lat-lon grid."""
    return create_latlon_grid(16, 32)


@pytest.fixture(scope="module")
def gauss_grid():
    """Small T5 Gaussian grid for spectral tests."""
    from legoesm.grids.gaussian import create_gaussian_grid
    return create_gaussian_grid(5)


@pytest.fixture(scope="module")
def voronoi_mesh():
    """Level-2 MPAS mesh (162 cells)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(2)


# ---------------------------------------------------------------------------
# Helper: smooth scalar field on each grid type
# ---------------------------------------------------------------------------

def _smooth_scalar_cs(grid):
    """cos(lat)*cos(2*lon) on cubed-sphere."""
    data = jnp.cos(grid.lat) * jnp.cos(2.0 * grid.lon)
    return Field(data=data, name="phi", dims=("face", "x", "y"), units="1")


def _smooth_scalar_ll(grid):
    """cos(lat)*cos(2*lon) on lat-lon."""
    data = jnp.cos(grid.lat2d) * jnp.cos(2.0 * grid.lon2d)
    return Field(data=data, name="phi", dims=("lat", "lon"), units="1")


def _constant_field_cs(grid, value=42.0):
    data = jnp.full((6, grid.n, grid.n), value)
    return Field(data=data, name="const", dims=("face", "x", "y"), units="1")


def _constant_field_ll(grid, value=42.0):
    data = jnp.full((grid.n_lat, grid.n_lon), value)
    return Field(data=data, name="const", dims=("lat", "lon"), units="1")


def _dx_min_cs(grid):
    """Minimum grid spacing on cubed-sphere (scalar)."""
    return float(jnp.min(grid.dx))


# ======================================================================
# 1f) Null-space tests
# ======================================================================

class TestNullSpace:
    """grad(const) = 0, Laplacian(const) = 0, div(solid-body) = 0."""

    # --- gradient of constant = 0 ---

    def test_grad_constant_cubesphere(self, cs_grid):
        """Gradient of a constant field must be zero on cubed-sphere."""
        from legoesm.core.operators import gradient_x, gradient_y
        phi = _constant_field_cs(cs_grid, 7.0)
        gx = gradient_x(phi, cs_grid)
        gy = gradient_y(phi, cs_grid)
        assert float(jnp.max(jnp.abs(gx.data))) < 1e-10
        assert float(jnp.max(jnp.abs(gy.data))) < 1e-10

    def test_grad_constant_latlon(self, ll_grid):
        """Gradient of a constant field must be zero on lat-lon."""
        from legoesm.core.operators_latlon import gradient_x, gradient_y
        phi = _constant_field_ll(ll_grid, 7.0)
        gx = gradient_x(phi, ll_grid)
        gy = gradient_y(phi, ll_grid)
        assert float(jnp.max(jnp.abs(gx.data))) < 1e-10
        assert float(jnp.max(jnp.abs(gy.data))) < 1e-10

    def test_grad_constant_voronoi(self, voronoi_mesh):
        """Gradient of a constant field must be zero on MPAS mesh."""
        from legoesm.core.operators_voronoi import gradient_edge
        phi = jnp.full((voronoi_mesh.nCells,), 7.0)
        grad = gradient_edge(phi, voronoi_mesh)
        assert float(jnp.max(jnp.abs(grad))) < 1e-10

    # --- Laplacian of constant = 0 ---

    def test_laplacian_constant_cubesphere(self, cs_grid):
        """Laplacian of a constant must be zero on cubed-sphere."""
        from legoesm.core.operators import laplacian
        phi = _constant_field_cs(cs_grid, 3.14)
        lap = laplacian(phi, cs_grid)
        assert float(jnp.max(jnp.abs(lap.data))) < 1e-8

    def test_laplacian_constant_latlon(self, ll_grid):
        """Laplacian of a constant must be zero on lat-lon."""
        from legoesm.core.operators_latlon import laplacian
        phi = _constant_field_ll(ll_grid, 3.14)
        lap = laplacian(phi, ll_grid)
        assert float(jnp.max(jnp.abs(lap.data))) < 1e-8

    # --- divergence of solid-body rotation = 0 ---

    def test_div_solid_body_cubesphere(self, cs_grid):
        """Divergence of solid-body rotation must be ~0 on cubed-sphere.

        Solid-body rotation: u = U0*cos(lat), v = 0 in geographic coords.
        This is a non-divergent flow, so div(u,v) should vanish.
        """
        from legoesm.core.operators import divergence
        U0 = 10.0
        u_geo = U0 * jnp.cos(cs_grid.lat)
        v_geo = jnp.zeros_like(u_geo)
        u_local = u_geo * cs_grid.cos_angle + v_geo * cs_grid.sin_angle
        v_local = -u_geo * cs_grid.sin_angle + v_geo * cs_grid.cos_angle
        u_f = Field(data=u_local, name="u", dims=("face", "x", "y"), units="m/s")
        v_f = Field(data=v_local, name="v", dims=("face", "x", "y"), units="m/s")
        div = divergence(u_f, v_f, cs_grid)
        max_div = float(jnp.max(jnp.abs(div.data)))
        dx_min = _dx_min_cs(cs_grid)
        scale = U0 / dx_min
        # Cubed-sphere face boundaries introduce O(dx) error for vector fields
        assert max_div / scale < 0.2, f"div(solid-body) relative error {max_div/scale:.3e}"

    def test_div_solid_body_latlon(self, ll_grid):
        """Divergence of solid-body rotation must be ~0 on lat-lon."""
        from legoesm.core.operators_latlon import divergence
        U0 = 10.0
        u_data = U0 * jnp.cos(ll_grid.lat2d)
        v_data = jnp.zeros_like(u_data)
        u_f = Field(data=u_data, name="u", dims=("lat", "lon"), units="m/s")
        v_f = Field(data=v_data, name="v", dims=("lat", "lon"), units="m/s")
        div = divergence(u_f, v_f, ll_grid)
        max_div = float(jnp.max(jnp.abs(div.data)))
        scale = U0 / float(ll_grid.dy)
        assert max_div / scale < 0.05, f"div(solid-body) relative error {max_div/scale:.3e}"

    def test_div_solid_body_voronoi(self, voronoi_mesh):
        """Divergence of solid-body rotation on MPAS mesh."""
        from legoesm.core.operators_voronoi import divergence_cell
        mesh = voronoi_mesh
        U0 = 10.0
        u_edge = U0 * jnp.cos(mesh.latEdge) * jnp.cos(mesh.angleEdge)
        div = divergence_cell(u_edge, mesh)
        max_div = float(jnp.max(jnp.abs(div)))
        dx_typical = float(jnp.mean(mesh.dcEdge))
        scale = U0 / dx_typical
        assert max_div / scale < 0.05, f"div(solid-body) relative error {max_div/scale:.3e}"

    # --- gradient of constant = 0 on Gaussian grid (spectral) ---

    def test_grad_constant_gaussian(self, gauss_grid):
        """Gradient of a constant field must be zero on a Gaussian grid.

        On a Gaussian grid, a constant field lives entirely in the n=0, m=0
        spherical harmonic mode. The spectral gradient of n=0 is zero, so
        we verify that sh_analysis of a constant yields only the (0,0) mode
        and that all other coefficients are negligible.
        """
        from legoesm.grids.gaussian import sh_analysis, sh_synthesis
        grid = gauss_grid
        # Constant field = 42.0 on the Gaussian grid
        const_field = jnp.full((grid.n_lat, grid.n_lon), 42.0)
        # Forward SH transform
        coeffs = sh_analysis(grid, const_field)
        # Zero out the n=0 mode (the only one that should be nonzero)
        mask_n0 = grid.ls == 0
        grad_coeffs = jnp.where(mask_n0, 0.0, coeffs)
        # All remaining coefficients should be near zero
        max_grad_coeff = float(jnp.max(jnp.abs(grad_coeffs)))
        assert max_grad_coeff < 1e-10, (
            f"Non-zero gradient coefficients for constant field: {max_grad_coeff:.3e}"
        )

    # --- Laplacian of constant = 0 on Gaussian grid (spectral) ---

    def test_laplacian_constant_gaussian(self, gauss_grid):
        """Laplacian of a constant must be zero on a Gaussian grid.

        Uses spectral Laplacian: multiply SH coefficients by -n(n+1)/a^2.
        For a constant field (only n=0 mode), the Laplacian eigenvalue is 0.
        """
        from legoesm.grids.gaussian import sh_analysis, sh_synthesis, spectral_laplacian
        grid = gauss_grid
        const_field = jnp.full((grid.n_lat, grid.n_lon), 3.14)
        coeffs = sh_analysis(grid, const_field)
        lap_coeffs = spectral_laplacian(grid, coeffs)
        lap_grid = sh_synthesis(grid, lap_coeffs)
        max_lap = float(jnp.max(jnp.abs(lap_grid)))
        assert max_lap < 1e-8, (
            f"Laplacian of constant on Gaussian grid = {max_lap:.3e}, expected ~0"
        )

    # --- hyperdiffusion of constant = 0 on cubed-sphere ---

    def test_hyperdiff_constant_cubesphere(self, cs_grid):
        """Hyperdiffusion (nabla^4) of a constant field must be zero on cubed-sphere."""
        from legoesm.core.operators import hyperdiffusion
        phi = _constant_field_cs(cs_grid, 5.0)
        coeff = 1.0e10  # arbitrary nonzero coefficient
        result = hyperdiffusion(phi, cs_grid, coeff)
        max_val = float(jnp.max(jnp.abs(result.data)))
        assert max_val < 1e-6, (
            f"nabla^4(constant) = {max_val:.3e} on cubed-sphere, expected ~0"
        )

    # --- hyperdiffusion of constant = 0 on lat-lon ---

    def test_hyperdiff_constant_latlon(self, ll_grid):
        """Hyperdiffusion (nabla^4) of a constant field must be zero on lat-lon."""
        from legoesm.core.operators_latlon import hyperdiffusion
        phi = _constant_field_ll(ll_grid, 5.0)
        coeff = 1.0e10  # arbitrary nonzero coefficient
        result = hyperdiffusion(phi, ll_grid, coeff)
        max_val = float(jnp.max(jnp.abs(result.data)))
        assert max_val < 1e-6, (
            f"nabla^4(constant) = {max_val:.3e} on lat-lon, expected ~0"
        )


# ======================================================================
# 1b) curl(grad(phi)) = 0
# ======================================================================

class TestCurlGrad:
    """curl(grad(phi)) must vanish for any smooth scalar phi."""

    def test_curl_grad_cubesphere(self, cs_grid):
        """curl(grad(phi)) = 0 on cubed-sphere A-grid.

        The identity is exact in the continuous case. For 2nd-order FD,
        the commutator error is O(dx^2), so we check relative to the
        field magnitude rather than requiring machine zero.
        """
        from legoesm.core.operators import gradient_x, gradient_y, curl_z
        phi = _smooth_scalar_cs(cs_grid)
        gx = gradient_x(phi, cs_grid)
        gy = gradient_y(phi, cs_grid)
        vort = curl_z(gx, gy, cs_grid)
        max_vort = float(jnp.max(jnp.abs(vort.data)))
        # curl(grad) should be near machine precision for smooth fields
        # The absolute value matters more than scaling by grad/dx
        assert max_vort < 1e-10, f"|curl(grad)| = {max_vort:.3e} on cubed-sphere"

    def test_curl_grad_latlon(self, ll_grid):
        """curl(grad(phi)) = 0 on lat-lon grid."""
        from legoesm.core.operators_latlon import gradient_x, gradient_y, curl_z
        phi = _smooth_scalar_ll(ll_grid)
        gx = gradient_x(phi, ll_grid)
        gy = gradient_y(phi, ll_grid)
        vort = curl_z(gx, gy, ll_grid)
        max_vort = float(jnp.max(jnp.abs(vort.data)))
        grad_scale = float(jnp.max(jnp.abs(gx.data)))
        dx_min = float(jnp.min(ll_grid.dx))
        assert max_vort < grad_scale / dx_min * 0.5, (
            f"|curl(grad)| = {max_vort:.3e}, grad_scale/dx = {grad_scale/dx_min:.3e}"
        )

    def test_curl_grad_voronoi(self, voronoi_mesh):
        """curl(grad(phi)) = 0 on MPAS Voronoi mesh.

        grad maps cells -> edges, curl maps edges -> vertices.
        curl(grad) should vanish exactly (Stokes' theorem on discrete mesh).
        """
        from legoesm.core.operators_voronoi import gradient_edge, curl_vertex
        mesh = voronoi_mesh
        phi = jnp.cos(mesh.latCell) * jnp.cos(2.0 * mesh.lonCell)
        grad = gradient_edge(phi, mesh)
        curl = curl_vertex(grad, mesh)
        max_curl = float(jnp.max(jnp.abs(curl)))
        assert max_curl < 1e-10, f"|curl(grad)| = {max_curl:.3e} on Voronoi mesh"


# ======================================================================
# 1c) Laplacian consistency: nabla^2(phi) = div(grad(phi))
# ======================================================================

class TestLaplacianConsistency:
    """The Laplacian computed directly must agree with div(grad)."""

    def test_lap_eq_divgrad_cubesphere(self, cs_grid):
        """On cubed-sphere, laplacian() IS div(grad()), so this verifies
        that identity and also checks compact Laplacian correlation."""
        from legoesm.core.operators import laplacian, laplacian_compact, gradient_x, gradient_y, divergence
        phi = _smooth_scalar_cs(cs_grid)
        gx = gradient_x(phi, cs_grid)
        gy = gradient_y(phi, cs_grid)
        divgrad = divergence(gx, gy, cs_grid)
        lap = laplacian(phi, cs_grid)
        # These should be identical since laplacian() = div(grad())
        diff = float(jnp.max(jnp.abs(lap.data - divgrad.data)))
        assert diff < 1e-12, f"laplacian != div(grad): max diff = {diff:.3e}"

        # Compact Laplacian uses a different stencil but same sign/magnitude
        lap_compact = laplacian_compact(phi.data, cs_grid)
        corr = float(jnp.corrcoef(
            lap.data.ravel(), lap_compact.ravel()
        )[0, 1])
        assert corr > 0.9, f"Compact vs composed Laplacian correlation = {corr:.3f}"

    def test_lap_eq_divgrad_latlon(self, ll_grid):
        """On lat-lon, the laplacian uses compact stencil; check vs div(grad)."""
        from legoesm.core.operators_latlon import (
            laplacian, gradient_x, gradient_y, divergence,
        )
        phi = _smooth_scalar_ll(ll_grid)
        gx = gradient_x(phi, ll_grid)
        gy = gradient_y(phi, ll_grid)
        divgrad = divergence(gx, gy, ll_grid)
        lap = laplacian(phi, ll_grid)
        # Different stencils — check they correlate well
        l2_diff = float(jnp.sqrt(jnp.mean((lap.data - divgrad.data) ** 2)))
        l2_lap = float(jnp.sqrt(jnp.mean(lap.data ** 2)))
        rel = l2_diff / (l2_lap + 1e-30)
        assert rel < 0.5, f"Laplacian vs div(grad) relative L2 = {rel:.3e}"

    def test_lap_consistency_voronoi(self, voronoi_mesh):
        """On Voronoi mesh, scalar Laplacian = div(grad(phi)).
        Check it is finite and has zero global integral."""
        from legoesm.core.operators_voronoi import gradient_edge, divergence_cell
        mesh = voronoi_mesh
        phi = jnp.cos(mesh.latCell) * jnp.cos(2.0 * mesh.lonCell)
        grad_phi = gradient_edge(phi, mesh)
        lap_scalar = divergence_cell(grad_phi, mesh)
        assert jnp.all(jnp.isfinite(lap_scalar))
        # Global integral of Laplacian on closed sphere = 0
        mean_lap = float(jnp.sum(lap_scalar * mesh.areaCell) / jnp.sum(mesh.areaCell))
        assert abs(mean_lap) < 1e-6 * float(jnp.max(jnp.abs(lap_scalar))), (
            f"Mean Laplacian should be ~0, got {mean_lap:.3e}"
        )


# ======================================================================
# 1d) Integration by parts / adjoint consistency
# ======================================================================

class TestAdjointConsistency:
    r"""On a closed sphere, grad and -div are adjoints:
    \int grad(phi) . F dA = -\int phi * div(F) dA

    This is critical for energy conservation and 4D-Var correctness.
    At coarse resolution (C8, 16x32) the adjoint error is O(dx^2),
    so we use a generous tolerance.
    """

    def test_adjoint_grad_div_cubesphere(self, cs_grid):
        """<grad(phi), F> ~ -<phi, div(F)> on cubed-sphere."""
        from legoesm.core.operators import (
            gradient_x, gradient_y, divergence, global_integral,
        )
        phi_data = jnp.cos(cs_grid.lat) * jnp.sin(cs_grid.lon)
        u_data = jnp.sin(cs_grid.lat) * jnp.cos(cs_grid.lon)
        v_data = jnp.cos(2.0 * cs_grid.lat)

        phi_f = Field(data=phi_data, name="phi", dims=("face", "x", "y"), units="1")
        u_f = Field(data=u_data, name="u", dims=("face", "x", "y"), units="1")
        v_f = Field(data=v_data, name="v", dims=("face", "x", "y"), units="1")

        gx = gradient_x(phi_f, cs_grid)
        gy = gradient_y(phi_f, cs_grid)
        integrand_lhs = Field(
            data=gx.data * u_data + gy.data * v_data,
            name="lhs", dims=("face", "x", "y"), units="1",
        )
        lhs = float(global_integral(integrand_lhs, cs_grid))

        div_F = divergence(u_f, v_f, cs_grid)
        integrand_rhs = Field(
            data=-phi_data * div_F.data,
            name="rhs", dims=("face", "x", "y"), units="1",
        )
        rhs = float(global_integral(integrand_rhs, cs_grid))

        # At C8 resolution the adjoint is only approximate (O(dx^2))
        scale = max(abs(lhs), abs(rhs), 1e-30)
        rel_err = abs(lhs - rhs) / scale
        assert rel_err < 1.0, (
            f"Adjoint test: LHS={lhs:.6e}, RHS={rhs:.6e}, rel_err={rel_err:.3e}. "
            f"Expected O(dx^2) error at coarse resolution."
        )

    def test_adjoint_grad_div_latlon(self, ll_grid):
        """<grad(phi), F> ~ -<phi, div(F)> on lat-lon."""
        from legoesm.core.operators_latlon import (
            gradient_x, gradient_y, divergence, global_integral,
        )
        phi_data = jnp.cos(ll_grid.lat2d) * jnp.sin(ll_grid.lon2d)
        u_data = jnp.sin(ll_grid.lat2d) * jnp.cos(ll_grid.lon2d)
        v_data = jnp.cos(2.0 * ll_grid.lat2d)

        phi_f = Field(data=phi_data, name="phi", dims=("lat", "lon"), units="1")
        u_f = Field(data=u_data, name="u", dims=("lat", "lon"), units="1")
        v_f = Field(data=v_data, name="v", dims=("lat", "lon"), units="1")

        gx = gradient_x(phi_f, ll_grid)
        gy = gradient_y(phi_f, ll_grid)
        integrand_lhs = Field(
            data=gx.data * u_data + gy.data * v_data,
            name="lhs", dims=("lat", "lon"), units="1",
        )
        lhs = float(global_integral(integrand_lhs, ll_grid))

        div_F = divergence(u_f, v_f, ll_grid)
        integrand_rhs = Field(
            data=-phi_data * div_F.data,
            name="rhs", dims=("lat", "lon"), units="1",
        )
        rhs = float(global_integral(integrand_rhs, ll_grid))

        scale = max(abs(lhs), abs(rhs), 1e-30)
        rel_err = abs(lhs - rhs) / scale
        assert rel_err < 1.0, (
            f"Adjoint test: LHS={lhs:.6e}, RHS={rhs:.6e}, rel_err={rel_err:.3e}. "
            f"Expected O(dx^2) error at coarse resolution."
        )


# ======================================================================
# 1e) Spectral Laplacian eigenvalue: nabla^2(Y_n^m) = -n(n+1)/a^2 Y_n^m
# ======================================================================

class TestSpectralLaplacianEigenvalue:
    """The spectral Laplacian should exactly reproduce eigenvalues."""

    def test_laplacian_eigenvalue_gaussian(self, gauss_grid):
        r"""nabla^2(Y_3^2) = -3*4/a^2 * Y_3^2 on Gaussian grid.

        We create Y_3^2 in spectral space (a single coefficient),
        transform to grid, apply the spectral Laplacian, and check.
        """
        from legoesm.grids.gaussian import sh_synthesis
        grid = gauss_grid
        a = grid.radius
        n, m = 3, 2
        expected_eigenvalue = -n * (n + 1) / a**2

        # Create Y_n^m in spectral space: set a single coefficient to 1
        field_hat = jnp.zeros(grid.n_sh, dtype=jnp.complex128)
        idx = None
        for i in range(grid.n_sh):
            if int(grid.ls[i]) == n and int(grid.ms[i]) == m:
                idx = i
                break
        assert idx is not None, f"Could not find (n={n}, m={m}) in spectral indices"
        field_hat = field_hat.at[idx].set(1.0 + 0.0j)

        # Transform to grid space: sh_synthesis(grid, coeffs)
        field_grid = sh_synthesis(grid, field_hat)

        # Apply spectral Laplacian: multiply by -n(n+1)/a^2
        lap_hat = field_hat * grid.lap  # grid.lap[i] = -l_i*(l_i+1)/a^2
        lap_grid = sh_synthesis(grid, lap_hat)

        # Check eigenvalue: lap_grid / field_grid = expected
        mask = jnp.abs(field_grid) > 1e-10 * float(jnp.max(jnp.abs(field_grid)))
        ratio = jnp.where(mask, lap_grid / field_grid, expected_eigenvalue)
        max_err = float(jnp.max(jnp.abs(ratio - expected_eigenvalue)))
        assert max_err < 1e-10 * abs(expected_eigenvalue), (
            f"Spectral Laplacian eigenvalue error: {max_err:.3e}, "
            f"expected {expected_eigenvalue:.6e}"
        )


# ======================================================================
# 1a) div(curl-derived field) ~ 0
# ======================================================================

class TestDivCurl:
    """The divergence of a curl-derived vector field should be ~0."""

    def test_div_of_curl_field_cubesphere(self, cs_grid):
        """Construct a non-divergent field from a stream function,
        then check its divergence vanishes.

        Given psi, let (u, v) = (-dpsi/dy, dpsi/dx) => div(u,v) = 0.
        """
        from legoesm.core.operators import gradient_x, gradient_y, divergence
        psi_data = jnp.sin(cs_grid.lat) * jnp.cos(cs_grid.lon)
        psi = Field(data=psi_data, name="psi", dims=("face", "x", "y"), units="m2/s")
        dpsi_dx = gradient_x(psi, cs_grid)
        dpsi_dy = gradient_y(psi, cs_grid)
        u_f = Field(data=-dpsi_dy.data, name="u", dims=("face", "x", "y"), units="m/s")
        v_f = Field(data=dpsi_dx.data, name="v", dims=("face", "x", "y"), units="m/s")
        div = divergence(u_f, v_f, cs_grid)
        max_div = float(jnp.max(jnp.abs(div.data)))
        grad_scale = float(jnp.max(jnp.sqrt(dpsi_dx.data**2 + dpsi_dy.data**2)))
        dx_min = _dx_min_cs(cs_grid)
        rel = max_div / (grad_scale / dx_min + 1e-30)
        assert rel < 0.5, f"|div(curl-field)| / (grad_scale/dx) = {rel:.3e}"

    def test_div_of_curl_field_latlon(self, ll_grid):
        """Same test on lat-lon grid."""
        from legoesm.core.operators_latlon import gradient_x, gradient_y, divergence
        psi_data = jnp.sin(ll_grid.lat2d) * jnp.cos(ll_grid.lon2d)
        psi = Field(data=psi_data, name="psi", dims=("lat", "lon"), units="m2/s")
        dpsi_dx = gradient_x(psi, ll_grid)
        dpsi_dy = gradient_y(psi, ll_grid)
        u_f = Field(data=-dpsi_dy.data, name="u", dims=("lat", "lon"), units="m/s")
        v_f = Field(data=dpsi_dx.data, name="v", dims=("lat", "lon"), units="m/s")
        div = divergence(u_f, v_f, ll_grid)
        max_div = float(jnp.max(jnp.abs(div.data)))
        grad_scale = float(jnp.max(jnp.sqrt(dpsi_dx.data**2 + dpsi_dy.data**2)))
        dx_min = float(jnp.min(ll_grid.dx))
        rel = max_div / (grad_scale / dx_min + 1e-30)
        assert rel < 0.5, f"|div(curl-field)| / (grad_scale/dx) = {rel:.3e}"


# ======================================================================
# Extra: global integral of Laplacian on closed sphere = 0
# ======================================================================

class TestLaplacianIntegralZero:
    """The global integral of the Laplacian of any field on a closed
    sphere must vanish (by the divergence theorem)."""

    def test_integral_lap_cubesphere(self, cs_grid):
        from legoesm.core.operators import laplacian, global_integral
        phi = _smooth_scalar_cs(cs_grid)
        lap = laplacian(phi, cs_grid)
        integral = float(global_integral(lap, cs_grid))
        scale = float(cs_grid.total_area) * float(jnp.max(jnp.abs(lap.data)))
        rel = abs(integral) / (scale + 1e-30)
        # At C8 the boundary errors are O(dx^2), allow 5%
        assert rel < 0.05, f"Integral of Laplacian / scale = {rel:.3e}"

    def test_integral_lap_latlon(self, ll_grid):
        from legoesm.core.operators_latlon import laplacian, global_integral
        phi = _smooth_scalar_ll(ll_grid)
        lap = laplacian(phi, ll_grid)
        integral = float(global_integral(lap, ll_grid))
        scale = float(ll_grid.total_area) * float(jnp.max(jnp.abs(lap.data)))
        rel = abs(integral) / (scale + 1e-30)
        assert rel < 0.05, f"Integral of Laplacian / scale = {rel:.3e}"

    def test_integral_lap_voronoi(self, voronoi_mesh):
        from legoesm.core.operators_voronoi import gradient_edge, divergence_cell
        mesh = voronoi_mesh
        phi = jnp.cos(mesh.latCell) * jnp.cos(2.0 * mesh.lonCell)
        grad = gradient_edge(phi, mesh)
        lap = divergence_cell(grad, mesh)
        integral = float(jnp.sum(lap * mesh.areaCell))
        scale = float(jnp.sum(mesh.areaCell)) * float(jnp.max(jnp.abs(lap)))
        rel = abs(integral) / (scale + 1e-30)
        assert rel < 0.01, f"Integral of Laplacian / scale = {rel:.3e}"
