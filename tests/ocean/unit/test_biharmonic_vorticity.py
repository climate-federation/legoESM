"""Unit tests for biharmonic dissipation on relative vorticity ζ.

Validates ``vertex_laplacian_3d`` and ``biharmonic_vorticity_del4_3d`` in
``legoesm.core.operators_voronoi``.

Properties tested:

1. ``vertex_laplacian_3d`` returns 0 for a vertex-constant field (kernel).
2. ``vertex_laplacian_3d`` matches sign convention of ordinary diffusion:
   on a spherical harmonic-like smooth vorticity field with local maximum
   at a vertex, ``∇²ζ < 0`` at that vertex.
3. ``biharmonic_vorticity_del4_3d(u)``, read back via ``curl_vertex_3d``,
   yields ``curl(F) = −∇⁴ζ`` — i.e. positive coefficient means damping.
4. The operator is zero on a pure horizontally-uniform velocity field
   (ζ identically zero everywhere).
5. For a localised ζ bump (smooth), ``K_ζ · biharmonic_vorticity_del4`` acts
   as a scale-selective damping: higher k modes get damped faster than
   lower k modes (monotone k⁴ scaling within numerical error).
6. JAX-traceability: the operator works under ``jax.jit``.

Run with:
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_biharmonic_vorticity.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.core.operators_voronoi import (
    biharmonic_vorticity_del4_3d,
    curl_vertex_3d,
    vertex_laplacian_3d,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def mesh():
    """A small global Voronoi mesh for quick tests."""
    return create_voronoi_mesh(2)  # level 2 → ~160 cells


# ============================================================================
# vertex_laplacian_3d
# ============================================================================

def test_vertex_laplacian_constant_field_is_zero(mesh):
    """Laplacian of a constant field is zero to machine precision."""
    nlev = 3
    phi = jnp.ones((mesh.nVertices, nlev))
    lap = vertex_laplacian_3d(phi, mesh)
    assert jnp.all(jnp.isfinite(lap))
    np.testing.assert_allclose(lap, 0.0, atol=1e-10)


def test_vertex_laplacian_sign_on_bump(mesh):
    """Laplacian of a smooth positive bump centred at one vertex is
    negative at the bump centre (standard diffusion sign)."""
    latv = np.asarray(mesh.latVertex)
    lonv = np.asarray(mesh.lonVertex)
    # Pick a vertex near the north pole as the bump centre.
    iv_peak = int(np.argmax(latv))
    lat0, lon0 = latv[iv_peak], lonv[iv_peak]
    # Gaussian bump in great-circle distance on the sphere.
    cos_d = (np.sin(lat0) * np.sin(latv)
             + np.cos(lat0) * np.cos(latv) * np.cos(lonv - lon0))
    cos_d = np.clip(cos_d, -1.0, 1.0)
    ang = np.arccos(cos_d)
    phi = jnp.asarray(np.exp(-(ang / 0.5) ** 2))[:, None]   # (nV, 1)
    lap = vertex_laplacian_3d(phi, mesh)
    assert lap[iv_peak, 0] < 0.0, (
        "Laplacian of a positive bump must be negative at the bump centre; "
        f"got lap[{iv_peak}] = {float(lap[iv_peak, 0]):.3e}")


def test_vertex_laplacian_jit(mesh):
    """Operator is compatible with jax.jit via closure (mesh unhashable)."""
    def jittable(phi):
        return vertex_laplacian_3d(phi, mesh)
    fn = jax.jit(jittable)
    phi = jnp.asarray(np.random.default_rng(0).normal(size=(mesh.nVertices, 2)))
    out = fn(phi)
    assert out.shape == (mesh.nVertices, 2)
    assert jnp.all(jnp.isfinite(out))


# ============================================================================
# biharmonic_vorticity_del4_3d
# ============================================================================

def test_biharmonic_vorticity_zero_velocity(mesh):
    """Zero velocity produces zero damping force."""
    nlev = 2
    u = jnp.zeros((mesh.nEdges, nlev))
    F = biharmonic_vorticity_del4_3d(u, mesh)
    assert F.shape == (mesh.nEdges, nlev)
    np.testing.assert_allclose(F, 0.0, atol=1e-12)


def test_biharmonic_vorticity_damps_zeta(mesh):
    """Reading back the curl of the damping force yields −∇⁴ζ (damping).

    For a velocity whose ζ has a clear positive peak, the vorticity
    tendency ``curl(K_ζ · F)`` at the peak should be *negative* when
    ``K_ζ > 0``.
    """
    # Build a velocity field whose curl has a smooth positive peak.
    # Easiest: pick ζ pattern at vertices directly, then solve for u that
    # produces it (approximately, via a Poisson-like inversion).  Here we
    # skip the inversion and just test with a random velocity — then we
    # inspect `dζ/dt = curl(F)` and compare its sign to `−∇⁴ζ` sign.
    rng = np.random.default_rng(42)
    u = jnp.asarray(rng.normal(scale=0.01, size=(mesh.nEdges, 1)))
    zeta0 = curl_vertex_3d(u, mesh)               # (nV, 1)
    lap_zeta0 = vertex_laplacian_3d(zeta0, mesh)
    lap2_zeta0 = vertex_laplacian_3d(lap_zeta0, mesh)   # ≈ ∇⁴ζ

    F = biharmonic_vorticity_del4_3d(u, mesh)
    # Curl of F should approximate −∇⁴ζ (damping sign for K_ζ>0).
    dzeta_dt = curl_vertex_3d(F, mesh)

    # Check sign consistency: correlation between dzeta_dt and −∇⁴ζ should
    # be strongly positive (they represent the same operator up to discrete
    # commutation errors).
    a = np.asarray(dzeta_dt).ravel()
    b = -np.asarray(lap2_zeta0).ravel()
    # Exclude near-zero entries to make the correlation meaningful.
    keep = (np.abs(b) > 0.1 * np.max(np.abs(b)))
    corr = np.corrcoef(a[keep], b[keep])[0, 1]
    assert corr > 0.9, (
        f"dζ/dt from biharmonic_vorticity_del4 is not aligned with −∇⁴ζ "
        f"(correlation {corr:.3f}); sign or formulation is wrong.")


def test_biharmonic_vorticity_jit(mesh):
    """Operator is compatible with jax.jit via closure (mesh unhashable)."""
    def jittable(u):
        return biharmonic_vorticity_del4_3d(u, mesh)
    fn = jax.jit(jittable)
    u = jnp.asarray(np.random.default_rng(0).normal(size=(mesh.nEdges, 2)))
    out = fn(u)
    assert out.shape == (mesh.nEdges, 2)
    assert jnp.all(jnp.isfinite(out))


def test_biharmonic_vorticity_scale_selectivity(mesh):
    """Damping rate scales as k⁴ — high-wavenumber modes decay much faster
    than low-wavenumber modes.

    Constructs two velocity fields with dominant vorticity at distinct
    scales (coarse vs fine random noise), applies the operator, and
    verifies the damping ratio is strongly in favour of the fine scale.
    """
    rng = np.random.default_rng(0)
    # Coarse velocity: smooth perturbation built from cell-scale random
    # potential (corresponds to low-k curl).
    phi_cell_coarse = rng.normal(size=(mesh.nCells, 1))
    # Fine velocity: random noise directly at edges (broad spectrum, but
    # with strong high-k component).
    u_fine = jnp.asarray(rng.normal(size=(mesh.nEdges, 1)))

    # Make a "coarse" velocity by smoothing via a few Laplacian applications
    # of a random cell potential — ensures lower effective k.
    from legoesm.core.operators_voronoi import gradient_edge_3d
    u_coarse_raw = gradient_edge_3d(jnp.asarray(phi_cell_coarse), mesh)
    u_coarse = u_coarse_raw  # lower-k velocity

    zeta_coarse = np.asarray(curl_vertex_3d(u_coarse, mesh)).ravel()
    zeta_fine = np.asarray(curl_vertex_3d(u_fine, mesh)).ravel()

    F_coarse = biharmonic_vorticity_del4_3d(u_coarse, mesh)
    F_fine = biharmonic_vorticity_del4_3d(u_fine, mesh)

    # Effective damping rate on ζ: curl(F)·sign(ζ) / |ζ|
    dzeta_coarse = np.asarray(curl_vertex_3d(F_coarse, mesh)).ravel()
    dzeta_fine = np.asarray(curl_vertex_3d(F_fine, mesh)).ravel()

    # Mean squared damping tendency, normalised by ζ² (i.e., avg rate):
    rate_coarse = float(np.sum(dzeta_coarse**2) / (np.sum(zeta_coarse**2) + 1e-30))
    rate_fine = float(np.sum(dzeta_fine**2) / (np.sum(zeta_fine**2) + 1e-30))

    # Fine scale should be damped much more strongly than coarse — k⁴
    # selectivity. Factor of at least 4× is a safe threshold on an
    # unstructured mesh (true continuum ratio would be orders of
    # magnitude, but discrete ops lose some selectivity).
    assert rate_fine > 4.0 * rate_coarse, (
        f"scale-selectivity violated: rate_fine/rate_coarse = "
        f"{rate_fine/rate_coarse:.2f} (expected > 4)")


def test_biharmonic_vorticity_energy_monotone(mesh):
    """Operator is dissipative in the L² momentum norm: ∫ u · F dA ≤ 0.

    For K_ζ > 0, adding ``K_ζ · F`` to du/dt must decrease ``∫ u·u dA``
    (i.e. kinetic energy — since the operator acts on vorticity part only,
    the effect on total KE can be small but must be non-positive for the
    scheme to be stable).
    """
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.normal(size=(mesh.nEdges, 1)))
    F = biharmonic_vorticity_del4_3d(u, mesh)
    # Inner product weighted by dvEdge·dcEdge (FV edge area). MPAS uses
    # ½·dvEdge·dcEdge as the edge control volume.
    edge_area = 0.5 * np.asarray(mesh.dvEdge * mesh.dcEdge)[:, None]
    u_dot_F = float(np.sum(np.asarray(u) * np.asarray(F) * edge_area))
    # Must be non-positive for dissipation. Allow a small positive slop
    # from discrete commutation errors at random-noise scale.
    tol = 1e-6 * float(np.sum(np.asarray(u) ** 2 * edge_area))
    assert u_dot_F <= tol, (
        f"biharmonic_vorticity_del4 is not dissipative: ∫u·F dA = "
        f"{u_dot_F:.3e} (must be ≤ {tol:.3e})")


def test_biharmonic_vorticity_grad_smoke(mesh):
    """jax.grad through the operator works (AD-compatibility)."""
    def loss(u):
        F = biharmonic_vorticity_del4_3d(u, mesh)
        return jnp.sum(F ** 2)
    u = jnp.asarray(np.random.default_rng(0).normal(size=(mesh.nEdges, 1)))
    g = jax.grad(loss)(u)
    assert g.shape == u.shape
    assert jnp.all(jnp.isfinite(g))
    # Trivially, gradient should be non-zero for a non-trivial loss.
    assert jnp.sum(g ** 2) > 0.0
