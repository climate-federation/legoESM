"""Units of the Voronoi biharmonic Smagorinsky coefficient.

``vector_laplacian_del2_3d`` is dimensional (1/m^2), so ``-del2(B del2 u)``
is a velocity tendency only when ``B`` is a biharmonic viscosity [m^4/s]:
``B = C^2 Delta^4 |D|``.  Two checks: (1) the coefficient the operator
applies is exactly that (a Laplacian-units ``(C Delta)^2 |D|`` would be off
by ``Delta^2`` ~ 1e11 on ico5); (2) for a smooth vector field |D| converges,
so RMS(smag)/RMS(unit del4) must fall off closer to ``Delta^4`` than to
``Delta^2`` across the ico4 -> ico5 refinement (the discrete del4 of a
low-wavenumber field is still truncation-noisy at these resolutions, hence
the log-space discriminator rather than a tight ratio).
"""

import functools
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from legoesm.core.operators_voronoi import (
    curl_vertex_3d,
    divergence_cell_3d,
    gradient_edge_3d,
    smagorinsky_biharmonic_3d,
    vector_laplacian_del2_3d,
    vector_laplacian_del4_3d,
)
from legoesm.grids.voronoi import create_voronoi_mesh

C_SMAG = 0.3


@functools.lru_cache(maxsize=None)
def _mesh(level):
    return create_voronoi_mesh(level, lloyd_iterations=0)


def _cell_lon_lat(mesh):
    x, y, z = (jnp.asarray(getattr(mesh, k), dtype=jnp.float64)
               for k in ("xCell", "yCell", "zCell"))
    return jnp.arctan2(y, x), jnp.arcsin(z / jnp.sqrt(x * x + y * y + z * z))


def _rms(x):
    return float(jnp.sqrt(jnp.mean(jnp.asarray(x) ** 2)))


def _run(level):
    mesh = _mesh(level)
    lon, lat = _cell_lon_lat(mesh)
    # gradient of a pole-regular smooth scalar (Y_3^3-like): irrotational,
    # its divergence converges with resolution.  float64: the cached meshes
    # are float32 and del2(del2(u)) underflows there.
    phi = 1.0e6 * jnp.cos(lat) ** 3 * jnp.sin(3.0 * lon)
    u = jnp.asarray(gradient_edge_3d(jnp.stack([phi, 0.5 * phi], axis=-1), mesh),
                    dtype=jnp.float64)
    t_smag = smagorinsky_biharmonic_3d(u, mesh, C_SMAG)
    t_bih = vector_laplacian_del4_3d(u, mesh)
    delta = jnp.sqrt(jnp.asarray(mesh.dcEdge, dtype=jnp.float64)
                     * jnp.asarray(mesh.dvEdge, dtype=jnp.float64))
    return mesh, u, t_smag, t_bih, delta


@pytest.mark.parametrize("level", [4, 5])
def test_finite(level):
    # No sign check: the sandwich form -del2(B del2 u) with a varying B is
    # not discretely energy-stable (documented for the lat-lon operator too).
    _, u, t_smag, _, _ = _run(level)
    assert t_smag.shape == u.shape
    assert jnp.all(jnp.isfinite(t_smag))


def test_coefficient_is_c2_delta4_deformation():
    mesh, u, t_smag, _, delta = _run(5)
    div_c = divergence_cell_3d(u, mesh)
    curl_v = curl_vertex_3d(u, mesh)
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    v0, v1 = mesh.verticesOnEdge[0], mesh.verticesOnEdge[1]
    deformation = jnp.sqrt((0.5 * (div_c[c1] + div_c[c2])) ** 2
                           + (0.5 * (curl_v[v0] + curl_v[v1])) ** 2 + 1e-30)
    B = (C_SMAG ** 2) * delta[:, None] ** 4 * deformation
    t_ref = -vector_laplacian_del2_3d(B * vector_laplacian_del2_3d(u, mesh), mesh)
    # the operator forms Delta from the float32 mesh arrays: ~1e-6 relative
    assert _rms(t_smag - t_ref) <= 1e-5 * _rms(t_ref)


def test_scaling_closer_to_delta4_than_delta2():
    _, _, t4_smag, t4_bih, d4 = _run(4)
    _, _, t5_smag, t5_bih, d5 = _run(5)
    step = float(jnp.mean(d5) / jnp.mean(d4))
    assert 0.45 < step < 0.55
    actual = (_rms(t5_smag) / _rms(t5_bih)) / (_rms(t4_smag) / _rms(t4_bih))
    d_four = abs(math.log(actual) - math.log(step ** 4))
    d_two = abs(math.log(actual) - math.log(step ** 2))
    assert d_four < d_two, (actual, step ** 4, step ** 2)
