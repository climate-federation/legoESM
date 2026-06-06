"""Regression guard: the FC-Gram cube-ocean velocity viscosity must use a
ROTATION-AWARE (vector) halo.

fv3_faithful (ocean): ``ocean_baroclinic_tendencies_fc`` computed the horizontal
viscosity / hyperdiffusion of the VELOCITY by stacking the face-local
components ``(u, v)`` into a passive ``nlev*2`` axis and halo-exchanging that
stack with the SCALAR ``pad_halo_4d`` — blending the two components across cube
panel seams WITHOUT rotating them into the neighbour face's local basis.  Every
RHS evaluation then added a panel-edge momentum imprint to ``du/dt, dv/dt``.
The fix routes the velocity halo through the rotation-aware
``fc_pad_halo_vector`` (the same vector halo the momentum tendency already
uses, and the analogue of the CD-grid atmosphere diffusion-halo fix) and runs
the Laplacian / hand-built biharmonic per component.

These tests fail if a refactor reverts the velocity viscosity to a scalar halo:
(a) the FC velocity Laplacian with the rotation-aware (vector) halo preserves
the zonal symmetry of a zonally-symmetric jet's viscous tendency *better* than a
scalar halo (the correct discriminator for a VECTOR field — a scalar 2nd-diff
"smoothness" metric is invalid because the face-local components genuinely jump
across a seam), and (b) the production source no longer scalar-pads a stacked
``(u, v)`` for the velocity viscosity.

Note: a non-rotated (scalar) velocity halo does NOT necessarily produce a
*rougher* field at seams — it produces a less ZONALLY-SYMMETRIC one.  The true
horizontal Laplacian of a zonal (``u_geo ∝ cos lat``) jet has a zonally-symmetric
magnitude, so the more-correct halo yields the lower non-zonal variance.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from tests.legoesm_paths import legoesm_source_path

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.grids.cubed_sphere import create_cubed_sphere  # noqa: E402
from legoesm.core.operators_fc import (  # noqa: E402
    build_fc_config, fc_pad_halo_vector,
)
from legoesm.core.operators_fc_3d import fc_laplacian_3d  # noqa: E402
from legoesm.grids.halo import pad_halo_4d  # noqa: E402
from legoesm.grids.regridding import (  # noqa: E402
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon,
)


def _nonzonal_fraction_of_magnitude(lap_u, lap_v, weights) -> float:
    """Non-zonal variance fraction of the vector-Laplacian MAGNITUDE (a
    rotation-invariant scalar) after regridding to lat-lon.  A zonally-
    symmetric input jet has a zonally-symmetric true |∇²V|, so a lower
    fraction == better zonal-symmetry preservation == more correct."""
    mag = np.asarray(jnp.sqrt(lap_u[..., 0] ** 2 + lap_v[..., 0] ** 2))
    ll = apply_cubedsphere_to_latlon(mag, weights)
    zm = np.nanmean(ll, axis=1, keepdims=True)
    tot = np.nansum((ll - np.nanmean(ll)) ** 2)
    return float(np.nansum((ll - zm) ** 2) / max(tot, 1e-30))


def test_fc_velocity_viscosity_vector_halo_preserves_zonal_symmetry():
    n, nlev = 24, 4
    grid = create_cubed_sphere(n)
    fc_config = build_fc_config(d=2, C=4, degree=5)

    # Smooth geographically-zonal jet (u_geo ∝ cos lat, v_geo = 0) rotated into
    # the cube cell-centre face-local basis — its face-local components are
    # discontinuous across panel seams, exposing a non-rotated velocity halo.
    lat = jnp.asarray(grid.lat)
    ca = jnp.asarray(grid.cos_angle)[..., None]
    sa = jnp.asarray(grid.sin_angle)[..., None]
    u_geo = (0.5 * jnp.cos(lat))[..., None] * jnp.ones((1, 1, 1, nlev))
    u = ca * u_geo
    v = -sa * u_geo
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)

    # Vector (rotation-aware) halo — what the fix uses.
    u_pad, v_pad = fc_pad_halo_vector(u, v, grid)
    lap_u_vec = fc_laplacian_3d(u, grid, fc_config, padded=u_pad)
    lap_v_vec = fc_laplacian_3d(v, grid, fc_config, padded=v_pad)
    nz_vec = _nonzonal_fraction_of_magnitude(lap_u_vec, lap_v_vec, weights)

    # Scalar halo on a stacked (u, v) — the bug.
    stk = jnp.stack([u, v], axis=-1).reshape(6, n, n, nlev * 2)
    stk_pad = pad_halo_4d(stk, halo=1, interp_offsets=grid.halo_interp_offsets)
    lap_stk = fc_laplacian_3d(stk, grid, fc_config, padded=stk_pad).reshape(
        6, n, n, nlev, 2)
    nz_scalar = _nonzonal_fraction_of_magnitude(
        lap_stk[..., 0], lap_stk[..., 1], weights)

    assert np.isfinite(nz_vec) and np.isfinite(nz_scalar)
    assert nz_vec < nz_scalar - 0.05, (
        f"vector-halo non-zonal fraction {nz_vec:.3f} is not clearly below the "
        f"scalar-halo {nz_scalar:.3f}: the FC velocity Laplacian halo is no "
        "longer rotation-aware — a scalar halo on stacked (u, v) breaks the "
        "zonal symmetry of the viscous tendency.  Use fc_pad_halo_vector."
    )


def test_fc_velocity_viscosity_source_uses_vector_halo():
    """Source guard: the velocity viscosity must NOT scalar-pad a stacked
    (u, v); it must use fc_pad_halo_vector."""
    src = legoesm_source_path("ocean/dynamics/ocean_pe_fc.py").read_text()
    # No stacked-(u,v) -> scalar pad_halo_4d for the velocity viscosity.
    bad = re.search(
        r"jnp\.stack\(\s*\[\s*u\s*\*\s*mask[^\]]*,\s*v\s*\*\s*mask[^\]]*\]"
        r"[\s\S]{0,400}?pad_halo_4d",
        src,
    )
    assert bad is None, (
        "found a stacked-(u, v) -> scalar pad_halo_4d velocity viscosity halo "
        "in ocean_pe_fc.py — this re-introduces the cube panel-seam momentum "
        "imprint; use fc_pad_halo_vector per component."
    )
    assert src.count("fc_pad_halo_vector(") >= 3, (
        "the FC velocity viscosity must vector-pad (u, v) via "
        "fc_pad_halo_vector for the Laplacian and the biharmonic inner/outer "
        "Laplacian."
    )
