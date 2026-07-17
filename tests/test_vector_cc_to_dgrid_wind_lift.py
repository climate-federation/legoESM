"""Regression guard: cube cell-centre → D-grid wind lift must be vector-aware.

fv3_faithful (iter-14) root-caused the cubed-sphere baroclinic v-imprint to the
PE ``_step_cell_centre`` / ``tendencies()`` paths converting cell-centre winds to
D-grid corners with a SCALAR ``interp_center_to_corner`` on the stacked ``(u, v)``
— which blends the face-local wind components across cube panel seams WITHOUT
rotation, producing D-grid winds ~167×/83× rougher at panel edges than the
interior and a relative vorticity ~229× rougher.  The fix routes the lift through
the rotation-aware ``center_to_dgrid_vector`` (the inverse of the vector-aware
exit ``dgrid_to_center_vector``).

These tests fail if a future refactor reverts any PE cc→D-grid VECTOR lift to the
scalar path: they assert (a) the vector lift yields a near-interior-smooth
relative vorticity at panel edges for a balanced zonal jet, while the scalar lift
does not, and (b) the production source no longer scalar-interpolates stacked
(u, v) in the PE cell-centre conversion paths.
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
from legoesm.grids.cubed_sphere_cdgrid import (  # noqa: E402
    create_cubed_sphere_cdgrid,
)
from legoesm.core.operators_cdgrid import (  # noqa: E402
    interp_center_to_corner,
    center_to_dgrid_vector,
    dgrid_vorticity,
)


def _edge_roughness_ratio(z0: np.ndarray) -> float:
    """RMS of the i-direction 2nd-difference at the panel-edge columns vs the
    deep interior — a grid-scale-noise / seam-imprint metric on a (6, n, n)
    field.  ~1 for a smooth field; ≫1 when the panel edges carry grid noise.
    """
    d2 = z0[:, 2:, :] - 2.0 * z0[:, 1:-1, :] + z0[:, :-2, :]
    edge = np.concatenate([d2[:, 0:1, :].ravel(), d2[:, -1:, :].ravel()])
    interior = d2[:, 2:-2, 2:-2].ravel()
    return float(
        np.sqrt(np.mean(edge ** 2)) / np.sqrt(np.mean(interior ** 2))
    )


def _balanced_jet_cc_winds(n: int, nlev: int):
    """Smooth zonal jet (u_geo ∝ cos lat, v_geo = 0) in the cube cell-centre
    face-local basis — the field whose face-local components are discontinuous
    across panel seams and so expose the scalar-vs-vector lift difference.
    """
    grid = create_cubed_sphere(n)
    cd = create_cubed_sphere_cdgrid(grid)
    lat = jnp.asarray(grid.lat)  # (6, n, n)
    u_geo = (30.0 * jnp.cos(lat))[..., None] * jnp.ones((1, 1, 1, nlev))
    v_geo = jnp.zeros_like(u_geo)
    # Rotate geographic (east, north) → cube cell-centre face-local basis so the
    # field matches what the model carries (use the cell-centre angles).
    from legoesm.grids.cubed_sphere_cdgrid import cell_centre_angles_from_4edge
    ca, sa = cell_centre_angles_from_4edge(cd)
    ca = ca[..., None]
    sa = sa[..., None]
    u_fl = ca * u_geo + sa * v_geo
    v_fl = -sa * u_geo + ca * v_geo
    return cd, u_fl, v_fl


def test_vector_lift_vorticity_is_smooth_scalar_is_not():
    n, nlev = 36, 4
    cd, u_fl, v_fl = _balanced_jet_cc_winds(n, nlev)

    # scalar (buggy) lift: stack (u, v) and interp as scalars
    uv = jnp.stack([u_fl, v_fl], axis=-1).reshape(6, n, n, nlev * 2)
    uvd = interp_center_to_corner(uv, cd).reshape(6, n + 1, n + 1, nlev, 2)
    z_scalar = np.asarray(dgrid_vorticity(uvd[..., 0], uvd[..., 1], cd))[..., 0]

    # vector-aware (fixed) lift
    u_d, v_d = center_to_dgrid_vector(u_fl, v_fl, cd)
    z_vec = np.asarray(dgrid_vorticity(u_d, v_d, cd))[..., 0]

    r_scalar = _edge_roughness_ratio(z_scalar)
    r_vec = _edge_roughness_ratio(z_vec)

    # The scalar lift injects a large panel-edge vorticity imprint; the vector
    # lift keeps the edge roughness near the interior level.
    assert r_scalar > 25.0, (
        f"expected the scalar (u,v) lift to be panel-edge-rough, got {r_scalar:.1f}"
    )
    assert r_vec < 5.0, (
        f"vector lift relative-vorticity edge roughness {r_vec:.1f} (>5×): the "
        "cc→D-grid wind lift is no longer rotation-aware — the cube baroclinic "
        "v-imprint will return.  Use center_to_dgrid_vector, not a scalar "
        "interp_center_to_corner on stacked (u, v)."
    )
    assert r_vec < r_scalar / 5.0


def test_pe_source_uses_vector_lift_not_scalar_stacked_uv():
    """Source guard: the PE cell-centre conversion paths must not lift the
    prognostic / tendency winds with a scalar ``interp_center_to_corner`` on a
    stacked ``(u, v)``.  Catches a silent revert of the iter-14 fix.
    """
    src = legoesm_source_path(
        "atmosphere/dynamics/gcm/primitive_eq_cdgrid.py"
    ).read_text()
    # The three fixed sites must call the vector helper.
    assert src.count("center_to_dgrid_vector(") >= 3, (
        "PE cc→D-grid wind/tendency lifts must use center_to_dgrid_vector at "
        "all three sites (_step_cell_centre, tendencies() entry, vector "
        "tendency blocks)."
    )
    # No stacked-(u,v)-then-scalar-interp pattern should remain.
    bad = re.search(
        r"jnp\.stack\(\s*\[\s*[^\]]*\bu[^\]]*,\s*[^\]]*\bv[^\]]*\]"
        r"[\s\S]{0,400}?interp_center_to_corner",
        src,
    )
    assert bad is None, (
        "found a stacked-(u,v) → scalar interp_center_to_corner lift in "
        "primitive_eq_cdgrid.py — this re-introduces the cube panel-seam wind "
        "imprint; use center_to_dgrid_vector."
    )
