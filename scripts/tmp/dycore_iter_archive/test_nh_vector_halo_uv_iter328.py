"""FV3_3D iter 328: opt-in FV3-faithful vector halo for the NH
cell-centre → D-grid corner interpolation of (u, v).

Audit
-----
The NH 3D path (``compressible_euler_cdgrid.py``) stores winds at
cell centres and interpolates to D-grid corners via
``interp_center_to_corner`` on a passive-stacked ``(u, v, lev)``
axis.  This applies SCALAR halo (with duogrid routing if active)
but does NOT rotate the (u, v) face-local components across cube-
face boundaries.  At cube edges the neighbouring face's e_x / e_y
basis differs from the local face's, so a scalar halo treats the
components as untransformed field values, leaving an O(1) basis-
mismatch error at cube edges that contributes directly to NH cube
imprint in u, v.

FV3 ``ext_vector`` (``fv_duogrid.F90:626-975``) rotates (u, v)
components to the neighbouring face's basis BEFORE the 4-point
average.  legoESM's ``pad_halo_vector`` / ``center_to_dgrid_vector``
implement the same rotation pattern.  PE 3D path stores winds at
corners (no center-to-corner interpolation needed), so this gap is
NH-only.

iter-328 adds opt-in flag ``use_fv3_vector_halo_uv: bool = False``.
When True, switches NH step 2 from
``interp_center_to_corner(stack(u, v))`` to
``center_to_dgrid_vector(u, v, cdgrid)``.

Tests
-----

1. ``test_baseline_bit_for_bit`` — flag=False reproduces baseline
   exactly.
2. ``test_vector_path_changes_state`` — flag=True measurably
   differs from flag=False (proves wiring active).
3. ``test_vector_path_finite`` — flag=True produces finite state.
4. ``test_vector_diff_concentrated_at_edges`` — diff between
   vector and scalar paths concentrates at panel edges (cube
   boundary cells).
5. ``test_vector_path_differentiable_at_rest`` — AD-safe at rest.
6. ``test_vector_path_with_duogrid_finite`` — vector + duogrid
   compose without explosion.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


def _build(grid, seed=328, amp=3.0):
    n = grid.n
    nlev = 5
    z_top = 30000.0
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-amp, amp, size=(6, n, n, nlev))
    v_p = rng.uniform(-amp, amp, size=(6, n, n, nlev))

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return height_coord, terrain_metric, state


def test_baseline_bit_for_bit():
    """flag=False is bit-for-bit baseline (no behavior change)."""
    grid = create_cubed_sphere(8)
    hc, tm, state = _build(grid)
    cfg_default = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    cfg_explicit_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_vector_halo_uv=False,
    )
    m_default = CDGridCompressibleEulerModel(grid, hc, tm, cfg_default)
    m_explicit_off = CDGridCompressibleEulerModel(
        grid, hc, tm, cfg_explicit_off,
    )
    s_default = m_default.step(state, 5.0)
    s_explicit = m_explicit_off.step(state, 5.0)
    np.testing.assert_array_equal(
        np.asarray(s_default.u.data), np.asarray(s_explicit.u.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_default.v.data), np.asarray(s_explicit.v.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_default.theta_prime.data),
        np.asarray(s_explicit.theta_prime.data),
    )


def test_vector_path_changes_state():
    """flag=True measurably differs from flag=False."""
    grid = create_cubed_sphere(8)
    hc, tm, state = _build(grid)
    cfg_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_vector_halo_uv=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_vector_halo_uv=True,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_off.u.data) - np.asarray(s_on.u.data),
    )))
    assert diff > 1e-12, (
        "iter-328 vector halo flag did NOT change state — wiring "
        "is silently a no-op."
    )


def test_vector_path_finite():
    """Vector halo path produces finite state."""
    grid = create_cubed_sphere(8)
    hc, tm, state = _build(grid)
    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_vector_halo_uv=True,
    )
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    s = m.step(state, 5.0)
    for fld in (s.u.data, s.v.data, s.w.data,
                s.theta_prime.data, s.rho_prime.data):
        assert np.all(np.isfinite(np.asarray(fld)))


def test_vector_diff_concentrated_at_edges():
    """Diff between vector and scalar halo paths concentrates at
    panel edges — proves the rotation impact is at cube
    boundaries (not bulk shift)."""
    grid = create_cubed_sphere(8)
    hc, tm, state = _build(grid)
    cfg_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        use_fv3_vector_halo_uv=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        use_fv3_vector_halo_uv=True,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)

    diff_v = np.asarray(s_on.v.data) - np.asarray(s_off.v.data)
    n = grid.n
    edge_width = 2
    i_idx = np.arange(n)
    edge_i = (i_idx < edge_width) | (i_idx >= n - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    edge_max = float(np.max(np.abs(diff_v[:, edge_mask, :])))
    interior_max = float(np.max(np.abs(diff_v[:, interior_mask, :])))
    assert edge_max > interior_max * 1.5, (
        f"vector vs scalar halo diff is NOT concentrated at panel "
        f"edges: edge_max={edge_max:.3e} ≤ 1.5 * interior_max="
        f"{1.5 * interior_max:.3e}.  Either the rotation is "
        f"silently affecting interior or is a bulk shift."
    )


def test_vector_path_differentiable_at_rest():
    """jax.grad flows finitely through 2 NH steps with vector halo
    flag at rest state."""
    grid = create_cubed_sphere(8)
    hc, tm, _ = _build(grid)
    n = 8
    nlev = 5
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    rest = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_vector_halo_uv=True,
    )
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)

    def loss(amp):
        s = rest._replace(
            u=rest.u.replace(data=amp * jnp.ones_like(rest.u.data)),
        )
        for _ in range(2):
            s = m.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)


def test_vector_path_with_duogrid_finite():
    """Vector halo + duogrid compose without divergence."""
    grid_duo = create_cubed_sphere(8, use_duogrid=True)
    hc, tm, state = _build(grid_duo)
    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_vector_halo_uv=True,
    )
    m = CDGridCompressibleEulerModel(grid_duo, hc, tm, cfg)
    s = m.step(state, 5.0)
    assert np.all(np.isfinite(np.asarray(s.u.data)))
    assert np.all(np.isfinite(np.asarray(s.v.data)))
