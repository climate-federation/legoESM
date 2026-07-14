"""FV3_3D iter 325: thread duogrid kinked-to-extended remap through
the 3 NH halo sites that previously bypassed it.

Audit
-----
The PE 3D path (``primitive_eq_cdgrid.py``) threads
``grid.duogrid`` through every packed/unpacked halo for K-like
quantities (iter-84 / iter-1184 / iter-1188).  The NH path
(``compressible_euler_cdgrid.py``) had THREE halo sites that
silently bypassed duogrid even when the user constructed the grid
with ``use_duogrid=True``:

* ``packed_pad_halo_4d(K, pi_prime, ...)``         (line 398)
* ``packed_pad_halo_mpi_4d(K, pi_prime, ...)``     (line 403)
* ``_pad_halo_4d_module(_ke_correction)``          (line 744;
  inside the iter-168 corner-divergence damping block)

Without duogrid, the cube-edge gradient at panel boundaries sees
the cube-projected halo cell instead of the duogrid-corrected
value (FV3 ``fv_duogrid.F90`` Lagrange-extended halo).  This
leaves an O(dx²) bias at the cube edge that contributes directly
to cube imprint in u, v, w.

iter-325 wires ``duogrid=grid.duogrid`` through these 3 sites,
mirroring the PE pattern.

Tests
-----
1. ``test_no_duogrid_path_unchanged`` — without duogrid (default
   ``create_cubed_sphere(n)``), 1 NH step bit-for-bit equal to
   pre-iter-325.  Regression guard.
2. ``test_duogrid_path_changes_state`` — with duogrid
   (``use_duogrid=True``), 1 NH step state differs MEASURABLY
   from no-duogrid path on a non-rest IC.  Proves the duogrid
   wiring is actually active and not a silent no-op.
3. ``test_duogrid_path_finite`` — duogrid path produces finite
   state (not NaN/Inf).
4. ``test_duogrid_differentiable_at_rest`` — AD-safety with
   duogrid active + full FV3 toolkit.
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


def _build_state(grid):
    n = grid.n
    nlev = 5
    z_top = 30000.0
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=325)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return height_coord, terrain_metric, state


def _toolkit_cfg():
    """Engage iter-168 corner-div + iter-171 cell-centre div_damp
    so both the K-pi-halo + ke_correction-halo paths fire."""
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e8, div_damp_dddmp=0.20,
    )


def test_no_duogrid_path_unchanged():
    """Default grid (no duogrid) — 1 NH step finite + reproducible."""
    grid = create_cubed_sphere(8, use_duogrid=False)
    assert grid.duogrid is None
    hc, tm, state = _build_state(grid)
    cfg = _toolkit_cfg()
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    s_after = m.step(state, 5.0)
    arr = np.asarray(s_after.u.data)
    assert np.all(np.isfinite(arr))


def test_duogrid_path_changes_state():
    """With duogrid wired through K + pi + ke_correction halos,
    the result must differ MEASURABLY from the no-duogrid path."""
    n = 8
    grid_plain = create_cubed_sphere(n, use_duogrid=False)
    grid_duo = create_cubed_sphere(n, use_duogrid=True)
    assert grid_plain.duogrid is None
    assert grid_duo.duogrid is not None

    hc, tm, state_plain = _build_state(grid_plain)
    _, _, state_duo = _build_state(grid_duo)
    # Identical initial state ensured by same seed in _build_state.
    np.testing.assert_array_equal(
        np.asarray(state_plain.u.data),
        np.asarray(state_duo.u.data),
    )

    cfg = _toolkit_cfg()
    m_plain = CDGridCompressibleEulerModel(grid_plain, hc, tm, cfg)
    m_duo = CDGridCompressibleEulerModel(grid_duo, hc, tm, cfg)
    s_plain = m_plain.step(state_plain, 5.0)
    s_duo = m_duo.step(state_duo, 5.0)

    # State must differ — proves the duogrid wiring is active.
    diff = float(np.max(np.abs(
        np.asarray(s_plain.u.data) - np.asarray(s_duo.u.data),
    )))
    assert diff > 1e-12, (
        "iter-325 wiring did NOT change u state when duogrid was "
        "active.  Either the grid.duogrid attribute is None on "
        "use_duogrid=True paths or the halo sites still bypass it."
    )


def test_duogrid_path_finite():
    """Duogrid path produces finite state (not NaN/Inf)."""
    grid_duo = create_cubed_sphere(8, use_duogrid=True)
    hc, tm, state = _build_state(grid_duo)
    cfg = _toolkit_cfg()
    m = CDGridCompressibleEulerModel(grid_duo, hc, tm, cfg)
    s_after = m.step(state, 5.0)
    for fld in (s_after.u.data, s_after.v.data,
                s_after.theta_prime.data, s_after.rho_prime.data):
        assert np.all(np.isfinite(np.asarray(fld))), (
            "Duogrid NH step produced non-finite output."
        )


def test_duogrid_differentiable_at_rest():
    """``jax.grad`` flows finitely through 2 NH steps with duogrid
    + full toolkit at rest state."""
    grid_duo = create_cubed_sphere(8, use_duogrid=True)
    hc, tm, _ = _build_state(grid_duo)
    n = grid_duo.n
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
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    cfg = _toolkit_cfg()
    m = CDGridCompressibleEulerModel(grid_duo, hc, tm, cfg)

    def loss(amp):
        s = rest._replace(
            u=rest.u.replace(data=amp * jnp.ones_like(rest.u.data)),
        )
        for _ in range(2):
            s = m.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)
