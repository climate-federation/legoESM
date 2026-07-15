"""FV3_3D iter 326: quantitative cube-imprint reduction from
iter-325's duogrid wiring on the NH 3D path.

iter-325 threaded ``grid.duogrid`` through the 3 NH halo sites that
previously bypassed the FV3 ``fv_duogrid.F90`` Lagrange-extended
halo.  iter-179 / iter-265 / iter-269 verify the FV3 damping
toolkit reduces the cube-imprint metric ``edge_std /
interior_std`` on the NH path.  iter-326 closes the wiring-impact
gap: with the FV3 toolkit OFF, the duogrid path alone must produce
a smaller (or equal) imprint ratio than the no-duogrid path.

This is the most-direct quantitative test that iter-325's wiring
actually reduces cube-edge artifacts on the NH path — beyond just
"changes the state" (already verified by iter-325).

Tests
-----

1. ``test_duogrid_reduces_or_equals_imprint_ratio`` — duogrid path
   ``edge_std / interior_std`` ≤ no-duogrid path × 1.05.  Tight
   margin since the bias correction is O(dx²) at C8.
2. ``test_duogrid_diff_v_concentrated_at_edges`` — the wind diff
   between duogrid and no-duogrid paths concentrates at panel
   edges (where the halo cells differ).  Catches a regression
   that would silently route both to the same halo.
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


def _build_random(grid, seed=326):
    n = grid.n
    nlev = 5
    z_top = 30000.0
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))

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


def _edge_interior_std_ratio(state, edge_width=2):
    """iter-179 cube-imprint metric on v field."""
    v = state.v.data
    n = v.shape[1]
    i_idx = jnp.arange(n)
    edge_i = (i_idx < edge_width) | (i_idx >= n - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    v_edge = v[:, edge_mask, :].reshape(-1)
    v_interior = v[:, interior_mask, :].reshape(-1)
    edge_std = float(jnp.std(v_edge))
    interior_std = float(jnp.std(v_interior))
    return edge_std / max(interior_std, 1e-30)


def _step_n(model, state, n_steps, dt=5.0):
    s = state
    for _ in range(n_steps):
        s = model.step(s, dt)
    return s


def test_duogrid_reduces_or_equals_imprint_ratio():
    """Duogrid path produces edge_std/interior_std ≤ no-duogrid
    path × 1.05 — duogrid corrects the cube-edge halo bias so
    spurious edge wind amplification cannot exceed the no-duogrid
    path by a meaningful margin (5 % slack absorbs C8 noise)."""
    n = 8
    grid_plain = create_cubed_sphere(n, use_duogrid=False)
    grid_duo = create_cubed_sphere(n, use_duogrid=True)

    hc_p, tm_p, state_p = _build_random(grid_plain)
    hc_d, tm_d, state_d = _build_random(grid_duo)

    cfg = CDGridCompressibleEulerConfig(
        # Toolkit OFF so we isolate duogrid's halo impact.
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        corner_div_damp_d2_bg=0.0,
        damp_v=0.0,
        div_damp_coeff=0.0,
        A_h=0.0,
    )
    m_plain = CDGridCompressibleEulerModel(grid_plain, hc_p, tm_p, cfg)
    m_duo = CDGridCompressibleEulerModel(grid_duo, hc_d, tm_d, cfg)

    s_plain = _step_n(m_plain, state_p, n_steps=5)
    s_duo = _step_n(m_duo, state_d, n_steps=5)

    ratio_plain = _edge_interior_std_ratio(s_plain)
    ratio_duo = _edge_interior_std_ratio(s_duo)

    assert ratio_duo <= ratio_plain * 1.05, (
        f"iter-325 duogrid wiring did NOT reduce the NH cube-"
        f"imprint metric: ratio_duo={ratio_duo:.4f} > "
        f"1.05 * ratio_plain={1.05 * ratio_plain:.4f}.  Either "
        f"the duogrid path is amplifying edge bias (broken "
        f"wiring) or the metric is measuring something else."
    )


def test_duogrid_diff_v_concentrated_at_edges():
    """The wind difference between duogrid and no-duogrid paths
    must be larger at panel edges than the face interior — proves
    the iter-325 wiring affects the EDGE halo specifically (not a
    bulk-shift no-op)."""
    n = 8
    grid_plain = create_cubed_sphere(n, use_duogrid=False)
    grid_duo = create_cubed_sphere(n, use_duogrid=True)

    hc_p, tm_p, state_p = _build_random(grid_plain)
    hc_d, tm_d, state_d = _build_random(grid_duo)

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        # Toolkit knobs OFF — isolate duogrid impact.
        corner_div_damp_d2_bg=0.0,
        damp_v=0.0,
        div_damp_coeff=0.0,
        A_h=0.0,
    )
    m_plain = CDGridCompressibleEulerModel(grid_plain, hc_p, tm_p, cfg)
    m_duo = CDGridCompressibleEulerModel(grid_duo, hc_d, tm_d, cfg)

    s_plain = _step_n(m_plain, state_p, n_steps=3)
    s_duo = _step_n(m_duo, state_d, n_steps=3)

    diff_v = np.asarray(s_duo.v.data) - np.asarray(s_plain.v.data)
    edge_width = 2
    i_idx = np.arange(n)
    edge_i = (i_idx < edge_width) | (i_idx >= n - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    edge_max = float(np.max(np.abs(diff_v[:, edge_mask, :])))
    interior_max = float(np.max(np.abs(diff_v[:, interior_mask, :])))

    # Edge difference must be strictly larger than interior diff
    # by a meaningful margin (≥ 1.5×); duogrid's halo correction
    # propagates inward over 3 steps, but the edge cells see the
    # halo correction directly while interior sees only secondary
    # effects.
    assert edge_max > interior_max * 1.5, (
        f"duogrid - no_duogrid diff is NOT concentrated at panel "
        f"edges: edge_max={edge_max:.3e} ≤ 1.5 * interior_max="
        f"{1.5 * interior_max:.3e}.  Either the duogrid wiring is "
        f"silently affecting interior cells (which would indicate "
        f"a propagation bug) or the wiring is a global shift "
        f"(no-op for cube-edge artifact reduction)."
    )
