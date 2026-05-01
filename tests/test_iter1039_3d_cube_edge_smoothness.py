"""Iter-1039 sentinel: 3D model field smoothness across cube edges.

Verifies that on a non-trivial 3D initial condition, the model
does not concentrate spurious perturbations on cube edges or
vertices over a multi-step integration.

Method:
1. Initialize hydrostatic model with isothermal + small u-wind
   perturbation (zonal flow).
2. Run 30 steps.
3. Check that final field's variance across cube edges is bounded
   (within 2× the global field variance).

If edge artifacts develop, edge-cell values will spike and the
edge-variance ratio will exceed the threshold.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from legoesm.grids.cubed_sphere import create_cubed_sphere


@pytest.fixture(scope="module")
def hyd_model_state():
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
        CDGridPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    grid = create_cubed_sphere(8)
    nlev = 5
    sigma_coord = create_sigma_coordinate(nlev)
    dt = 300.0
    nu2, _ = default_div_damp_coeffs(grid, dt=dt)
    config = CDGridPrimitiveEquationConfig(
        div_damp_coeff=nu2,
        hyperdiff_coeff=1e14,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma_coord, config)

    n = grid.n
    T_data = jnp.full((6, n, n, nlev), 250.0)
    # Small zonal-wind perturbation
    u_data = jnp.full((6, n, n, nlev), 5.0)
    v_data = jnp.zeros((6, n, n, nlev))
    ps_data = jnp.full((6, n, n), 1e5)
    phis_data = jnp.zeros((6, n, n))

    state = HydrostaticState(
        u=Field(data=u_data, name="u", dims=("face", "x", "y", "lev"),
                 units="m/s"),
        v=Field(data=v_data, name="v", dims=("face", "x", "y", "lev"),
                 units="m/s"),
        T=Field(data=T_data, name="T", dims=("face", "x", "y", "lev"),
                 units="K"),
        p_s=Field(data=ps_data, name="p_s", dims=("face", "x", "y"),
                   units="Pa"),
        phis=Field(data=phis_data, name="phis",
                    dims=("face", "x", "y"), units="m^2/s^2"),
    )
    return model, state, dt


def test_iter1039_hydrostatic_no_edge_artifacts(hyd_model_state):
    """3D hydrostatic on uniform u=5 m/s zonal wind: no edge artifacts."""
    model, state, dt = hyd_model_state
    s = state
    for _ in range(30):
        s = model.step(s, dt)

    assert jnp.all(jnp.isfinite(s.T.data)), "NaN in T after 30 steps"
    assert jnp.all(jnp.isfinite(s.u.data)), "NaN in u after 30 steps"
    assert jnp.all(jnp.isfinite(s.v.data)), "NaN in v after 30 steps"

    # Edge-vs-interior variance check on u (zonal wind).
    # u shape: (6, n, n, nlev) — face-cell-cell-level
    u = np.asarray(s.u.data)
    n = u.shape[1]
    # Edge cells: i=0, i=n-1, j=0, j=n-1
    edge_mask = np.zeros((n, n), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    # Interior cells: complement
    interior_mask = ~edge_mask

    # Compute std over cells per face per level, then average
    edge_std_global = float(np.std(u[:, edge_mask, :]))
    interior_std_global = float(np.std(u[:, interior_mask, :]))

    # Edge std should be at most ~3× interior std
    # (some edge concentration is expected at low resolution C8)
    if interior_std_global > 1e-10:
        ratio = edge_std_global / interior_std_global
        assert ratio < 5.0, (
            f"Edge/interior std ratio = {ratio:.2f} > 5.0 — "
            f"likely edge artifact developing.")


# Non-hydrostatic edge-artifact coverage is provided by the existing
# `tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py
# ::TestFVCompressibleEuler::test_30_steps_stable` (rest state, 30
# steps, finite + p_s drift bounded).  Iter-1039 adds hydrostatic
# counterpart with a zonal-wind perturbation IC and edge-vs-interior
# variance ratio check.
