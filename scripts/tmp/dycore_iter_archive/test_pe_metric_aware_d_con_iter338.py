"""FV3_3D iter 338: opt-in metric-aware d_con form for PE
``damp_v_d_con`` site (closes gap #1 partial).

Audit
-----
iter-208 PE ``damp_v_d_con`` (and iter-209 NH analogue) uses the
simpler form ``ΔKE = u·du + 0.5·du² + v·dv + 0.5·dv²`` at corners
+ ``interp_corner_to_center``.  iter-238 audit documented this as
a fidelity gap vs FV3 ``sw_core.F90:1980``::

    heat = -0.25 * d_con * rsin2 * (
        sum_4_edges(ub², vb²) + 2 * sum_4_edges(gy, gx)
        - cosa_s * (u2 * dv2 + v2 * du2 + du2 * dv2))

The simpler form conserves GLOBAL energy exactly; LOCAL heat
distribution differs at cube edges where ``cosa_s ≠ 0``.  iter-324
verified the metric prerequisites (``cosa_cell``, ``rsin2_cell``,
``rdxa``, ``rdya``) are available on ``CubedSphereCDGrid``.

iter-338 adds opt-in ``use_fv3_metric_aware_d_con: bool = False``
to PE config.  When True, the iter-208 site switches to the
FV3-faithful metric-aware form.  Edge-stagger ``rdx`` / ``rdy``
approximated via cell-centre ``rdxa`` / ``rdya`` broadcast
(documented limitation; FV3 has edge-native ``rdx`` / ``rdy``).

Tests
-----

1. ``test_baseline_bit_for_bit`` — flag=False reproduces baseline.
2. ``test_metric_changes_T`` — flag=True changes T measurably
   when damp_v_d_con active.
3. ``test_metric_diff_concentrated_at_edges`` — diff between
   metric and simple forms concentrates at panel edges (where
   cosa_s differs from 0).
4. ``test_metric_finite`` — flag=True produces finite tendency.
5. ``test_metric_differentiable_at_rest`` — AD-safe at rest.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def pe_state():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=338)
    n_corners = n + 1
    u_p = rng.uniform(-25.0, 25.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-25.0, 25.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def _cfg(use_metric: bool):
    return CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0, A_h=0.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        use_fv3_metric_aware_d_con=use_metric,
    )


def test_baseline_bit_for_bit(pe_state):
    """flag=False reproduces baseline bit-for-bit."""
    grid, cdgrid, coord, state = pe_state
    cfg_default = _cfg(use_metric=False)
    cfg_explicit_default = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0, A_h=0.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        # explicit False = default
    )
    m_d = CDGridPrimitiveEquationModel(grid, coord,cfg_default)
    m_e = CDGridPrimitiveEquationModel(
        grid, coord, cfg_explicit_default,
    )
    s_d = m_d.step(state, 100.0)
    s_e = m_e.step(state, 100.0)
    np.testing.assert_array_equal(
        np.asarray(s_d.T.data), np.asarray(s_e.T.data),
    )


def test_metric_changes_T(pe_state):
    """flag=True changes T measurably vs flag=False."""
    grid, cdgrid, coord, state = pe_state
    m_off = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(use_metric=False),
    )
    m_on = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(use_metric=True),
    )
    s_off = m_off.step(state, 100.0)
    s_on = m_on.step(state, 100.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.T.data) - np.asarray(s_off.T.data),
    )))
    assert diff > 1e-12, (
        "iter-338 metric-aware d_con flag did NOT change T when "
        "flipped — wiring is silent no-op."
    )


def test_metric_diff_concentrated_at_edges(pe_state):
    """Diff between metric and simple forms concentrates at panel
    edges (where cosa_s ≠ 0).  Proves the metric correction lands
    at cube boundaries, not bulk shift."""
    grid, cdgrid, coord, state = pe_state
    m_off = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(use_metric=False),
    )
    m_on = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(use_metric=True),
    )
    s_off = m_off.step(state, 100.0)
    s_on = m_on.step(state, 100.0)
    diff_T = np.asarray(s_on.T.data) - np.asarray(s_off.T.data)
    n = grid.n
    edge_width = 2
    i_idx = np.arange(n)
    edge_i = (i_idx < edge_width) | (i_idx >= n - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    edge_max = float(np.max(np.abs(diff_T[:, edge_mask, :])))
    interior_max = float(np.max(np.abs(diff_T[:, interior_mask, :])))
    # Metric correction's cosa_s factor concentrates near panel
    # boundaries where cosa_s is largest (cube vertices).  Require
    # edge dominance.
    assert edge_max > interior_max * 1.2, (
        f"Metric correction is NOT concentrated at panel edges: "
        f"edge_max={edge_max:.3e}, interior_max={interior_max:.3e}.  "
        f"cosa_s metric should make the diff edge-dominant."
    )


def test_metric_finite(pe_state):
    """flag=True produces finite T."""
    grid, cdgrid, coord, state = pe_state
    m = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(use_metric=True),
    )
    s = m.step(state, 100.0)
    assert np.all(np.isfinite(np.asarray(s.T.data)))
    assert np.all(np.isfinite(np.asarray(s.u_d.data)))


def test_metric_differentiable_at_rest(pe_state):
    """jax.grad finite through 2 PE steps with metric flag at rest."""
    grid, cdgrid, coord, state = pe_state
    rest = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )
    m = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(use_metric=True),
    )

    def loss(amp):
        s = rest._replace(
            u_d=rest.u_d.replace(
                data=amp * jnp.ones_like(rest.u_d.data),
            ),
        )
        for _ in range(2):
            s = m.step(s, 100.0)
        return jnp.mean(s.T.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)
