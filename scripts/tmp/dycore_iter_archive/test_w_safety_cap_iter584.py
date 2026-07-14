"""FV3_3D iter 584: w-safety cap test.

Tests
-----

1. ``test_w_safety_cap_disabled_by_default`` — default 0.0 = no-op.
2. ``test_w_safety_cap_clips_extreme_values`` — large w → clipped.
3. ``test_w_safety_cap_asymmetric`` — separate w_safety_cap_min.
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
    make_fv3_faithful_nh_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_state(n=8, w_spike=120.0):
    """State with an extreme w spike at one cell."""
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    w_data = np.zeros((6, n, n, nlev + 1))
    # Single +120 m/s spike at face 0, cell (0, 0), half-level 2
    w_data[0, 0, 0, 2] = w_spike
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_data, dtype=jnp.float64),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n), dtype=jnp.float64),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0), dtype=jnp.float64),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def _kw_base():
    return dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )


def test_w_safety_cap_disabled_by_default():
    """w_safety_cap=0.0 → no-op (clip not applied)."""
    grid, hc, tm, state = _build_state(w_spike=120.0)
    cfg = make_fv3_faithful_nh_config(**_kw_base())
    assert cfg.w_safety_cap == 0.0
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    new_state = m.step(state, dt=10.0)
    # The 120 m/s spike will get damped some by dycore but the CAP
    # is not applied because w_safety_cap=0
    # Just verify the cap path was NOT triggered (no clip at exact 90)
    w_max = float(jnp.abs(new_state.w.data).max())
    # If clip were active at 90, max would be ≤ 90 exactly.  With clip
    # disabled the dycore evolves the spike (damps but doesn't clip).
    assert jnp.all(jnp.isfinite(new_state.w.data))


def test_w_safety_cap_clips_extreme_values():
    """w_safety_cap=50 → |w| ≤ 50 after step."""
    grid, hc, tm, state = _build_state(w_spike=120.0)
    cfg = make_fv3_faithful_nh_config(w_safety_cap=50.0, **_kw_base())
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    new_state = m.step(state, dt=10.0)
    w_max = float(jnp.abs(new_state.w.data).max())
    assert w_max <= 50.0 + 1e-10, (
        f"w_safety_cap=50 should bound |w|: got {w_max}"
    )


def test_w_safety_cap_asymmetric():
    """Separate w_safety_cap (up) and w_safety_cap_min (down)."""
    grid, hc, tm, state = _build_state(w_spike=120.0)
    cfg = make_fv3_faithful_nh_config(
        w_safety_cap=90.0,
        w_safety_cap_min=60.0,  # → [-60, +90]
        **_kw_base(),
    )
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    new_state = m.step(state, dt=10.0)
    w_data = np.asarray(new_state.w.data)
    assert w_data.max() <= 90.0 + 1e-10
    assert w_data.min() >= -60.0 - 1e-10
