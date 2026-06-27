"""Ocean surface freezing-point floor (``config.freeze_floor``).

Sea-ice thermodynamic surrogate for the full-PE lat-lon C-grid ocean: with no
prognostic ice, an exposed high-latitude cell super-cools below the freezing
point of seawater (~-1.8 C), which is unphysical and 3-5 C colder than NEMO
(whose LIM ice caps SST).  The floor clamps T at ``freeze_floor_temp_c`` each
step -- the same ``jnp.maximum(T, T_freeze)`` clamp the slab oceans already use
(``simple_ocean.py``).

Tests:
  * the leaf ``_apply_freeze_floor`` floors sub-freezing cells, leaves warm /
    deep cells untouched, and preserves array shape;
  * the default ``freeze_floor_temp_c`` equals ``T_freeze_ocean - T_freeze``
    from ``constants`` (no hardcoded -1.8 literal);
  * gating: ``config.freeze_floor=False`` (default) leaves a sub-freezing state
    unchanged through ``step``; ``True`` floors it.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants


def _model(freeze_floor=False, n_lat=8, n_lon=16, **cfg_kw):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, n_barotropic_substeps=8,
        enable_runtime_checks=False, freeze_floor=freeze_floor, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def test_default_freeze_temp_from_constants():
    """Default floor = T_freeze_ocean - T_freeze (= -1.8 C), not a literal."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig.from_flat()
    expected = constants.T_freeze_ocean - constants.T_freeze
    assert cfg.freeze_floor_temp_c == pytest.approx(expected)
    assert cfg.freeze_floor_temp_c == pytest.approx(-1.8, abs=1e-9)
    assert cfg.freeze_floor is False  # off by default (bit-exact legacy)


def test_apply_freeze_floor_clamps_subfreezing_only():
    """The leaf floors sub-freezing cells; warm/deep cells are untouched."""
    state, model = _model(freeze_floor=True)
    floor = model.config.freeze_floor_temp_c
    T = np.array(state.T.data)
    # surface (k=0): a cold patch BELOW freezing + a warm patch ABOVE.
    T[..., 0] = -5.0                      # everywhere super-cooled at surface
    T[0, 0, 0] = 12.0                     # one warm surface cell
    T[..., 1:] = 3.0                      # deep cells well above freezing
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))

    out = model._apply_freeze_floor(state)
    To = np.asarray(out.T.data)

    assert To.shape == T.shape
    # sub-freezing surface cells lifted exactly to the floor
    assert np.isclose(To[1, 0, 0], floor)
    assert np.all(To[..., 0] >= floor - 1e-12)
    # the warm surface cell is unchanged (maximum keeps the larger value)
    assert np.isclose(To[0, 0, 0], 12.0)
    # deep cells (above freezing) are bit-identical -- floor is a no-op there
    assert np.allclose(To[..., 1:], T[..., 1:])


def test_apply_freeze_floor_noop_when_all_warm():
    """No cell below freezing -> output is bit-identical to input."""
    state, model = _model(freeze_floor=True)
    T = np.array(state.T.data)
    T[...] = 5.0
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    out = model._apply_freeze_floor(state)
    assert np.array_equal(np.asarray(out.T.data), T)


def test_step_gating_floor_off_vs_on():
    """A single step from a super-frozen surface: floor OFF leaves cells below
    freezing; floor ON guarantees T >= floor everywhere."""
    floor_temp = constants.T_freeze_ocean - constants.T_freeze
    dt = 900.0

    def _frozen_state(model, state):
        T = np.array(state.T.data)
        T[..., 0] = -6.0  # surface deeply super-cooled
        return state._replace(T=state.T.replace(data=jnp.asarray(T)))

    s_off, m_off = _model(freeze_floor=False)
    s_off = _frozen_state(m_off, s_off)
    out_off = m_off.step(s_off, dt)
    T_off = np.asarray(out_off.T.data[..., 0])
    # without the floor, the super-cooled surface stays below freezing
    assert np.nanmin(T_off) < floor_temp

    s_on, m_on = _model(freeze_floor=True)
    s_on = _frozen_state(m_on, s_on)
    out_on = m_on.step(s_on, dt)
    # with the floor, no SURFACE cell remains below freezing (surface-only cap)
    T_on_sfc = np.asarray(out_on.T.data[..., 0])
    finite = np.isfinite(T_on_sfc)
    assert np.all(T_on_sfc[finite] >= floor_temp - 1e-9)
    # subsurface is left to the dycore (not clamped)
    assert out_on.T.data.shape == s_on.T.data.shape
