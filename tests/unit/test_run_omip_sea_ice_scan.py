"""Prognostic sea ice wired into the run_omip JRA55 scan block loop.

``--jra55-sea-ice`` makes the block scan carry ``(ocean_state, ice_state)`` and
replaces the freeze-cap SST stand-in with a slab sea-ice tile (open-ocean bulk
fluxes scale by f_ocean=1-A; the ice tile feeds basal heat / melt-freeze
freshwater / brine salt / stress to the ocean).  These tests exercise the wired
``_build_jra55_block_fn`` scan path:

* ice ON: the block returns ``(ocean_state, ice_state)``, the ocean stays
  finite, and the ice concentration stays a valid fraction;
* ice OFF (default): the block returns the ocean state alone (the carry is
  unchanged) and ``enable_freeze_cap`` is kept — i.e. wiring is opt-in and the
  default path is structurally identical.

Reuses the synthetic-cache + tiny-latlon infra from the JRA55 dispatch tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

_DAY23 = Path(__file__).resolve().parent / "test_run_omip_jra55_dispatch.py"
_spec = importlib.util.spec_from_file_location("_day23_iceomip", _DAY23)
_day23 = importlib.util.module_from_spec(_spec)
sys.modules["_day23_iceomip"] = _day23
_spec.loader.exec_module(_day23)

run_omip = _day23.run_omip
_make_synthetic_cache = _day23._make_synthetic_cache
_make_tiny_latlon_setup = _day23._make_tiny_latlon_setup
_argparse_namespace = _day23._argparse_namespace
_make_woa_like_targets = _day23._make_woa_like_targets

jax.config.update("jax_enable_x64", True)


def _setup(tmp_path, *, sea_ice: bool, n_lat=8, n_lon=16):
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                  n_records=64)
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(n_lat=n_lat,
                                                         n_lon=n_lon)
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache))
    args.jra55_sea_ice = sea_ice
    jra55_state = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    return model, state, jra55_state


def test_setup_builds_slab_ice_and_disables_freeze_cap(tmp_path):
    """--jra55-sea-ice ⇒ enable_sea_ice, a slab SeaIceConfig + zero initial ice
    state, and the freeze-cap stand-in forced OFF (no double-capping)."""
    from legoesm.ice.config import SeaIceConfig
    from legoesm.ice.state import SeaIceState

    _, _, js = _setup(tmp_path, sea_ice=True)
    assert js["enable_sea_ice"] is True
    assert js["enable_freeze_cap"] is False
    assert isinstance(js["ice_config"], SeaIceConfig)
    assert js["ice_config"].dynamics == "none"          # slab
    ice0 = js["ice_state_init"]
    assert isinstance(ice0, SeaIceState)
    assert float(jnp.max(ice0.concentration.data)) == 0.0   # zero cold start
    # Ice lives on the full ocean-surface grid (lat_2d x lon_2d broadcast).
    expected = np.broadcast_shapes(np.asarray(js["lat_2d"]).shape,
                                   np.asarray(js["lon_2d"]).shape)
    assert ice0.concentration.data.shape == expected


def test_off_default_keeps_freeze_cap_and_no_ice(tmp_path):
    _, _, js = _setup(tmp_path, sea_ice=False)
    assert js["enable_sea_ice"] is False
    assert js["enable_freeze_cap"] is True            # stand-in retained
    assert "ice_state_init" not in js


def test_scan_block_carries_and_evolves_ice(tmp_path):
    """The wired block scan returns (ocean_state, ice_state); the ocean stays
    finite and the ice concentration stays a valid [0,1] fraction."""
    model, state, js = _setup(tmp_path, sea_ice=True)
    dt = 600.0
    block_fn = run_omip._build_jra55_block_fn(model, js, dt)
    atm_stack, runoff_stack = run_omip._preload_jra55_forcing_block(
        0, 4, dt, js,
    )
    out = block_fn(state, atm_stack, runoff_stack, jnp.int32(0),
                   js["ice_state_init"])
    assert isinstance(out, tuple) and len(out) == 2, "scan must carry ice"
    new_state, new_ice = out
    assert bool(jnp.all(jnp.isfinite(new_state.T.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.S.data)))
    A = np.asarray(new_ice.concentration.data)
    assert np.all(np.isfinite(A))
    assert np.all(A >= 0.0) and np.all(A <= 1.0)
    assert np.all(np.asarray(new_ice.h_ice.data) >= 0.0)


def test_scan_block_off_returns_ocean_state_only(tmp_path):
    """Ice OFF: the block returns the ocean state alone (carry unchanged)."""
    model, state, js = _setup(tmp_path, sea_ice=False)
    dt = 600.0
    block_fn = run_omip._build_jra55_block_fn(model, js, dt)
    atm_stack, runoff_stack = run_omip._preload_jra55_forcing_block(
        0, 4, dt, js,
    )
    out = block_fn(state, atm_stack, runoff_stack, jnp.int32(0))
    # Not a (state, ice) tuple — the ocean LatLonCGridOceanState itself.
    assert hasattr(out, "T") and hasattr(out, "S")
    assert bool(jnp.all(jnp.isfinite(out.T.data)))
