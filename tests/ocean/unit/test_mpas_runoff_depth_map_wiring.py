"""MPAS/Voronoi Arctic freshwater-retention lever: the NEMO ln_rnf_depth_ini
per-cell runoff spread-depth MAP wired into the MPAS ocean freshwater path.

The MPAS runoff path already spread the runoff dilution over a FLAT scalar
depth (``runoff_depth_spread_m``); a flat 150 m dilutes small Siberian/Arctic
rivers through the whole shelf column and leaves the Arctic SSS several PSU too
salty.  This wires the PER-CELL map (``runoff_depth_spread_map``, NEMO
``ln_rnf_depth_ini``: small rivers stay near-surface, the Amazon spreads to
~150 m) into ``MPASOceanConfig`` and routes both cores through ONE shared
selector, ``resolve_runoff_spread_arg`` (no per-grid copy-paste of the
map/scalar selection + mutual-exclusion guard).

These tests cover the NEW code only — the grid-agnostic tendency
``runoff_spread_virtual_salt_tendency_3d`` (which the MPAS state feeds as its
flattened ``(nCells,)`` / ``(nCells, nlev)`` analogue) is already exercised
per-cell in ``test_runoff_depth_spread.py``.

Sign convention: freshwater is +INTO the ocean, so the runoff virtual-salt
tendency is NEGATIVE (dilution lowers salinity); a SHALLOWER spread depth
concentrates that dilution nearer the surface (more shelf-surface freshening =
more freshwater retained at the Arctic shelf).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import types

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    resolve_runoff_spread_arg,
    runoff_spread_virtual_salt_tendency_3d,
)

S_REF, RHO0 = 35.0, 1025.0


def _fw(runoff):
    n = runoff.shape[0]
    z = jnp.zeros(n, dtype=jnp.float64)
    return FreshwaterForcing(
        precip=z, evap=z, runoff=jnp.asarray(runoff, dtype=jnp.float64),
        ice_fw=z, restoring=z)


# ---------------------------------------------------------------------------
# 1. Shared selector: map/scalar/None selection + mutual-exclusion guard
# ---------------------------------------------------------------------------

def test_resolve_selects_map():
    cfg = types.SimpleNamespace(
        runoff_depth_spread_m=0.0,
        runoff_depth_spread_map=np.array([10.0, 20.0, 30.0]))
    got = resolve_runoff_spread_arg(cfg)
    assert got is not None
    np.testing.assert_allclose(np.asarray(got), [10.0, 20.0, 30.0])


def test_resolve_selects_scalar():
    cfg = types.SimpleNamespace(
        runoff_depth_spread_m=150.0, runoff_depth_spread_map=None)
    assert resolve_runoff_spread_arg(cfg) == 150.0


def test_resolve_none_is_legacy_topcell():
    cfg = types.SimpleNamespace(
        runoff_depth_spread_m=0.0, runoff_depth_spread_map=None)
    assert resolve_runoff_spread_arg(cfg) is None
    # a config missing the fields entirely is also legacy (robust getattr)
    assert resolve_runoff_spread_arg(types.SimpleNamespace()) is None
    # runoff_depth_spread_m=None (defensive) is treated as 0.0, not a crash
    assert resolve_runoff_spread_arg(
        types.SimpleNamespace(runoff_depth_spread_m=None,
                              runoff_depth_spread_map=None)) is None


def test_resolve_map_and_scalar_mutually_exclusive():
    cfg = types.SimpleNamespace(
        runoff_depth_spread_m=150.0,
        runoff_depth_spread_map=np.array([10.0]))
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_runoff_spread_arg(cfg)


# ---------------------------------------------------------------------------
# 2. MPASOceanConfig carries the field and threads it through the selector
# ---------------------------------------------------------------------------

def test_mpas_config_has_field_defaulting_none():
    from legoesm.ocean.mpas_config import MPASOceanConfig
    assert "runoff_depth_spread_map" in MPASOceanConfig._fields
    c = MPASOceanConfig()
    assert c.runoff_depth_spread_map is None            # legacy default
    # the MPAS core reads exactly this field name via the shared selector
    h = jnp.asarray(np.full(12, 30.0))
    got = resolve_runoff_spread_arg(c._replace(runoff_depth_spread_map=h))
    np.testing.assert_allclose(np.asarray(got), np.asarray(h))


def test_mpas_config_map_scalar_mutual_exclusion_via_selector():
    from legoesm.ocean.mpas_config import MPASOceanConfig
    c = MPASOceanConfig()._replace(
        runoff_depth_spread_m=150.0,
        runoff_depth_spread_map=jnp.asarray(np.full(6, 20.0)))
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_runoff_spread_arg(c)


# ---------------------------------------------------------------------------
# 3. The retention physics on the MPAS-flattened (nCells, nlev) state
# ---------------------------------------------------------------------------

def test_shallow_map_retains_more_surface_freshwater_than_flat_deep():
    """The Arctic fix: a shallow per-cell map concentrates the runoff dilution
    at the surface (stronger negative dS at level 0) vs a flat-deep spread,
    while the deep Amazon cell is unchanged and the column integral (total
    salt tendency) is IDENTICAL — only the vertical distribution shifts."""
    n, nlev = 6, 6
    h_k = jnp.asarray(np.full((n, nlev), 50.0))          # 50 m layers, 300 m
    mask = jnp.ones(n)
    arctic, amazon = 1, 4
    runoff = np.zeros(n)
    runoff[arctic] = 1.0e-4
    runoff[amazon] = 1.0e-4
    fw = _fw(runoff)

    # (a) current MPAS lever: flat 150 m -> dilutes the whole shelf column
    dS_flat = np.asarray(runoff_spread_virtual_salt_tendency_3d(
        fw, S_REF, h_k, RHO0, mask, runoff_spread_m=150.0))
    # (b) NEMO ln_rnf_depth_ini: shallow at the Arctic cell, deep at the Amazon
    depth_map = np.full(n, 150.0)
    depth_map[arctic] = 20.0
    dS_map = np.asarray(runoff_spread_virtual_salt_tendency_3d(
        fw, S_REF, h_k, RHO0, mask, runoff_spread_m=jnp.asarray(depth_map)))

    # Arctic surface freshening is STRONGER (more negative) with the shallow map
    assert dS_map[arctic, 0] < dS_flat[arctic, 0] < 0.0
    # Amazon cell unchanged (map == flat there)
    np.testing.assert_allclose(dS_map[amazon], dS_flat[amazon], rtol=1e-12)
    # conservation: column-integral salt tendency identical for both closures
    col_flat = np.sum(dS_flat * np.asarray(h_k), axis=-1)
    col_map = np.sum(dS_map * np.asarray(h_k), axis=-1)
    np.testing.assert_allclose(col_map, col_flat, rtol=1e-12)


def test_end_to_end_nemo_map_from_climatology_on_mpas_shapes():
    """runoff_depth.nemo_runoff_depth_map (grid-agnostic, (nCells,) out) ->
    MPASOceanConfig -> resolve_runoff_spread_arg -> tendency: the full lever
    path the run_omip_core2 gate now wires for --grid mpas."""
    from legoesm.ocean.forcing.runoff_depth import nemo_runoff_depth_map
    from legoesm.ocean.mpas_config import MPASOceanConfig
    n, nlev = 6, 6
    runoff_monthly = np.zeros((12, n))                   # (12, nCells)
    arctic, amazon = 1, 4
    runoff_monthly[:, arctic] = 1.0e-3                   # small Siberian river
    runoff_monthly[:, amazon] = 0.05                     # Amazon ~ rn_rnf_max
    H_bathy = np.full(n, 4000.0)                          # (nCells,)
    h_rnf = nemo_runoff_depth_map(runoff_monthly, H_bathy)
    assert h_rnf.shape == (n,)
    assert h_rnf[arctic] < 10.0                          # Arctic stays shallow
    assert h_rnf[amazon] == pytest.approx(150.0, rel=1e-6)   # Amazon -> 150 m

    cfg = MPASOceanConfig()._replace(
        runoff_depth_spread_map=jnp.asarray(h_rnf))
    arg = resolve_runoff_spread_arg(cfg)
    assert arg is not None

    h_k = jnp.asarray(np.full((n, nlev), 50.0))
    mask = jnp.ones(n)
    fw = _fw(runoff_monthly[0])
    dS = np.asarray(runoff_spread_virtual_salt_tendency_3d(
        fw, S_REF, h_k, RHO0, mask, runoff_spread_m=arg))
    # Arctic runoff stays in the top cell (shallow plume); nothing below 50 m
    assert dS[arctic, 0] < 0.0
    assert dS[arctic, 2] == 0.0
    # Amazon spreads to 150 m -> reaches level 2 (100-150 m)
    assert dS[amazon, 2] < 0.0
