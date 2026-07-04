"""River-runoff depth spreading (NEMO sbcrnf rn_dep_max) + river-mouth
SSS-restoring gate (NEMO sbcssr (1-2*rnfmsk)).

NEMO injects river runoff uniformly over the levels down to
``min(rn_dep_max, local depth)`` and disables SSS restoring at river mouths;
legoESM previously diluted a single surface cell while the restoring fought
the plume toward coarse WOA — the Amazon SSS artifact.

Tests:
  * CONSERVATION IDENTITY: the column-integral salt tendency of the spread
    closure equals the legacy top-cell closure exactly (with and without the
    global normalization) — only the vertical distribution differs;
  * spread structure: runoff tendency uniform over the included levels, zero
    below the spread depth / on dry levels; shallow columns degrade to the
    top-cell form;
  * non-runoff channels (P-E, restoring) stay at the top cell;
  * river-mouth gate: restoring zero where runoff > threshold, unchanged
    elsewhere; None runoff = bit-identical legacy.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    normalized_virtual_salt_flux,
    runoff_spread_virtual_salt_tendency_3d,
    virtual_salt_flux,
)

S_REF, RHO0 = 35.0, 1025.0


def _fw(n, runoff_amp=2.0e-4, seed=0):
    rng = np.random.default_rng(seed)
    z = lambda a: jnp.asarray(a, dtype=jnp.float64)
    runoff = np.zeros(n)
    runoff[[2, 5]] = runoff_amp                       # two "river mouths"
    return FreshwaterForcing(
        precip=z(rng.uniform(0, 1e-4, n)),
        evap=z(rng.uniform(0, 1e-4, n)),
        runoff=z(runoff),
        ice_fw=z(np.zeros(n)),
        restoring=z(rng.uniform(-2e-5, 2e-5, n)),
    )


def _column_geometry(n=8, nlev=6):
    """Mixed-depth columns: deep, mid, shallow (single wet level)."""
    dz = np.full((n, nlev), 50.0)                     # 50 m layers, 300 m max
    dz[6, 1:] = 0.0                                   # shallow column (50 m)
    dz[7, 3:] = 0.0                                   # mid column (150 m)
    mask = np.ones(n)
    mask[4] = 0.0                                     # one land cell
    dz[4, :] = 0.0
    area = np.full(n, 1.0e9)
    return jnp.asarray(dz), jnp.asarray(mask), jnp.asarray(area)


@pytest.mark.parametrize("normalize", [False, True])
def test_column_integral_matches_legacy_topcell(normalize):
    """sum_k dS3d[k]*h_k == dS_top_legacy * h_0 per column (exact salt
    conservation: the spread only redistributes vertically)."""
    n, nlev = 8, 6
    h_k, mask, area = _column_geometry(n, nlev)
    fw = _fw(n)
    dS3 = runoff_spread_virtual_salt_tendency_3d(
        fw, S_REF, h_k, RHO0, mask, runoff_spread_m=150.0,
        area=area, normalize=normalize)
    if normalize:
        dS_top = normalized_virtual_salt_flux(
            fw, S_REF, h_k[..., 0], RHO0, area, mask)
    else:
        dS_top = virtual_salt_flux(fw, S_REF, h_k[..., 0], RHO0)
    col_spread = np.asarray(jnp.sum(dS3 * h_k, axis=-1))
    col_legacy = np.asarray(dS_top * h_k[..., 0])
    assert np.allclose(col_spread, col_legacy, rtol=1e-12, atol=1e-18)


def test_spread_structure_uniform_and_capped():
    n, nlev = 8, 6
    h_k, mask, area = _column_geometry(n, nlev)
    fw = _fw(n)
    dS3 = np.asarray(runoff_spread_virtual_salt_tendency_3d(
        fw, S_REF, h_k, RHO0, mask, runoff_spread_m=150.0))
    # river cell 2 (deep): spread over exactly the top 3 (150 m / 50 m) levels
    r = float(fw.runoff[2])
    expect = -S_REF * r / (RHO0 * 150.0)
    # levels 1,2 carry ONLY the runoff term (non-runoff channels are top-cell)
    assert dS3[2, 1] == pytest.approx(expect, rel=1e-12)
    assert dS3[2, 2] == pytest.approx(expect, rel=1e-12)
    assert dS3[2, 3] == 0.0 and dS3[2, 5] == 0.0      # below spread depth
    # non-river cell: nothing below the surface
    assert np.all(dS3[0, 1:] == 0.0)
    # land cell: nothing anywhere
    assert np.all(dS3[4] == 0.0)


def test_straddling_level_gets_fractional_weight():
    """h=[100,100,...], spread=150: the second level carries HALF weight so
    h_rnf = 150 exactly (NEMO min(rn_dep_max, depth)), not 200 (codex MED:
    whole-level inclusion would over-spread on coarse vertical grids)."""
    n, nlev = 4, 4
    h_k = jnp.asarray(np.full((n, nlev), 100.0))
    mask = jnp.ones(n)
    runoff = np.zeros(n); runoff[1] = 2.0e-4
    z = lambda a: jnp.asarray(a, dtype=jnp.float64)
    fw = FreshwaterForcing(precip=z(np.zeros(n)), evap=z(np.zeros(n)),
                           runoff=z(runoff), ice_fw=z(np.zeros(n)),
                           restoring=z(np.zeros(n)))
    dS3 = np.asarray(runoff_spread_virtual_salt_tendency_3d(
        fw, S_REF, h_k, RHO0, mask, runoff_spread_m=150.0))
    dS_col = -S_REF * 2.0e-4 / (RHO0 * 150.0)
    assert dS3[1, 0] == pytest.approx(dS_col, rel=1e-12)          # full
    assert dS3[1, 1] == pytest.approx(0.5 * dS_col, rel=1e-12)    # half
    assert dS3[1, 2] == 0.0
    # column integral still exact
    assert float(np.sum(dS3[1] * np.asarray(h_k[1]))) == pytest.approx(
        -S_REF * 2.0e-4 / RHO0, rel=1e-12)


def test_helper_masks_land_itself():
    """Land cells return exactly zero from the helper (codex LOW), even with
    positive h_k."""
    n, nlev = 8, 4
    h_k = jnp.asarray(np.full((n, nlev), 100.0))   # land cell has h_k > 0
    mask = jnp.asarray([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    fw = _fw(n)
    dS3 = np.asarray(runoff_spread_virtual_salt_tendency_3d(
        fw, S_REF, h_k, RHO0, mask, runoff_spread_m=150.0))
    assert np.all(dS3[1] == 0.0)


def test_shallow_column_degrades_to_topcell():
    """A 50 m column spreads over its single wet level = legacy top-cell."""
    n, nlev = 8, 6
    h_k, mask, area = _column_geometry(n, nlev)
    fw = _fw(n)
    # move a river onto the shallow column 6
    runoff = np.zeros(n); runoff[6] = 3.0e-4
    fw = fw._replace(runoff=jnp.asarray(runoff))
    dS3 = np.asarray(runoff_spread_virtual_salt_tendency_3d(
        fw, S_REF, h_k, RHO0, mask, runoff_spread_m=150.0))
    dS_top = np.asarray(virtual_salt_flux(fw, S_REF, h_k[..., 0], RHO0))
    assert dS3[6, 0] == pytest.approx(dS_top[6], rel=1e-12)
    assert np.all(dS3[6, 1:] == 0.0)


def test_requires_positive_spread():
    n, nlev = 8, 6
    h_k, mask, _ = _column_geometry(n, nlev)
    with pytest.raises(ValueError, match="runoff_spread_m > 0"):
        runoff_spread_virtual_salt_tendency_3d(
            _fw(n), S_REF, h_k, RHO0, mask, runoff_spread_m=0.0)


# ---------------------------------------------------------------------------
# River-mouth restoring gate
# ---------------------------------------------------------------------------

def test_river_mouth_gate_zeroes_restoring_at_mouths():
    from legoesm.ocean.forcing.sss_restoring import (
        SSSRestoringConfig, compute_sss_restoring_flux,
    )
    n_lat, n_lon = 10, 16
    lat = jnp.broadcast_to(jnp.linspace(-60, 60, n_lat)[:, None], (n_lat, n_lon))
    lon = jnp.broadcast_to(jnp.linspace(0, 350, n_lon)[None, :], (n_lat, n_lon))
    S_t = jnp.full((n_lat, n_lon), 35.0)
    S_m = S_t + 1.0
    ice = jnp.zeros((n_lat, n_lon))
    cfg = SSSRestoringConfig(enabled=True)
    runoff = np.zeros((n_lat, n_lon))
    runoff[4, 7] = 5.0e-4                              # Amazon-like cell
    out_gated = compute_sss_restoring_flux(
        S_m, S_t, lat, lon, ice, cfg, river_runoff=jnp.asarray(runoff))
    out_legacy = compute_sss_restoring_flux(S_m, S_t, lat, lon, ice, cfg)
    g = np.asarray(out_gated["dS_dt_top"]); l = np.asarray(out_legacy["dS_dt_top"])
    assert g[4, 7] == 0.0                              # gated at the mouth
    assert l[4, 7] != 0.0
    m = np.ones_like(g, dtype=bool); m[4, 7] = False
    assert np.array_equal(g[m], l[m])                  # elsewhere identical
    # None runoff -> bit-identical legacy
    out_none = compute_sss_restoring_flux(
        S_m, S_t, lat, lon, ice, cfg, river_runoff=None)
    assert np.array_equal(np.asarray(out_none["dS_dt_top"]), l)


def test_per_cell_map_equals_scalar_per_cell():
    """A per-cell spread-depth MAP (NEMO ln_rnf_depth_ini mode) must
    reproduce, cell by cell, what the scalar mode gives when called with
    that cell's depth — same math, broadcast over cells."""
    from legoesm.ocean.freshwater import (
        runoff_spread_virtual_salt_tendency_3d,
    )
    n, nlev = 8, 6
    fw = _fw(n)
    h_k, mask, _area = _column_geometry(n, nlev)
    rng = np.random.default_rng(3)
    depth_map = jnp.asarray(rng.uniform(3.0, 200.0, mask.shape))
    got = runoff_spread_virtual_salt_tendency_3d(
        fw, 35.0, h_k, 1026.0, mask, runoff_spread_m=depth_map)
    # reference: scalar call per unique depth, assembled cell-wise
    ref = np.zeros_like(np.asarray(got))
    for idx in np.ndindex(*mask.shape):
        one = runoff_spread_virtual_salt_tendency_3d(
            fw, 35.0, h_k, 1026.0, mask,
            runoff_spread_m=float(depth_map[idx]))
        ref[idx] = np.asarray(one)[idx]
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-12)


def test_per_cell_map_conserves_column_integral():
    from legoesm.ocean.freshwater import (
        runoff_spread_virtual_salt_tendency_3d, virtual_salt_flux,
    )
    n, nlev = 8, 6
    fw = _fw(n)
    h_k, mask, _area = _column_geometry(n, nlev)
    depth_map = jnp.asarray(
        np.random.default_rng(4).uniform(1.0, 150.0, mask.shape))
    dS3 = runoff_spread_virtual_salt_tendency_3d(
        fw, 35.0, h_k, 1026.0, mask, runoff_spread_m=depth_map)
    col = jnp.sum(dS3 * h_k, axis=-1)
    dS_top_legacy = virtual_salt_flux(
        fw, S_ref=35.0, dz_0=h_k[..., 0], rho_0=1026.0)
    np.testing.assert_allclose(
        np.asarray(col), np.asarray(dS_top_legacy * h_k[..., 0]),
        rtol=1e-10)
