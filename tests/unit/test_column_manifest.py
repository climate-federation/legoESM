"""Unit tests for :mod:`legoesm.training.column_manifest`.

Pins Stage 3 of ``docs/COMPARE_REANALYSIS.md``: worst-column manifest assembly
+ environment tagging (SST, CAPE, bulk shear), index/coordinate bookkeeping,
the valid-column drop, and JSON round-trip.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.column_era5_metrics import ColumnErrorFields
from legoesm.training.column_manifest import (
    ColumnEnvironmentFields,
    EnvironmentConfig,
    build_worst_column_manifest,
    compute_bulk_shear,
    compute_column_environment,
    dicts_to_manifest,
    manifest_to_dicts,
    read_manifest,
    write_manifest,
)


def _make_error_fields(combined):
    combined = jnp.asarray(combined)
    z = jnp.zeros_like(combined)
    return ColumnErrorFields(
        T_rmse_K=combined,  # reuse so records carry recognizable values
        qv_rmse_kg_kg=z,
        wind_rmse_m_s=z,
        precip_err_mm_day=z,
        combined_score=combined,
    )


def _make_env(shape):
    return ColumnEnvironmentFields(
        sst_K=jnp.full(shape, 300.0),
        cape_J_kg=jnp.full(shape, 1500.0),
        bulk_shear_m_s=jnp.full(shape, 10.0),
    )


def test_bulk_shear_vector_difference():
    nlev = 4
    sigma = jnp.array([0.2, 0.4, 0.7, 0.95])  # surface-last
    # lower ~0.85 -> level 3 (0.95); upper ~0.25 -> level 0 (0.2)
    u = jnp.zeros((1, nlev)).at[0, 0].set(3.0)  # upper u=3
    v = jnp.zeros((1, nlev)).at[0, 0].set(4.0)  # upper v=4 (lower 0)
    shear = float(compute_bulk_shear(u, v, sigma)[0])
    assert shear == pytest.approx(5.0, abs=1e-10)


def test_bulk_shear_level_selection():
    sigma = jnp.array([0.1, 0.3, 0.5, 0.85])
    cfg = EnvironmentConfig(shear_lower_sigma=0.85, shear_upper_sigma=0.1)
    u = jnp.array([[10.0, 0.0, 0.0, 2.0]])  # upper(idx0)=10, lower(idx3)=2
    v = jnp.zeros((1, 4))
    shear = float(compute_bulk_shear(u, v, sigma, cfg)[0])
    assert shear == pytest.approx(8.0, abs=1e-10)


def test_bulk_shear_jit_with_traced_sigma():
    """Level selection must compose under jax.jit with sigma as a traced arg."""
    import jax

    sigma = jnp.array([0.2, 0.4, 0.7, 0.95])
    u = jnp.zeros((2, 4)).at[:, 0].set(3.0)
    v = jnp.zeros((2, 4)).at[:, 0].set(4.0)
    jitted = jax.jit(compute_bulk_shear)
    shear = jitted(u, v, sigma)
    assert float(shear[0]) == pytest.approx(5.0, abs=1e-10)


def test_compute_column_environment_shapes_and_cape_sign():
    nlat, nlon, nlev = 2, 3, 6
    shape = (nlat, nlon)
    sigma = jnp.linspace(0.05, 0.98, nlev)
    p_s = 1.0e5
    p_full = jnp.broadcast_to(sigma * p_s, (nlat, nlon, nlev))
    sigma_half = jnp.concatenate(
        [jnp.array([0.0]), 0.5 * (sigma[1:] + sigma[:-1]), jnp.array([1.0])]
    )
    p_half = jnp.broadcast_to(sigma_half * p_s, (nlat, nlon, nlev + 1))
    # A conditionally unstable warm humid column -> positive CAPE.
    T = jnp.broadcast_to(jnp.linspace(220.0, 300.0, nlev), (nlat, nlon, nlev))
    q_v = jnp.broadcast_to(jnp.linspace(1e-5, 1.6e-2, nlev), (nlat, nlon, nlev))
    u = jnp.zeros((nlat, nlon, nlev))
    v = jnp.zeros((nlat, nlon, nlev))
    sst = jnp.full(shape, 301.0)

    env = compute_column_environment(
        T=T, q_v=q_v, u=u, v=v, p_full=p_full, p_half=p_half,
        sst=sst, sigma_full=sigma,
    )
    assert env.cape_J_kg.shape == shape
    assert env.sst_K.shape == shape
    assert env.bulk_shear_m_s.shape == shape
    assert float(jnp.min(env.cape_J_kg)) >= 0.0  # CAPE is non-negative
    assert float(jnp.max(env.cape_J_kg)) > 0.0


def test_compute_column_environment_cape_responds_to_p_full():
    """CAPE is the buoyancy integral over the PASSED p_full/p_half (the mechanism the iter-342
    hybrid fix relies on — `column_environment_grid` feeds the coordinate's pressures).  Two
    calls with the SAME T/q/u/v/sigma_full but DIFFERENT p_full give DIFFERENT CAPE — a VACUITY
    lock (cf. iter 345): if CAPE re-derived pressures from sigma_full internally, the iter-342
    fix would silently do nothing and this would fail."""
    nlat, nlon, nlev = 1, 1, 6
    sigma = jnp.linspace(0.05, 0.98, nlev)
    sigma_half = jnp.concatenate(
        [jnp.array([0.0]), 0.5 * (sigma[1:] + sigma[:-1]), jnp.array([1.0])])
    T = jnp.broadcast_to(jnp.linspace(220.0, 300.0, nlev), (nlat, nlon, nlev))  # noqa: N806
    q_v = jnp.broadcast_to(jnp.linspace(1e-5, 1.6e-2, nlev), (nlat, nlon, nlev))
    u = jnp.zeros((nlat, nlon, nlev))
    v = jnp.zeros((nlat, nlon, nlev))
    sst = jnp.full((nlat, nlon), 301.0)

    def _cape(p_s):     # DIFFERENT p_full via p_s, SAME sigma_full
        p_full = jnp.broadcast_to(sigma * p_s, (nlat, nlon, nlev))
        p_half = jnp.broadcast_to(sigma_half * p_s, (nlat, nlon, nlev + 1))
        env = compute_column_environment(
            T=T, q_v=q_v, u=u, v=v, p_full=p_full, p_half=p_half, sst=sst, sigma_full=sigma)
        return float(env.cape_J_kg[0, 0])

    assert abs(_cape(1.0e5) - _cape(7.0e4)) > 1.0     # CAPE responds to p_full (not ignored)


def test_compute_column_environment_places_each_columns_tags_at_its_own_cell():
    """``compute_column_environment`` must place each column's CAPE/shear at THAT
    column's grid cell — the per-column reshape correspondence.

    The shapes/sign test above uses a UNIFORM (broadcast) field — every column
    identical — so a column-mixing bug in the ``reshape(-1, nlev) → reshape(lead)``
    round-trip (or an i↔j transpose) could not be caught.  This is the computation
    analog of the iter-221 manifest-gather lock (different function, different
    reshape).  Here exactly ONE cell of a NON-SQUARE 3×4 grid — (1,2), non-origin
    and non-diagonal — carries an unstable, sheared profile and the rest are stable
    + windless, so the CAPE and bulk-shear fields must peak at (1,2) and be ~0
    elsewhere; a transpose (→(2,1)) or F-order flatten (→(0,2)) would misplace them.
    """
    nlat, nlon, nlev = 3, 4, 6
    flat = 1 * nlon + 2                                   # the (1,2) cell, row-major
    sigma = jnp.linspace(0.05, 0.98, nlev)
    p_s = 1.0e5
    p_full = jnp.broadcast_to(sigma * p_s, (nlat, nlon, nlev))
    sigma_half = jnp.concatenate(
        [jnp.array([0.0]), 0.5 * (sigma[1:] + sigma[:-1]), jnp.array([1.0])])
    p_half = jnp.broadcast_to(sigma_half * p_s, (nlat, nlon, nlev + 1))
    # Stable everywhere (surface-cold inversion), dry, windless …
    T = jnp.broadcast_to(jnp.linspace(300.0, 260.0, nlev), (nlat, nlon, nlev))
    q_v = jnp.broadcast_to(jnp.full(nlev, 1e-6), (nlat, nlon, nlev))
    u = jnp.zeros((nlat, nlon, nlev))
    v = jnp.zeros((nlat, nlon, nlev))
    # … except the single (1,2) cell: warm-humid-surface (positive CAPE) + sheared.
    T = T.at[1, 2, :].set(jnp.linspace(220.0, 300.0, nlev))
    q_v = q_v.at[1, 2, :].set(jnp.linspace(1e-5, 1.6e-2, nlev))
    u = u.at[1, 2, :].set(jnp.linspace(25.0, 0.0, nlev))   # top→surface wind gradient
    sst = 290.0 + jnp.arange(nlat * nlon, dtype=jnp.float64).reshape(nlat, nlon)

    env = compute_column_environment(
        T=T, q_v=q_v, u=u, v=v, p_full=p_full, p_half=p_half,
        sst=sst, sigma_full=sigma)

    cape = np.asarray(env.cape_J_kg).reshape(-1)
    shear = np.asarray(env.bulk_shear_m_s).reshape(-1)
    # CAPE + shear peak at (1,2) and vanish elsewhere → the per-column tags landed
    # at the right cell (not a transposed/F-order one).
    assert int(np.argmax(cape)) == flat and cape[flat] > 100.0
    assert np.allclose(np.delete(cape, flat), 0.0)
    assert int(np.argmax(shear)) == flat and shear[flat] > 1.0
    assert np.allclose(np.delete(shear, flat), 0.0)
    # SST passes through per-cell (no reshape, but lock it isn't transposed).
    assert float(np.asarray(env.sst_K)[1, 2]) == pytest.approx(290.0 + flat)


def test_build_manifest_selects_worst_and_tags():
    combined = jnp.array([[0.1, 0.9], [0.5, 0.2]])
    errs = _make_error_fields(combined)
    env = _make_env((2, 2))
    lat = jnp.array([10.0, 20.0])
    lon = jnp.array([100.0, 110.0])
    records = build_worst_column_manifest(
        error_fields=errs, environment=env,
        lat_deg=lat, lon_deg=lon, time_index=7, n=2,
    )
    assert len(records) == 2
    worst = records[0]
    assert worst.combined_score == pytest.approx(0.9)
    assert worst.grid_index == (0, 1)
    assert worst.lat_deg == pytest.approx(10.0)
    assert worst.lon_deg == pytest.approx(110.0)
    assert worst.time_index == 7
    assert worst.environment.cape_J_kg == pytest.approx(1500.0)
    assert worst.environment.sst_K == pytest.approx(300.0)


def test_build_manifest_environment_tag_matches_worst_columns_own_cell():
    """The worst column's environment tag (SST/CAPE/shear) must be gathered from
    THAT column's cell — not a transposed / wrongly-flattened neighbour.

    ``test_build_manifest_selects_worst_and_tags`` uses a UNIFORM environment, so
    its sst/cape asserts are vacuous for the GATHER: a mis-indexed env lookup (a
    C-vs-F flatten or an i↔j transpose between the ranking flat-index and the
    environment field) would still return the uniform value.  The env tag drives
    the LES forcing AND the env-kernel clustering/deploy, so a mis-gather would
    spin off the LES under a DIFFERENT column's climate and diagnose a coefficient
    for the wrong regime.  Here every cell carries a UNIQUE sst/cape/shear and the
    worst column sits at a NON-origin, NON-diagonal cell of a NON-SQUARE grid
    (1,2) so a transpose (→(2,1), out of range) or an F-order flatten (→(0,2))
    lands on a different cell with different values — a decisive catch.
    """
    nlat, nlon = 3, 4
    combined = jnp.zeros((nlat, nlon)).at[1, 2].set(9.0)   # unique worst at (1,2)
    errs = _make_error_fields(combined)
    i_idx = jnp.arange(nlat)[:, None]
    j_idx = jnp.arange(nlon)[None, :]
    env = ColumnEnvironmentFields(                          # every cell distinct
        sst_K=290.0 + 10.0 * i_idx + 1.0 * j_idx,          # (1,2) → 302
        cape_J_kg=1000.0 + 100.0 * i_idx + 10.0 * j_idx,   # (1,2) → 1120
        bulk_shear_m_s=(1.0 * i_idx + 0.1 * j_idx) + jnp.zeros((nlat, nlon)),  # (1,2) → 1.2
    )
    lat = jnp.array([10.0, 20.0, 30.0])
    lon = jnp.array([100.0, 110.0, 120.0, 130.0])
    records = build_worst_column_manifest(
        error_fields=errs, environment=env,
        lat_deg=lat, lon_deg=lon, time_index=0, n=1,
    )
    worst = records[0]
    assert worst.grid_index == (1, 2)
    assert worst.lat_deg == pytest.approx(20.0)            # lat[1], not lat[2]
    assert worst.lon_deg == pytest.approx(120.0)           # lon[2], not lon[1]
    # The env tag is THIS cell's value — a transpose/F-order gather would differ.
    assert worst.environment.sst_K == pytest.approx(302.0)
    assert worst.environment.cape_J_kg == pytest.approx(1120.0)
    assert worst.environment.bulk_shear_m_s == pytest.approx(1.2)


def test_build_manifest_drops_invalid_columns():
    combined = jnp.array([[0.1, 0.9], [0.5, 0.2]])
    errs = _make_error_fields(combined)
    env = _make_env((2, 2))
    lat = jnp.array([10.0, 20.0])
    lon = jnp.array([100.0, 110.0])
    mask = jnp.array([[True, False], [False, False]])  # one valid column
    records = build_worst_column_manifest(
        error_fields=errs, environment=env,
        lat_deg=lat, lon_deg=lon, time_index=0, n=3, valid_mask=mask,
    )
    assert len(records) == 1
    assert records[0].grid_index == (0, 0)


def test_build_manifest_accepts_grid_shaped_coords():
    combined = jnp.array([[0.1, 0.9], [0.5, 0.2]])
    errs = _make_error_fields(combined)
    env = _make_env((2, 2))
    lat2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    lon2d = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    records = build_worst_column_manifest(
        error_fields=errs, environment=env,
        lat_deg=lat2d, lon_deg=lon2d, time_index=0, n=1,
    )
    assert records[0].lat_deg == pytest.approx(2.0)  # (0,1)
    assert records[0].lon_deg == pytest.approx(6.0)


def test_build_manifest_rejects_bad_coord_shapes():
    combined = jnp.zeros((6, 4, 4))  # cubed-sphere-like; no 1-D lat/lon meaning
    errs = _make_error_fields(combined)
    env = _make_env((6, 4, 4))
    with pytest.raises(ValueError):
        build_worst_column_manifest(
            error_fields=errs, environment=env,
            lat_deg=jnp.zeros((6,)), lon_deg=jnp.zeros((4,)),
            time_index=0, n=1,
        )


def test_manifest_json_round_trip(tmp_path):
    combined = jnp.array([[0.1, 0.9], [0.5, 0.2]])
    errs = _make_error_fields(combined)
    env = _make_env((2, 2))
    lat = jnp.array([10.0, 20.0])
    lon = jnp.array([100.0, 110.0])
    records = build_worst_column_manifest(
        error_fields=errs, environment=env,
        lat_deg=lat, lon_deg=lon, time_index=3, n=2,
    )
    path = str(tmp_path / "manifest.json")
    write_manifest(records, path)
    loaded = read_manifest(path)
    assert loaded == records  # NamedTuples compare by value
    # dict layer is plain-JSON friendly.
    dicts = manifest_to_dicts(records)
    assert isinstance(dicts[0]["grid_index"], list)
    assert dicts_to_manifest(dicts) == records


def test_manifest_round_trip_preserves_nan_precip(tmp_path):
    """DEFENSIVE codec lock: the manifest must round-trip a non-finite field FAITHFULLY
    — write→read keeps a NaN precip NaN while every other field is exactly equal; it
    must NOT silently coerce NaN→0 or drop the column. (NB in PRODUCTION precip_err is
    NEVER NaN — score_columns masks a missing precip obs to 0 and sets 0 when precip is
    absent; see test_score_columns_nan_precip_obs_does_not_exclude_bad_column — so a
    valid worst column with NaN precip cannot actually arise. This guards the JSON codec
    against ANY future non-finite leaf, which the equality-based round-trip above can't
    cover since NaN != NaN.) Forward-compatible with a future null↔NaN codec change
    (the round-trip VALUE stays NaN either way)."""
    import math

    combined = jnp.array([[0.1, 0.9], [0.5, 0.2]])
    errs = ColumnErrorFields(
        T_rmse_K=combined, qv_rmse_kg_kg=jnp.zeros_like(combined),
        wind_rmse_m_s=jnp.zeros_like(combined),
        precip_err_mm_day=jnp.full_like(combined, jnp.nan),   # no ERA5 precip ⇒ NaN
        combined_score=combined)
    records = build_worst_column_manifest(
        error_fields=errs, environment=_make_env((2, 2)),
        lat_deg=jnp.array([10.0, 20.0]), lon_deg=jnp.array([100.0, 110.0]),
        time_index=3, n=2)
    assert records and all(math.isnan(r.precip_err_mm_day) for r in records)  # builder kept NaN
    path = str(tmp_path / "manifest_nan.json")
    write_manifest(records, path)
    loaded = read_manifest(path)
    assert len(loaded) == len(records)
    for lo, rec in zip(loaded, records):
        assert math.isnan(lo.precip_err_mm_day)               # NaN preserved, not coerced
        # every OTHER field exactly equal (neutralize the not-self-equal precip slot).
        assert lo._replace(precip_err_mm_day=0.0) == rec._replace(precip_err_mm_day=0.0)
